"""RunPod queue worker. No public ComfyUI port and no credentials in requests."""
import base64
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import uuid

import requests
import runpod
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
from huggingface_hub import hf_hub_download

from workflow import CHECKPOINT, ENCODER, build_workflow, validate_input

ROOT = Path("/comfyui")
VOLUME = Path("/runpod-volume")
API = "http://127.0.0.1:8188"
PROCESS = None


EXPECTED_CHECKPOINT_SHA256 = "8ae14327130c6ffdc87705b02c8e7654aa5c6d9a7f28a52d0acc1c30cb0d2932"


def valid_model_file(path, expected_sha256=None):
    """Validate safetensors structure and, for the checkpoint, its published SHA256."""
    from safetensors import safe_open, SafetensorError
    try:
        with safe_open(str(path), framework="pt", device="cpu") as model:
            if not list(model.keys()):
                return False
    except (OSError, ValueError, RuntimeError, SafetensorError):
        return False
    if expected_sha256:
        with open(path, "rb") as model_file:
            actual_sha256 = hashlib.file_digest(model_file, "sha256").hexdigest()
        if actual_sha256 != expected_sha256:
            print("Checkpoint SHA256 mismatch; will download a fresh copy", flush=True)
            return False
    return True


def download_model(repo, filename, cache, token, expected_sha256=None):
    print("Checking model cache: " + filename, flush=True)
    path = hf_hub_download(repo_id=repo, filename=filename, cache_dir=str(cache), token=token)
    if not valid_model_file(path, expected_sha256):
        print("Incomplete or mismatched model cache; downloading a fresh copy once: " + filename, flush=True)
        path = hf_hub_download(repo_id=repo, filename=filename, cache_dir=str(cache), token=token, force_download=True)
        if not valid_model_file(path, expected_sha256):
            raise RuntimeError("Model failed integrity validation after one fresh download: " + filename)
    print("Model file validated: " + filename + " (" + str(Path(path).stat().st_size) + " bytes)", flush=True)
    return path


def prepare():
    global PROCESS
    if PROCESS is not None and PROCESS.poll() is None:
        return
    if not VOLUME.is_dir():
        raise RuntimeError("Attach a RunPod Network Volume at /runpod-volume before running jobs")
    cache = VOLUME / "hf-cache"
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
    models = [
        ("Lightricks/LTX-2", CHECKPOINT, "checkpoints", CHECKPOINT, EXPECTED_CHECKPOINT_SHA256),
        ("Comfy-Org/ltx-2", "split_files/text_encoders/" + ENCODER, "text_encoders", ENCODER, None),
    ]
    for repo, filename, category, name, expected_sha256 in models:
        # No token is passed to the Docker build or saved in a model URL.
        path = download_model(repo, filename, cache, token, expected_sha256)
        dest = ROOT / "models" / category / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink():
            dest.unlink()
        elif dest.exists():
            raise RuntimeError("Unexpected file in the container model directory")
        dest.symlink_to(path)
    PROCESS = subprocess.Popen([
        sys.executable, str(ROOT / "main.py"), "--listen", "127.0.0.1", "--port", "8188",
        "--lowvram", "--reserve-vram", "2", "--preview-method", "none",
        "--disable-auto-launch", "--disable-all-custom-nodes",
    ], cwd=ROOT)
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        if PROCESS.poll() is not None:
            raise RuntimeError("ComfyUI stopped during startup; inspect worker logs")
        try:
            r = requests.get(API + "/system_stats", timeout=3)
            if r.ok:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    PROCESS.terminate()
    PROCESS.wait(timeout=30)
    raise RuntimeError("ComfyUI startup timed out")


def run_graph(graph):
    response = requests.post(API + "/prompt", json={"prompt": graph, "client_id": uuid.uuid4().hex}, timeout=30)
    if not response.ok:
        raise RuntimeError("ComfyUI rejected the workflow: " + response.text[:3000])
    prompt_id = response.json()["prompt_id"]
    deadline = time.monotonic() + 1500
    report_at = time.monotonic() + 30
    print("ComfyUI generation started: " + prompt_id, flush=True)
    while time.monotonic() < deadline:
        if PROCESS.poll() is not None:
            raise RuntimeError("ComfyUI stopped during generation; inspect memory usage and worker logs")
        response = requests.get(API + "/history/" + prompt_id, timeout=15)
        response.raise_for_status()
        entry = response.json().get(prompt_id)
        if entry:
            if entry.get("status", {}).get("status_str") == "error":
                messages = entry.get("status", {}).get("messages", [])
                details = next((m[1] for m in messages if m[0] == "execution_error"), {})
                raise RuntimeError("Generation failed: " + str(details.get("exception_message", "inspect logs"))[:1000])
            for output in entry.get("outputs", {}).values():
                # SaveVideo reports UI artifacts under 'images' in core ComfyUI.
                for key in ("images", "videos", "gifs"):
                    for artifact in output.get(key, []):
                        if artifact.get("type") != "output" or not artifact.get("filename", "").endswith(".mp4"):
                            continue
                        root = (ROOT / "output").resolve()
                        path = (root / artifact.get("subfolder", "") / artifact["filename"]).resolve()
                        if root not in path.parents or not path.is_file():
                            raise RuntimeError("Invalid output artifact path")
                        return path, prompt_id
            raise RuntimeError("Generation ended without an MP4 output")
        if time.monotonic() >= report_at:
            print("Waiting for ComfyUI job " + prompt_id + "; result not ready yet", flush=True)
            report_at = time.monotonic() + 30
        time.sleep(2)
    requests.post(API + "/interrupt", timeout=10).raise_for_status()
    # Avoid picking up another job while the timed-out graph is still executing.
    PROCESS.terminate()
    PROCESS.wait(timeout=30)
    raise RuntimeError("Generation timed out")


def transcode(source, destination):
    subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-t", "10", "-map", "0:v:0", "-map", "0:a:0",
        "-vf", "scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,setsar=1",
        "-r", "24", "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-maxrate", "3500k", "-bufsize", "7000k", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(destination),
    ], check=True, timeout=180)
    info = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(destination)
    ], timeout=30))
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    if video["width"] != 720 or video["height"] != 1280:
        raise RuntimeError("Unexpected output dimensions")
    if abs(float(info["format"]["duration"]) - 10) > 0.1:
        raise RuntimeError("Unexpected output duration")
    if not any(s["codec_type"] == "audio" for s in info["streams"]):
        raise RuntimeError("Output is missing its audio track")
    if destination.stat().st_size > 7 * 1024 * 1024:
        raise RuntimeError("MP4 exceeds inline response limit")


def handler(job):
    prompt, seed = validate_input(job.get("input"))
    seed = secrets.randbelow(2**63) if seed is None else seed
    prepare()
    source = destination = None
    prompt_id = None
    try:
        source, prompt_id = run_graph(build_workflow(prompt, seed))
        destination = source.with_name(uuid.uuid4().hex + "-720p.mp4")
        transcode(source, destination)
        return {
            "filename": "short-10s.mp4", "mime_type": "video/mp4",
            "width": 720, "height": 1280, "duration_seconds": 10, "fps": 24,
            "seed": seed, "video_base64": base64.b64encode(destination.read_bytes()).decode("ascii"),
            "audio_note": "Ambient/effects-only prompt applied; review output for accidental voices.",
        }
    finally:
        for path in (source, destination):
            if path:
                path.unlink(missing_ok=True)
        if prompt_id:
            try:
                requests.post(API + "/history", json={"delete": [prompt_id]}, timeout=10)
            except requests.RequestException:
                pass


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
