"""RunPod queue worker. No public ComfyUI port and no credentials in requests."""
import base64
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
from huggingface_hub import hf_hub_download

from workflow import CHECKPOINT, ENCODER, build_workflow, validate_input

ROOT = Path("/comfyui")
VOLUME = Path("/runpod-volume")
API = "http://127.0.0.1:8188"
PROCESS = None


def prepare():
    global PROCESS
    if PROCESS is not None and PROCESS.poll() is None:
        return
    if not VOLUME.is_dir():
        raise RuntimeError("Attach a RunPod Network Volume at /runpod-volume before running jobs")
    cache = VOLUME / "hf-cache"
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
    models = [
        ("Lightricks/LTX-2", CHECKPOINT, "checkpoints", CHECKPOINT),
        ("Comfy-Org/ltx-2", "split_files/text_encoders/" + ENCODER, "text_encoders", ENCODER),
    ]
    for repo, filename, category, name in models:
        # No token is passed to the Docker build or saved in a model URL.
        path = hf_hub_download(repo_id=repo, filename=filename, cache_dir=str(cache), token=token)
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
