"""Build-time ComfyUI MP4/audio integration test; no model download or GPU."""
import json
from pathlib import Path
import subprocess
import sys
import time
import requests
from workflow import build_workflow


# Exercise the exact production cache validation, invalidation, and retry without network.
import struct
import tempfile
import handler as worker
with tempfile.TemporaryDirectory() as directory:
    valid = Path(directory) / 'valid.safetensors'
    broken = Path(directory) / 'broken.safetensors'
    header = json.dumps({'test': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
    header += b' ' * ((-len(header)) % 8)
    valid.write_bytes(struct.pack('<Q', len(header)) + header + struct.pack('<f', 1.0))
    broken.write_bytes(b'incomplete')
    assert worker.inspect_model_file(valid) == (True, 'size=' + str(valid.stat().st_size) + ' bytes')
    assert not worker.inspect_model_file(broken)[0]
    original = worker.hf_hub_download
    calls = []
    def fake_download(**kwargs):
        calls.append(kwargs.get('force_download', False))
        return str(valid if calls[-1] else broken)
    worker.hf_hub_download = fake_download
    try:
        assert worker.download_model('test/repo', 'model.safetensors', directory, None) == str(valid)
        assert calls == [False, True], calls
        calls.clear()
        broken.write_bytes(b'incomplete')
        def always_broken(**kwargs):
            calls.append(kwargs.get('force_download', False))
            return str(broken)
        worker.hf_hub_download = always_broken
        try:
            worker.download_model('test/repo', 'model.safetensors', directory, None)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Incomplete model must not be accepted after retry')
        assert calls == [False, True], calls
    finally:
        worker.hf_hub_download = original
print('PASS: invalid model cache is deleted, one clean retry repairs it, repeated corruption fails before GPU inference', flush=True)

root = Path('/comfyui')
audio = root / 'input' / 'export-smoke.wav'
subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'anullsrc=r=48000:cl=stereo', '-t', '1', str(audio)], check=True)
process = subprocess.Popen([sys.executable, str(root / 'main.py'), '--cpu', '--listen', '127.0.0.1', '--port', '8188', '--disable-all-custom-nodes'], cwd=root)
api = 'http://127.0.0.1:8188'
try:
    for attempt in range(180):
        if process.poll() is not None:
            raise RuntimeError('ComfyUI exited during export smoke test')
        try:
            if requests.get(api + '/system_stats', timeout=2).ok:
                break
        except requests.RequestException:
            pass
        time.sleep(1)
    else:
        raise RuntimeError('CPU ComfyUI startup timed out')
    save = build_workflow('export smoke test', 42)['19']
    save['inputs']['video'] = ['3', 0]
    save['inputs']['filename_prefix'] = 'export-smoke/test'
    graph = {
        '1': {'class_type': 'EmptyImage', 'inputs': {'width': 64, 'height': 64, 'batch_size': 24, 'color': 0}},
        '2': {'class_type': 'LoadAudio', 'inputs': {'audio': audio.name}},
        '3': {'class_type': 'CreateVideo', 'inputs': {'images': ['1', 0], 'audio': ['2', 0], 'fps': 24.0}},
        '19': save,
    }
    result = requests.post(api + '/prompt', json={'prompt': graph}, timeout=30)
    result.raise_for_status()
    prompt_id = result.json()['prompt_id']
    for attempt in range(120):
        response = requests.get(api + '/history/' + prompt_id, timeout=5)
        response.raise_for_status()
        entry = response.json().get(prompt_id)
        if entry:
            if entry.get('status', {}).get('status_str') == 'error':
                raise RuntimeError('CPU export failed: ' + json.dumps(entry.get('status')))
            break
        time.sleep(1)
    else:
        raise RuntimeError('CPU MP4 export timed out')
    files = list((root / 'output' / 'export-smoke').glob('*.mp4'))
    if len(files) != 1:
        raise RuntimeError('Expected exactly one MP4 artifact')
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(files[0])]))
    streams = {stream['codec_type'] for stream in probe['streams']}
    if not {'video', 'audio'} <= streams:
        raise RuntimeError('Export is missing video or audio')
    print('PASS: actual ComfyUI SaveVideo exported MP4 with video AND audio, using production workflow inputs', flush=True)
    files[0].unlink()
finally:
    process.terminate()
    process.wait(timeout=30)
    audio.unlink(missing_ok=True)
