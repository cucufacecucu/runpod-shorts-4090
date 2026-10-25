"""python3 decode_result.py result.json short.mp4 (RunPod /status response)."""
import base64
import json
from pathlib import Path
import sys

if __name__ == "__main__":
    response = json.loads(Path(sys.argv[1]).read_text())
    if response.get("status") != "COMPLETED":
        raise SystemExit("Job is not COMPLETED")
    Path(sys.argv[2]).write_bytes(base64.b64decode(response["output"]["video_base64"], validate=True))
    print("Saved", sys.argv[2])
