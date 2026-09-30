from __future__ import annotations

import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "api" / "model"
URL = "https://github.com/Errordevz/demo-ai/releases/download/cloud-latest/demo-ai-cloud-runtime.zip"

print("downloading cloud runtime:", URL, flush=True)
with urllib.request.urlopen(URL, timeout=120) as response:
    payload = response.read()

tmp = ROOT / "build" / "cloud-runtime.zip"
tmp.parent.mkdir(parents=True, exist_ok=True)
tmp.write_bytes(payload)

if MODEL_DIR.exists():
    shutil.rmtree(MODEL_DIR)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    archive.extractall(MODEL_DIR)

manifest = (MODEL_DIR / "manifest.json").read_text(encoding="utf-8")
print("cloud runtime ready:", len(payload), "bytes")
print(manifest[:500])
