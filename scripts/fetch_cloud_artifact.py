from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import os, tempfile
url=os.environ.get("DEMO_AI_CLOUD_ARTIFACT_URL","https://github.com/Errordevz/demo-ai/releases/download/cloud-latest/demo-ai-cloud-runtime.zip")
root=Path(__file__).resolve().parents[1]/"api"/"model"; root.mkdir(parents=True,exist_ok=True)
with tempfile.NamedTemporaryFile(suffix=".zip",delete=False) as f:
    with urlopen(url,timeout=120) as r: f.write(r.read())
    tmp=f.name
with ZipFile(tmp) as z: z.extractall(root)
Path(tmp).unlink(missing_ok=True)
print("fetched cloud artifact into",root)
