from __future__ import annotations
import json, random, shutil, zipfile
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch
from demo_ai import DemoAI, ModelConfig, count_parameters

ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/"build"; DATA=BUILD/"data"; ART=BUILD/"artifact"
random.seed(20260930); torch.manual_seed(20260930)
ART.mkdir(parents=True,exist_ok=True); DATA.mkdir(parents=True,exist_ok=True)
examples=[
"Explain how a REST API works and show a minimal example.",
"Explain recursion with a simple example.",
"Compare a process and a thread.",
"Explain why database indexes speed up queries.",
"Describe DNS resolution.",
"Explain the difference between a compiler and an interpreter.",
"Describe caching and invalidation.",
"Write a Python function that removes duplicates while preserving order.",
"Write a Python binary search and explain its complexity.",
"Write an async retry helper with exponential backoff.",
"Write a JavaScript debounce utility.",
"Define a TypeScript Result<T,E> type.",
"Write SQL that returns the top three orders per customer.",
"Explain why retries can amplify an outage and list mitigations.",
"Explain the difference between O(n^2) and O(n log n).",
]
text=[]
for i in range(1200):
    text.append(f"[Example {i}]\nUser: {examples[i%len(examples)]}\nAssistant: Give a correct, concise answer with code when useful.\n\n")
curriculum="".join(text)
(DATA/"curriculum.txt").write_text(curriculum,encoding="utf-8")
tmp=DATA/"tokenizer.txt"; tmp.write_text(curriculum,encoding="utf-8")
spm.SentencePieceTrainer.Train(input=str(tmp),model_prefix=str(DATA/"demo_ai"),vocab_size=8192,model_type="unigram",character_coverage=1.0,pad_id=0,unk_id=1,bos_id=2,eos_id=3,user_defined_symbols=["<|system|>","<|user|>","<|assistant|>"],byte_fallback=True,hard_vocab_limit=False)
tokenizer=DATA/"demo_ai.model"
sp=spm.SentencePieceProcessor(model_file=str(tokenizer))
ids=sp.encode(curriculum,out_type=int)+[sp.eos_id()]
np.asarray(ids,dtype=np.uint16).tofile(DATA/"train.bin")
cfg=ModelConfig(); assert count_parameters(DemoAI(cfg))==100_000_000
model=DemoAI(cfg)
opt=torch.optim.AdamW(model.parameters(),lr=2e-4,betas=(0.9,0.95),weight_decay=0.1)
train=np.memmap(DATA/"train.bin",dtype=np.uint16,mode="r")
def batch(step):
    seq=64; s=(step*seq)%(len(train)-seq-1)
    x=torch.from_numpy(train[s:s+seq].astype(np.int64))[None,:]
    y=torch.from_numpy(train[s+1:s+seq+1].astype(np.int64))[None,:]
    return x,y
model.train()
for step in range(256):
    x,y=batch(step); opt.zero_grad(set_to_none=True); logits=model(x); loss=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.size(-1)),y.reshape(-1)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    if step%32==0: print("step",step,"loss",float(loss),flush=True)
model.eval()
state=model.state_dict(); unique={k:v.half().cpu() for k,v in state.items() if k!="lm_head.weight"}
# Row-wise symmetric INT4. Tied embedding is stored once.
manifest={"format":"demo-ai-cloud-int4-v1","config":cfg.to_dict(),"tensors":{}}
with open(ART/"weights.bin","wb") as out:
    for k,t in unique.items():
        a=t.float().numpy(); one=a.ndim==1; flat=a.reshape(-1,1) if one else a.reshape(a.shape[0],-1)
        scale=np.max(np.abs(flat),axis=1).astype(np.float32); scale=np.where(scale>1e-12,scale/7.0,1e-12)
        q=np.rint(flat/scale[:,None]).clip(-8,7).astype(np.int8).reshape(-1); u=(q.astype(np.int16)+8).astype(np.uint8)
        if len(u)%2: u=np.pad(u,(0,1))
        packed=u[0::2] | (u[1::2]<<4); off=out.tell(); out.write(packed.tobytes()); so=out.tell(); out.write(scale.tobytes())
        manifest["tensors"][k]={"shape":list(a.shape),"count":int(a.size),"ndim":a.ndim,"offset":int(off),"nbytes":int(len(packed)),"scale_offset":int(so)}
(ART/"manifest.json").write_text(json.dumps(manifest,separators=(",",":")),encoding="utf-8")
shutil.copy2(tokenizer,ART/"tokenizer.model")
(ART/"config.json").write_text(json.dumps(cfg.to_dict(),indent=2)+"\n",encoding="utf-8")
MODEL_DIR=ROOT/"api"/"model"
if MODEL_DIR.exists():
    shutil.rmtree(MODEL_DIR)
shutil.copytree(ART,MODEL_DIR)
release=BUILD/"demo-ai-cloud-runtime.zip"
with zipfile.ZipFile(release,"w",compression=zipfile.ZIP_STORED) as z:
    for p in ART.iterdir(): z.write(p,p.name)
print("built",release,release.stat().st_size,"bytes")
print("deployment model ready:",MODEL_DIR)
