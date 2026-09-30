from __future__ import annotations
import json, os, secrets, threading
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch
from demo_ai import DemoAI, ModelConfig

ROOT=Path(__file__).resolve().parent
MODEL_DIR=Path(os.getenv("DEMO_AI_MODEL_DIR",ROOT/"model"))

class QuantStore:
    def __init__(self,root):
        self.manifest=json.loads((root/"manifest.json").read_text())
        self.mm=np.memmap(root/"weights.bin",mode="r",dtype=np.uint8); self.cache={}
    def weight(self,key):
        if key in self.cache: return self.cache[key]
        m=self.manifest["tensors"][key]; raw=np.frombuffer(self.mm,dtype=np.uint8,count=m["nbytes"],offset=m["offset"])
        lo,hi=raw&15,raw>>4; vals=np.empty(raw.size*2,dtype=np.int8); vals[0::2]=lo.astype(np.int8)-8; vals[1::2]=hi.astype(np.int8)-8; vals=vals[:m["count"]].reshape(m["shape"])
        scales=np.frombuffer(self.mm,dtype=np.float32,count=m["shape"][0],offset=m["scale_offset"])
        if m.get("ndim",len(m["shape"]))==1: arr=(vals.astype(np.float16)*scales.astype(np.float16)).astype(np.float16)
        else: arr=(vals.astype(np.float16)*scales.astype(np.float16)[:,None]).astype(np.float16)
        t=torch.from_numpy(np.ascontiguousarray(arr)); self.cache[key]=t; return t

class CloudDemoAI:
    def __init__(self,root):
        self.store=QuantStore(root); self.cfg=ModelConfig(); self.model=DemoAI(self.cfg)
        state={k:self.store.weight(k) for k in self.model.state_dict().keys() if k in self.store.manifest["tensors"]}
        state["lm_head.weight"]=self.store.weight("tok_emb.weight")
        self.model.load_state_dict(state,strict=True); self.model.eval()
        self.tokenizer=spm.SentencePieceProcessor(model_file=str(root/"tokenizer.model")); self.valid_vocab=self.tokenizer.get_piece_size()
    @torch.inference_mode()
    def generate(self,prompt,max_new=96,temperature=.7,top_k=40,seed=None):
        max_new=max(1,min(int(max_new),192)); temperature=max(.05,min(float(temperature),1.5)); top_k=max(1,min(int(top_k),min(128,self.valid_vocab)))
        g=torch.Generator(device="cpu"); g.manual_seed(secrets.randbits(32) if seed is None else int(seed))
        ids=self.tokenizer.encode(prompt,out_type=int)[-self.cfg.context_length:]; ids=ids or [self.tokenizer.bos_id()]
        x=torch.tensor([ids],dtype=torch.long); past=None
        for _ in range(max_new):
            logits,past=self.model(x,past_kvs=past,use_cache=True); s=logits[:,-1,:].float()/temperature; s[:,self.valid_vocab:]=-float("inf")
            vals,inds=torch.topk(s,k=top_k,dim=-1); probs=torch.softmax(vals,dim=-1); nxt=inds.gather(1,torch.multinomial(probs,1,generator=g)); token=int(nxt.item()); ids.append(token)
            if token==self.tokenizer.eos_id() or len(ids)>=self.cfg.context_length: break
            x=nxt
        return self.tokenizer.decode(ids)

_model=None; _lock=threading.Lock()
def get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None: _model=CloudDemoAI(MODEL_DIR)
    return _model

class RateLimiter:
    def __init__(self): self.lock=threading.Lock(); self.hits={}
    def allow(self,key,limit,window=60):
        import time; now=time.time()
        with self.lock:
            a=[t for t in self.hits.get(key,[]) if now-t<window]
            if len(a)>=limit: self.hits[key]=a; return False
            a.append(now); self.hits[key]=a; return True
rate_limiter=RateLimiter()

SYSTEM="You are Demo AI, a compact 100M-parameter language model. Be helpful, direct, and concise. Do not claim capabilities you do not have."
def build_prompt(messages):
    parts=[f"### System\n{SYSTEM}\n"]
    for m in messages[-12:]:
        if not isinstance(m,dict) or m.get("role") not in {"user","assistant"}: continue
        c=str(m.get("content","")).strip()
        if c: parts.append(("### Assistant" if m["role"]=="assistant" else "### User")+f"\n{c[:8000]}\n")
    parts.append("### Assistant\n"); return "\n".join(parts)
