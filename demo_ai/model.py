from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import ModelConfig

def rotate_half(x):
    x1=x[...,::2]; x2=x[...,1::2]
    return torch.stack((-x2,x1),dim=-1).flatten(-2)

class RMSNorm(nn.Module):
    def __init__(self,dim,eps=1e-6):
        super().__init__(); self.weight=nn.Parameter(torch.ones(dim)); self.eps=eps
    def forward(self,x):
        v=x.float().pow(2).mean(dim=-1,keepdim=True)
        return self.weight*(x*torch.rsqrt(v+self.eps)).to(dtype=x.dtype)

class RotaryEmbedding(nn.Module):
    def __init__(self,head_dim,max_seq_len,theta=10000.0):
        super().__init__()
        inv=1/(theta**(torch.arange(0,head_dim,2).float()/head_dim))
        t=torch.arange(max_seq_len,dtype=torch.float32)
        f=torch.outer(t,inv)
        # rotate_half() operates on adjacent pairs, so duplicate each frequency
        # as [f0,f0,f1,f1,...], not [f0,f1,...,f0,f1,...].
        e=torch.stack((f,f),dim=-1).reshape(max_seq_len,head_dim)
        self.register_buffer("cos",e.cos()[None,None,:,:],persistent=False)
        self.register_buffer("sin",e.sin()[None,None,:,:],persistent=False)
    def forward(self,q,k,start_pos=0):
        end=start_pos+q.size(-2)
        c=self.cos[:,:,start_pos:end,:].to(dtype=q.dtype,device=q.device)
        s=self.sin[:,:,start_pos:end,:].to(dtype=q.dtype,device=q.device)
        return q*c+rotate_half(q)*s,k*c+rotate_half(k)*s

class SwiGLU(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.gate=nn.Linear(cfg.emb_dim,cfg.ffn_dim,bias=False)
        self.up=nn.Linear(cfg.emb_dim,cfg.ffn_dim,bias=False)
        self.down=nn.Linear(cfg.ffn_dim,cfg.emb_dim,bias=False)
    def forward(self,x): return self.down(F.silu(self.gate(x))*self.up(x))

class GQA(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.n_heads=cfg.n_heads; self.n_kv_heads=cfg.n_kv_heads; self.head_dim=cfg.head_dim
        self.q_proj=nn.Linear(cfg.emb_dim,cfg.emb_dim,bias=False)
        self.k_proj=nn.Linear(cfg.emb_dim,cfg.kv_dim,bias=False)
        self.v_proj=nn.Linear(cfg.emb_dim,cfg.kv_dim,bias=False)
        self.out_proj=nn.Linear(cfg.emb_dim,cfg.emb_dim,bias=False)
        self.rope=RotaryEmbedding(cfg.head_dim,cfg.context_length,cfg.rope_theta)
    def _repeat(self,x):
        return x.repeat_interleave(self.n_heads//self.n_kv_heads,dim=1)
    def forward(self,x,past_kv=None,use_cache=False):
        b,s,_=x.shape
        q=self.q_proj(x).view(b,s,self.n_heads,self.head_dim).transpose(1,2)
        k=self.k_proj(x).view(b,s,self.n_kv_heads,self.head_dim).transpose(1,2)
        v=self.v_proj(x).view(b,s,self.n_kv_heads,self.head_dim).transpose(1,2)
        past=0 if past_kv is None else past_kv[0].size(-2)
        q,k=self.rope(q,k,past)
        if past_kv is not None:
            k=torch.cat((past_kv[0],k),dim=-2); v=torch.cat((past_kv[1],v),dim=-2)
        present=(k,v) if use_cache else None
        ka=self._repeat(k); va=self._repeat(v); total=ka.size(-2)
        if past_kv is None:
            mask=None; causal=True
        else:
            qp=torch.arange(past,past+s,device=x.device)[:,None]
            kp=torch.arange(total,device=x.device)[None,:]
            mask=kp<=qp; causal=False
        y=F.scaled_dot_product_attention(q,ka,va,attn_mask=mask,is_causal=causal)
        y=y.transpose(1,2).contiguous().view(b,s,-1)
        return self.out_proj(y),present

class Block(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.norm1=RMSNorm(cfg.emb_dim); self.attn=GQA(cfg); self.norm2=RMSNorm(cfg.emb_dim); self.ffn=SwiGLU(cfg)
    def forward(self,x,past=None,use_cache=False):
        a,p=self.attn(self.norm1(x),past_kv=past,use_cache=use_cache)
        x=x+a; x=x+self.ffn(self.norm2(x)); return x,p

class DemoAI(nn.Module):
    def __init__(self,cfg=None):
        super().__init__(); self.cfg=cfg or ModelConfig()
        self.tok_emb=nn.Embedding(self.cfg.vocab_size,self.cfg.emb_dim)
        self.blocks=nn.ModuleList([Block(self.cfg) for _ in range(self.cfg.n_layers)])
        self.final_norm=RMSNorm(self.cfg.emb_dim)
        self.lm_head=nn.Linear(self.cfg.emb_dim,self.cfg.vocab_size,bias=False)
        self.lm_head.weight=self.tok_emb.weight
        self.apply(self._init)
    def _init(self,m):
        if isinstance(m,nn.Linear): nn.init.normal_(m.weight,0,0.02)
        elif isinstance(m,nn.Embedding): nn.init.normal_(m.weight,0,0.02)
        elif isinstance(m,RMSNorm): nn.init.ones_(m.weight)
    def forward(self,input_ids,targets=None,past_kvs=None,use_cache=False):
        x=self.tok_emb(input_ids); presents=[]
        for i,b in enumerate(self.blocks):
            past=None if past_kvs is None else past_kvs[i]
            x,p=b(x,past,use_cache)
            if use_cache: presents.append(p)
        logits=self.lm_head(self.final_norm(x))
        return (logits,presents) if use_cache else logits
    def num_parameters(self): return sum(p.numel() for p in self.parameters())

def count_parameters(model): return sum(p.numel() for p in model.parameters())
