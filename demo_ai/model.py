from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = x.float().pow(2).mean(dim=-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(v + self.eps)).to(dtype=x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv)
        emb = torch.stack((freqs, freqs), dim=-1).reshape(max_seq_len, head_dim)
        self.register_buffer("cos", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0):
        end = start_pos + q.size(-2)
        c = self.cos[:, :, start_pos:end, :].to(dtype=q.dtype, device=q.device)
        s = self.sin[:, :, start_pos:end, :].to(dtype=q.dtype, device=q.device)
        return q * c + rotate_half(q) * s, k * c + rotate_half(k) * s


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class SparseMoE(nn.Module):
    """Top-k token routing. Only the selected experts execute for each token."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k
        self.router = nn.Linear(cfg.emb_dim, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(cfg.emb_dim, cfg.expert_ffn_dim) for _ in range(cfg.n_experts)]
        )

    def forward(self, x: torch.Tensor):
        b, s, d = x.shape
        flat = x.reshape(-1, d)
        router_probs = F.softmax(self.router(flat), dim=-1)
        top_values, top_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_weights = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        out = torch.zeros_like(flat)

        for expert_idx, expert in enumerate(self.experts):
            positions, slots = torch.where(top_indices == expert_idx)
            if positions.numel() == 0:
                continue
            expert_out = expert(flat.index_select(0, positions))
            weights = top_weights[positions, slots].unsqueeze(-1).to(expert_out.dtype)
            out.index_add_(0, positions, expert_out * weights)

        importance = router_probs.mean(dim=0)
        load = torch.stack([(top_indices == i).float().mean() for i in range(self.n_experts)])
        aux_loss = self.n_experts * torch.sum(importance * load)
        return out.reshape(b, s, d), aux_loss


class GQA(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.q_proj = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=False)
        self.k_proj = nn.Linear(cfg.emb_dim, cfg.kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.emb_dim, cfg.kv_dim, bias=False)
        self.out_proj = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=False)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.context_length, cfg.rope_theta)

    def _repeat(self, x: torch.Tensor):
        return x.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

    def forward(self, x: torch.Tensor, past_kv=None, use_cache=False):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        past = 0 if past_kv is None else past_kv[0].size(-2)
        q, k = self.rope(q, k, past)
        if past_kv is not None:
            k, v = torch.cat((past_kv[0], k), dim=-2), torch.cat((past_kv[1], v), dim=-2)
        present = (k, v) if use_cache else None
        ka, va = self._repeat(k), self._repeat(v)
        total = ka.size(-2)
        if past_kv is None:
            mask, causal = None, True
        else:
            qp = torch.arange(past, past + s, device=x.device)[:, None]
            kp = torch.arange(total, device=x.device)[None, :]
            mask, causal = kp <= qp, False
        y = F.scaled_dot_product_attention(q, ka, va, attn_mask=mask, is_causal=causal)
        y = y.transpose(1, 2).contiguous().view(b, s, -1)
        return self.out_proj(y), present


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.emb_dim)
        self.attn = GQA(cfg)
        self.norm2 = RMSNorm(cfg.emb_dim)
        self.moe = SparseMoE(cfg)

    def forward(self, x: torch.Tensor, past=None, use_cache=False):
        attn_out, present = self.attn(self.norm1(x), past_kv=past, use_cache=use_cache)
        x = x + attn_out
        moe_out, aux_loss = self.moe(self.norm2(x))
        return x + moe_out, present, aux_loss


class DemoAI(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.tok_emb = nn.Embedding(self.cfg.vocab_size, self.cfg.emb_dim)
        self.blocks = nn.ModuleList([Block(self.cfg) for _ in range(self.cfg.n_layers)])
        self.final_norm = RMSNorm(self.cfg.emb_dim)
        self.lm_head = nn.Linear(self.cfg.emb_dim, self.cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init)

    def _init(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, 0.0, 0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, input_ids, targets=None, past_kvs=None, use_cache=False, return_aux=False):
        x = self.tok_emb(input_ids)
        presents, aux_losses = [], []
        for i, block in enumerate(self.blocks):
            past = None if past_kvs is None else past_kvs[i]
            x, present, aux = block(x, past, use_cache)
            aux_losses.append(aux)
            if use_cache:
                presents.append(present)
        logits = self.lm_head(self.final_norm(x))
        aux_loss = torch.stack(aux_losses).mean() if aux_losses else logits.new_zeros(())
        result = (logits, presents) if use_cache else logits
        return (result, aux_loss) if return_aux else result

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
