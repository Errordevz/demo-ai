from __future__ import annotations

import json
import math
import os
import secrets
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import sentencepiece as spm


ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("DEMO_AI_MODEL_DIR", ROOT / "model"))


class QuantStore:
    """
    Lazy INT4 -> float32 dequantization.

    Shared/attention tensors stay cached because they are used for every token.
    Expert tensors use a bounded LRU so the 100M model does not require keeping
    every expert expanded to float32 in RAM at once.
    """

    def __init__(self, root: Path, max_expert_tensors: int = 72):
        self.manifest = json.loads((root / "manifest.json").read_text())
        self.mm = np.memmap(root / "weights.bin", mode="r", dtype=np.uint8)
        self.shared_cache = {}
        self.expert_cache = OrderedDict()
        self.max_expert_tensors = max(3, int(max_expert_tensors))
        self.lock = threading.Lock()

    def _dequantize(self, meta):
        raw = np.frombuffer(
            self.mm,
            dtype=np.uint8,
            count=meta["nbytes"],
            offset=meta["offset"],
        )
        lo = raw & 0x0F
        hi = raw >> 4
        vals = np.empty(raw.size * 2, dtype=np.int8)
        vals[0::2] = lo.astype(np.int8) - 8
        vals[1::2] = hi.astype(np.int8) - 8
        vals = vals[: meta["count"]].reshape(meta["shape"])

        scales = np.frombuffer(
            self.mm,
            dtype=np.float32,
            count=meta["shape"][0],
            offset=meta["scale_offset"],
        )

        if meta.get("ndim", len(meta["shape"])) == 1:
            return vals.astype(np.float32) * scales.astype(np.float32)

        return vals.astype(np.float32) * scales.astype(np.float32)[:, None]

    def weight(self, key):
        if ".moe.experts." in key:
            with self.lock:
                cached = self.expert_cache.get(key)
                if cached is not None:
                    self.expert_cache.move_to_end(key)
                    return cached
                meta = self.manifest["tensors"][key]
                result = self._dequantize(meta)
                self.expert_cache[key] = result
                self.expert_cache.move_to_end(key)
                while len(self.expert_cache) > self.max_expert_tensors:
                    self.expert_cache.popitem(last=False)
                return result

        with self.lock:
            cached = self.shared_cache.get(key)
            if cached is not None:
                return cached
            meta = self.manifest["tensors"][key]
            result = self._dequantize(meta)
            self.shared_cache[key] = result
            return result


class CloudDemoAI:
    def __init__(self, root: Path):
        self.root = root
        self.store = QuantStore(
            root,
            max_expert_tensors=int(os.getenv("DEMO_AI_EXPERT_CACHE_TENSORS", "72")),
        )
        cfg = self.store.manifest["config"]
        self.model_name = self.store.manifest.get("model", "demo-ai-100m-moe")
        self.vocab = int(cfg["vocab_size"])
        self.context_length = int(cfg["context_length"])
        self.emb_dim = int(cfg["emb_dim"])
        self.n_layers = int(cfg["n_layers"])
        self.n_heads = int(cfg["n_heads"])
        self.n_kv_heads = int(cfg["n_kv_heads"])
        self.n_experts = int(cfg["n_experts"])
        self.top_k = int(cfg["top_k"])
        self.expert_ffn_dim = int(cfg["expert_ffn_dim"])
        self.head_dim = int(cfg.get("head_dim", self.emb_dim // self.n_heads))
        self.theta = float(cfg["rope_theta"])
        self.active_expert_fraction = self.top_k / self.n_experts

        self.sp = spm.SentencePieceProcessor(model_file=str(root / "tokenizer.model"))
        self.valid_vocab = min(self.sp.get_piece_size(), self.vocab)
        self.inv_freq = (
            1.0
            / (
                self.theta
                ** (
                    np.arange(0, self.head_dim, 2, dtype=np.float32)
                    / self.head_dim
                )
            )
        ).astype(np.float32)

    @staticmethod
    def rmsnorm(x, w):
        xf = x.astype(np.float32, copy=False)
        y = xf / np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + 1e-6)
        return y * w

    @staticmethod
    def linear(x, w):
        return x @ w.T

    def rope(self, x, pos):
        t = np.arange(pos, pos + x.shape[1], dtype=np.float32)[:, None]
        freqs = t * self.inv_freq[None, :]
        emb = np.stack((freqs, freqs), axis=-1).reshape(x.shape[1], self.head_dim)
        c = np.cos(emb)[None, :, :]
        s = np.sin(emb)[None, :, :]
        even, odd = x[..., 0::2], x[..., 1::2]
        rotated = np.stack((-odd, even), axis=-1).reshape(x.shape)
        return x * c + rotated * s

    @staticmethod
    def softmax(x):
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=-1, keepdims=True)

    def sparse_moe_token(self, x, prefix):
        router_w = self.store.weight(prefix + ".moe.router.weight")
        probs = self.softmax(self.linear(x, router_w))[0]
        top_ids = np.argpartition(probs, -self.top_k)[-self.top_k:]
        top_ids = top_ids[np.argsort(probs[top_ids])[::-1]]
        top_scores = probs[top_ids]
        top_scores = top_scores / max(float(top_scores.sum()), 1e-9)
        out = np.zeros_like(x, dtype=np.float32)

        for expert_id, mix in zip(top_ids.tolist(), top_scores.tolist()):
            ep = f"{prefix}.moe.experts.{expert_id}"
            gate = self.linear(x, self.store.weight(ep + ".gate.weight"))
            up = self.linear(x, self.store.weight(ep + ".up.weight"))
            swish = gate / (1.0 + np.exp(-np.clip(gate, -40.0, 40.0)))
            ff = swish * up
            expert_out = self.linear(ff, self.store.weight(ep + ".down.weight"))
            out += float(mix) * expert_out
        return out

    def run_token(self, token_id, pos, caches):
        x = self.store.weight("tok_emb.weight")[token_id][None, :]

        for li in range(self.n_layers):
            prefix = f"blocks.{li}"
            n1 = self.rmsnorm(x, self.store.weight(prefix + ".norm1.weight"))
            q = self.linear(
                n1, self.store.weight(prefix + ".attn.q_proj.weight")
            ).reshape(1, self.n_heads, self.head_dim).transpose(1, 0, 2)
            k = self.linear(
                n1, self.store.weight(prefix + ".attn.k_proj.weight")
            ).reshape(1, self.n_kv_heads, self.head_dim).transpose(1, 0, 2)
            v = self.linear(
                n1, self.store.weight(prefix + ".attn.v_proj.weight")
            ).reshape(1, self.n_kv_heads, self.head_dim).transpose(1, 0, 2)

            q = self.rope(q, pos)
            k = self.rope(k, pos)
            if caches[li] is None:
                keys, values = k, v
            else:
                past_k, past_v = caches[li]
                keys = np.concatenate([past_k, k], axis=1)
                values = np.concatenate([past_v, v], axis=1)
            caches[li] = (keys, values)

            key_attn = np.repeat(keys, self.n_heads // self.n_kv_heads, axis=0)
            value_attn = np.repeat(values, self.n_heads // self.n_kv_heads, axis=0)
            scores = np.matmul(q, key_attn.transpose(0, 2, 1)) / math.sqrt(self.head_dim)
            attn = self.softmax(scores)
            attn_out = (
                np.matmul(attn, value_attn)
                .transpose(1, 0, 2)
                .reshape(1, self.emb_dim)
            )
            x = x + self.linear(
                attn_out,
                self.store.weight(prefix + ".attn.out_proj.weight"),
            )

            n2 = self.rmsnorm(x, self.store.weight(prefix + ".norm2.weight"))
            x = x + self.sparse_moe_token(n2, prefix)

        x = self.rmsnorm(x, self.store.weight("final_norm.weight"))
        return self.linear(x, self.store.weight("tok_emb.weight"))[0]

    def generate(self, prompt, max_new=20, temperature=0.7, top_k=24, seed=None):
        max_new = max(1, min(int(max_new), 32))
        temperature = max(0.05, min(float(temperature), 1.5))
        top_k = max(1, min(int(top_k), min(64, self.valid_vocab)))
        prompt_ids = self.sp.encode(prompt, out_type=int)[-self.context_length:]
        prompt_ids = prompt_ids or [self.sp.bos_id()]
        caches = [None] * self.n_layers

        for pos, token in enumerate(prompt_ids[:-1]):
            self.run_token(token, pos, caches)

        last_pos = len(prompt_ids) - 1
        next_logits = self.run_token(prompt_ids[-1], last_pos, caches)
        generated = []
        rng = np.random.default_rng(
            secrets.randbits(32) if seed is None else int(seed)
        )

        for _ in range(max_new):
            scores = next_logits / temperature
            scores[self.valid_vocab:] = -np.inf
            k = min(top_k, self.valid_vocab)
            idx = np.argpartition(scores, -k)[-k:]
            vals = scores[idx]
            probs = np.exp(vals - vals.max())
            probs /= probs.sum()
            token = int(rng.choice(idx, p=probs))
            generated.append(token)
            if token == self.sp.eos_id():
                break
            if len(prompt_ids) + len(generated) >= self.context_length:
                break
            next_logits = self.run_token(
                token, last_pos + len(generated), caches
            )

        return self.sp.decode(generated)


_model = None
_lock = threading.Lock()
_inference_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _model = CloudDemoAI(MODEL_DIR)
    return _model


class RateLimiter:
    def __init__(self):
        self.lock = threading.Lock()
        self.hits = {}

    def allow(self, key, limit, window=60):
        import time
        now = time.time()
        with self.lock:
            recent = [t for t in self.hits.get(key, []) if now - t < window]
            if len(recent) >= limit:
                self.hits[key] = recent
                return False
            recent.append(now)
            self.hits[key] = recent
            return True


rate_limiter = RateLimiter()

SYSTEM = (
    "You are Demo AI, a compact 100M-parameter sparse Mixture-of-Experts "
    "language model. Be helpful, direct, and concise. Do not claim capabilities "
    "you do not have."
)


def build_prompt(messages):
    parts = [f"### System\n{SYSTEM}\n"]
    for message in messages[-12:]:
        if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
            continue
        content = str(message.get("content", "")).strip()
        if content:
            parts.append(
                ("### Assistant" if message["role"] == "assistant" else "### User")
                + f"\n{content[:8000]}\n"
            )
    parts.append("### Assistant\n")
    return "\n".join(parts)
