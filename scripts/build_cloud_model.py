from __future__ import annotations

import os
import json
import random
import shutil
import zipfile
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch

from demo_ai import DemoAI, ModelConfig, count_parameters


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
DATA = BUILD / "data"
ART = BUILD / "artifact"
random.seed(20260930)
torch.manual_seed(20260930)
ART.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

examples = [
    "Explain how a REST API works and show a minimal example.",
    "Explain recursion with a simple example.",
    "Compare a process and a thread.",
    "Explain why database indexes speed up queries.",
    "Describe DNS resolution step by step.",
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
    "Implement a breadth-first search in Python.",
    "Implement a stack and queue without external packages.",
    "Explain HTTP status codes with examples.",
    "Explain JSON serialization and parsing.",
    "Write a safe file-loading function with validation.",
]
code_templates = [
    "def add(a, b):\n    return a + b",
    "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)",
    "async def fetch_with_retry(fetch, attempts=3):\n    for _ in range(attempts):\n        try:\n            return await fetch()\n        except Exception:\n            continue",
    "SELECT customer_id, COUNT(*) AS orders FROM orders GROUP BY customer_id ORDER BY orders DESC;",
    "const debounce = (fn, delay) => { let timer; return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), delay); }; };",
]
text_parts = []
for i in range(6000):
    prompt = examples[i % len(examples)]
    code = code_templates[i % len(code_templates)]
    text_parts.append(
        f"[Example {i}]\n### User\n{prompt}\n### Assistant\n"
        f"Provide a correct answer. When useful, include this code pattern:\n{code}\n\n"
    )
curriculum = "".join(text_parts)
(DATA / "curriculum.txt").write_text(curriculum, encoding="utf-8")
token_text = DATA / "tokenizer.txt"
token_text.write_text(curriculum, encoding="utf-8")

spm.SentencePieceTrainer.Train(
    input=str(token_text),
    model_prefix=str(DATA / "demo_ai"),
    vocab_size=8191,
    model_type="unigram",
    character_coverage=1.0,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    user_defined_symbols=["<|system|>", "<|user|>", "<|assistant|>"],
    byte_fallback=True,
    hard_vocab_limit=False,
)
tokenizer = DATA / "demo_ai.model"
sp = spm.SentencePieceProcessor(model_file=str(tokenizer))
ids = sp.encode(curriculum, out_type=int) + [sp.eos_id()]
np.asarray(ids, dtype=np.uint16).tofile(DATA / "train.bin")

cfg = ModelConfig()
model = DemoAI(cfg)
total_params = count_parameters(model)
assert total_params == 100_000_000, total_params

opt = torch.optim.AdamW(
    model.parameters(),
    lr=2e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)
train = np.memmap(DATA / "train.bin", dtype=np.uint16, mode="r")


def batch(step):
    seq = 96
    max_start = len(train) - seq - 2
    start = (step * seq * 3) % max_start
    x = torch.from_numpy(train[start : start + seq].astype(np.int64))[None, :]
    y = torch.from_numpy(train[start + 1 : start + seq + 1].astype(np.int64))[None, :]
    return x, y


model.train()
bootstrap_steps = int(os.environ.get("DEMO_AI_BOOTSTRAP_STEPS", "64"))
print("bootstrap training steps:", bootstrap_steps, flush=True)
for step in range(bootstrap_steps):
    x, y = batch(step)
    opt.zero_grad(set_to_none=True)
    logits, aux = model(x, return_aux=True)
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
    )
    loss = ce + cfg.router_aux_loss_weight * aux
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 32 == 0:
        print(
            "step",
            step,
            "ce",
            float(ce.detach()),
            "aux",
            float(aux.detach()),
            "loss",
            float(loss.detach()),
            flush=True,
        )

model.eval()
state = model.state_dict()
unique = {k: v.half().cpu() for k, v in state.items() if k != "lm_head.weight"}
manifest = {
    "format": "demo-ai-cloud-int4-moe-v2",
    "model": "demo-ai-100m-moe",
    "config": cfg.to_dict(),
    "parameters": total_params,
    "tokenizer_pieces": sp.get_piece_size(),
    "moe": {
        "experts": cfg.n_experts,
        "top_k": cfg.top_k,
        "expert_ffn_dim": cfg.expert_ffn_dim,
        "active_expert_fraction": cfg.active_expert_fraction,
    },
    "tensors": {},
}

with open(ART / "weights.bin", "wb") as out:
    for key, tensor in unique.items():
        array = tensor.float().numpy()
        one_dimensional = array.ndim == 1
        flat = array.reshape(-1, 1) if one_dimensional else array.reshape(array.shape[0], -1)
        scales = np.max(np.abs(flat), axis=1).astype(np.float32)
        scales = np.where(scales > 1e-12, scales / 7.0, 1e-12)
        q = np.rint(flat / scales[:, None]).clip(-8, 7).astype(np.int8).reshape(-1)
        unsigned = (q.astype(np.int16) + 8).astype(np.uint8)
        if len(unsigned) % 2:
            unsigned = np.pad(unsigned, (0, 1))
        packed = unsigned[0::2] | (unsigned[1::2] << 4)
        offset = out.tell()
        out.write(packed.tobytes())
        scale_offset = out.tell()
        out.write(scales.tobytes())
        manifest["tensors"][key] = {
            "shape": list(array.shape),
            "count": int(array.size),
            "ndim": array.ndim,
            "offset": int(offset),
            "nbytes": int(len(packed)),
            "scale_offset": int(scale_offset),
        }

(ART / "manifest.json").write_text(
    json.dumps(manifest, separators=(",", ":")),
    encoding="utf-8",
)
shutil.copy2(tokenizer, ART / "tokenizer.model")
(ART / "config.json").write_text(
    json.dumps(cfg.to_dict(), indent=2) + "\n",
    encoding="utf-8",
)

model_dir = ROOT / "api" / "model"
if model_dir.exists():
    shutil.rmtree(model_dir)
shutil.copytree(ART, model_dir)

release = BUILD / "demo-ai-cloud-runtime.zip"
with zipfile.ZipFile(release, "w", compression=zipfile.ZIP_STORED) as z:
    for file in ART.iterdir():
        z.write(file, file.name)

print("built", release, release.stat().st_size, "bytes")
print("deployment model ready:", model_dir)
