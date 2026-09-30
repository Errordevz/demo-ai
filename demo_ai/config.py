from dataclasses import dataclass, asdict
from pathlib import Path
import json

@dataclass
class ModelConfig:
    vocab_size: int = 8192
    context_length: int = 1024
    emb_dim: int = 800
    n_layers: int = 17
    n_heads: int = 8
    n_kv_heads: int = 2
    ffn_dim: int = 1623
    dropout: float = 0.05
    rope_theta: float = 10000.0
    @property
    def head_dim(self): return self.emb_dim // self.n_heads
    @property
    def kv_dim(self): return self.n_kv_heads * self.head_dim
    def to_dict(self): return asdict(self)
    def save(self,path):
        Path(path).write_text(json.dumps(self.to_dict(),indent=2)+"\n",encoding="utf-8")