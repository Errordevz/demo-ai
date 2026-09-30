from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass
class ModelConfig:
    # Exact 100,000,000-parameter sparse-MoE budget:
    # vocab=8191, dim=800, layers=12, experts=4, top_k=2,
    # expert_ffn_dim=644, GQA=8 query heads / 2 KV heads.
    vocab_size: int = 8191
    context_length: int = 1024
    emb_dim: int = 800
    n_layers: int = 12
    n_heads: int = 8
    n_kv_heads: int = 2
    n_experts: int = 4
    top_k: int = 2
    expert_ffn_dim: int = 644
    router_aux_loss_weight: float = 0.02
    dropout: float = 0.05
    rope_theta: float = 10000.0

    @property
    def head_dim(self):
        return self.emb_dim // self.n_heads

    @property
    def kv_dim(self):
        return self.n_kv_heads * self.head_dim

    @property
    def active_expert_fraction(self):
        return self.top_k / self.n_experts

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
