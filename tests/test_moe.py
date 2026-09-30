from __future__ import annotations

import torch

from demo_ai import DemoAI, ModelConfig, count_parameters


def main():
    cfg = ModelConfig()
    model = DemoAI(cfg)
    total = count_parameters(model)
    assert total == 100_000_000, total

    x = torch.randint(0, cfg.vocab_size, (1, 12))
    logits, aux = model(x, return_aux=True)
    assert logits.shape == (1, 12, cfg.vocab_size)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(aux)
    assert cfg.n_experts == 4
    assert cfg.top_k == 2
    assert cfg.top_k < cfg.n_experts

    print("PASS exact parameters:", total)
    print("PASS shape:", tuple(logits.shape))
    print("PASS sparse routing:", f"top-{cfg.top_k}/{cfg.n_experts}")
    print("PASS aux loss:", float(aux))


if __name__ == "__main__":
    main()
