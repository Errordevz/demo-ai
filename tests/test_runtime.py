from __future__ import annotations

import os
import time

os.environ["DEMO_AI_MODEL_DIR"] = "api/model"

from api.runtime import CloudDemoAI


def main():
    model = CloudDemoAI(__import__("pathlib").Path("api/model"))
    started = time.perf_counter()
    text = model.generate(
        "### System\nYou are Demo AI.\n### User\nhi\n### Assistant\n",
        max_new=2,
        temperature=0.7,
        top_k=8,
        seed=0,
    )
    elapsed = time.perf_counter() - started
    assert isinstance(text, str)
    assert text.strip(), "runtime returned an empty response"
    print("PASS runtime response:", repr(text))
    print("PASS runtime seconds:", round(elapsed, 3))


if __name__ == "__main__":
    main()
