"""Rebuild reports/comparison.json (models vs reality) from the saved models,
without retraining. `run_pipeline.py` also writes it.

    python scripts/build_comparison.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ticktock import config  # noqa: E402
from ticktock.compare import build_comparison  # noqa: E402
from ticktock.dataset import build_dataset  # noqa: E402
from ticktock.models import load_model  # noqa: E402


def main():
    meta = json.loads((config.MODELS_DIR / "meta.json").read_text())
    models = [load_model(n) for n in meta["models"]]
    df = build_dataset()
    test = df[df["split"] == "test"]
    out = build_comparison(models, test)
    path = config.REPORTS_DIR / "comparison.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB, {len(test):,} test orders)")


if __name__ == "__main__":
    main()
