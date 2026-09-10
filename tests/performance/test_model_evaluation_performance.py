"""Performance test for bounded Stage4 Issue3 final evaluation."""

from __future__ import annotations

import json
import time
from pathlib import Path

from tests.functional.test_model_evaluation import _stacking_artifacts
from src.models.model_evaluation import evaluate_final_candidates


def test_model_evaluation_performance(tmp_path: Path) -> None:
    """Record test-evaluation runtime after fixed Issue2.5 artifacts are available."""

    split_dir, features, models, manifest = _stacking_artifacts(tmp_path)
    started = time.perf_counter()
    result = evaluate_final_candidates(split_dir, features, manifest, models, tmp_path / "evaluation_reports")
    wall_seconds = time.perf_counter() - started
    summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    summary["test_wall_runtime_seconds"] = round(wall_seconds, 6)
    (tmp_path / "evaluation_performance.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    assert summary["test_evaluation_count"] == 1
    assert 0 < wall_seconds < 60
