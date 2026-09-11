"""Performance test for Stage 5 bounded local batch inference."""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.inference.batch_inference import run_batch_inference
from tests.functional.test_batch_inference import _write_inference_fixture


def test_batch_inference_performance(tmp_path: Path) -> None:
    """Record runtime, throughput, CPU, and RSS for a bounded synthetic batch."""

    fixture = _write_inference_fixture(tmp_path, rows=50_000)
    output = tmp_path / "predictions.csv"
    summary_path = tmp_path / "batch_inference_performance.json"
    started = time.perf_counter()
    summary = run_batch_inference(
        fixture["input"],
        output,
        fixture["config"],
        summary_path,
        batch_size=10_000,
        project_root=tmp_path,
    )
    wall_seconds = time.perf_counter() - started
    saved = json.loads(summary_path.read_text(encoding="utf-8"))

    print("\n===== Stage 5 Batch Inference Performance =====")
    print(f"rows: {saved['row_count']}")
    print(f"runtime_seconds: {saved['runtime_seconds']}")
    print(f"throughput_rows_per_second: {saved['throughput_rows_per_second']}")
    print(f"process_cpu_seconds: {saved['process_cpu_seconds']}")
    print(f"peak_process_rss_bytes: {saved['peak_process_rss_bytes']}")
    print("===============================================")

    assert summary["row_count"] == 50_000
    assert 0 < wall_seconds < 30
    assert saved["throughput_rows_per_second"] > 0
    assert saved["peak_process_rss_bytes"] > 0
    assert output.is_file()
