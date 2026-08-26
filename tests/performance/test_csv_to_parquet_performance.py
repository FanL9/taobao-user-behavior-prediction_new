"""Performance test for the stage-one conversion path."""

from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path

import pandas as pd
import psutil

from src.data import convert_csv_to_parquet


def _sample_memory(
    process: psutil.Process,
    stop_event: threading.Event,
    samples: list[int],
) -> None:
    """Sample resident memory until conversion finishes."""

    while not stop_event.wait(0.01):
        try:
            samples.append(process.memory_info().rss)
        except psutil.Error:
            return


def _run_benchmark(rows: int, chunksize: int, report_path: Path) -> dict[str, object]:
    """Convert synthetic data and save runtime, CPU, memory, and GPU status."""

    process = psutil.Process()
    stop_event = threading.Event()
    memory_samples = [process.memory_info().rss]

    with tempfile.TemporaryDirectory(prefix="taobao-stage1-benchmark-") as temp_dir:
        temp_path = Path(temp_dir)
        csv_path = temp_path / "input.csv"
        parquet_path = temp_path / "output.parquet"
        row_numbers = pd.RangeIndex(rows)
        pd.DataFrame(
            {
                "time": "2025-11-18 00",
                "user_id": row_numbers % 10_000 + 1,
                "item_id": row_numbers + 1,
                "item_category": row_numbers % 1_000 + 1,
                "behavior_type": row_numbers % 4 + 1,
            }
        ).to_csv(csv_path, index=False)

        cpu_before = process.cpu_times()
        sampler = threading.Thread(
            target=_sample_memory,
            args=(process, stop_event, memory_samples),
            daemon=True,
        )
        sampler.start()
        started_at = time.perf_counter()
        result = convert_csv_to_parquet(
            csv_path,
            parquet_path,
            chunksize=chunksize,
        )
        runtime_seconds = time.perf_counter() - started_at
        stop_event.set()
        sampler.join()
        memory_samples.append(process.memory_info().rss)
        cpu_after = process.cpu_times()

    cpu_time_seconds = (
        cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system
    )
    record: dict[str, object] = {
        "rows": result.row_count,
        "runtime_seconds": round(runtime_seconds, 6),
        "process_cpu_time_seconds": round(cpu_time_seconds, 6),
        "peak_process_rss_bytes": max(memory_samples),
        "gpu_used": False,
    }
    report_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def test_conversion_performance_is_recorded(tmp_path) -> None:
    report_path = tmp_path / "benchmark.json"

    record = _run_benchmark(
        rows=10_000,
        chunksize=2_000,
        report_path=report_path,
    )

    assert record["rows"] == 10_000
    assert 0 < record["runtime_seconds"] < 30
    assert record["process_cpu_time_seconds"] >= 0
    assert record["peak_process_rss_bytes"] > 0
    assert record["gpu_used"] is False
    assert json.loads(report_path.read_text(encoding="utf-8")) == record
