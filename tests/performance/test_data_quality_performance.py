"""Performance record test for read-only data quality inspection."""

from __future__ import annotations

import json
import threading
import time

import pandas as pd
import psutil

from src.data import check_csv_quality


def _sample_memory(
    process: psutil.Process,
    stop_event: threading.Event,
    samples: list[int],
) -> None:
    """Record process memory while the quality check is running."""

    while not stop_event.wait(0.01):
        try:
            samples.append(process.memory_info().rss)
        except psutil.Error:
            return


def test_quality_check_records_runtime_cpu_memory_and_gpu(tmp_path) -> None:
    rows = 10_000
    source = tmp_path / "user_behavior_processed.csv"
    report_path = tmp_path / "performance.json"
    row_numbers = pd.RangeIndex(rows)
    pd.DataFrame(
        {
            "time": "2025-11-18 00",
            "user_id": row_numbers % 1_000 + 1,
            "item_id": row_numbers + 1,
            "item_category": row_numbers % 100 + 1,
            "behavior_type": row_numbers % 4 + 1,
        }
    ).to_csv(source, index=False)

    process = psutil.Process()
    stop_event = threading.Event()
    memory_samples = [process.memory_info().rss]
    cpu_before = process.cpu_times()
    sampler = threading.Thread(
        target=_sample_memory,
        args=(process, stop_event, memory_samples),
        daemon=True,
    )
    sampler.start()
    started_at = time.perf_counter()
    quality_report = check_csv_quality(
        source,
        chunksize=2_000,
        duplicate_partitions=4,
    )
    runtime_seconds = time.perf_counter() - started_at
    stop_event.set()
    sampler.join()
    memory_samples.append(process.memory_info().rss)
    cpu_after = process.cpu_times()

    performance = {
        "rows": quality_report["scale"]["row_count"],
        "runtime_seconds": round(runtime_seconds, 6),
        "process_cpu_time_seconds": round(
            cpu_after.user
            + cpu_after.system
            - cpu_before.user
            - cpu_before.system,
            6,
        ),
        "peak_process_rss_bytes": max(memory_samples),
        "gpu_used": False,
    }
    report_path.write_text(
        json.dumps(performance, indent=2) + "\n",
        encoding="utf-8",
    )

    assert performance["rows"] == rows
    assert 0 < performance["runtime_seconds"] < 30
    assert performance["process_cpu_time_seconds"] >= 0
    assert performance["peak_process_rss_bytes"] > 0
    assert performance["gpu_used"] is False
    assert json.loads(report_path.read_text(encoding="utf-8")) == performance
