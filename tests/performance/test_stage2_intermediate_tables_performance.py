"""Performance test for the four stage-two intermediate tables."""

from __future__ import annotations

import json
import threading
import time

import pandas as pd
import psutil

from src.features.stage2_intermediate_tables import generate_intermediate_tables


def _sample_memory(
    process: psutil.Process,
    stop_event: threading.Event,
    samples: list[int],
) -> None:
    """Sample process memory until generation ends.

    Args:
        process: Current Python process whose RSS is sampled.
        stop_event: Signal set when table generation has finished.
        samples: Mutable output list receiving RSS values in bytes.

    Returns:
        None. Samples are appended to ``samples``.
    """

    while not stop_event.wait(0.01):
        try:
            samples.append(process.memory_info().rss)
        except psutil.Error:
            return


def test_stage2_intermediate_generation_performance(tmp_path) -> None:
    """Record runtime and resource use of the four-table generation interface.

    Args:
        tmp_path: Pytest temporary directory for input and output Parquet files.

    Returns:
        None. The test writes a temporary metrics record and verifies it.
    """

    rows = 50_000
    row_index = pd.RangeIndex(rows)
    event_time = pd.Timestamp("2025-11-18") + pd.to_timedelta(
        row_index % (20 * 24), unit="h"
    )
    item_id = row_index % 5_000 + 1
    clean_data = pd.DataFrame(
        {
            "time": event_time.strftime("%Y-%m-%d %H"),
            "user_id": row_index % 1_000 + 1,
            "item_id": item_id,
            "category_id": item_id % 100 + 1,
            "behavior_type": row_index % 4 + 1,
            "behavior_date": event_time.strftime("%Y-%m-%d"),
            "behavior_hour": event_time.hour,
            "weekday": event_time.weekday,
        }
    )
    input_path = tmp_path / "user_behavior_clean.parquet"
    output_directory = tmp_path / "stage2"
    clean_data.to_parquet(input_path, index=False)

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
    outputs = generate_intermediate_tables(input_path, output_directory)
    runtime_seconds = time.perf_counter() - started_at
    stop_event.set()
    sampler.join()
    memory_samples.append(process.memory_info().rss)
    cpu_after = process.cpu_times()

    performance = {
        "runtime_seconds": round(runtime_seconds, 6),
        "process_cpu_time_seconds": round(
            cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system,
            6,
        ),
        "peak_process_rss_bytes": max(memory_samples),
        "gpu_used": False,
    }
    report_path = tmp_path / "stage2_performance.json"
    report_path.write_text(
        json.dumps(performance, indent=2) + "\n",
        encoding="utf-8",
    )

    assert len(outputs) == 4
    assert 0 < performance["runtime_seconds"] < 30
    assert performance["process_cpu_time_seconds"] >= 0
    assert performance["peak_process_rss_bytes"] > 0
    assert performance["gpu_used"] is False
    assert json.loads(report_path.read_text(encoding="utf-8")) == performance
