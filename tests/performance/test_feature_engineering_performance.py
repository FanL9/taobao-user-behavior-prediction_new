"""Performance test for the first four stage-two feature tables."""

from __future__ import annotations

import json
import threading
import time

import pandas as pd
import psutil

from src.features.feature import generate_all_feature_tables


def _sample_memory(
    process: psutil.Process,
    stop_event: threading.Event,
    samples: list[int],
) -> None:
    """Sample process RSS until feature generation finishes.

    Args:
        process: Current Python process whose RSS is sampled.
        stop_event: Signal set after generation finishes.
        samples: Mutable output list receiving RSS values in bytes.

    Returns:
        None. RSS values are appended to ``samples``.
    """

    while not stop_event.wait(0.01):
        try:
            samples.append(process.memory_info().rss)
        except psutil.Error:
            return


def test_stage2_feature_generation_performance(tmp_path) -> None:
    """Record runtime and resources for the four-table generation interface.

    Args:
        tmp_path: Pytest temporary directory for Parquet input and outputs.

    Returns:
        None. The test writes a temporary JSON metrics record and validates it.
    """

    rows = 50_000
    valid_hours = pd.date_range("2025-11-18", "2025-12-07 23:00", freq="h").append(
        pd.date_range("2025-12-09", "2025-12-14 23:00", freq="h")
    ).append(pd.date_range("2025-12-16", "2025-12-17 23:00", freq="h"))
    row_index = pd.RangeIndex(rows)
    event_time = valid_hours.take(row_index % len(valid_hours))
    item_id = row_index % 5_000 + 1
    behavior_names = pd.Series(["pv", "fav", "cart", "buy"])
    clean_data = pd.DataFrame(
        {
            "time": event_time.strftime("%Y-%m-%d %H"),
            "user_id": row_index % 1_000 + 1,
            "item_id": item_id,
            "category_id": item_id % 100 + 1,
            "behavior_name": behavior_names.take(row_index % 4).to_numpy(),
            "behavior_date": event_time.strftime("%Y-%m-%d"),
        }
    )
    input_path = tmp_path / "user_behavior_clean.parquet"
    output_directory = tmp_path / "features"
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
    outputs = generate_all_feature_tables(input_path, output_directory)
    runtime_seconds = time.perf_counter() - started_at
    stop_event.set()
    sampler.join()
    memory_samples.append(process.memory_info().rss)
    cpu_after = process.cpu_times()

    metrics = {
        "runtime_seconds": round(runtime_seconds, 6),
        "process_cpu_time_seconds": round(
            cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system,
            6,
        ),
        "peak_process_rss_bytes": max(memory_samples),
        "gpu_used": False,
    }
    record_path = tmp_path / "stage2_feature_performance.json"
    record_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))

    assert len(outputs) == 8
    assert 0 < metrics["runtime_seconds"] < 30
    assert metrics["process_cpu_time_seconds"] >= 0
    assert metrics["peak_process_rss_bytes"] > 0
    assert metrics["gpu_used"] is False
    assert json.loads(record_path.read_text(encoding="utf-8")) == metrics
