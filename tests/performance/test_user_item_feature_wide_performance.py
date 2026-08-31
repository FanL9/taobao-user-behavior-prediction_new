"""Performance test for the stage-two user-item feature-wide merge."""

from __future__ import annotations

import json
import threading
import time

import pandas as pd
import psutil

from src.features.feature import generate_all_feature_tables
from src.features.user_item_feature_wide import generate_user_item_feature_wide


def _sample_memory(
    process: psutil.Process,
    stop_event: threading.Event,
    samples: list[int],
) -> None:
    """Append current-process RSS samples until the merge finishes.

    Args:
        process: Current Python process.
        stop_event: Signal set after generation finishes.
        samples: Mutable RSS sample list in bytes.

    Returns:
        None. Samples are appended in place.
    """

    while not stop_event.wait(0.01):
        try:
            samples.append(process.memory_info().rss)
        except psutil.Error:
            return


def test_user_item_feature_wide_performance(tmp_path) -> None:
    """Record runtime and resources for only the eight-table wide merge.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None. Assertions validate metrics and output status.
    """

    rows = 50_000
    valid_hours = pd.date_range(
        "2025-11-18", "2025-12-07 23:00", freq="h"
    ).append(
        pd.date_range("2025-12-09", "2025-12-14 23:00", freq="h")
    ).append(pd.date_range("2025-12-16", "2025-12-17 23:00", freq="h"))
    row_index = pd.RangeIndex(rows)
    event_time = valid_hours.take(row_index % len(valid_hours))
    item_id = (row_index * 7 + row_index // 1_000) % 5_000 + 1
    behavior_names = pd.Series(["pv", "fav", "cart", "buy"])
    clean = pd.DataFrame(
        {
            "time": event_time.strftime("%Y-%m-%d %H"),
            "user_id": row_index % 1_000 + 1,
            "item_id": item_id,
            "category_id": item_id % 100 + 1,
            "behavior_name": behavior_names.take(row_index % 4).to_numpy(),
            "behavior_date": event_time.strftime("%Y-%m-%d"),
        }
    )
    clean_path = tmp_path / "clean.parquet"
    feature_directory = tmp_path / "features"
    clean.to_parquet(clean_path, index=False)
    generate_all_feature_tables(clean_path, feature_directory)

    process = psutil.Process()
    stop_event = threading.Event()
    samples = [process.memory_info().rss]
    cpu_before = process.cpu_times()
    sampler = threading.Thread(
        target=_sample_memory,
        args=(process, stop_event, samples),
        daemon=True,
    )
    sampler.start()
    started_at = time.perf_counter()
    result = generate_user_item_feature_wide(
        feature_directory,
        tmp_path / "user_item_feature_wide.parquet",
        tmp_path / "quality.json",
        batch_size=10_000,
    )
    runtime_seconds = time.perf_counter() - started_at
    stop_event.set()
    sampler.join()
    samples.append(process.memory_info().rss)
    cpu_after = process.cpu_times()

    metrics = {
        "runtime_seconds": round(runtime_seconds, 6),
        "process_cpu_time_seconds": round(
            cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system,
            6,
        ),
        "peak_process_rss_bytes": max(samples),
        "gpu_used": False,
    }
    metrics_path = tmp_path / "user_item_wide_performance.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, sort_keys=True))

    assert result["quality_report"]["status"] == "passed"
    assert result["quality_report"]["row_count"] > 0
    assert 0 < metrics["runtime_seconds"] < 30
    assert metrics["process_cpu_time_seconds"] >= 0
    assert metrics["peak_process_rss_bytes"] > 0
    assert metrics["gpu_used"] is False
