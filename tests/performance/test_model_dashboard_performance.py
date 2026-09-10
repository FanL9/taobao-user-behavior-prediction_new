"""Performance check for read-only Stage4 dashboard data loading."""

from __future__ import annotations

import time

import psutil

from dashboards.eda.model_dashboard import read_model_dashboard_outputs
from tests.functional.test_model_dashboard import _write_outputs


def test_model_dashboard_loads_compact_outputs_quickly(tmp_path) -> None:
    """Persisted dashboard outputs should load without invoking model pipelines."""

    _write_outputs(tmp_path)
    process = psutil.Process()
    rss_before = process.memory_info().rss
    started = time.perf_counter()
    loaded = read_model_dashboard_outputs(tmp_path)
    elapsed = time.perf_counter() - started
    rss_delta = process.memory_info().rss - rss_before
    assert elapsed < 2.0
    assert rss_delta < 64 * 1024 * 1024
    assert len(loaded["metrics"]) == 1
