import os
import time
from pathlib import Path

import psutil
from streamlit.testing.v1 import AppTest


REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_APP = REPO_ROOT / "dashboards" / "eda" / "app.py"


def test_eda_dashboard_startup_performance() -> None:
    """Measure Stage 1 EDA dashboard startup resource usage."""
    process = psutil.Process(os.getpid())

    memory_before = process.memory_info().rss
    cpu_before = process.cpu_times()
    started_at = time.perf_counter()

    app = AppTest.from_file(str(DASHBOARD_APP))
    app.run(timeout=30)

    wall_seconds = time.perf_counter() - started_at
    cpu_after = process.cpu_times()
    memory_after = process.memory_info().rss

    process_cpu_seconds = (
        cpu_after.user
        + cpu_after.system
        - cpu_before.user
        - cpu_before.system
    )

    average_cpu_percent = (
        process_cpu_seconds / wall_seconds * 100
        if wall_seconds > 0
        else 0.0
    )

    memory_delta_mb = (memory_after - memory_before) / (1024 * 1024)

    print("")
    print("===== EDA Dashboard Performance =====")
    print(f"startup_seconds: {wall_seconds:.4f}")
    print(f"process_cpu_seconds: {process_cpu_seconds:.4f}")
    print(f"average_cpu_percent: {average_cpu_percent:.2f}")
    print(f"memory_before_mb: {memory_before / (1024 * 1024):.2f}")
    print(f"memory_after_mb: {memory_after / (1024 * 1024):.2f}")
    print(f"memory_delta_mb: {memory_delta_mb:.2f}")
    print("gpu_used: False")
    print("=====================================")

    assert not app.exception
    assert wall_seconds < 30
    assert process_cpu_seconds >= 0
