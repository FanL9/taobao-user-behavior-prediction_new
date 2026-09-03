import json
import os
import time
from pathlib import Path

import psutil
from streamlit.testing.v1 import AppTest


REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_APP = REPO_ROOT / "dashboards" / "eda" / "app.py"
PERFORMANCE_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "performance"
    / "runtime"
    / "eda_dashboard_stage2_performance.json"
)


def test_eda_dashboard_startup_performance() -> None:
    """Measure dashboard startup wall time, CPU usage, and memory delta."""
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
        cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system
    )
    average_cpu_percent = (
        process_cpu_seconds / wall_seconds * 100 if wall_seconds > 0 else 0.0
    )
    memory_delta_mb = (memory_after - memory_before) / (1024 * 1024)

    result = {
        "startup_seconds": round(wall_seconds, 4),
        "process_cpu_seconds": round(process_cpu_seconds, 4),
        "average_cpu_percent": round(average_cpu_percent, 2),
        "memory_before_mb": round(memory_before / (1024 * 1024), 2),
        "memory_after_mb": round(memory_after / (1024 * 1024), 2),
        "memory_delta_mb": round(memory_delta_mb, 2),
        "gpu_used": False,
    }

    PERFORMANCE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PERFORMANCE_OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("")
    print("===== EDA Dashboard Performance =====")
    for key, value in result.items():
        print(f"{key}: {value}")
    print(f"result_file: {PERFORMANCE_OUTPUT}")
    print("=====================================")

    assert not app.exception
    assert wall_seconds < 30
    assert process_cpu_seconds >= 0
