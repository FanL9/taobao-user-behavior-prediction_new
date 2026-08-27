import time
import os
import subprocess
import sys
import csv


# ============================================================
# Configuration
# ============================================================

EDA_FILE = "src/data/EDA_analysis.py"

OUTPUT_FILE = "data/EDA/performance_test_result.csv"


# ============================================================
# CPU Monitoring
# ============================================================

def get_cpu_usage():
    """
    Get current CPU usage using Windows PowerShell.
    """

    try:
        command = [
            "powershell",
            "-Command",
            "(Get-Counter '\\Processor(_Total)\\% Processor Time').CounterSamples.CookedValue"
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            value = result.stdout.strip()

            if value:
                return f"{float(value):.2f}%"

        return "N/A"

    except Exception:
        return "N/A"


# ============================================================
# GPU Monitoring
# ============================================================

def get_gpu_usage():
    """
    Get current NVIDIA GPU utilization.
    """

    try:
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits"
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            values = result.stdout.strip().splitlines()

            if values:
                return ", ".join(
                    value.strip() + "%"
                    for value in values
                )

        return "N/A"

    except Exception:
        return "N/A"


# ============================================================
# Performance Test
# ============================================================

def run_performance_test():
    """
    Run the EDA analysis and record runtime,
    CPU usage, and GPU usage.
    """

    print("=" * 60)
    print("EDA PERFORMANCE TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Check whether EDA_analysis.py exists
    # --------------------------------------------------------

    if not os.path.exists(EDA_FILE):

        print("\nERROR: EDA_analysis.py was not found.")

        print("Expected path:")
        print(EDA_FILE)

        return

    # --------------------------------------------------------
    # Record system status before EDA
    # --------------------------------------------------------

    print("\nRecording system status...")

    cpu_before = get_cpu_usage()
    gpu_before = get_gpu_usage()

    # --------------------------------------------------------
    # Run EDA analysis
    # --------------------------------------------------------

    print("\nRunning EDA analysis...")

    start_time = time.perf_counter()

    result = subprocess.run(
        [
            sys.executable,
            EDA_FILE
        ],
        capture_output=True,
        text=True
    )

    end_time = time.perf_counter()

    # --------------------------------------------------------
    # Calculate runtime
    # --------------------------------------------------------

    runtime_seconds = end_time - start_time

    # --------------------------------------------------------
    # Record system status after EDA
    # --------------------------------------------------------

    cpu_after = get_cpu_usage()
    gpu_after = get_gpu_usage()

    # --------------------------------------------------------
    # Determine execution status
    # --------------------------------------------------------

    if result.returncode == 0:
        status = "Success"
    else:
        status = "Failed"

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PERFORMANCE RESULTS")
    print("=" * 60)

    print(f"\nEDA execution status: {status}")

    print(
        f"Runtime: "
        f"{runtime_seconds:.2f} seconds"
    )

    print(
        f"CPU usage before EDA: "
        f"{cpu_before}"
    )

    print(
        f"CPU usage after EDA: "
        f"{cpu_after}"
    )

    print(
        f"GPU usage before EDA: "
        f"{gpu_before}"
    )

    print(
        f"GPU usage after EDA: "
        f"{gpu_after}"
    )

    # --------------------------------------------------------
    # Save performance results
    # --------------------------------------------------------

    result_data = {
        "status": status,
        "runtime_seconds": round(
            runtime_seconds,
            2
        ),
        "cpu_usage_before": cpu_before,
        "cpu_usage_after": cpu_after,
        "gpu_usage_before": gpu_before,
        "gpu_usage_after": gpu_after
    }

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=result_data.keys()
        )

        writer.writeheader()
        writer.writerow(result_data)

    print(
        "\nPerformance result saved to:"
    )

    print(OUTPUT_FILE)

    # --------------------------------------------------------
    # Show EDA error if execution failed
    # --------------------------------------------------------

    if result.returncode != 0:

        print("\nEDA ERROR OUTPUT:")

        print(result.stderr)

    print("\n" + "=" * 60)
    print("PERFORMANCE TEST COMPLETED")
    print("=" * 60)

    return result_data


# ============================================================
# Run Performance Test
# ============================================================

run_performance_test()
