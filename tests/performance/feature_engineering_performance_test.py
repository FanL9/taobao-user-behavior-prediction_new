import os
import time
import subprocess
import threading
import psutil
import sys

# ============================================================
# Configuration
# ============================================================

FEATURE_SCRIPT = "src/features/feature.py"

PYTHON_EXE = sys.executable


# ============================================================
# Resource Monitor
# ============================================================

class ResourceMonitor:

    def __init__(self, process):

        self.process = process

        self.running = False

        self.cpu_samples = []
        self.memory_samples = []

        self.thread = None

    def monitor(self):

        while self.running:

            try:

                cpu = self.process.cpu_percent(
                    interval=0.5
                )

                memory = (
                    self.process.memory_info().rss
                    / 1024
                    / 1024
                )

                self.cpu_samples.append(cpu)
                self.memory_samples.append(memory)

            except psutil.NoSuchProcess:

                break

    def start(self):

        self.running = True

        self.thread = threading.Thread(
            target=self.monitor,
            daemon=True
        )

        self.thread.start()

    def stop(self):

        self.running = False

        if self.thread is not None:

            self.thread.join(
                timeout=2
            )

    def average_cpu(self):

        if not self.cpu_samples:

            return 0

        return (
            sum(self.cpu_samples)
            / len(self.cpu_samples)
        )

    def peak_memory(self):

        if not self.memory_samples:

            return 0

        return max(
            self.memory_samples
        )


# ============================================================
# GPU Monitoring
# ============================================================

def get_gpu_usage():

    try:

        result = subprocess.run(

            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits"
            ],

            capture_output=True,

            text=True,

            timeout=5
        )

        if result.returncode != 0:

            return None

        values = []

        for line in result.stdout.splitlines():

            line = line.strip()

            if line:

                values.append(
                    float(line)
                )

        if not values:

            return None

        return (
            sum(values)
            / len(values)
        )

    except Exception:

        return None


# ============================================================
# Main Performance Test
# ============================================================

def main():

    print("=" * 70)

    print(
        "FEATURE.PY PERFORMANCE TEST"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Check feature.py
    # --------------------------------------------------------

    if not os.path.exists(
        FEATURE_SCRIPT
    ):

        print("\nERROR:")
        print(
            "feature.py not found:"
        )
        print(
            FEATURE_SCRIPT
        )

        return

    print("\nTarget script:")
    print(
        FEATURE_SCRIPT
    )

    print("\nStarting feature.py...")
    print(
        "Resource monitoring started."
    )

    print("-" * 70)

    # --------------------------------------------------------
    # Start feature.py
    # --------------------------------------------------------

    start_time = time.perf_counter()

    process = subprocess.Popen(

        [
            PYTHON_EXE,
            FEATURE_SCRIPT
        ],

        stdout=None,

        stderr=None
    )

    ps_process = psutil.Process(
        process.pid
    )

    # --------------------------------------------------------
    # Start monitoring
    # --------------------------------------------------------

    monitor = ResourceMonitor(
        ps_process
    )

    monitor.start()

    # --------------------------------------------------------
    # Monitor GPU
    # --------------------------------------------------------

    gpu_samples = []

    while process.poll() is None:

        gpu_usage = get_gpu_usage()

        if gpu_usage is not None:

            gpu_samples.append(
                gpu_usage
            )

        time.sleep(1)

    # --------------------------------------------------------
    # Stop monitoring
    # --------------------------------------------------------

    monitor.stop()

    end_time = time.perf_counter()

    total_runtime = (
        end_time
        - start_time
    )

    exit_code = process.returncode

    # --------------------------------------------------------
    # GPU result
    # --------------------------------------------------------

    if gpu_samples:

        average_gpu = (
            sum(gpu_samples)
            / len(gpu_samples)
        )

        peak_gpu = max(
            gpu_samples
        )

    else:

        average_gpu = None

        peak_gpu = None

    # ========================================================
    # Final Results
    # ========================================================

    print("\n")

    print("=" * 70)

    print(
        "PERFORMANCE TESTING RESULTS"
    )

    print("=" * 70)

    print(
        f"Total runtime      : "
        f"{total_runtime:.2f} seconds"
    )

    print(
        f"Average CPU usage  : "
        f"{monitor.average_cpu():.2f}%"
    )

    print(
        f"Peak memory usage  : "
        f"{monitor.peak_memory():.2f} MB"
    )

    if average_gpu is not None:

        print(
            f"Average GPU usage  : "
            f"{average_gpu:.2f}%"
        )

        print(
            f"Peak GPU usage     : "
            f"{peak_gpu:.2f}%"
        )

    else:

        print(
            "Average GPU usage  : N/A"
        )

        print(
            "Peak GPU usage     : N/A"
        )

    print(
        f"Process exit code  : "
        f"{exit_code}"
    )

    if exit_code == 0:

        print(
            "Status             : SUCCESS"
        )

    else:

        print(
            "Status             : FAILED"
        )

    print("=" * 70)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    main()