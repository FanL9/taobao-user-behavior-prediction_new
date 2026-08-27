# Stage 1 EDA Dashboard Test Results

## 1. Scope

This report records the functional and performance validation results for Issue #5, the Stage 1 EDA visualization dashboard.

Dashboard entry point:

`dashboards/eda/app.py`

Data loader:

`dashboards/eda/data_loader.py`

---

## 2. Functional Tests

Functional test file:

`tests/functional/test_eda_dashboard.py`

The test suite covers:

- loading lightweight Stage 1 EDA outputs
- missing required output handling
- behavior distribution contract
- user purchase summary contract
- Top Items contract
- Top Categories contract
- Daily Trend data contract
- Hourly Trend data contract
- descriptive funnel contract
- Streamlit dashboard smoke run

Targeted functional test command:

`python -m pytest .\tests\functional\test_eda_dashboard.py -q`

Observed result:

`8 passed in 2.87s`

The standalone dashboard smoke test also passed successfully.

---

## 3. Performance Test

Performance test file:

`tests/performance/test_eda_dashboard_performance.py`

Test command:

`python -m pytest .\tests\performance\test_eda_dashboard_performance.py -q -s`

Observed result:

- startup_seconds: 1.7705
- process_cpu_seconds: 1.5469
- average_cpu_percent: 87.37
- memory_before_mb: 76.41
- memory_after_mb: 151.81
- memory_delta_mb: 75.40
- gpu_used: False

Pytest result:

`1 passed in 2.70s`

---

## 4. Performance Interpretation

The dashboard startup test completed in under two seconds in the measured local test run.

The dashboard uses CPU processing during initial rendering and does not use GPU acceleration.

The measured process memory increase during the startup test was approximately 75.40 MB.

The dashboard intentionally loads lightweight Stage 1 aggregate outputs instead of the full 12,256,906-row clean dataset.

`item_statistics.csv` is validated as an upstream EDA output but is not loaded during normal dashboard startup.

This design avoids loading the multi-million-row item detail table when the dashboard only requires Top-N item statistics.

---

## 5. Streamlit Test Environment Note

The performance test may emit a `missing ScriptRunContext` warning when Streamlit AppTest runs in bare test mode.

This warning did not cause an application exception or test failure.

The previous deprecated `use_container_width` usage was replaced with `width="stretch"` before the final performance measurement.

---

## 6. Current Validation Status

- Dashboard data contract tests: PASS
- Missing-file handling test: PASS
- Dashboard smoke run: PASS
- Dashboard startup performance test: PASS
- GPU usage: not used

Full repository test suite: PASS

Final full-suite result:

`33 passed in 35.83s`
