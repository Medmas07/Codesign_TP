# TP-CODESIGN Delivery

Final delivery package for OpenCL matrix multiplication benchmarks and multi-device analysis.

## Structure

- `src/`
  - Python scripts and OpenCL kernels used in the project.
- `notebooks/`
  - Final notebooks used for experiments, sweeps, and plots.
- `report/`
  - Final report PDF (if provided).

## Key notebooks

- `notebooks/opencl_examples/multi_device_analysis/multi_device_alpha_sweep.ipynb`
- `notebooks/opencl_examples/multi_device_optimized_both/multi_device_optimized_both.ipynb`
- `notebooks/opencl_examples/tp0_benchmark/tp0_opencl_matmul_benchmark.ipynb`
- `notebooks/opencl_examples/tp0_benchmark/matmul_benchmark_dashboard.ipynb`

## Notes

- TP0 notebooks use scripts in:
  - `notebooks/opencl_examples/tp0_benchmark/prof_files/`
- Multi-device notebooks run their local script in the same folder.
- Generated artifacts (`.csv`, caches, logs) are ignored by `.gitignore`.

