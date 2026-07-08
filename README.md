# Stabilo Optimize

[![GitHub Release](https://img.shields.io/github/v/release/rfonod/stabilo-optimize?include_prereleases)](https://github.com/rfonod/stabilo-optimize/releases) [![PyPI Version](https://img.shields.io/pypi/v/stabilo-optimize)](https://pypi.org/project/stabilo-optimize/) [![PyPI - Total Downloads](https://img.shields.io/pepy/dt/stabilo-optimize?label=total%20downloads)](https://pepy.tech/project/stabilo-optimize) [![PyPI - Downloads per Month](https://img.shields.io/pypi/dm/stabilo-optimize?color=%234c1)](https://pypi.org/project/stabilo-optimize/) [![CI](https://github.com/rfonod/stabilo-optimize/actions/workflows/ci.yml/badge.svg)](https://github.com/rfonod/stabilo-optimize/actions/workflows/ci.yml) [![Python](https://img.shields.io/badge/python-3.9--3.13-blue)](https://www.python.org/) [![License](https://img.shields.io/github/license/rfonod/stabilo-optimize)](https://github.com/rfonod/stabilo-optimize/blob/main/LICENSE) [![GitHub Issues](https://img.shields.io/github/issues/rfonod/stabilo-optimize)](https://github.com/rfonod/stabilo-optimize/issues) [![Open Access](https://img.shields.io/badge/Journal-10.1016%2Fj.trc.2025.105205-blue)](https://doi.org/10.1016/j.trc.2025.105205) [![arXiv](https://img.shields.io/badge/arXiv-2411.02136-b31b1b.svg)](https://arxiv.org/abs/2411.02136) [![Archived Code](https://img.shields.io/badge/Zenodo-Software%20Archive-blue)](https://zenodo.org/doi/10.5281/zenodo.13828430)

**Stabilo-Optimize** is a Python benchmarking tool designed specifically to evaluate and tune methods and hyperparameters of the [Stabilo](https://github.com/rfonod/stabilo) 🌀 library for video and track stabilization tasks. It systematically generates performance evaluations through random perturbations, eliminating the need for ground-truth homographies. This tool significantly simplifies the optimization of stabilization techniques, making it ideal for high-precision tasks in fields such as urban monitoring, traffic analysis, and drone imagery processing.

![Benchmark Campaign Illustration](https://raw.githubusercontent.com/rfonod/stabilo-optimize/main/assets/benchmark_visualization.webp)

## Why Stabilo-Optimize

- **Ground Truth-Free Benchmarking**: Randomly generates photometric and homographic perturbations (brightness variations, Gaussian blur, saturation adjustments, fog effects, rotations, translations, scales, and perspective shifts).
- **Hierarchical Benchmarking Strategy**: Encourages users to systematically vary hyperparameters hierarchically for efficient parameter optimization.
- **Flexible JSON Configuration**: Customize extensive parameter settings using nested dictionaries (see [comprehensive_benchmark.json](experiments/sample_experiment/comprehensive_benchmark.json) or [simple_benchmark.json](experiments/sample_experiment/simple_benchmark.json) for examples).
- **Result Visualization**: Generates comprehensive performance plots and benchmarking process visualizations.

![Benchmarking Process Diagram](https://raw.githubusercontent.com/rfonod/stabilo-optimize/main/assets/registration_campaign.png)

## Install

Create and activate a **Python virtual environment** (Python 3.9–3.13), then install from [PyPI](https://pypi.org/project/stabilo-optimize/):

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install stabilo-optimize
```

Also works with [uv](https://docs.astral.sh/uv/) (`uv pip install stabilo-optimize`) and [conda](https://www.anaconda.com/docs/getting-started/miniconda/install).

For development, clone the repository and install in editable mode with the `dev` extra:

```bash
git clone https://github.com/rfonod/stabilo-optimize.git
cd stabilo-optimize && pip install -e '.[dev]'
```

## Quick Start

A sample benchmark (`simple_benchmark.json`) with provided scenes and vehicle bounding box masks is included in the `experiments/sample_experiment` directory. To reproduce the results, run:

```bash
stabilo-optimize benchmark experiments/sample_experiment/simple_benchmark.json -sp -sv -o
```

- `-sp`: Save performance plots.
- `-sv`: Save benchmark visualization video.
- `-o`: Overwrite previous results.

Add `-l <path>`/`--log-file <path>` (also available on `plot`) to additionally write console output to a file — parent directories are created automatically, color codes are stripped in the file, the resolved absolute path is printed to the console, and log files are gitignored (`*.log`) by default.

`-v`/`--verbosity` (`benchmark` only) controls how much detail is printed, including from Stabilo's own internal logging: `0`=quiet (top-level status only, Stabilo's own messages fully suppressed), `1`=minimal (+ per-run header/summary table, Stabilo errors only), `2`=detailed (+ per-scene summary table, Stabilo warnings and up), `3`=debug (+ per-trial output, Stabilo info and up).

Use `stabilo-optimize benchmark --help` to explore additional command-line options, or re-plot existing results without re-running the benchmark:

```bash
stabilo-optimize plot experiments/sample_experiment/simple_benchmark.json
```

**Note:** This example is limited to three scenes for demonstration purposes. Users should define their own benchmarks with a more representative selection of scenes for meaningful evaluation.

## Custom Benchmarking

To set up your own benchmark, create a new experiment directory within `experiments` containing:

- `benchmark.json`: Configuration specifying methods/hyperparameters and number of random trials (`N`) per scene. For reliable results, set `N > 100`.
- `scenes`: Directory containing input images (and optional exclusion masks in YOLO format). Ensure selected scenes adequately represent your stabilization tasks. To obtain reliable benchmarking results, include a diverse set of scenes covering different lighting conditions and camera  viewpoints.

Example structure:

```text
experiments
└─custom_experiment
  ├─benchmark.json
  └─scenes
    ├ image1.jpg
    ├ image1.txt
    ├ image2.jpg
    ├ image2.txt
    ├ ...
```

**Note**: A comprehensive configuration file (`comprehensive_benchmark.json`) is included for illustration purposes. Due to computational costs, users should avoid directly running such an extensive parameter search. Instead, adopt a hierarchical parameter search approach by fixing some hyperparameters and varying others.

Refer to the [Stabilo](https://github.com/rfonod/stabilo) 🚀 library and the associated [article](https://doi.org/10.1016/j.trc.2025.105205) for detailed descriptions of available methods and hyperparameters.

**GPU acceleration:** setting `gpu: true` in the config runs (parts of) Stabilo's pipeline on an NVIDIA GPU — see [`docs/cuda.md`](https://github.com/rfonod/stabilo/blob/main/docs/cuda.md) in the Stabilo repo for building a CUDA-enabled OpenCV and setting it up. This mainly affects `Computation_time`, not tuning outcomes: RANSAC-based homography/affine estimation always runs on CPU, so the HEA/MIoU accuracy metrics are essentially unaffected by this setting.

## Benchmarking Metrics

Benchmarks use metrics like Homography Estimation Accuracy (HEA) and Mean Intersection over Union (MIoU). MIoU specifically evaluates the accuracy of object-level registration and requires bounding box masks for calculation. Detailed metric definitions and analysis are provided in the manuscript.

## Citation

If you use **Stabilo-Optimize** in your research, software, or product, please cite the following resources appropriately:

1. **Preferred Citation:** Please cite the associated article for any use of the Stabilo-Optimize, including research, applications, and derivative work:

    ```bibtex
    @article{fonod2025advanced,
      title = {Advanced computer vision for extracting georeferenced vehicle trajectories from drone imagery},
      author = {Fonod, Robert and Cho, Haechan and Yeo, Hwasoo and Geroliminis, Nikolas},
      journal = {Transportation Research Part C: Emerging Technologies},
      volume = {178},
      pages = {105205},
      year = {2025},
      publisher = {Elsevier},
      doi = {10.1016/j.trc.2025.105205},
      url = {https://doi.org/10.1016/j.trc.2025.105205}
    }
    ```

2. **Repository Citation:** If you reference, modify, or build upon the Stabilo-Optimize software itself, please also cite the corresponding Zenodo release:

    ```bibtex
    @software{fonod2026stabilo-optimize,
      author = {Fonod, Robert},
      license = {MIT},
      month = jul,
      title = {Stabilo Optimize: A Framework for Comprehensive Evaluation and Analysis for the Stabilo Library},
      url = {https://github.com/rfonod/stabilo-optimize},
      doi = {10.5281/zenodo.13828430},
      version = {1.1.0},
      year = {2026}
    }
    ```

## Contributing

Contributions from the community are welcome! If you encounter any issues or have suggestions for improvements, please open a [GitHub Issue](https://github.com/rfonod/stabilo-optimize/issues) or submit a pull request.

## License

This project is distributed under the MIT License. See the [LICENSE](LICENSE) file for more details.
