# Stabilo Optimize

[![GitHub Release](https://img.shields.io/github/v/release/rfonod/stabilo-optimize?include_prereleases)](https://github.com/rfonod/stabilo-optimize/releases) [![License](https://img.shields.io/github/license/rfonod/stabilo-optimize)](https://github.com/rfonod/stabilo-optimize/blob/main/LICENSE) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13828430.svg)](https://doi.org/10.5281/zenodo.13828430) [![arXiv](https://img.shields.io/badge/arXiv-2411.02136-b31b1b.svg?style=flat)](https://arxiv.org/abs/2411.02136) [![Development Status](https://img.shields.io/badge/development-active-brightgreen)](https://github.com/rfonod/stabilo-optimize)

**Stabilo-Optimize** is a Python benchmarking tool designed specifically to evaluate and tune methods and hyperparameters of the [stabilo](https://github.com/rfonod/stabilo) 🚀 library for video and track stabilization tasks. It systematically generates performance evaluations through random perturbations, eliminating the need for ground-truth homographies. This tool significantly simplifies the optimization of stabilization techniques, making it ideal for high-precision tasks in fields such as urban monitoring, traffic analysis, and drone imagery processing.

![Benchmark Campaign Illustration](assets/benchmark_visualization.gif?raw=True)

## Key Features

- **Ground Truth-Free Benchmarking**: Randomly generates photometric and homographic perturbations (brightness variations, Gaussian blur, saturation adjustments, fog effects, rotations, translations, scales, and perspective shifts).
- **Hierarchical Benchmarking Strategy**: Encourages users to systematically vary hyperparameters hierarchically for efficient parameter optimization.
- **Flexible JSON Configuration**: Customize extensive parameter settings using nested dictionaries (see [comprehensive_benchmark.json](experiments/sample_experiment/comprehensive_benchmark.json) or [simple_benchmark.json](experiments/sample_experiment/simple_benchmark.json) for examples).
- **Result Visualization**: Generates comprehensive performance plots and benchmarking process visualizations.

![Benchmarking Process Diagram](assets/registration_campaign.png)

## Installation

1. **Create and activate a Python virtual environment** (Python >= 3.9), e.g., using [Miniconda3](https://docs.anaconda.com/free/miniconda/):
    ```bash
    conda create -n stabilo-optimize python=3.9 -y
    conda activate stabilo-optimize
    ```

2. **Clone or fork the repository**:
    ```bash
    git clone https://github.com/rfonod/stabilo-optimize.git
    cd stabilo-optimize
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Example Usage

A sample benchmark (`simple_benchmark.json`) with provided scenes and vehicle bounding box masks is included in the `experiments/sample_experiment` directory. To reproduce the results, run:

```bash
python benchmark.py experiments/sample_experiment/simple_benchmark.json -sp -sv -o
```

- `-sp`: Save performance plots.
- `-sv`: Save benchmark visualization video.
- `-o`: Overwrite previous results.

Use `python benchmark.py --help` to explore additional command-line options.

**Note:** This example is limited to three scenes for demonstration purposes. Users should define their own benchmarks with a more representative selection of scenes for meaningful evaluation.

## Custom Benchmarking

To set up your own benchmark, create a new experiment directory within `experiments` containing:

- `benchmark.json`: Configuration specifying methods/hyperparameters and number of random trials (`N`) per scene. For reliable results, set `N > 100`. 
- `scenes`: Directory containing input images (and optional exclusion masks in YOLO format). Ensure selected scenes adequately represent your stabilization tasks. To obtain reliable benchmarking results, include a diverse set of scenes covering different lighting conditions and camera  viewpoints. 

Example structure:

```
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

Refer to the [stabilo](https://github.com/rfonod/stabilo) library and the associated [manuscript](https://arxiv.org/abs/2411.02136) for detailed descriptions of available methods and hyperparameters.

## Benchmarking Metrics

Benchmarks use metrics like Homography Estimation Accuracy (HEA) and Mean Intersection over Union (MIoU). MIoU specifically evaluates the accuracy of object-level registration and requires bounding box masks for calculation. Detailed metric definitions and analysis are provided in the manuscript.

## Citing This Work

If using this tool for research or commercial applications, please cite appropriately:

**Preferred Citation:** 

```bibtex
@misc{fonod2025advanced,
  title={Advanced computer vision for extracting georeferenced vehicle trajectories from drone imagery}, 
  author={Robert Fonod and Haechan Cho and Hwasoo Yeo and Nikolas Geroliminis},
  year={2025},
  eprint={2411.02136},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2411.02136},
  doi={https://doi.org/10.48550/arXiv.2411.02136}
}
```
**Repository Citation:**

```bibtex
@software{fonod2025stabilo-optimize,
  author = {Fonod, Robert},
  license = {MIT},
  month = mar,
  title = {Stabilo Optimize: A Framework for Comprehensive Evaluation and Analysis for the Stabilo Library},
  url = {https://github.com/rfonod/stabilo-optimize},
  doi = {10.5281/zenodo.13828430},
  version = {1.0.0},
  year = {2025}
}
```

A [CITATION.cff](CITATION.cff) file is provided for consistent referencing.

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please open a [GitHub Issue](https://github.com/rfonod/stabilo-optimize/issues) or submit a pull request. Your contributions are greatly appreciated!

## License

This project is distributed under the MIT License. Refer to the [LICENSE](LICENSE) file for detailed terms.