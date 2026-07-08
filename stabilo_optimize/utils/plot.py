#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
plot.py - Plot the results of the benchmarking process.

This script reads the results of the benchmarking process and plots the results
for each detector, mask, and CLAHE configuration. The plots show the Mean Bounding
Box IoU (MIoU), Homography Estimation Accuracy (HEA), and computation time for each
configuration. The plots can be saved to disk or displayed interactively.

Usage:
    stabilo-optimize plot <source> [options]

Arguments:
    source: Filepath to a JSON configuration file or directory with JSON files

Options:
    --experiment-dir, -e <dir>: Directory to load benchmark results (defaults to parent of config file)
    --show-plots, -s: Whether to display plots interactively
    --save-plots, -sp: Whether to save plots to disk (in the benchmark/plots directory)
    --overwrite, -o: Whether to overwrite existing plots
    --hea-threshold, -ht <float>: Threshold (in pixels) for Homography Estimation Accuracy (default: 1.0)
    --miou-detail, -md <int>: Detail level for MIoU plots (1=average only, 2=average+minimum)
    --quiet, -q: Whether to suppress console output

Example:
    stabilo-optimize plot experiments/example/sample.json -s -md 2
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_results(source: Path, experiment_dir: Optional[Path] = None, show_plots: bool = False, save_plots: bool = False, overwrite: bool = False, hea_threshold: float = 1.0, miou_detail: int = 1, quiet: bool = False) -> None:
    """
    Plot the results of the benchmarking process.
    """
    if source.is_dir():
        config_files = [f for f in source.iterdir() if f.suffix.lower() == '.json']
        for config_file in config_files:
            plot_results(config_file, experiment_dir, show_plots, save_plots, overwrite, hea_threshold, miou_detail, quiet)
        return

    if experiment_dir is None:
        experiment_dir = source.parent

    benchmark_name = source.stem
    results_dir = experiment_dir / 'results' / benchmark_name

    # Process the benchmark and plot results if data is valid
    process_benchmark(benchmark_name, results_dir, experiment_dir, show_plots, save_plots, hea_threshold, miou_detail, quiet)


def process_benchmark(benchmark_name: str, results_dir: Path, experiment_dir: Path, show_plots: bool, save_plots: bool, hea_threshold: float, miou_detail: int, quiet: bool) -> None:
    """
    Process a single benchmark and generate plots.
    """
    runs_filepath = results_dir / 'RUNS.json'

    if not runs_filepath.exists():
        print(f"\033[91mERROR: The benchmark '{benchmark_name}' has not been executed yet\033[0m")
        return

    # Read the RUNS JSON file
    with open(runs_filepath) as f:
        runs = json.load(f)

    # Check if the benchmark has completed all runs
    last_run_number = len(runs['run_number'])
    run_filepath = results_dir / f'{int(last_run_number):07}.json'
    if not run_filepath.exists():
        print(f"\033[93mWARNING: Benchmark '{benchmark_name}' incomplete. Check logs in results directory.\033[0m")

        # Ask the user if they want to continue with plotting the incomplete benchmark results
        user_input = input("Do you want to continue with plotting the incomplete benchmark results? (y/n): ")
        if user_input.lower() != 'y':
            return

    # Process all runs and collect results
    df = collect_benchmark_data(runs, results_dir, hea_threshold)
    if df is None or df.empty:
        print(f"\033[91mERROR: No valid data found for benchmark '{benchmark_name}'\033[0m")
        return

    # Find out which hyperparameter has been varied the most and use it as the x-axis
    metrics_cols = ['CT', 'MBB_IOU_AVG', 'MBB_IOU_MIN', 'HEA', 'HEE', 'detector_name']
    x_axis = df.drop(metrics_cols, axis=1).nunique().idxmax()

    # Print analysis if not in quiet mode
    if not quiet:
        print_benchmark_analysis(df, x_axis)

    # Plot the results
    plot_results_per_detector(df, x_axis, save_plots, show_plots, benchmark_name, experiment_dir, hea_threshold, miou_detail)


def collect_benchmark_data(runs: Dict[str, Any], results_dir: Path, hea_threshold: float) -> pd.DataFrame:
    """
    Collect and process data from all benchmark runs.
    """
    dfs = []

    for run_number in runs['run_number']:
        # Read the results of the current run
        run_filepath = results_dir / f'{int(run_number):07}.json'
        if not run_filepath.exists():
            print(f"\033[93mWARNING: File {run_filepath} does not exist\033[0m")
            continue

        with open(run_filepath) as f:
            run_results = json.load(f)['results']

        # Aggregate the metrics for this run
        CT, MBB_IOU_AVG, MBB_IOU_MIN, HEA, HEE = aggregate_results(run_results, hea_threshold)

        # Create a result dictionary for this run
        run_params = runs['run_number'][run_number].copy()
        run_params.pop('N')
        run_params.pop('seed')
        run_params.pop('transformation_type')

        # Add the metrics to the run parameters
        run_params['CT'] = CT
        run_params['MBB_IOU_AVG'] = MBB_IOU_AVG
        run_params['MBB_IOU_MIN'] = MBB_IOU_MIN
        run_params['HEA'] = HEA
        run_params['HEE'] = HEE

        # Convert to DataFrame and append to list
        dfs.append(pd.DataFrame.from_dict(run_params, orient='index').T)

    # Combine all run DataFrames
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None


def print_benchmark_analysis(df: pd.DataFrame, x_axis: str) -> None:
    """
    Print analysis of benchmark results.
    """
    print(df)

    # Group by detector, mask_use, and clahe and find best configurations
    groups = ['detector_name', 'mask_use', 'clahe']

    # For each metric, show the best configuration
    for metric, label in [
        ('MBB_IOU_AVG', 'Best Average Mean IoU:'),
        ('MBB_IOU_MIN', 'Best Minimum Mean IoU:'),
        ('HEA', 'Best Homography Estimation Accuracy:'),
    ]:
        best_rows = df.groupby(groups).apply(lambda x, metric=metric: x[x[metric] == x[metric].max()])

        print(f"\n{label}")
        columns_to_show = [x_axis, metric]
        print(best_rows[columns_to_show])


def plot_results_per_detector(df: pd.DataFrame, x_axis: str, save_plots: bool, show_plots: bool, benchmark_name: str, experiment_dir: Path, hea_threshold: float, miou_detail: int) -> None:
    """
    Plot the results of the benchmarking process.
    """
    for mask_use in df['mask_use'].unique():
        for clahe_use in df['clahe'].unique():
            plot_results_per_detector_and_mask_clahe(df, x_axis, mask_use, clahe_use, save_plots, benchmark_name, experiment_dir, hea_threshold, miou_detail)

    if show_plots:
        plt.show()


def plot_results_per_detector_and_mask_clahe(df: pd.DataFrame, x_axis: str, mask_use: bool, clahe_use: bool, save_plots: bool, benchmark_name: str, experiment_dir: Path, hea_threshold: float, miou_detail: int) -> None:
    """
    Plot benchmark results for specific mask and CLAHE configuration.
    """
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab10.colors)

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), gridspec_kw={'height_ratios': [2.2, 1]})

    set_log_scale = False

    # Plot data for each detector
    for i, detector in enumerate(sorted(df['detector_name'].unique())):
        # Select rows for current configuration
        df_detector = df[(df['detector_name'] == detector) & (df['mask_use'] == mask_use) & (df['clahe'] == clahe_use)]

        # Extract data for plotting
        x_data = df_detector[x_axis]
        if x_axis == 'downsample_ratio':
            x_data = 1 / x_data

        y_data = {
            'MBB_IOU_AVG': df_detector['MBB_IOU_AVG'],
            'MBB_IOU_MIN': df_detector['MBB_IOU_MIN'],
            'HEA': df_detector['HEA'],
            'CT': df_detector['CT'] * 1000,  # Convert to milliseconds
        }
        MIoU_available = y_data['MBB_IOU_AVG'].notnull().all()

        if y_data['CT'].max() > 1000:
            set_log_scale = True

        # Plot metrics in first subplot
        ax1.plot(x_data, y_data['HEA'], color=f'C{i}', linestyle='--', marker='s')
        if MIoU_available:
            ax1.plot(x_data, y_data['MBB_IOU_AVG'], color=f'C{i}', linestyle=':', marker='o')
            if miou_detail == 2:
                ax1.plot(x_data, y_data['MBB_IOU_MIN'], color=f'C{i}', linestyle='-.', marker='x')

        # Add legend for the first detector only
        if i == 0:
            if MIoU_available:
                if miou_detail == 1:
                    ax1.plot([], [], color='gray', linestyle='--', marker='o', label='MIoU')
                else:
                    ax1.plot([], [], color='gray', linestyle='--', marker='o', label='Avg(MIoU)')
                    ax1.plot([], [], color='gray', linestyle='-.', marker='x', label='Min(MIoU)')
            ax1.plot([], [], color='gray', linestyle=':', marker='s', label='HEA')

        # Plot computation time in second subplot
        ax2.plot(x_data, y_data['CT'], color=f'C{i}', linestyle='-', label=detector)
        ax2.scatter(x_data, y_data['CT'], color=f'C{i}', marker='o')

    # Add titles and labels
    configure_plot_appearance(fig, ax1, ax2, mask_use, clahe_use, x_axis, save_plots, set_log_scale, hea_threshold, MIoU_available)

    # Get handles and labels for legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    # Increase font size of ticks
    ax1.tick_params(axis='both', which='major', labelsize=13)
    ax2.tick_params(axis='both', which='major', labelsize=13)

    # Combine legends in ax1
    lines = lines1 + lines2
    labels = labels1 + labels2
    ax1.legend(lines, labels, loc='best', fontsize=13)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show(block=False)

    # Save the plot if requested
    if save_plots:
        save_benchmark_plot(fig, benchmark_name, experiment_dir, x_axis, mask_use, clahe_use)


def configure_plot_appearance(fig: plt.Figure, ax1: plt.Axes, ax2: plt.Axes, mask_use: bool, clahe_use: bool, x_axis: str, save_plots: bool, set_log_scale: bool, hea_threshold: float, MIoU_available
) -> None:
    """
    Configure appearance of benchmark plots.
    """
    # Create checkmark/cross for mask and CLAHE settings
    mask_text = '✔' if mask_use else '✘'
    clahe_text = '✔' if clahe_use else '✘'

    # Set titles and annotations
    if save_plots:
        text_position = (0.76, 0.97) if x_axis == 'downsample_ratio' else (0.107, 0.97)
        fig.text(
            *text_position,
            f"({mask_text} Mask, {clahe_text} CLAHE)",
            ha='left',
            va='top',
            fontsize=14,
            bbox=dict(facecolor='white', alpha=0.4),
        )
    else:
        if MIoU_available:
            fig.suptitle(f"MBBIOU & HEA vs. {x_axis.replace('_', ' ').title()} ({mask_text} Mask, {clahe_text} CLAHE)")
        else:
            fig.suptitle(f"HEA vs. {x_axis.replace('_', ' ').title()} ({mask_text} Mask, {clahe_text} CLAHE)")
        ax2.set_title(f"Avg. comp. time vs. {x_axis.replace('_', ' ').title()} ({'Mask' if mask_use else 'No Mask'}, {'CLAHE' if clahe_use else 'No CLAHE'})")

    # Set y-axis scale and ticks for computation time
    if set_log_scale:
        ax2.set_yscale('log')
        ax2.text(
            0.985,
            0.95,
            'log scale (y-axis)',
            transform=ax2.transAxes,
            fontsize=14,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.4),
        )

    # Set axis labels
    if MIoU_available:
        ax1.set_ylabel("MIoU & HEA (" + r'$\varepsilon$' + f" = {round(hea_threshold)} pixel)", fontsize=14)
    else:
        ax1.set_ylabel("HEA (" + r'$\varepsilon$' + f" = {round(hea_threshold)} pixel)", fontsize=14)
    ax2.set_ylabel("Avg. computation time (ms)", fontsize=14)

    # Set x-axis labels with special formatting for certain parameters
    if x_axis == 'downsample_ratio':
        ax1.set_xlabel('')
        ax2.set_xlabel(r'Downscaling factor ($\rho^{-1}$)', fontsize=14)
    elif x_axis == 'filter_ratio':
        ax1.set_xlabel('')
        ax2.set_xlabel(r'Filter ratio threshold ($\theta_{\text{SNN}}$)', fontsize=14)
    else:
        ax2.set_xlabel(x_axis.replace('_', ' ').title(), fontsize=14)


def save_benchmark_plot(fig: plt.Figure, benchmark_name: str, experiment_dir: Path, x_axis: str, mask_use: bool, clahe_use: bool) -> None:
    """
    Save benchmark plot to file.
    """
    plots_dir = experiment_dir / 'plots' / benchmark_name
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_filename = (
        f'{x_axis.replace(" ", "_")}_{"Mask" if mask_use else "No_Mask"}_{"CLAHE" if clahe_use else "No_CLAHE"}.pdf'
    )
    fig.savefig(plots_dir / plot_filename)


def aggregate_results(run_results: Dict[str, Any], hea_threshold: float, aggregation_type: str = 'mean') -> Tuple[float, float, float, float, float]:
    """
    Aggregate benchmark results across all scenes.
    """
    CT, MBB_IOU, HEA, HEE = [], [], [], []

    # Collect metrics from all scenes
    for scene in run_results:
        CT.extend(run_results[scene]['Computation_time'])
        MBB_IOU.extend(run_results[scene]['Mean_IoU'])
        HEA.extend(run_results[scene]['Mean_corner_error'])
        HEE.extend(run_results[scene]['Homography_error_norm'])

    # Compute aggregated metrics
    CT = np.mean(CT)
    MBB_IOU_AVG = np.mean(MBB_IOU)
    MBB_IOU_MIN = np.min(MBB_IOU)
    HEA = sum([c_error < hea_threshold for c_error in HEA]) / len(HEA)

    if aggregation_type == 'mean':
        HEE = np.mean(HEE)
    else:  # median
        HEE = np.median(HEE)

    return CT, MBB_IOU_AVG, MBB_IOU_MIN, HEA, HEE


def parse_cli_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Plot the results of the benchmarking process")

    # Required arguments
    parser.add_argument("source", type=Path, help="Filepath to a .json configuration file or a directory containing multiple .json configuration files")

    # Processing arguments
    parser.add_argument("--experiment-dir", "-e", type=Path, default=None, help="Directory to load the benchmark results [default: parent directory of the config file(s)]")
    parser.add_argument("--show-plots", "-s", action="store_true", help="Show the plots [default: False]")
    parser.add_argument("--save-plots", "-sp", action="store_true", help="Save the plots to the benchmark/plots directory")
    parser.add_argument("--overwrite", "-o", action="store_true", help="Overwrite the existing plots")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress the output")

    # Plotting arguments
    parser.add_argument("--hea-threshold", "-ht", type=float, default=1.0, help="Threshold (in pixels) for the Homography Estimation Accuracy (HEA)")
    parser.add_argument("--miou-detail", "-md", type=int, default=1, help="Detail level for MIoU plots: 1 for average only, 2 for both average and minimum")

    return parser.parse_args()


def main() -> None:
    """Entry point for the 'stabilo-optimize plot' subcommand."""
    args = parse_cli_args()
    plot_results(**vars(args))


if __name__ == "__main__":
    main()
