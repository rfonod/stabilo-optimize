#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
benchmark.py - Benchmark various image matching methods and different hyperparameters.

This script benchmarks various image matching methods and different hyperparameters on a set of image scenes.
The benchmark results are saved in a .json file for each run and summarized at the end of the benchmark.
The benchmark can be resumed from the last run or overwrite the results of the previous benchmark run.

Usage:
    stabilo-optimize benchmark <source> [options]

Arguments:
    source          Filepath to a .json configuration file or a directory containing multiple configuration files

Options:
    -e, --experiment-dir    Directory to save the benchmark results [default: parent directory of the config file(s)]
    -o, --overwrite         Overwrite the results of the previous benchmark run
    -r, --resume            Resume the benchmark from the last run
    -v, --verbosity         Verbosity level [default: 0]
                                0: quiet    - top-level status only; stabilo's own log messages fully suppressed
                                1: minimal  - + per-run hyperparameter header and summary table; stabilo errors only
                                2: detailed - + per-scene summary table; stabilo warnings and up
                                3: debug    - + per-trial output; stabilo info and up
    -msc, --mask-start-col  The column index where the mask values start in the bounding box file [default: 1]
    -sp, --save-plots       Save the plots to the benchmark/plots directory
    -s, --show-visualization Visualize the benchmarking process
    -sv, --save-visualization Save the visualization of the benchmarking process
    -l, --log-file          Filepath to also write console output to (prints the resolved path used) [default: none]

Note:
- All scenes should be placed in the 'scenes' directory of the given experiment directory
    - Masks (bounding boxes) should be saved in the same directory as the images with the same filename but a .txt extension
    - Masks should be saved in the format: x y w h (x, y: center coordinates, w, h: width and height)
- If masks (bounding boxes) are unavailable, set "mask_use" to False in the configuration file
- For the '--save-visualization' option to run properly, all scenes should have the same resolution
- The benchmark results are saved in the 'results' directory of the given experiment directory
- Each run is fully reproducible for a given "seed" (numpy and OpenCV's RNG are both seeded); only
  Computation_time varies between repeat runs, since it measures actual wall-clock time
- "gpu": true requires a CUDA-enabled OpenCV build (see stabilo's docs/cuda.md); it mainly affects
  Computation_time, not HEA/MIoU, since RANSAC-based homography/affine estimation always runs on CPU
"""

import argparse
import itertools
import json
import platform
import sys
import time
import traceback
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import psutil
import torch
from stabilo import Stabilizer
from tqdm import tqdm

from stabilo_optimize.utils.logging_utils import configure_stabilo_logging, tee_stdout_to_file
from stabilo_optimize.utils.plot import plot_results
from stabilo_optimize.utils.visualize import get_video_writer, render_stabilization_visuals

IMG_SUFFIXES: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']  # supported image formats


def run_benchmarks(source: Path, args: argparse.Namespace) -> None:
    """
    Run the benchmark with the given configuration file(s).
    """
    configure_stabilo_logging(args.verbosity)

    # Check if the source is a directory
    if source.is_dir():
        config_files = [f for f in source.iterdir() if f.suffix.lower() == '.json']
        for config_file in config_files:
            run_single_benchmark(config_file, args)
        return

    # Run the benchmark for a single configuration file
    run_single_benchmark(source, args)


def run_single_benchmark(config_file: Path, args: argparse.Namespace) -> None:
    """
    Run a single benchmark with the given configuration file.
    """

    # Load the benchmark configuration file
    config = load_config(config_file)
    if config is None:
        return

    # Get the experiment directory
    experiment_dir = config_file.parent if args.experiment_dir is None else args.experiment_dir

    # Get benchmark name from the config file
    benchmark_name = config_file.stem

    # Get the results directory path
    results_dir = experiment_dir / 'results' / benchmark_name

    # Check if results from a previous run should be overwritten
    if not args.overwrite and (results_dir / 'SUCCESS.txt').exists():
        print(f"\033[33mSkipping '{benchmark_name}' as results already exist and overwrite is disabled.\033[0m")
        return

    # Create results directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)

    # Generate all combinations of methods and hyperparameters
    runs = create_runs(config, results_dir, args.resume)

    # Define scenes directory path
    scenes_dir = experiment_dir / 'scenes'

    # Print the configuration file
    print(f"\033[92mBenchmarking '{benchmark_name}.json' with {len(runs)} combinations of methods and hyperparameters.\033[0m")
    print(f"\033[92mEach combination will be evaluated {config['N'][0]} times on each scene in the 'scenes' directory.\033[0m")
    print(f"\033[92mResults will be saved in '{results_dir}'\033[0m")

    # Create visualization directory only if needed
    visualization_directory = None
    if args.save_visualization:
        visualization_directory = experiment_dir / 'visualizations' / benchmark_name
        visualization_directory.mkdir(parents=True, exist_ok=True)

    # Iterate over combinations and create dictionaries for each run
    try:
        for run_number in tqdm(runs, desc="Benchmark run", unit="runs", leave=True):
            # Get the methods and hyperparameters for the current run
            run_configuration = runs[run_number]

            # Skip the current run if the results file already exists
            if args.resume and (results_dir / f"{run_number:07}.json").exists():
                tqdm.write(f"Skipping benchmark run {run_number}/{len(runs)} as the results file already exists.")
                continue

            # Run benchmark with current parameters
            results = run_single(run_configuration.copy(), run_number, scenes_dir, visualization_directory, args)

            # Save results
            save_results(results, results_dir, run_number)
    except Exception as e:
        exception_handler(run_number, run_configuration, len(runs), results_dir, e)
    else:
        logging_handler(run_number, len(runs), results_dir, args.resume)

    if args.save_plots:
        # Create plots directory when needed
        plots_directory = experiment_dir / 'plots' / benchmark_name
        plots_directory.mkdir(parents=True, exist_ok=True)
        plot_results(config_file, save_plots=True, overwrite=args.overwrite, quiet=args.verbosity == 0)


def run_single(
    run_configuration: Dict[str, Any], run_number: int, scenes_dir: Path,
    visualization_directory: Optional[Path], args: argparse.Namespace
) -> Dict[str, Any]:
    """
    Run a single benchmark with the given parameters and return the results.
    """

    # Print the methods and hyperparameters for the current run
    if args.verbosity > 0:
        msg = [f"{method: <25} - {param}" for method, param in run_configuration.items()]
        print(
            f"{48*'-'}\nMethods and hyperparameters for the current run:\n{48*'-'}\n",
            '\n'.join(msg),
            f"\n{48*'-'}",
            sep='',
        )

    # Set seed for reproducibility. np.random.seed covers the injected perturbations
    # (generate_random_homography, apply_random_photometric_distortion); cv2.setRNGSeed covers
    # OpenCV's own RNG, used internally by stabilo's RANSAC-based homography/affine estimation
    # (cv2.findHomography/estimateAffinePartial2D), which np.random.seed does not reach.
    seed = run_configuration.pop('seed')
    np.random.seed(seed)
    cv2.setRNGSeed(seed)

    # Get number of runs per scene
    N = run_configuration.pop('N')

    # Initialize results dictionary
    results = {'results': {}, 'params': run_configuration.copy()}

    # Initialize video writer
    video_writer = None

    try:
        # Get the paths to the image scenes saved in the scenes directory
        scenes_path = [s for s in scenes_dir.iterdir() if s.is_file() and s.suffix.lower() in IMG_SUFFIXES]

        if not scenes_path:
            raise FileNotFoundError(f"No valid image files found in {scenes_dir}")

        # Initialize a video writer
        if args.save_visualization:
            try:
                scene = cv2.imread(str(scenes_path[0]))
                if scene is None:
                    raise ValueError(f"Could not read image: {scenes_path[0]}")
                video_height, video_width = scene.shape[:2]
                video_writer = get_video_writer(visualization_directory, run_configuration, run_number, video_width, video_height)
            except Exception as e:
                print(f"\033[91mFailed to initialize the video writer: {e}\033[0m")
                args.save_visualization = False  # Disable saving visualization

        # Iterate over the scenes
        for scene_filepath in tqdm(scenes_path, desc="Scene", disable=args.verbosity > 2, unit="scenes", leave=False):
            scene_name = scene_filepath.name

            try:
                # Initialize a stabilizer object
                stabilizer = Stabilizer(
                    benchmark=True,
                    viz=True if (args.show_visualization or args.save_visualization) else False,
                    min_good_match_count_warning=20 if args.verbosity > 2 else -1,
                    min_inliers_match_count_warning=10 if args.verbosity > 2 else -1,
                    **run_configuration,
                )

                # Load the scene and the mask
                scene, boxes, h, w = load_image_and_boxes(scene_filepath, args.mask_start_col)
                if scene is None:
                    raise ValueError(f"Could not load scene: {scene_filepath}")

                # Set the reference frame
                stabilizer.set_ref_frame(scene, boxes)

                mean_ious, H_errors, mean_corner_errors, times = [], [], [], []
                for run_num in range(N):
                    try:
                        # Generate a random homography matrix
                        H_orig = generate_random_homography(w, h)

                        # Apply random photometric distortion to the original scene
                        scene_distorted = apply_random_photometric_distortion(scene)

                        # Warp the distorted scene and boxes using the generated homography matrix
                        scene_distorted_warped = warp_scene(scene_distorted, H_orig, w, h)
                        boxes_warped = stabilizer.transform_boxes(boxes, H_orig) if boxes is not None else None

                        # Clip the boxes to the image boundaries
                        boxes_warped_clipped = clip_boxes(boxes_warped, w, h)

                        # Stabilize the warped scene back to the reference frame
                        start_time = time.time()
                        stabilizer.stabilize(scene_distorted_warped, boxes_warped_clipped)
                        times.append(time.time() - start_time)

                        # Get the estimated homography matrix
                        H_est = stabilizer.get_cur_trans_matrix()

                        # Transform the boxes back to the reference frame
                        boxes_warped_back = stabilizer.transform_boxes(boxes_warped, H_est, 'xywh', 'xywh') if boxes_warped is not None else None

                        # Transform the four corners of the reference frame to the warped frame and back
                        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 2)
                        corners_warped = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H_orig).reshape(-1, 2)
                        corners_warped_back = cv2.perspectiveTransform(corners_warped.reshape(-1, 1, 2), H_est).reshape(-1, 2)

                        # Compute metrics
                        mean_iou, H_error, mean_corner_error = compute_metrics(boxes, boxes_warped_back, H_orig, H_est, corners, corners_warped_back)

                        # Print the homography error, mIoU, and the mean time for each scene
                        if args.verbosity > 2:
                            print(f"{run_num+1:3}/{N} - {scene_name: <24} - {times[-1]:.6f} | {mean_iou:8.6f} | {mean_corner_error:.6f} | {H_error:.6f} ")

                        # Store the results per run
                        mean_ious.append(mean_iou)
                        H_errors.append(H_error)
                        mean_corner_errors.append(mean_corner_error)

                        # Visualize the stabilization process
                        if args.show_visualization or args.save_visualization:
                            imgs = render_stabilization_visuals(stabilizer)

                            # Save the visualization
                            if args.save_visualization and video_writer is not None:
                                imgs = cv2.resize(imgs, (video_width, video_height))
                                video_writer.write(imgs)

                            # Show the visualization
                            if args.show_visualization:
                                cv2.imshow("Stabilization process visualization (Press any key to continue)", imgs)
                                if cv2.waitKey(0) & 0xFF == ord('q'):
                                    args.show_visualization = False
                                    cv2.destroyAllWindows()
                    except Exception as e:
                        tqdm.write(f"\033[93mWarning: Failed on run {run_num+1}/{N} for scene {scene_name}: {e}\033[0m")
                        continue

                # Store the results per scene
                if mean_ious:  # Only store if we have results
                    results['results'][scene_name] = {
                        'Mean_IoU': mean_ious,
                        'Homography_error_norm': H_errors,
                        'Mean_corner_error': mean_corner_errors,
                        'Computation_time': times,
                    }
                else:
                    tqdm.write(f"\033[93mWarning: No valid results for scene {scene_name}\033[0m")

            except Exception as e:
                tqdm.write(f"\033[93mWarning: Failed to process scene {scene_name}: {str(e)}\033[0m")
                # Store empty results for this scene
                results['results'][scene_name] = {
                    'Mean_IoU': [],
                    'Homography_error_norm': [],
                    'Mean_corner_error': [],
                    'Computation_time': [],
                }
                if args.verbosity > 1:
                    tqdm.write(traceback.format_exc())

        # Summarize the results only if there are valid results
        if any(results['results']):
            summarize_results(results, args.verbosity)
        else:
            print("\033[91mNo valid results were produced for any scene\033[0m")

    except Exception as e:
        print(f"\033[91mCritical error in run_single: {str(e)}\033[0m")
        if args.verbosity > 0:
            print(traceback.format_exc())
        # Ensure we return at least a basic result structure
        results['error'] = str(e)
        results['traceback'] = traceback.format_exc()

    finally:
        # Release the video writer
        if video_writer is not None:
            try:
                video_writer.release()
            except Exception as e:
                print(f"\033[93mWarning: Failed to release video writer: {e}\033[0m")

        # Close all windows
        if args.show_visualization:
            cv2.destroyAllWindows()

    return results


def load_config(config_filepath: Path) -> Optional[Dict[str, Any]]:
    """
    Load the benchmark configuration from a .json file.
    """

    # Load the benchmark configuration
    try:
        with open(config_filepath, 'r') as f:
            try:
                config = json.load(f)
            except json.JSONDecodeError as e:
                print(
                    f"\033[91mFailed to load the benchmark configuration from {config_filepath}. Invalid JSON format!\033[0m\n{e}"
                )
                return None
            except Exception as e:
                print(f"\033[91mFailed to load the benchmark configuration from {config_filepath}!\033[0m\n{e}")
                return None
    except Exception as e:
        print(f"\033[91mCould not open configuration file {config_filepath}!\033[0m\n{e}")
        return None

    # Check if the config file contains all required keys
    required_keys = {
        'N',
        'seed',
        'detector_name',
        'matcher_name',
        'filter_type',
        'transformation_type',
        'clahe',
        'downsample_ratio',
        'max_features',
        'ref_multiplier',
        'mask_use',
        'filter_ratio',
        'ransac_method',
        'ransac_epipolar_threshold',
        'ransac_max_iter',
        'ransac_confidence',
        'gpu',
    }

    missing_keys = required_keys - set(config.keys())
    if missing_keys:
        print(f"\033[91mMissing keys: {', '.join(missing_keys)}\033[0m")
        return None

    return config


def create_runs(config: Dict[str, Any], results_dir: Path, resume: bool) -> Dict[int, Dict[str, Any]]:
    """
    Generate all combinations of methods and hyperparameters as specified in the config file.
    """
    # Create a dictionary to store system metadata
    system_metadata = {
        'start_date': time.strftime("%d/%m/%Y"),
        'start_time': time.strftime("%H:%M:%S"),
        'os': {
            'system': platform.system(),
            'platform': platform.platform(),
            'release': platform.release(),
            'version': platform.version(),
        },
        'software': {
            'python': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'stabilo': version("stabilo"),
            'opencv': cv2.__version__,
            'numpy': np.__version__,
            'torch': torch.__version__,
        },
        'hardware': {
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'cpu_cores': f"{psutil.cpu_count(logical=False)} core(s)",
            'ram': f"{psutil.virtual_memory().total / 1024**3:.2f} GB",
        },
    }

    # Add GPU information if available
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        system_metadata['hardware'].update(
            {
                'gpu': f"{gpu_count} device(s)",
                'gpu_name': [torch.cuda.get_device_name(i) for i in range(gpu_count)],
                'gpu_memory': [
                    f"{torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB" for i in range(gpu_count)
                ],
            }
        )
    else:
        system_metadata['hardware'].update({'gpu': "0 devices", 'gpu_name': [], 'gpu_memory': []})

    # Initialize runs dictionary with metadata
    runs = {'metadata': system_metadata}

    # Generate all combinations of methods and hyperparameters (values only)
    all_combinations = list(itertools.product(*config.values()))

    # Get the index of the detector in the config dictionary
    detector_index_in_config = list(config.keys()).index('detector_name')

    # Create a dictionary to store the methods and hyperparameters for each run
    runs['run_number'] = {}
    for run_number, configuration in enumerate(all_combinations, 1):
        # If a parameter in the configuration is a dictionary, replace it with the value from the corresponding detector
        configuration_list = list(configuration)
        for i, parameter in enumerate(configuration_list):
            if isinstance(parameter, dict):
                configuration_list[i] = parameter[configuration_list[detector_index_in_config]]

        # Create a dictionary to link the methods and hyperparameters to the run number inside the runs dictionary
        run_params = dict(zip(config.keys(), configuration_list))

        # Create a dictionary to store the methods and hyperparameters for the current run
        runs['run_number'][run_number] = run_params

    if resume:
        # Load the methods and hyperparameters from the saved .json file
        with open(results_dir / 'RUNS.json', 'r') as f:
            runs_saved = json.load(f)

        # Verify that the saved run parameters match the generated run parameters
        runs_saved['run_number'] = {int(k): v for k, v in runs_saved['run_number'].items()}
        if runs_saved['run_number'] != runs['run_number']:
            raise ValueError("Mismatch between saved and generated run parameters. Check the config file or delete RUNS.json to start fresh.")

        # Add the new metadata to the runs dictionary
        resume_number = len([key for key in runs_saved.keys() if key.startswith('metadata - resume')]) + 1
        runs_saved[f"metadata - resume {resume_number:02}"] = runs['metadata']
        runs = runs_saved

    # Save the methods, hyperparameters and (updated) metadata to a .json file
    with open(results_dir / 'RUNS.json', 'w') as f:
        json.dump(runs, f, indent=4)

    return runs['run_number']


def load_image_and_boxes(
    scene_path: Path, mask_start_col: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, int]:
    """
    Load an image and its associated bounding boxes.
    """

    # Load the scene
    scene = cv2.imread(str(scene_path))

    # Get the scene height and width
    h, w = scene.shape[:2]

    # Load the bounding boxes, if they exist
    boxes = None
    boxes_path = scene_path.with_suffix('.txt')
    if boxes_path.exists():
        boxes = np.loadtxt(str(boxes_path), delimiter=' ')
        boxes = boxes[:, mask_start_col : mask_start_col + 4]

        # If boxes are in normalized coordinates, convert them to absolute coordinates
        if np.max(boxes) <= 1:
            boxes[:, 0] *= w
            boxes[:, 1] *= h
            boxes[:, 2] *= w
            boxes[:, 3] *= h

    return scene, boxes, h, w


def generate_random_homography(w: Union[int, float], h: Union[int, float]) -> np.ndarray:
    """
    Generate a random homography matrix for image transformation.
    """

    # Generate random translation (+/- 10% of the image size)
    tx = np.random.uniform(-w / 10, w / 10)
    ty = np.random.uniform(-h / 10, h / 10)
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

    # Generate random rotation (+/- 15 degrees) around the center of the image
    angle = np.random.uniform(-15, 15)
    R = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    R = np.vstack((R, [0, 0, 1]))

    # Generate random scale (+/- 5%)
    s = np.random.uniform(0.95, 1.05)
    S = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]])

    # Generate random projective transformation
    p1 = np.random.uniform(-0.00005, 0.00005)
    p2 = np.random.uniform(-0.00005, 0.00005)
    P = np.array([[1, 0, 0], [0, 1, 0], [p1, p2, 1]])

    # Generate the homography matrix
    H = T @ R @ S @ P

    return H


def apply_random_photometric_distortion(img: np.ndarray) -> np.ndarray:
    """
    Apply random photometric distortion to the image.
    """

    # Adjust the brightness randomly
    img = adjust_random_brightness(img)

    # Adjust the saturation randomly
    img = adjust_random_saturation(img)

    # Apply random blur
    img = apply_random_blur(img)

    # Add random fog
    img = add_random_fog(img)

    return img


def adjust_random_brightness(img: np.ndarray) -> np.ndarray:
    """
    Adjust the brightness of an image randomly.
    """
    factor = np.random.uniform(0.75, 1.25)
    return cv2.convertScaleAbs(img, alpha=factor, beta=0)


def adjust_random_saturation(img: np.ndarray) -> np.ndarray:
    """
    Adjust the saturation of an image randomly.
    """
    saturation_factor = np.random.uniform(0.95, 1.05)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1] * saturation_factor, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)


def apply_random_blur(img: np.ndarray, max_blur: int = 5) -> np.ndarray:
    """
    Apply random blur to an image.
    """
    blur_factor = np.random.randint(0, max_blur)
    ksize = 2 * (blur_factor // 2) + 1  # Ensure odd kernel size
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def add_random_fog(img: np.ndarray, max_fog_density: float = 0.1) -> np.ndarray:
    """
    Add random fog effect to an image.
    """
    fog = np.ones_like(img) * 255
    alpha = np.random.uniform(0, max_fog_density)
    return cv2.addWeighted(img, 1 - alpha, fog, alpha, 0)


def warp_scene(img: np.ndarray, H: np.ndarray, w: int, h: int) -> np.ndarray:
    """
    Warp an image using a homography matrix.
    """
    img = cv2.warpPerspective(img, H, (w, h))
    return img


def clip_boxes(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    """
    Clip bounding boxes to the image boundaries.
    """
    if boxes is None:
        return None

    boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h)

    return boxes


def compute_metrics(
    boxes: Optional[np.ndarray],
    boxes_warped_back: Optional[np.ndarray],
    H: np.ndarray,
    H_est: np.ndarray,
    corners: np.ndarray,
    corners_warped_back: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute the metrics for the benchmark.
    """

    # Calculate the mean IoU between the original and the warped-back boxes
    mean_iou = np.float64(calculate_mean_iou(boxes, boxes_warped_back))

    # Calculate the error between the estimated homography matrix and the generated one
    H_error_norm = np.float64(np.linalg.norm(np.linalg.inv(H) - H_est, 'fro'))

    # Calculate the mean pixel error between the original and the warped-back image corners
    mean_corner_error = np.float64(np.linalg.norm(corners - corners_warped_back, axis=1).mean())

    return mean_iou, H_error_norm, mean_corner_error


def calculate_mean_iou(box1: Optional[np.ndarray], box2: Optional[np.ndarray]) -> float:
    """
    Calculate the mean intersection over union between two sets of bounding boxes.
    """

    if box1 is None or box2 is None:
        return np.nan

    # Calculate the top left and bottom right coordinates of each box
    box1_tl = box1[:, :2] - box1[:, 2:] / 2
    box1_br = box1[:, :2] + box1[:, 2:] / 2
    box2_tl = box2[:, :2] - box2[:, 2:] / 2
    box2_br = box2[:, :2] + box2[:, 2:] / 2

    # Calculate the intersection coordinates
    tl = np.maximum(box1_tl, box2_tl)
    br = np.minimum(box1_br, box2_br)

    # Calculate the intersection area
    intersection = np.maximum(0, br - tl)
    intersection_area = intersection[:, 0] * intersection[:, 1]

    # Calculate the union area
    box1_area = box1[:, 2] * box1[:, 3]
    box2_area = box2[:, 2] * box2[:, 3]
    union_area = box1_area + box2_area - intersection_area

    # Calculate the intersection over union
    iou = intersection_area / union_area

    # Calculate the mean iou
    mean_iou = np.mean(iou)

    return mean_iou


def save_results(results: Dict[str, Any], results_dir: Path, run_number: int) -> None:
    """
    Save the results of the benchmark in a .json file.
    """
    filename = results_dir / f"{run_number:07}.json"
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(f"Failed to save the results for {filename}!\n{e}")
        return


def summarize_results(results: Dict[str, Any], verbosity: int) -> None:
    """
    Summarize the results of the benchmark.
    """
    if verbosity > 1:
        print(
            f"\n{'Scene filename':26} Avg. comp. time |   Avg. mIoU  Avg. C error Avg. H error | Median mIoU  Median C. error Median H error"
        )
        print(129 * "-")
        for scene_name, scene_results in results['results'].items():
            print(
                f"{scene_name:24}   "
                f"{np.mean(scene_results['Computation_time']):15.6f} | "
                f"{np.mean(scene_results['Mean_IoU']):11.6f}  "
                f"{np.mean(scene_results['Mean_corner_error']):12.6f} | "
                f"{np.mean(scene_results['Homography_error_norm']):12.6f} "
                f"{np.median(scene_results['Mean_IoU']):11.6f}  "
                f"{np.median(scene_results['Mean_corner_error']):16.6f}"
                f"{np.median(scene_results['Homography_error_norm']):14.6f}"
            )

    # Print the average error, mIoU, and time over all scenes
    if verbosity > 0:
        print(
            "\nAvg. comp. time | Avg. mIoU | Avg. C error | Avg. H error | Median mIoU | Median C. error | Median H error"
        )
        print(106 * "-")
        print(
            f"{np.mean([scene_results['Computation_time'] for scene_results in results['results'].values()]):15.6f} | "
            f"{np.mean([scene_results['Mean_IoU'] for scene_results in results['results'].values()]):9.6f} | "
            f"{np.mean([scene_results['Mean_corner_error'] for scene_results in results['results'].values()]):12.6f} | "
            f"{np.mean([scene_results['Homography_error_norm'] for scene_results in results['results'].values()]):12.6f} | "
            f"{np.median([scene_results['Mean_IoU'] for scene_results in results['results'].values()]):11.6f} | "
            f"{np.median([scene_results['Mean_corner_error'] for scene_results in results['results'].values()]):15.6f} | "
            f"{np.median([scene_results['Homography_error_norm'] for scene_results in results['results'].values()]):14.6f}\n"
        )


def exception_handler(
    run_number: int, run_params: Dict[str, Any], runs_total: int, results_dir: Path, e: Exception
) -> None:
    """
    Handle exceptions during benchmark execution.
    """
    msg = f"FAILURE: The benchmark failed at run {run_number}/{runs_total}!\n\n"
    msg += f"Date and time: {time.strftime('%d/%m/%Y %H:%M:%S')}\n"
    msg += f"{'-'*54}\nError message:\n{'-'*54}\n"
    msg += f"Error message:\n {traceback.format_exc()}\n"
    msg += f"{'-'*54}\nMethods and hyperparameters for the failed run {run_number}:\n{'-'*54}\n"
    msg += '\n'.join([f"{method: <25} - {param}" for method, param in run_params.items()])
    msg += f"\n{'-'*54}\n\n"

    print(msg, e)
    with open(results_dir / 'FAILURE.txt', 'a') as f:
        f.write(msg)
    exit(1)


def logging_handler(run_number: int, runs_total: int, results_dir: Path, resume: bool) -> None:
    """
    Handle benchmark logging and create appropriate log files.
    """
    if resume:
        filenames = ['FAILURE.txt', 'WARNING.txt']
        [(results_dir / filename).unlink() for filename in filenames if (results_dir / filename).exists()]

    if run_number == runs_total:
        msg = "SUCCESS: All benchmark runs completed successfully!\n"
        msg += f"The last run was {run_number}/{runs_total}"
        filename = results_dir / 'SUCCESS.txt'
    else:
        msg = "WARNING: Benchmark not fully completed!\n"
        msg += f"The last run was {run_number}, while the total number of planned runs was {runs_total}."
        filename = results_dir / 'WARNING.txt'

    print(msg)
    with open(filename, 'w') as f:
        f.write(msg)


def parse_cli_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Benchmark image matching methods and hyperparameters")

    # Main arguments
    parser.add_argument("source", type=Path, help="Filepath to a .json configuration file or directory containing configuration files",)
    parser.add_argument("--experiment-dir", "-e", type=Path, default=None, help="Directory to save the benchmark results [default: parent directory of the config file(s)]",)

    # Processing and output options
    exclusive_group = parser.add_mutually_exclusive_group()
    exclusive_group.add_argument("-o", "--overwrite", action="store_true", help="Overwrite previous benchmark results")
    exclusive_group.add_argument("--resume", "-r", action="store_true", help="Resume the benchmark from the last run")
    parser.add_argument(
        "--verbosity", "-v", type=int, default=0, choices=[0, 1, 2, 3],
        help="Verbosity level: 0=quiet (top-level status only; stabilo's own log messages fully suppressed), "
             "1=minimal (+ per-run hyperparameter header and summary table; stabilo errors only), "
             "2=detailed (+ per-scene summary table; stabilo warnings and up), "
             "3=debug (+ per-trial output; stabilo info and up) [default: 0]",
    )

    # Mask options
    parser.add_argument("--mask-start-col", "-msc", type=int, default=1, help="Column index where mask values start in the bbox file")

    # Visualization options
    parser.add_argument("--save-plots", "-sp", action="store_true", help="Save plots to the 'plots' directory")
    parser.add_argument("--show-visualization", "-s", action="store_true", help="Visualize the benchmarking process")
    parser.add_argument("--save-visualization", "-sv", action="store_true", help="Save the visualization of the benchmarking process")

    # Logging options
    parser.add_argument("--log-file", "-l", type=Path, default=None, help="Filepath to also write console output to (parent directories are created if needed)")

    return parser.parse_args()


def main() -> None:
    """Entry point for the 'stabilo-optimize benchmark' subcommand."""
    args = parse_cli_args()
    with tee_stdout_to_file(args.log_file):
        run_benchmarks(args.source, args)


if __name__ == "__main__":
    main()
