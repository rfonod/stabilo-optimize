# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from stabilo_optimize.benchmark import (
    calculate_mean_iou,
    clip_boxes,
    compute_metrics,
    create_runs,
    generate_random_homography,
    load_config,
    run_single,
)

SAMPLE_SCENES_DIR = Path(__file__).resolve().parent.parent / 'experiments' / 'sample_experiment' / 'scenes'

REQUIRED_CONFIG_KEYS = [
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
]


def test_generate_random_homography_shape_and_invertible():
    np.random.seed(0)
    H = generate_random_homography(640, 480)

    assert H.shape == (3, 3)
    assert H[2, 2] == pytest.approx(1.0)
    assert np.linalg.det(H) != 0


def test_clip_boxes_clips_to_image_boundaries():
    boxes = np.array([[-5.0, -5.0, 15.0, 15.0], [50.0, 50.0, 60.0, 60.0]])

    clipped = clip_boxes(boxes, w=10, h=10)

    np.testing.assert_array_equal(clipped, np.array([[0.0, 0.0, 10.0, 10.0], [10.0, 10.0, 10.0, 10.0]]))


def test_clip_boxes_none_passthrough():
    assert clip_boxes(None, w=10, h=10) is None


def test_calculate_mean_iou_identical_boxes():
    box = np.array([[5.0, 5.0, 10.0, 10.0]])

    assert calculate_mean_iou(box, box.copy()) == pytest.approx(1.0)


def test_calculate_mean_iou_partial_overlap():
    box1 = np.array([[5.0, 5.0, 10.0, 10.0]])  # spans [0,10] x [0,10]
    box2 = np.array([[10.0, 5.0, 10.0, 10.0]])  # spans [5,15] x [0,10]

    assert calculate_mean_iou(box1, box2) == pytest.approx(1 / 3)


def test_calculate_mean_iou_none_returns_nan():
    assert np.isnan(calculate_mean_iou(None, np.array([[0.0, 0.0, 1.0, 1.0]])))


def test_compute_metrics_identity_transform_is_error_free():
    boxes = np.array([[5.0, 5.0, 10.0, 10.0]])
    corners = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    H = np.eye(3)

    mean_iou, H_error_norm, mean_corner_error = compute_metrics(boxes, boxes.copy(), H, H, corners, corners.copy())

    assert mean_iou == pytest.approx(1.0)
    assert H_error_norm == pytest.approx(0.0)
    assert mean_corner_error == pytest.approx(0.0)


def test_create_runs_expands_cartesian_product_and_resolves_detector_dict(tmp_path):
    config = {
        'N': [50],
        'seed': [42],
        'detector_name': ['orb', 'sift'],
        'ransac_epipolar_threshold': [{'orb': 1.0, 'sift': 2.0}],
    }

    runs = create_runs(config, tmp_path, resume=False)

    assert len(runs) == 2
    assert runs[1] == {'N': 50, 'seed': 42, 'detector_name': 'orb', 'ransac_epipolar_threshold': 1.0}
    assert runs[2] == {'N': 50, 'seed': 42, 'detector_name': 'sift', 'ransac_epipolar_threshold': 2.0}
    assert (tmp_path / 'RUNS.json').exists()


def test_load_config_valid_round_trip(tmp_path):
    config = {key: [0] for key in REQUIRED_CONFIG_KEYS}
    config_path = tmp_path / 'valid.json'
    config_path.write_text(json.dumps(config))

    assert load_config(config_path) == config


def test_load_config_missing_keys_returns_none(tmp_path):
    config_path = tmp_path / 'missing_keys.json'
    config_path.write_text(json.dumps({'N': [1]}))

    assert load_config(config_path) is None


def test_load_config_invalid_json_returns_none(tmp_path):
    config_path = tmp_path / 'invalid.json'
    config_path.write_text('{not valid json')

    assert load_config(config_path) is None


def test_run_single_is_reproducible_given_same_seed():
    """Same seed -> bit-identical metrics (Computation_time excepted) across repeated runs.

    Regression test: cv2's own RNG (used by stabilo's RANSAC-based homography estimation)
    is not covered by np.random.seed alone and must be seeded separately.
    """
    args = argparse.Namespace(verbosity=0, mask_start_col=1, save_visualization=False, show_visualization=False)
    run_configuration = {
        'seed': 0,
        'N': 5,
        'detector_name': 'orb',
        'matcher_name': 'bf',
        'filter_type': 'ratio',
        'transformation_type': 'projective',
        'clahe': False,
        'downsample_ratio': 0.5,
        'max_features': 2000,
        'ref_multiplier': 2.0,
        'mask_use': True,
        'filter_ratio': 0.5,
        'ransac_method': 38,
        'ransac_epipolar_threshold': 2.0,
        'ransac_max_iter': 5000,
        'ransac_confidence': 0.999999,
        'gpu': False,
    }

    results_a = run_single(run_configuration.copy(), 1, SAMPLE_SCENES_DIR, None, args)
    results_b = run_single(run_configuration.copy(), 1, SAMPLE_SCENES_DIR, None, args)

    assert results_a['results'] and set(results_a['results']) == set(results_b['results'])
    for scene, metrics_a in results_a['results'].items():
        metrics_b = results_b['results'][scene]
        for key in ('Mean_IoU', 'Homography_error_norm', 'Mean_corner_error'):
            assert metrics_a[key] == metrics_b[key], f'{scene}/{key} differs between identically-seeded runs'
