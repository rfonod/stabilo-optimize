# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

import pytest

from stabilo_optimize.utils.plot import aggregate_results

RUN_RESULTS = {
    'scene1': {
        'Computation_time': [0.1, 0.2],
        'Mean_IoU': [0.8, 0.9],
        'Mean_corner_error': [0.5, 1.5],
        'Homography_error_norm': [0.01, 0.03],
    },
    'scene2': {
        'Computation_time': [0.3],
        'Mean_IoU': [0.7],
        'Mean_corner_error': [2.0],
        'Homography_error_norm': [0.05],
    },
}


def test_aggregate_results_mean():
    CT, MBB_IOU_AVG, MBB_IOU_MIN, HEA, HEE = aggregate_results(RUN_RESULTS, hea_threshold=1.0)

    assert CT == pytest.approx(0.2)
    assert MBB_IOU_AVG == pytest.approx(0.8)
    assert MBB_IOU_MIN == pytest.approx(0.7)
    assert HEA == pytest.approx(1 / 3)  # only 0.5 (of 0.5, 1.5, 2.0) is below the 1.0 threshold
    assert HEE == pytest.approx(0.03)


def test_aggregate_results_median():
    _, _, _, _, HEE = aggregate_results(RUN_RESULTS, hea_threshold=1.0, aggregation_type='median')

    assert HEE == pytest.approx(0.03)  # median of [0.01, 0.03, 0.05]
