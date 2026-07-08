#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
Visualization utilities for the benchmarking process.
Provides functions to visualize the matching process and save results to video.
"""

import json
import platform
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

# Platform detection
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ['Darwin', 'Linux', 'Windows'])

# Visualization constants
RANDOM_COLORS = np.random.randint(0, 256, (100, 3))  # Random colors for feature matches
DEFAULT_FPS = 5
DEFAULT_FONT = cv2.FONT_HERSHEY_PLAIN

# Text rendering settings
TEXT_CONFIG = {
    'title': {'scale': 6, 'thickness': 4, 'color': (0, 255, 0), 'bg_color': (0, 0, 0)},
    'info': {'scale': 4, 'thickness': 4, 'color': (255, 0, 255), 'bg_color': (0, 0, 0)},
    'help': {'scale': 2, 'thickness': 4, 'color': (255, 255, 255), 'bg_color': (0, 0, 0)},
}

# Line colors
INLIER_COLOR = [0, 255, 0]  # Green
OUTLIER_COLOR = [0, 0, 255]  # Red


def get_video_writer(
    visualization_directory: Path,
    run_params: Dict[str, Any],
    run_number: int,
    video_width: int,
    video_height: int,
    fps: int = DEFAULT_FPS,
) -> cv2.VideoWriter:
    """
    Initialize a video writer for saving visualization results.
    """
    # Create filename based on platform
    filename = f"run_{run_number:07}"
    suffix = '.mp4' if MACOS else '.avi' if WINDOWS else '.mp4'
    video_filepath = str(visualization_directory / filename) + suffix

    # Initialize video writer with platform-specific codec
    fourcc = 'avc1' if MACOS else 'WMV2' if WINDOWS else 'mp4v'
    fourcc = cv2.VideoWriter_fourcc(*fourcc)
    video_writer = cv2.VideoWriter(video_filepath, fourcc, fps, (video_width, video_height))

    # Save run parameters
    with open(visualization_directory / (filename + '.json'), 'w') as f:
        json.dump(run_params, f, indent=4)

    return video_writer


def draw_text(
    img: np.ndarray, text: str, pos: Tuple[int, int] = (0, 0), config: Dict[str, Any] = TEXT_CONFIG['title']
) -> None:
    """
    Draw text on image with background rectangle.
    """
    x, y = pos
    text_size, _ = cv2.getTextSize(text, DEFAULT_FONT, config['scale'], config['thickness'])
    text_w, text_h = text_size

    cv2.rectangle(img, pos, (x + text_w, y + text_h), config['bg_color'], -1)
    cv2.putText(
        img,
        text,
        (x, y + text_h + config['scale'] - 1),
        DEFAULT_FONT,
        config['scale'],
        config['color'],
        config['thickness'],
    )


def draw_feature_visualization(
    frame: np.ndarray, points: Optional[np.ndarray] = None, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draw feature points and mask on a frame.
    """
    # Apply mask if provided
    if mask is not None:
        frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert to BGR for colored visualization
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Draw feature points if provided
    if points is not None:
        for i, pt in enumerate(points):
            x, y = pt.ravel()
            cv2.circle(frame, (int(x), int(y)), 9, RANDOM_COLORS[i % 100].tolist(), 6)

    return frame


def draw_matching_lines(
    combined_img: np.ndarray,
    ref_pts: Optional[np.ndarray],
    cur_pts: Optional[np.ndarray],
    inliers_mask: Optional[np.ndarray],
    inliers_count: int,
) -> Tuple[np.ndarray, str, int]:
    """
    Draw matching lines between reference and current frames.
    """
    match_count = 'N/A'

    if ref_pts is not None and cur_pts is not None:
        w = combined_img.shape[1] // 2
        for i, (pt1, pt2) in enumerate(zip(ref_pts, cur_pts)):
            x1, y1 = pt1.ravel()
            x2, y2 = pt2.ravel()
            color = INLIER_COLOR if inliers_mask[i] else OUTLIER_COLOR
            cv2.line(combined_img, (int(x1), int(y1)), (w + int(x2), int(y2)), color, 1, cv2.LINE_AA)
        match_count = str(len(ref_pts))

    return combined_img, match_count, inliers_count


def render_stabilization_visuals(stabilizer: Any) -> np.ndarray:
    """
    Render comprehensive visualization of the stabilization process.
    """
    # Prepare reference frame visualization
    ref_frame = draw_feature_visualization(stabilizer.ref_frame_gray, stabilizer.ref_pts, stabilizer.ref_mask)
    draw_text(ref_frame, "Reference scene")

    # Prepare current frame visualization
    cur_frame = draw_feature_visualization(stabilizer.cur_frame_gray, stabilizer.cur_pts, stabilizer.cur_mask)

    # Combine frames horizontally
    upper_visualization = np.hstack((ref_frame, cur_frame))

    # Draw matching lines and get statistics
    upper_visualization, match_count, inliers_count = draw_matching_lines(
        upper_visualization,
        stabilizer.ref_pts,
        stabilizer.cur_pts,
        stabilizer.cur_inliers,
        stabilizer.cur_inliers_count,
    )

    # Add information text
    draw_text(
        upper_visualization,
        f"Number of matched points after pre-filtering: {match_count}",
        pos=(ref_frame.shape[1] // 2, 10),
        config=TEXT_CONFIG['info'],
    )
    draw_text(
        upper_visualization,
        f"Number of inliers after RANSAC: {inliers_count}",
        pos=(ref_frame.shape[1] // 2, 80),
        config=TEXT_CONFIG['info'],
    )
    draw_text(upper_visualization, "Press 'q' to close the visualization", pos=(10, 80), config=TEXT_CONFIG['help'])

    # Combine with original frames
    lower_visualization = np.hstack((stabilizer.ref_frame, stabilizer.cur_frame))

    return np.vstack((upper_visualization, lower_visualization))
