import math

import numpy as np
import pytest

from terrain_navigation_pkg.path_clearance_core import evaluate_path_clearance
from terrain_navigation_pkg.path_clearance_core import PathPose2D
from terrain_navigation_pkg.path_clearance_core import sample_path
from terrain_navigation_pkg.path_clearance_core import (
    trim_path_to_nearest_position,
)


def evaluate(points, poses=None):
    if poses is None:
        poses = [PathPose2D(0.0, 0.0, 0.0)]
    return evaluate_path_clearance(
        poses,
        np.asarray(points, dtype=float).reshape((-1, 2)),
        vehicle_front_m=2.4,
        vehicle_rear_m=2.5,
        vehicle_half_width_m=1.0,
        hard_clearance_m=0.35,
        preferred_clearance_m=0.75,
    )


def test_obstacle_inside_hard_margin_is_invalid():
    result = evaluate([[0.0, 1.20]])
    assert result.minimum_clearance_m == pytest.approx(0.20)
    assert not result.hard_valid
    assert not result.preferred_valid


def test_obstacle_between_hard_and_preferred_margin_is_hard_only():
    result = evaluate([[0.0, 1.50]])
    assert result.minimum_clearance_m == pytest.approx(0.50)
    assert result.hard_valid
    assert not result.preferred_valid


def test_obstacle_beyond_preferred_margin_is_preferred_valid():
    result = evaluate([[0.0, 1.90]])
    assert result.minimum_clearance_m == pytest.approx(0.90)
    assert result.hard_valid
    assert result.preferred_valid


def test_clearance_uses_rotated_vehicle_body():
    result = evaluate(
        [[-1.20, 0.0]],
        [PathPose2D(0.0, 0.0, math.pi / 2.0)],
    )
    assert result.minimum_clearance_m == pytest.approx(0.20)
    assert not result.hard_valid


def test_path_sampling_is_bounded_and_uses_tangent_heading():
    sampled = sample_path(
        [PathPose2D(0.0, 0.0, 0.0), PathPose2D(0.0, 10.0, 0.0)],
        spacing_m=0.25,
        maximum_length_m=3.0,
    )
    assert sampled[-1].distance <= 3.0
    assert len(sampled) >= 12
    assert all(pose.yaw == pytest.approx(math.pi / 2.0) for pose in sampled)


def test_empty_cloud_is_clear():
    result = evaluate([])
    assert math.isinf(result.minimum_clearance_m)
    assert result.hard_valid
    assert result.preferred_valid


def test_trim_path_projects_vehicle_and_discards_completed_prefix():
    trimmed = trim_path_to_nearest_position(
        [
            PathPose2D(0.0, 0.0, 0.0),
            PathPose2D(10.0, 0.0, 0.0),
            PathPose2D(20.0, 0.0, 0.0),
        ],
        vehicle_x=12.0,
        vehicle_y=1.0,
    )
    assert trimmed.start_segment_index == 1
    assert trimmed.nearest_distance_m == pytest.approx(1.0)
    assert len(trimmed.poses) == 2
    assert trimmed.poses[0].x == pytest.approx(12.0)
    assert trimmed.poses[0].y == pytest.approx(0.0)
    assert trimmed.poses[-1].x == pytest.approx(20.0)


def test_trimmed_path_sampling_starts_at_current_vehicle_progress():
    trimmed = trim_path_to_nearest_position(
        [
            PathPose2D(0.0, 0.0, 0.0),
            PathPose2D(10.0, 0.0, 0.0),
            PathPose2D(20.0, 0.0, 0.0),
        ],
        vehicle_x=12.0,
        vehicle_y=0.0,
    )
    sampled = sample_path(trimmed.poses, 0.25, 30.0)
    assert sampled[0].x == pytest.approx(12.0)
    assert sampled[0].distance == pytest.approx(0.0)
    assert sampled[-1].x == pytest.approx(20.0)
    assert sampled[-1].distance == pytest.approx(8.0)


def test_trim_empty_path_is_explicit():
    trimmed = trim_path_to_nearest_position([], 1.0, 2.0)
    assert trimmed.poses == tuple()
    assert math.isinf(trimmed.nearest_distance_m)
    assert trimmed.start_segment_index == -1
