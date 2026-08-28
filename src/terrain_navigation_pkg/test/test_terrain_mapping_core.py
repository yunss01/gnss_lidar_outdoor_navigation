import math

import numpy as np
import pytest

from terrain_navigation_pkg.terrain_mapping_core import RollingVoxelMap
from terrain_navigation_pkg.terrain_mapping_core import build_terrain_layers
from terrain_navigation_pkg.terrain_mapping_core import filter_sensor_points
from terrain_navigation_pkg.terrain_mapping_core import (
    quaternion_rotation_matrix,
)
from terrain_navigation_pkg.terrain_mapping_core import transform_xyzi


def test_transform_xyzi_rotates_translates_and_preserves_intensity():
    half_angle = 0.25 * math.pi
    rotation = quaternion_rotation_matrix(
        0.0, 0.0, math.sin(half_angle), math.cos(half_angle)
    )
    points = np.asarray([[1.0, 0.0, 2.0, 0.7]], dtype=np.float32)

    transformed = transform_xyzi(
        points, rotation, np.asarray([10.0, 20.0, 1.0])
    )

    np.testing.assert_allclose(
        transformed[0], [10.0, 21.0, 3.0, 0.7], atol=1e-6
    )


def test_filter_sensor_points_removes_ego_and_bad_ranges():
    points = np.asarray([
        [0.0, 0.0, -1.0, 1.0],
        [3.0, 0.0, -1.0, 2.0],
        [60.0, 0.0, 0.0, 3.0],
        [math.nan, 0.0, 0.0, 4.0],
    ], dtype=np.float32)

    filtered = filter_sensor_points(
        points, 0.5, 50.0, -2.8, 2.8, 1.25
    )

    assert filtered.shape == (1, 4)
    assert filtered[0, 0] == 3.0


def test_rolling_voxel_map_fuses_and_prunes():
    voxel_map = RollingVoxelMap(1.0)
    voxel_map.update(np.asarray([
        [0.1, 0.1, 0.1, 1.0],
        [0.9, 0.8, 0.2, 3.0],
        [5.1, 0.0, 0.0, 4.0],
    ], dtype=np.float32), 1.0)

    assert len(voxel_map) == 2
    close_point = voxel_map.points()[
        np.argmin(np.linalg.norm(voxel_map.points()[:, :2], axis=1))
    ]
    np.testing.assert_allclose(
        close_point, [0.5, 0.45, 0.15, 2.0], atol=1e-6
    )

    removed = voxel_map.prune((0.0, 0.0), 2.0, 2.0)
    assert removed == 1
    assert len(voxel_map) == 1


def _grid_points(height_function):
    values = []
    for y_value in np.arange(-1.0, 1.01, 0.25):
        for x_value in np.arange(-1.0, 1.01, 0.25):
            values.append([
                x_value,
                y_value,
                height_function(x_value, y_value),
                1.0,
            ])
    return np.asarray(values, dtype=np.float32)


def test_flat_terrain_has_low_cost():
    layers = build_terrain_layers(
        _grid_points(lambda _x, _y: 0.0),
        (0.0, 0.0),
        3.0,
        0.25,
        1,
        8.0,
        25.0,
        0.10,
        0.35,
    )
    known = layers.traversability_cost >= 0
    assert np.count_nonzero(known) > 0
    assert np.max(layers.traversability_cost[known]) == 0


def test_steep_terrain_and_step_reach_lethal_cost():
    slope_layers = build_terrain_layers(
        _grid_points(lambda x_value, _y: x_value),
        (0.0, 0.0),
        3.0,
        0.25,
        1,
        8.0,
        25.0,
        0.10,
        0.35,
    )
    assert np.max(slope_layers.traversability_cost) == 100

    step_layers = build_terrain_layers(
        _grid_points(
            lambda x_value, _y: 0.0 if x_value < 0.0 else 0.5
        ),
        (0.0, 0.0),
        3.0,
        0.25,
        1,
        8.0,
        25.0,
        0.10,
        0.35,
    )
    assert np.max(step_layers.traversability_cost) == 100


def test_bad_threshold_order_is_rejected():
    with pytest.raises(ValueError):
        build_terrain_layers(
            _grid_points(lambda _x, _y: 0.0),
            (0.0, 0.0),
            3.0,
            0.25,
            1,
            25.0,
            8.0,
            0.10,
            0.35,
        )
