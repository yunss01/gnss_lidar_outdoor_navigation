import math

import numpy as np

from terrain_navigation_pkg.navigation_learning_recorder_core import (
    BevGeometry,
    build_lidar_bev,
    occupancy_grid_to_vehicle_bev,
    world_to_vehicle_xy,
)


def test_world_to_vehicle_xy_rotates_about_vehicle_pose():
    result = world_to_vehicle_xy(
        np.array([[10.0, 12.0], [8.0, 10.0]], dtype=np.float32),
        10.0,
        10.0,
        math.pi / 2.0,
    )
    assert np.allclose(result[0], [2.0, 0.0], atol=1e-6)
    assert np.allclose(result[1], [0.0, 2.0], atol=1e-6)


def test_build_lidar_bev_places_forward_left_point():
    geometry = BevGeometry(
        x_min_m=-2.0,
        x_max_m=2.0,
        y_min_m=-2.0,
        y_max_m=2.0,
        resolution_m=1.0,
        z_min_m=-1.0,
        z_max_m=2.0,
    )
    points = np.array([
        [1.5, 1.5, 0.2],
        [1.4, 1.4, 0.8],
        [-1.5, -1.5, 0.0],
    ], dtype=np.float32)
    bev = build_lidar_bev(points, geometry)
    assert bev.shape == (4, 4, 4)
    assert bev[0, 0, 0] == 1.0
    assert np.isclose(bev[2, 0, 0], 0.8)
    assert np.isclose(bev[3, 0, 0], 0.6)
    assert bev[0, 3, 3] == 1.0


def test_occupancy_grid_is_sampled_into_vehicle_bev():
    geometry = BevGeometry(
        x_min_m=0.0,
        x_max_m=2.0,
        y_min_m=0.0,
        y_max_m=2.0,
        resolution_m=1.0,
    )
    grid = np.array([
        [0, 10, 20],
        [30, 40, 50],
        [60, 70, 80],
    ], dtype=np.int16)
    output = occupancy_grid_to_vehicle_bev(
        grid=grid,
        grid_resolution=1.0,
        grid_origin_x=0.0,
        grid_origin_y=0.0,
        grid_origin_yaw=0.0,
        vehicle_x=0.0,
        vehicle_y=0.0,
        vehicle_yaw=0.0,
        geometry=geometry,
    )
    assert output.tolist() == [[40, 10], [30, 0]]
