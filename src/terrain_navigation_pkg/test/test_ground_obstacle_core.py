import numpy as np

from terrain_navigation_pkg.ground_obstacle_core import (
    extract_ground_relative_obstacles,
)


def _default_extract(points, **overrides):
    parameters = {
        'fixed_minimum_z_m': -1.4,
        'fixed_maximum_z_m': 1.0,
        'ground_fit_minimum_range_m': 2.0,
        'ground_fit_maximum_range_m': 20.0,
        'ground_fit_minimum_z_m': -2.5,
        'ground_fit_maximum_z_m': -0.8,
        'ground_seed_quantile': 0.35,
        'ground_inlier_tolerance_m': 0.06,
        'minimum_ground_points': 80,
        'low_obstacle_minimum_height_m': 0.07,
        'low_obstacle_maximum_height_m': 0.45,
    }
    parameters.update(overrides)
    return extract_ground_relative_obstacles(points, **parameters)


def test_sloped_road_is_removed_but_ten_centimetre_curb_is_retained():
    x_grid, y_grid = np.meshgrid(
        np.linspace(2.0, 18.0, 81), np.linspace(-5.0, 5.0, 51)
    )
    road_z = -1.80 + 0.012 * x_grid - 0.006 * y_grid
    road = np.column_stack(
        (x_grid.ravel(), y_grid.ravel(), road_z.ravel())
    )

    curb_x, curb_y = np.meshgrid(
        np.linspace(5.0, 15.0, 41), np.linspace(1.5, 2.0, 5)
    )
    curb_ground = -1.80 + 0.012 * curb_x - 0.006 * curb_y
    curb = np.column_stack(
        (curb_x.ravel(), curb_y.ravel(), (curb_ground + 0.10).ravel())
    )

    wall_x, wall_z = np.meshgrid(
        np.linspace(7.0, 9.0, 9), np.linspace(-1.0, 0.8, 12)
    )
    wall = np.column_stack(
        (wall_x.ravel(), np.full(wall_x.size, -3.0), wall_z.ravel())
    )
    points = np.vstack((road, curb, wall))

    result = _default_extract(points)
    road_mask = result.obstacle_mask[: road.shape[0]]
    curb_start = road.shape[0]
    curb_mask = result.obstacle_mask[
        curb_start: curb_start + curb.shape[0]
    ]
    wall_mask = result.obstacle_mask[-wall.shape[0]:]

    assert result.ground_plane is not None
    assert result.ground_point_count >= 3000
    assert np.count_nonzero(road_mask) == 0
    assert np.count_nonzero(curb_mask) == curb.shape[0]
    assert np.count_nonzero(wall_mask) == wall.shape[0]
    assert result.low_obstacle_point_count == curb.shape[0]


def test_local_ground_rejects_curved_road_and_retains_curb_edge():
    x_grid, y_grid = np.meshgrid(
        np.linspace(2.0, 18.0, 81), np.linspace(-5.0, 5.0, 51)
    )
    # A road crown/mesh warp cannot be represented by one plane.  Its total
    # height change exceeds the 7 cm curb threshold, but it changes gradually
    # within the local neighbourhood.
    road_z = -1.82 + 0.0007 * (x_grid - 10.0) ** 2 + 0.001 * y_grid ** 2
    road = np.column_stack(
        (x_grid.ravel(), y_grid.ravel(), road_z.ravel())
    )
    curb_x, curb_y = np.meshgrid(
        np.linspace(5.0, 15.0, 41), np.linspace(1.5, 2.0, 5)
    )
    curb_ground = (
        -1.82 + 0.0007 * (curb_x - 10.0) ** 2 + 0.001 * curb_y ** 2
    )
    curb = np.column_stack(
        (curb_x.ravel(), curb_y.ravel(), (curb_ground + 0.10).ravel())
    )
    points = np.vstack((road, curb))

    result = _default_extract(
        points,
        local_ground_enabled=True,
        local_ground_resolution_m=0.75,
        local_ground_radius_m=1.50,
        local_ground_quantile=0.25,
        local_ground_plane_enabled=True,
        local_ground_plane_residual_tolerance_m=0.055,
    )
    road_mask = result.obstacle_mask[: road.shape[0]]
    curb_mask = result.obstacle_mask[road.shape[0]:]

    assert np.count_nonzero(road_mask) == 0
    assert np.count_nonzero(curb_mask) == curb.shape[0]


def test_local_plane_follows_combined_pitch_and_crown():
    x_grid, y_grid = np.meshgrid(
        np.linspace(2.0, 18.0, 81), np.linspace(-5.0, 5.0, 51)
    )
    road_z = (
        -1.82 + 0.02 * x_grid + 0.001 * y_grid ** 2
        + 0.0007 * (x_grid - 10.0) ** 2
    )
    road = np.column_stack(
        (x_grid.ravel(), y_grid.ravel(), road_z.ravel())
    )
    curb_x, curb_y = np.meshgrid(
        np.linspace(5.0, 15.0, 41), np.linspace(1.5, 2.0, 5)
    )
    curb_ground = (
        -1.82 + 0.02 * curb_x + 0.001 * curb_y ** 2
        + 0.0007 * (curb_x - 10.0) ** 2
    )
    curb = np.column_stack((
        curb_x.ravel(), curb_y.ravel(), (curb_ground + 0.10).ravel()
    ))
    points = np.vstack((road, curb))

    result = _default_extract(
        points,
        local_ground_enabled=True,
        local_ground_resolution_m=0.75,
        local_ground_radius_m=1.75,
        local_ground_quantile=0.25,
        local_ground_plane_enabled=True,
        local_ground_plane_residual_tolerance_m=0.055,
    )

    assert np.count_nonzero(result.obstacle_mask[:road.shape[0]]) == 0
    # Boundary cells have fewer than four ground neighbours; retain at least
    # 98% of the curb while avoiding a dangerous threshold increase.
    assert np.count_nonzero(result.obstacle_mask[road.shape[0]:]) >= 201


def test_uphill_road_above_fixed_z_cut_is_still_ground():
    x_grid, y_grid = np.meshgrid(
        np.linspace(2.0, 18.0, 81), np.linspace(-4.0, 4.0, 41)
    )
    # The far road reaches z=-1.36 m and therefore crosses the old -1.4 m
    # fixed obstacle threshold despite being a continuous planar surface.
    road_z = -1.82 + 0.0255 * x_grid
    road = np.column_stack(
        (x_grid.ravel(), y_grid.ravel(), road_z.ravel())
    )
    result = _default_extract(
        road,
        local_ground_enabled=True,
        local_ground_plane_enabled=True,
        local_ground_resolution_m=0.75,
        local_ground_radius_m=1.50,
    )

    assert np.count_nonzero(road[:, 2] >= -1.4) > 0
    assert np.count_nonzero(result.obstacle_mask) == 0


def test_disabled_mode_exactly_matches_legacy_fixed_z_crop():
    points = np.array(
        [
            [3.0, 0.0, -1.8],
            [3.0, 0.0, -1.3],
            [3.0, 0.0, 0.5],
            [3.0, 0.0, 1.2],
        ]
    )
    result = _default_extract(points, enabled=False)
    np.testing.assert_array_equal(
        result.obstacle_mask, np.array([False, True, True, False])
    )
    assert result.ground_plane is None


def test_insufficient_ground_points_falls_back_to_legacy_crop():
    points = np.array(
        [
            [3.0, 0.0, -1.8],
            [3.0, 0.0, -1.3],
            [4.0, 0.0, 0.2],
        ]
    )
    result = _default_extract(points)
    np.testing.assert_array_equal(
        result.obstacle_mask, np.array([False, True, True])
    )
    assert result.ground_plane is None


def test_empty_cloud_is_supported():
    result = _default_extract([])
    assert result.obstacle_mask.shape == (0,)
    assert result.ground_plane is None
