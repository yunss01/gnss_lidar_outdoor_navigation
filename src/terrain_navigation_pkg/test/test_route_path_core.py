import math

import pytest

from terrain_navigation_pkg.navigation_core import EnuPoint
from terrain_navigation_pkg.route_path_core import compute_path_tracking_geometry
from terrain_navigation_pkg.route_path_core import cumulative_path_lengths
from terrain_navigation_pkg.route_path_core import generate_smooth_route
from terrain_navigation_pkg.route_path_core import nearest_path_index


def _minimum_distance(point, path):
    return min(
        math.hypot(
            candidate.east_m - point.east_m,
            candidate.north_m - point.north_m,
        )
        for candidate in path
    )


def test_smooth_route_stays_inside_waypoint_arrival_radius():
    control_points = [
        EnuPoint(0.0, 0.0),
        EnuPoint(10.0, 0.0),
        EnuPoint(15.0, 8.0),
        EnuPoint(20.0, 12.0),
    ]
    path = generate_smooth_route(
        control_points,
        start_yaw_rad=0.0,
        spacing_m=0.5,
        tangent_scale=0.5,
    )
    assert _minimum_distance(control_points[0], path) == pytest.approx(0.0)
    assert _minimum_distance(control_points[-1], path) == pytest.approx(0.0)
    for waypoint in control_points[1:-1]:
        assert _minimum_distance(waypoint, path) <= 2.0


def test_smooth_route_rounds_a_right_angle_without_overshooting_far():
    control_points = [
        EnuPoint(0.0, 0.0),
        EnuPoint(10.0, 0.0),
        EnuPoint(10.0, 10.0),
    ]
    path = generate_smooth_route(control_points, 0.0, 0.25, 0.5)
    near_corner = [
        point for point in path
        if 8.0 <= point.east_m <= 11.0 and -1.0 <= point.north_m <= 2.0
    ]
    assert near_corner
    assert any(
        point.east_m < 10.0 and point.north_m > 0.0
        for point in near_corner
    )
    assert max(point.east_m for point in path) < 11.0


def test_pure_pursuit_turns_left_and_obeys_curvature_limit():
    control_points = [
        EnuPoint(0.0, 0.0),
        EnuPoint(8.0, 0.0),
        EnuPoint(12.0, 6.0),
    ]
    path = generate_smooth_route(control_points, 0.0, 0.5, 0.5)
    lengths = cumulative_path_lengths(path)
    geometry = compute_path_tracking_geometry(
        path,
        lengths,
        EnuPoint(6.5, 0.0),
        current_yaw_rad=0.0,
        previous_index=0,
        lookahead_distance_m=4.0,
        maximum_curvature_per_m=0.20,
    )
    assert geometry.curvature_per_m > 0.0
    assert geometry.curvature_per_m <= 0.20
    assert geometry.target_index > geometry.nearest_index


def test_nearest_path_progress_never_moves_backward():
    path = [EnuPoint(float(index), 0.0) for index in range(20)]
    assert nearest_path_index(
        path,
        EnuPoint(3.2, 0.0),
        previous_index=8,
    ) == 8


def test_nearest_path_progress_cannot_jump_to_nearby_future_segment():
    # The return leg is spatially close to the start.  Without the progress
    # budget, a nearest-point search can mistake it for the current leg.
    path = [
        EnuPoint(0.0, 0.0),
        EnuPoint(1.0, 0.0),
        EnuPoint(2.0, 0.0),
        EnuPoint(2.0, 10.0),
        EnuPoint(1.0, 0.1),
        EnuPoint(0.0, 0.1),
    ]
    lengths = cumulative_path_lengths(path)
    unrestricted = compute_path_tracking_geometry(
        path,
        lengths,
        EnuPoint(1.0, 0.1),
        current_yaw_rad=0.0,
        previous_index=0,
        lookahead_distance_m=1.0,
        maximum_curvature_per_m=0.2,
    )
    bounded = compute_path_tracking_geometry(
        path,
        lengths,
        EnuPoint(1.0, 0.1),
        current_yaw_rad=0.0,
        previous_index=0,
        lookahead_distance_m=1.0,
        maximum_curvature_per_m=0.2,
        maximum_progress_distance_m=2.0,
    )
    assert unrestricted.nearest_index == 4
    assert bounded.nearest_index <= 2


def test_path_geometry_reports_real_distance_to_final_goal():
    path = [
        EnuPoint(0.0, 0.0),
        EnuPoint(10.0, 0.0),
        EnuPoint(20.0, 0.0),
    ]
    lengths = cumulative_path_lengths(path)
    geometry = compute_path_tracking_geometry(
        path,
        lengths,
        EnuPoint(3.0, 4.0),
        current_yaw_rad=0.0,
        previous_index=0,
        lookahead_distance_m=2.0,
        maximum_curvature_per_m=0.2,
    )
    assert geometry.final_distance_m == pytest.approx(math.hypot(17.0, 4.0))
    assert geometry.nearest_distance_m == pytest.approx(5.0)


def test_path_tracking_uses_physical_u_turn_not_in_place_rotation():
    path = [EnuPoint(0.0, 0.0), EnuPoint(10.0, 0.0)]
    lengths = cumulative_path_lengths(path)
    geometry = compute_path_tracking_geometry(
        path,
        lengths,
        EnuPoint(0.0, 0.0),
        current_yaw_rad=math.pi,
        previous_index=0,
        lookahead_distance_m=5.0,
        maximum_curvature_per_m=0.2,
    )
    assert abs(geometry.curvature_per_m) == pytest.approx(0.2)
