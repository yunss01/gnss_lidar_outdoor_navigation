import math

import numpy as np
import pytest

from terrain_navigation_pkg.local_avoidance_core import LEFT
from terrain_navigation_pkg.local_avoidance_core import LocalTrajectoryPlan
from terrain_navigation_pkg.local_avoidance_core import RIGHT
from terrain_navigation_pkg.local_avoidance_core import (
    associate_progress_interval,
)
from terrain_navigation_pkg.local_avoidance_core import Point2D
from terrain_navigation_pkg.local_avoidance_core import Pose2D
from terrain_navigation_pkg.local_avoidance_core import choose_avoidance_side
from terrain_navigation_pkg.local_avoidance_core import (
    choose_feasible_trajectory_side,
)
from terrain_navigation_pkg.local_avoidance_core import (
    clamp_tracked_progress_interval,
)
from terrain_navigation_pkg.local_avoidance_core import command_to_point
from terrain_navigation_pkg.local_avoidance_core import (
    compute_trajectory_planning_horizon,
)
from terrain_navigation_pkg.local_avoidance_core import detour_lateral_offset
from terrain_navigation_pkg.local_avoidance_core import observe_detour_corridor
from terrain_navigation_pkg.local_avoidance_core import observe_curved_corridor
from terrain_navigation_pkg.local_avoidance_core import (
    observe_inner_side_passage,
)
from terrain_navigation_pkg.local_avoidance_core import (
    obstacle_requires_detour,
)
from terrain_navigation_pkg.local_avoidance_core import quintic_smoothstep
from terrain_navigation_pkg.local_avoidance_core import pure_pursuit_command
from terrain_navigation_pkg.local_avoidance_core import (
    sample_constant_curvature_path,
)
from terrain_navigation_pkg.local_avoidance_core import measure_path_alignment
from terrain_navigation_pkg.local_avoidance_core import (
    measure_polyline_alignment,
)
from terrain_navigation_pkg.local_avoidance_core import (
    polyline_lookahead_target,
)
from terrain_navigation_pkg.local_avoidance_core import (
    observe_commanded_trajectory,
)
from terrain_navigation_pkg.local_avoidance_core import (
    observe_swept_vehicle_footprint,
)
from terrain_navigation_pkg.local_avoidance_core import (
    select_collision_free_trajectory,
)
from terrain_navigation_pkg.local_avoidance_core import (
    update_lateral_shift_confirmation,
)
from terrain_navigation_pkg.local_avoidance_core import (
    trajectory_handoff_ready,
)
from terrain_navigation_pkg.local_avoidance_core import (
    trajectory_recommit_required,
)
from terrain_navigation_pkg.local_avoidance_core import (
    trajectory_side_lock_release_permitted,
)
from terrain_navigation_pkg.local_avoidance_core import (
    update_trajectory_side_lock,
)
from terrain_navigation_pkg.safety_core import ForwardObstacleObservation


def observation(count):
    return ForwardObstacleObservation(count, count, 5.0)


def test_side_choice_uses_open_side_and_rejects_two_blocked_sides():
    assert choose_avoidance_side(observation(0), observation(30), 20) == LEFT
    assert choose_avoidance_side(observation(30), observation(0), 20) == RIGHT
    assert choose_avoidance_side(observation(30), observation(30), 20) is None


def test_side_choice_uses_preference_for_equal_open_sides():
    assert choose_avoidance_side(
        observation(0), observation(0), 20, preferred_side=RIGHT
    ) == RIGHT


def test_tracked_obstacle_interval_cannot_absorb_long_city_geometry():
    minimum, maximum = clamp_tracked_progress_interval(
        minimum_progress_m=12.0,
        maximum_progress_m=31.4,
        seed_progress_m=16.2,
        maximum_backward_extent_m=0.75,
        maximum_forward_extent_m=6.0,
    )
    assert minimum == pytest.approx(15.45)
    assert maximum == pytest.approx(22.2)


def test_goal_guard_ignores_obstacle_beyond_stopping_point():
    assert obstacle_requires_detour(8.0, 20.0, 3.0, 0.5)
    assert not obstacle_requires_detour(11.6, 6.0, 3.0, 0.5)
    assert obstacle_requires_detour(3.4, 6.0, 3.0, 0.5)


def test_intermediate_waypoint_keeps_full_trajectory_horizon():
    assert compute_trajectory_planning_horizon(
        configured_horizon_m=10.0,
        minimum_horizon_m=8.0,
        goal_distance_m=5.12,
        goal_stop_distance_m=3.0,
        goal_obstacle_margin_m=0.5,
        terminal_goal=False,
    ) == pytest.approx(10.0)


def test_terminal_goal_never_shrinks_below_safety_clear_range():
    assert compute_trajectory_planning_horizon(
        configured_horizon_m=10.0,
        minimum_horizon_m=8.0,
        goal_distance_m=5.12,
        goal_stop_distance_m=3.0,
        goal_obstacle_margin_m=0.5,
        terminal_goal=True,
    ) == pytest.approx(8.0)


def test_distant_terminal_goal_uses_configured_trajectory_horizon():
    assert compute_trajectory_planning_horizon(
        configured_horizon_m=10.0,
        minimum_horizon_m=8.0,
        goal_distance_m=30.0,
        goal_stop_distance_m=3.0,
        goal_obstacle_margin_m=0.5,
        terminal_goal=True,
    ) == pytest.approx(10.0)


def test_trajectory_handoff_waits_until_obstacle_clears_inside_side():
    assert not trajectory_handoff_ready(
        direct_path_clear=True,
        recovery_aligned=True,
        obstacle_side_lock_active=True,
    )


def test_trajectory_handoff_requires_clear_aligned_unlocked_path():
    assert trajectory_handoff_ready(True, True, False)
    assert not trajectory_handoff_ready(False, True, False)
    assert not trajectory_handoff_ready(True, False, False)


def test_trajectory_side_lock_clears_after_consecutive_open_scans():
    count = 0
    active = True
    for _ in range(5):
        count, active = update_trajectory_side_lock(
            active, count, 19, 20, 5
        )
    assert count == 5
    assert not active


def test_cleared_trajectory_side_lock_never_reactivates_during_recovery():
    count, active = update_trajectory_side_lock(
        lock_active=False,
        clear_count=6,
        obstacle_point_count=120,
        minimum_obstacle_points=20,
        clear_required_scans=5,
    )
    assert count == 6
    assert not active


def test_trajectory_side_lock_cannot_clear_while_direct_path_is_blocked():
    count, active = update_trajectory_side_lock(
        lock_active=True,
        clear_count=4,
        obstacle_point_count=0,
        minimum_obstacle_points=20,
        clear_required_scans=5,
        release_permitted=False,
    )
    assert count == 0
    assert active


def test_new_obstacle_after_clear_interval_requires_fresh_commitment():
    assert not trajectory_recommit_required(True, 4, 5)
    assert trajectory_recommit_required(True, 5, 5)
    assert not trajectory_recommit_required(False, 50, 5)
    assert trajectory_recommit_required(
        True, 0, 5, handoff_active=True
    )
    assert trajectory_recommit_required(
        True,
        5,
        5,
        cross_track_m=0.14,
        maximum_cross_track_m=1.5,
    )
    assert not trajectory_recommit_required(
        True,
        9,
        5,
        cross_track_m=8.67,
        maximum_cross_track_m=1.5,
    )


def test_side_lock_waits_for_real_shift_then_allows_safe_return():
    assert not trajectory_side_lock_release_permitted(
        center_blocked=True,
        committed_side=LEFT,
        cross_track_m=0.1,
        minimum_lateral_shift_m=3.0,
    )
    assert trajectory_side_lock_release_permitted(
        center_blocked=True,
        committed_side=LEFT,
        cross_track_m=3.0,
        minimum_lateral_shift_m=3.0,
    )
    assert not trajectory_side_lock_release_permitted(
        center_blocked=False,
        committed_side=RIGHT,
        cross_track_m=0.0,
        minimum_lateral_shift_m=3.0,
    )
    assert trajectory_side_lock_release_permitted(
        center_blocked=False,
        committed_side=RIGHT,
        cross_track_m=-3.0,
        minimum_lateral_shift_m=3.0,
    )


def _trajectory_plan(proximity_points=0, valid_candidates=7, score=1.0):
    return LocalTrajectoryPlan(
        curvature_per_m=0.1,
        desired_curvature_per_m=0.0,
        collision_point_count=0,
        safety_validation_point_count=0,
        proximity_point_count=proximity_points,
        nearest_path_m=math.inf,
        score=score,
        valid_candidate_count=valid_candidates,
        candidate_count=13,
        path_points=(),
    )


def test_feasible_side_tie_uses_material_corridor_point_advantage():
    plan = _trajectory_plan()
    assert choose_feasible_trajectory_side(
        plan,
        plan,
        preferred_side=LEFT,
        left_corridor_points=50,
        right_corridor_points=0,
        corridor_advantage_points=20,
    ) == RIGHT


def test_exact_arc_clearance_overrules_broad_corridor_tie_breaker():
    left = _trajectory_plan(proximity_points=0)
    right = _trajectory_plan(proximity_points=10)
    assert choose_feasible_trajectory_side(
        left,
        right,
        preferred_side=RIGHT,
        left_corridor_points=50,
        right_corridor_points=0,
        corridor_advantage_points=20,
    ) == LEFT


def test_quintic_smoothstep_has_gentle_symmetric_transition():
    assert quintic_smoothstep(0.0) == pytest.approx(0.0)
    assert quintic_smoothstep(0.5) == pytest.approx(0.5)
    assert quintic_smoothstep(1.0) == pytest.approx(1.0)
    assert quintic_smoothstep(0.25) == pytest.approx(
        1.0 - quintic_smoothstep(0.75)
    )


def test_detour_corridor_checks_full_vehicle_width_along_s_curve():
    points = np.array([
        [11.0, 0.0, 0.0],
        [11.0, 3.5, 0.0],
        [11.0, -3.5, 0.0],
        [6.0, 1.4, 0.0],
    ], dtype=np.float32)
    left = observe_detour_corridor(
        points,
        LEFT,
        minimum_x_m=2.5,
        maximum_x_m=12.0,
        lateral_offset_m=3.5,
        shift_forward_m=7.0,
        corridor_half_width_m=1.4,
        vehicle_front_m=2.4,
        minimum_z_m=-1.4,
        maximum_z_m=1.0,
    )
    right = observe_detour_corridor(
        points,
        RIGHT,
        minimum_x_m=2.5,
        maximum_x_m=12.0,
        lateral_offset_m=3.5,
        shift_forward_m=7.0,
        corridor_half_width_m=1.4,
        vehicle_front_m=2.4,
        minimum_z_m=-1.4,
        maximum_z_m=1.0,
    )
    assert left.clear_point_count == 2
    assert right.clear_point_count == 1
    assert left.nearest_distance_m == pytest.approx(6.0)
    assert detour_lateral_offset(7.0, LEFT, 3.5, 7.0) == 3.5


@pytest.mark.parametrize(
    ('side', 'inside_y'),
    ((LEFT, -3.0), (RIGHT, 3.0)),
)
def test_passage_observation_is_symmetric_and_detects_rear_clearance(
    side,
    inside_y,
):
    points = np.array([
        [4.0, inside_y, 0.0],
        [-2.0, inside_y, 0.0],
        [-4.0, inside_y, 0.0],
        [-6.0, inside_y, 0.0],
        [2.0, -inside_y, 0.0],
        [2.0, inside_y, -2.0],
    ], dtype=np.float32)
    observation = observe_inner_side_passage(
        points,
        avoidance_side=side,
        vehicle_rear_m=2.5,
        rear_clearance_m=1.0,
        minimum_lateral_m=0.6,
        maximum_lateral_m=7.0,
        rear_detection_m=12.0,
        forward_detection_m=17.0,
        minimum_z_m=-1.4,
        maximum_z_m=1.0,
    )
    assert observation.not_passed_point_count == 2
    assert observation.behind_point_count == 2


def test_passage_observation_reports_empty_cloud_as_clear():
    observation = observe_inner_side_passage(
        np.empty((0, 3), dtype=np.float32),
        avoidance_side=LEFT,
        vehicle_rear_m=2.5,
        rear_clearance_m=1.0,
        minimum_lateral_m=0.6,
        maximum_lateral_m=7.0,
        rear_detection_m=12.0,
        forward_detection_m=17.0,
        minimum_z_m=-1.4,
        maximum_z_m=1.0,
    )
    assert observation.not_passed_point_count == 0
    assert observation.behind_point_count == 0


def test_tracked_interval_expands_through_connected_target_not_city_clutter():
    progress = np.array(
        [15.8, 16.8, 18.0, 19.2, 30.0, 45.0],
        dtype=np.float64,
    )
    minimum, maximum, associated = associate_progress_interval(
        progress,
        np.ones(progress.shape, dtype=bool),
        minimum_progress_m=16.0,
        maximum_progress_m=17.0,
        association_gap_m=1.5,
    )
    assert minimum == pytest.approx(15.8)
    assert maximum == pytest.approx(19.2)
    assert associated.tolist() == [True, True, True, True, False, False]


def test_pure_pursuit_steers_toward_left_lookahead_without_saturation():
    command = pure_pursuit_command(
        Pose2D(0.0, 0.0, 0.0),
        Point2D(4.0, 1.0),
        speed_mps=1.0,
        maximum_yaw_rate_rps=0.35,
        minimum_heading_speed_ratio=0.25,
    )
    assert command.speed_mps > 0.9
    assert command.yaw_rate_rps == pytest.approx(
        command.speed_mps * 2.0 / 17.0
    )
    assert command.heading_error_rad > 0.0


def test_pure_pursuit_respects_ackermann_curvature_limit():
    maximum_curvature = math.tan(math.radians(35.0)) / 2.85
    command = pure_pursuit_command(
        Pose2D(0.0, 0.0, 0.0),
        Point2D(1.0, 2.0),
        speed_mps=1.0,
        maximum_yaw_rate_rps=0.35,
        minimum_heading_speed_ratio=0.25,
        maximum_curvature_per_m=maximum_curvature,
    )
    assert abs(command.yaw_rate_rps) / command.speed_mps == pytest.approx(
        maximum_curvature
    )


def test_lateral_shift_requires_repeated_measurements_near_3_5_metres():
    count = 0
    confirmed = False
    for measurement in (3.0, 3.31, 3.35, 3.42):
        count, confirmed = update_lateral_shift_confirmation(
            measurement,
            target_lateral_m=3.5,
            tolerance_m=0.2,
            previous_count=count,
            required_count=3,
        )
    assert count == 3
    assert confirmed

    count, confirmed = update_lateral_shift_confirmation(
        3.29,
        target_lateral_m=3.5,
        tolerance_m=0.2,
        previous_count=count,
        required_count=3,
    )
    assert count == 0
    assert not confirmed


def test_command_to_left_point_has_positive_ros_yaw_rate():
    command = command_to_point(
        Pose2D(0.0, 0.0, 0.0),
        Point2D(4.0, 3.0),
        speed_mps=0.8,
        heading_kp=1.2,
        maximum_yaw_rate_rps=0.45,
        minimum_heading_speed_ratio=0.25,
    )
    assert command.speed_mps > 0.0
    assert command.yaw_rate_rps > 0.0
    assert command.distance_m == pytest.approx(5.0)


def test_constant_curvature_path_uses_ros_left_positive_convention():
    path = sample_constant_curvature_path(0.1, 10.0, 0.5)

    assert path[0, 0] == pytest.approx(0.0)
    assert path[0, 1] == pytest.approx(0.0)
    assert path[-1, 0] == pytest.approx(math.sin(1.0) / 0.1)
    assert path[-1, 1] == pytest.approx((1.0 - math.cos(1.0)) / 0.1)
    assert path[-1, 2] == pytest.approx(1.0)


def test_path_alignment_reports_cross_track_and_heading_errors():
    alignment = measure_path_alignment(
        Pose2D(5.0, -0.6, math.radians(7.0)),
        Point2D(0.0, 0.0),
        0.0,
    )
    assert alignment.progress_m == pytest.approx(5.0)
    assert alignment.cross_track_m == pytest.approx(-0.6)
    assert math.degrees(alignment.heading_error_rad) == pytest.approx(7.0)


def test_polyline_alignment_uses_curve_tangent_and_forward_progress():
    points = [
        Point2D(0.0, 0.0),
        Point2D(10.0, 0.0),
        Point2D(10.0, 10.0),
        Point2D(20.0, 10.0),
    ]
    pose = Pose2D(10.5, 4.0, math.radians(95.0))
    alignment = measure_polyline_alignment(
        pose,
        points,
        previous_segment_index=1,
    )
    assert alignment.segment_index == 1
    assert alignment.progress_m == pytest.approx(14.0)
    assert alignment.cross_track_m == pytest.approx(-0.5)
    assert math.degrees(alignment.heading_error_rad) == pytest.approx(5.0)


def test_polyline_lookahead_crosses_segments_without_going_backward():
    points = [
        Point2D(0.0, 0.0),
        Point2D(10.0, 0.0),
        Point2D(10.0, 10.0),
    ]
    alignment = measure_polyline_alignment(
        Pose2D(8.0, 1.0, 0.0),
        points,
    )
    target = polyline_lookahead_target(points, alignment, 5.0)
    assert target.x_m == pytest.approx(10.0)
    assert target.y_m == pytest.approx(3.0)


def test_polyline_alignment_does_not_jump_to_earlier_loop_segment():
    points = [
        Point2D(0.0, 0.0),
        Point2D(10.0, 0.0),
        Point2D(10.0, 10.0),
        Point2D(0.0, 10.0),
        Point2D(0.0, 0.1),
    ]
    alignment = measure_polyline_alignment(
        Pose2D(0.0, 0.2, -0.5 * math.pi),
        points,
        previous_segment_index=3,
    )
    assert alignment.segment_index == 3


def test_swept_footprint_catches_outer_front_corner_missed_by_centerline():
    # On a left arc the vehicle centre-line stays away from this point, but
    # the long outside front corner sweeps through it.
    point = np.repeat([[5.5, 0.9, 0.0]], 20, axis=0)
    centerline = observe_curved_corridor(
        point, 0.18, 2.5, 6.0, 8.0, 1.35, -1.4, 1.0, 0.25
    )
    footprint = observe_swept_vehicle_footprint(
        point,
        0.18,
        2.5,
        6.0,
        8.0,
        1.35,
        2.4,
        2.5,
        -1.4,
        1.0,
        0.25,
    )
    assert centerline.stop_point_count == 0
    assert footprint.stop_point_count == 20


def test_swept_footprint_rejects_ego_body_but_keeps_door_side_obstacle():
    points = np.vstack((
        np.repeat([[1.5, 0.3, 0.0]], 20, axis=0),
        np.repeat([[0.0, 1.25, 0.0]], 20, axis=0),
    ))
    footprint = observe_swept_vehicle_footprint(
        points,
        0.18,
        2.5,
        6.0,
        8.0,
        1.35,
        2.4,
        2.5,
        -1.4,
        1.0,
        0.25,
        ego_half_width_m=1.0,
    )
    assert footprint.stop_point_count == 20


def test_commanded_trajectory_finds_obstacle_on_curved_vehicle_path():
    path = sample_constant_curvature_path(0.18, 8.0, 0.25)
    curve_point = path[-5]
    points = np.repeat(
        [[curve_point[0], curve_point[1], 0.0]],
        repeats=25,
        axis=0,
    ).astype(np.float32)
    result = observe_commanded_trajectory(
        points,
        linear_speed_mps=2.0,
        yaw_rate_rps=0.36,
        previous_curvature_per_m=0.18,
        maximum_curvature_per_m=0.25,
        curvature_uncertainty_per_m=0.04,
        minimum_x_m=2.5,
        stop_path_m=7.9,
        clear_path_m=8.0,
        half_width_m=1.65,
        extra_width_m=0.25,
        vehicle_front_m=2.4,
        vehicle_rear_m=2.5,
        vehicle_width_m=2.0,
        minimum_z_m=-1.4,
        maximum_z_m=1.0,
        sample_spacing_m=0.25,
    )
    assert result.commanded_curvature_per_m == pytest.approx(0.18)
    assert result.nominal.clear_point_count == 25
    assert result.envelope.clear_point_count == 25
    assert result.tested_curvature_count >= 3
    assert result.nominal_stop_points.shape == (25, 3)
    assert np.allclose(result.nominal_stop_points[0], points[0])


def test_uncertainty_envelope_is_distinct_from_nominal_hard_path():
    points = np.repeat([[6.0, 1.8, 0.0]], 25, axis=0).astype(np.float32)
    result = observe_commanded_trajectory(
        points,
        linear_speed_mps=2.0,
        yaw_rate_rps=0.0,
        previous_curvature_per_m=0.0,
        maximum_curvature_per_m=0.25,
        curvature_uncertainty_per_m=0.0,
        minimum_x_m=2.5,
        stop_path_m=7.9,
        clear_path_m=8.0,
        half_width_m=1.65,
        extra_width_m=0.25,
        vehicle_front_m=2.4,
        vehicle_rear_m=2.5,
        vehicle_width_m=2.0,
        minimum_z_m=-1.4,
        maximum_z_m=1.0,
        sample_spacing_m=0.25,
    )
    assert result.nominal.clear_point_count == 0
    assert result.envelope.clear_point_count == 25
    assert result.nominal_stop_points.shape == (0, 3)


def test_commanded_trajectory_rejects_padded_ego_returns_at_progress_zero():
    # Roof-LiDAR returns from a mirror/body edge may sit outside the nominal
    # 1.0 m half-width but inside the 1.35 m padded safety footprint.  They
    # are part of the ego vehicle, so they must not produce a permanent
    # nearest_distance=0 emergency stop.  An obstacle beyond the ego nose
    # must remain visible to the same test.
    ego_returns = np.repeat([[0.5, 1.2, -0.5]], 25, axis=0)
    forward_obstacle = np.repeat([[3.2, 0.0, 0.0]], 25, axis=0)

    ego_result = observe_commanded_trajectory(
        ego_returns,
        linear_speed_mps=1.0,
        yaw_rate_rps=0.0,
        previous_curvature_per_m=0.0,
        maximum_curvature_per_m=0.25,
        curvature_uncertainty_per_m=0.04,
        minimum_x_m=2.5,
        stop_path_m=4.0,
        clear_path_m=6.0,
        half_width_m=1.35,
        extra_width_m=0.25,
        vehicle_front_m=2.4,
        vehicle_rear_m=2.5,
        vehicle_width_m=2.0,
        minimum_z_m=-1.4,
        maximum_z_m=1.0,
        sample_spacing_m=0.25,
    )
    obstacle_result = observe_commanded_trajectory(
        forward_obstacle,
        linear_speed_mps=1.0,
        yaw_rate_rps=0.0,
        previous_curvature_per_m=0.0,
        maximum_curvature_per_m=0.25,
        curvature_uncertainty_per_m=0.04,
        minimum_x_m=2.5,
        stop_path_m=4.0,
        clear_path_m=6.0,
        half_width_m=1.35,
        extra_width_m=0.25,
        vehicle_front_m=2.4,
        vehicle_rear_m=2.5,
        vehicle_width_m=2.0,
        minimum_z_m=-1.4,
        maximum_z_m=1.0,
        sample_spacing_m=0.25,
    )

    assert ego_result.nominal.stop_point_count == 0
    assert math.isinf(ego_result.nominal.nearest_distance_m)
    assert obstacle_result.nominal.stop_point_count == 25
    assert obstacle_result.nominal.nearest_distance_m >= 0.0


def test_curved_corridor_checks_commanded_arc_not_fixed_forward_box():
    points = np.repeat([[5.0, 0.0, 0.0]], 20, axis=0)
    straight = observe_curved_corridor(
        points, 0.0, 2.5, 6.0, 8.0, 1.4, -1.4, 1.0
    )
    left_turn = observe_curved_corridor(
        points, 0.2, 2.5, 6.0, 8.0, 1.4, -1.4, 1.0
    )

    assert straight.stop_point_count == 20
    assert left_turn.stop_point_count == 0


def test_trajectory_selector_avoids_short_center_obstacle():
    center_obstacle = np.repeat([[7.0, 0.0, 0.0]], 30, axis=0)
    plan = select_collision_free_trajectory(
        center_obstacle,
        desired_curvature_per_m=0.0,
        previous_curvature_per_m=None,
        maximum_curvature_per_m=0.18,
        candidate_count=13,
        horizon_m=10.0,
        sample_spacing_m=0.35,
        minimum_path_m=2.5,
        corridor_half_width_m=1.4,
        proximity_margin_m=0.8,
        minimum_obstacle_points=20,
        minimum_z_m=-1.4,
        maximum_z_m=1.0,
        preferred_side=LEFT,
    )

    assert plan is not None
    assert plan.curvature_per_m > 0.0
    assert plan.collision_point_count < 20


def test_trajectory_selector_follows_open_side_of_long_wall():
    wall_x = np.linspace(3.0, 11.0, 100)
    # A diagonal wall blocks straight/right candidates while leaving a
    # sustained left-turning corridor.
    wall = np.column_stack((wall_x, -0.2 * wall_x + 1.0, np.zeros(100)))
    plan = select_collision_free_trajectory(
        wall,
        desired_curvature_per_m=0.0,
        previous_curvature_per_m=0.08,
        maximum_curvature_per_m=0.18,
        candidate_count=13,
        horizon_m=10.0,
        sample_spacing_m=0.35,
        minimum_path_m=2.5,
        corridor_half_width_m=1.4,
        proximity_margin_m=0.8,
        minimum_obstacle_points=20,
        minimum_z_m=-1.4,
        maximum_z_m=1.0,
        preferred_side=LEFT,
    )

    assert plan is not None
    assert plan.curvature_per_m > 0.0
    assert plan.valid_candidate_count > 0


def test_trajectory_selector_stops_when_every_candidate_is_blocked():
    angles = np.linspace(-1.3, 1.3, 500)
    wall = np.column_stack((
        4.0 * np.cos(angles),
        4.0 * np.sin(angles),
        np.zeros(angles.shape),
    ))
    plan = select_collision_free_trajectory(
        wall,
        desired_curvature_per_m=0.0,
        previous_curvature_per_m=None,
        maximum_curvature_per_m=0.18,
        candidate_count=13,
        horizon_m=10.0,
        sample_spacing_m=0.35,
        minimum_path_m=2.5,
        corridor_half_width_m=1.4,
        proximity_margin_m=0.8,
        minimum_obstacle_points=20,
        minimum_z_m=-1.4,
        maximum_z_m=1.0,
    )

    assert plan is None


def test_trajectory_selector_respects_committed_avoidance_side():
    plan = select_collision_free_trajectory(
        np.empty((0, 3), dtype=np.float32),
        desired_curvature_per_m=-0.12,
        previous_curvature_per_m=-0.12,
        maximum_curvature_per_m=0.18,
        candidate_count=13,
        horizon_m=10.0,
        sample_spacing_m=0.35,
        minimum_path_m=2.5,
        corridor_half_width_m=1.4,
        proximity_margin_m=0.8,
        minimum_obstacle_points=1,
        minimum_z_m=-1.4,
        maximum_z_m=1.0,
        allowed_side=LEFT,
    )

    assert plan is not None
    assert plan.curvature_per_m >= 0.0


def test_selector_final_validation_matches_safety_raw_cloud_contract():
    # The coarse 0.35 m planning samples miss this rear/outside corner, while
    # the final safety gate's 0.25 m samples see all 25 raw returns.
    points = np.repeat([[-1.5, 1.9, 0.0]], 25, axis=0).astype(np.float32)
    common = dict(
        xyz_points=points,
        desired_curvature_per_m=-0.18,
        previous_curvature_per_m=-0.18,
        maximum_curvature_per_m=0.18,
        candidate_count=3,
        horizon_m=10.0,
        sample_spacing_m=0.35,
        minimum_path_m=2.5,
        corridor_half_width_m=1.65,
        proximity_margin_m=0.8,
        minimum_obstacle_points=1,
        minimum_z_m=-1.4,
        maximum_z_m=1.0,
        voxel_size_m=0.2,
        vehicle_front_m=2.4,
        vehicle_rear_m=2.5,
        ego_half_width_m=1.0,
        allowed_side=RIGHT,
    )
    coarse = select_collision_free_trajectory(**common)
    assert coarse is not None
    assert coarse.curvature_per_m == pytest.approx(-0.18)

    validated = select_collision_free_trajectory(
        **common,
        validation_clear_path_m=8.0,
        validation_sample_spacing_m=0.25,
        validation_minimum_obstacle_points=20,
    )
    assert validated is not None
    assert validated.curvature_per_m != pytest.approx(-0.18)
    assert validated.safety_validation_point_count < 20

    safety = observe_commanded_trajectory(
        points,
        linear_speed_mps=1.0,
        yaw_rate_rps=validated.curvature_per_m,
        previous_curvature_per_m=validated.curvature_per_m,
        maximum_curvature_per_m=0.25,
        curvature_uncertainty_per_m=0.04,
        minimum_x_m=2.5,
        stop_path_m=6.0,
        clear_path_m=8.0,
        half_width_m=1.65,
        extra_width_m=0.25,
        vehicle_front_m=2.4,
        vehicle_rear_m=2.5,
        vehicle_width_m=2.0,
        minimum_z_m=-1.4,
        maximum_z_m=1.0,
        sample_spacing_m=0.25,
    )
    assert safety.nominal.clear_point_count < 20


def test_selector_temporarily_excludes_safety_rejected_curvature():
    plan = select_collision_free_trajectory(
        np.empty((0, 3), dtype=np.float32),
        desired_curvature_per_m=0.18,
        previous_curvature_per_m=0.18,
        maximum_curvature_per_m=0.18,
        candidate_count=13,
        horizon_m=10.0,
        sample_spacing_m=0.35,
        minimum_path_m=2.5,
        corridor_half_width_m=1.65,
        proximity_margin_m=0.8,
        minimum_obstacle_points=1,
        minimum_z_m=-1.4,
        maximum_z_m=1.0,
        excluded_curvatures=(0.18,),
        excluded_curvature_tolerance_per_m=0.012,
    )
    assert plan is not None
    assert plan.curvature_per_m == pytest.approx(0.15)


def test_feasible_side_choice_prefers_actual_clearer_arc_family():
    points = np.repeat([[6.0, -2.0, 0.0]], 25, axis=0).astype(np.float32)
    common = dict(
        xyz_points=points,
        desired_curvature_per_m=0.0,
        previous_curvature_per_m=None,
        maximum_curvature_per_m=0.18,
        candidate_count=13,
        horizon_m=10.0,
        sample_spacing_m=0.35,
        minimum_path_m=2.5,
        corridor_half_width_m=1.65,
        proximity_margin_m=0.8,
        minimum_obstacle_points=1,
        minimum_z_m=-1.4,
        maximum_z_m=1.0,
        voxel_size_m=0.2,
        vehicle_front_m=2.4,
        vehicle_rear_m=2.5,
        ego_half_width_m=1.0,
        validation_clear_path_m=8.0,
        validation_sample_spacing_m=0.25,
        validation_minimum_obstacle_points=20,
    )
    left_plan = select_collision_free_trajectory(
        **common, allowed_side=LEFT
    )
    right_plan = select_collision_free_trajectory(
        **common, allowed_side=RIGHT
    )
    assert choose_feasible_trajectory_side(
        left_plan, right_plan, preferred_side=RIGHT
    ) == LEFT


def test_polyline_recovery_search_has_metric_forward_limit():
    points = [
        Point2D(0.0, 0.0),
        Point2D(10.0, 0.0),
        Point2D(10.0, 10.0),
        Point2D(0.0, 10.0),
        Point2D(0.0, 1.0),
    ]
    pose = Pose2D(0.0, 1.0, 0.0)
    unlimited = measure_polyline_alignment(
        pose, points, previous_segment_index=0, search_ahead=80
    )
    limited = measure_polyline_alignment(
        pose,
        points,
        previous_segment_index=0,
        search_ahead=80,
        search_ahead_distance_m=12.0,
    )
    assert unlimited.segment_index == 3
    assert limited.segment_index == 0
