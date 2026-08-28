import math

import numpy as np

from terrain_navigation_pkg.safety_core import CollisionStopLatch
from terrain_navigation_pkg.safety_core import EmergencyStopHysteresis
from terrain_navigation_pkg.safety_core import ForwardObstacleObservation
from terrain_navigation_pkg.safety_core import RecoveryCurvatureHold
from terrain_navigation_pkg.safety_core import observe_forward_corridor


def test_forward_corridor_rejects_ground_side_and_distant_points():
    points = np.array([
        [4.0, 0.0, -0.5],
        [5.0, 0.5, 0.0],
        [7.0, 0.0, -0.2],
        [4.0, 2.0, 0.0],
        [4.0, 0.0, -1.8],
        [9.0, 0.0, 0.0],
    ], dtype=np.float32)
    result = observe_forward_corridor(
        points, 1.0, 6.0, 8.0, 1.35, -1.4, 1.0
    )
    assert result.stop_point_count == 2
    assert result.clear_point_count == 3
    assert result.nearest_distance_m == 4.0


def test_empty_corridor_has_infinite_nearest_distance():
    result = observe_forward_corridor(
        np.empty((0, 3)), 1.0, 6.0, 8.0, 1.35, -1.4, 1.0
    )
    assert result.stop_point_count == 0
    assert result.clear_point_count == 0
    assert math.isinf(result.nearest_distance_m)


def test_emergency_stop_is_immediate_and_resume_is_debounced():
    monitor = EmergencyStopHysteresis(
        minimum_points=3,
        clear_required_scans=3,
    )
    blocked = ForwardObstacleObservation(3, 3, 4.0)
    hysteresis_only = ForwardObstacleObservation(0, 3, 7.0)
    clear = ForwardObstacleObservation(0, 0, math.inf)

    assert monitor.update(blocked)
    assert monitor.update(hysteresis_only)
    assert monitor.update(clear)
    assert monitor.update(clear)
    assert not monitor.update(clear)


def test_rechecking_same_cloud_does_not_release_stop_hysteresis():
    monitor = EmergencyStopHysteresis(
        minimum_points=20,
        clear_required_scans=3,
    )
    blocked = ForwardObstacleObservation(20, 20, 4.0)
    clear = ForwardObstacleObservation(0, 0, math.inf)

    assert monitor.update(blocked)
    for _ in range(10):
        assert monitor.update(clear, allow_clear_progress=False)
    assert monitor.clear_scan_count == 0
    assert monitor.update(clear)
    assert monitor.update(clear)
    assert not monitor.update(clear)


def test_collision_stop_latch_requires_explicit_reset():
    latch = CollisionStopLatch(minimum_intensity=1.0)
    assert not latch.observe(0.5)
    assert latch.observe(120.0)
    assert latch.observe(0.0)
    latch.reset()
    assert not latch.latched


def test_collision_stop_latch_ignores_non_finite_event():
    latch = CollisionStopLatch(minimum_intensity=0.0)
    assert not latch.observe(math.nan)


def test_recovery_hold_keeps_direction_and_smooths_same_side_candidates():
    hold = RecoveryCurvatureHold(0.06, 1.0, 8.0, 0.5)
    hold.observe_safe(-0.20, 10.0)
    assert hold.activate_or_current(10.2) == -0.20
    hold.observe_safe(-0.10, 10.3)
    assert math.isclose(hold.activate_or_current(10.4), -0.15)
    hold.observe_safe(0.18, 10.5)
    assert math.isclose(hold.activate_or_current(10.6), -0.15)
    assert hold.activate_or_current(11.4) is None


def test_recovery_hold_expires_without_same_direction_refresh():
    hold = RecoveryCurvatureHold(0.06, 1.0, 8.0, 0.35)
    hold.observe_safe(0.2, 4.0)
    assert hold.activate_or_current(4.5) == 0.2
    assert hold.activate_or_current(5.1) is None


def test_recovery_hold_rejects_straight_candidate_and_can_be_invalidated():
    hold = RecoveryCurvatureHold(0.06, 1.0, 8.0, 0.35)
    hold.observe_safe(0.02, 1.0)
    assert hold.activate_or_current(1.1) is None
    hold.observe_safe(0.2, 1.2)
    assert hold.activate_or_current(1.3) == 0.2
    hold.invalidate()
    assert hold.activate_or_current(1.4) is None
