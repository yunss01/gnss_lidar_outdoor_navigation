"""Pure helpers for a short, odometry-anchored obstacle detour."""

from dataclasses import dataclass
import math
from typing import Optional, Sequence

import numpy as np

from .navigation_core import normalize_angle_rad
from .safety_core import ForwardObstacleObservation


LEFT = 1
RIGHT = -1


@dataclass(frozen=True)
class Pose2D:
    x_m: float
    y_m: float
    yaw_rad: float


@dataclass(frozen=True)
class Point2D:
    x_m: float
    y_m: float


@dataclass(frozen=True)
class LocalCommand:
    speed_mps: float
    yaw_rate_rps: float
    heading_error_rad: float
    distance_m: float


@dataclass(frozen=True)
class PassageObservation:
    """Obstacle returns on the inside of an active avoidance manoeuvre."""

    not_passed_point_count: int
    behind_point_count: int


@dataclass(frozen=True)
class LocalTrajectoryPlan:
    """Best collision-free constant-curvature path in the LiDAR frame."""

    curvature_per_m: float
    desired_curvature_per_m: float
    collision_point_count: int
    safety_validation_point_count: int
    proximity_point_count: int
    nearest_path_m: float
    score: float
    valid_candidate_count: int
    candidate_count: int
    path_points: tuple


@dataclass(frozen=True)
class CommandedTrajectoryObservation:
    """Nominal and uncertainty-envelope observations for one command."""

    nominal: ForwardObstacleObservation
    envelope: ForwardObstacleObservation
    commanded_curvature_per_m: float
    tested_curvature_count: int
    nominal_stop_points: np.ndarray


@dataclass(frozen=True)
class PathAlignment:
    """Vehicle alignment relative to a fixed world-frame reference line."""

    progress_m: float
    cross_track_m: float
    heading_error_rad: float


@dataclass(frozen=True)
class PolylineAlignment:
    """Vehicle alignment to one segment of a forward-only polyline."""

    progress_m: float
    cross_track_m: float
    heading_error_rad: float
    segment_index: int
    segment_ratio: float


def sample_constant_curvature_path(
    curvature_per_m: float,
    horizon_m: float,
    spacing_m: float,
):
    """Sample an Ackermann-feasible arc in ROS FLU vehicle coordinates."""
    if not all(math.isfinite(value) for value in (
        curvature_per_m,
        horizon_m,
        spacing_m,
    )):
        raise ValueError('trajectory sampling inputs must be finite')
    if horizon_m <= 0.0 or spacing_m <= 0.0:
        raise ValueError('trajectory horizon and spacing must be positive')
    sample_count = max(1, int(math.ceil(horizon_m / spacing_m)))
    distance = np.linspace(0.0, horizon_m, sample_count + 1)
    if abs(curvature_per_m) < 1e-9:
        x = distance
        y = np.zeros_like(distance)
        yaw = np.zeros_like(distance)
    else:
        yaw = curvature_per_m * distance
        x = np.sin(yaw) / curvature_per_m
        y = (1.0 - np.cos(yaw)) / curvature_per_m
    return np.column_stack((x, y, yaw))


def trajectory_handoff_ready(
    direct_path_clear: bool,
    recovery_aligned: bool,
    obstacle_side_lock_active: bool,
) -> bool:
    """Allow route handoff only after the passed obstacle clears the side.

    A detour can momentarily intersect the saved route while the obstacle is
    still beside the vehicle.  Releasing control at that instant makes the
    route follower steer back into the obstacle.  The side lock is cleared by
    consecutive LiDAR scans only after that obstacle is no longer present in
    the inside corridor.
    """
    return bool(
        direct_path_clear
        and recovery_aligned
        and not obstacle_side_lock_active
    )


def update_trajectory_side_lock(
    lock_active: bool,
    clear_count: int,
    obstacle_point_count: int,
    minimum_obstacle_points: int,
    clear_required_scans: int,
    release_permitted: bool = True,
):
    """Advance the one-way side lock used during one avoidance episode.

    Once consecutive scans establish that the original obstacle has passed,
    later returns must be handled as current geometry by collision checking;
    they must not resurrect the old direction constraint during recovery.
    """
    if clear_count < 0:
        raise ValueError('side-lock clear count must be non-negative')
    if obstacle_point_count < 0:
        raise ValueError('obstacle point count must be non-negative')
    if minimum_obstacle_points < 1 or clear_required_scans < 1:
        raise ValueError('side-lock thresholds must be positive')
    if not lock_active:
        return clear_count, False
    # The side corridor opposite the chosen turn is often empty before the
    # vehicle has even shifted around the obstacle. It must not release the
    # direction lock while the original navigation path is still blocked.
    if not release_permitted:
        return 0, True
    if obstacle_point_count < minimum_obstacle_points:
        clear_count += 1
    else:
        clear_count = 0
    return clear_count, clear_count < clear_required_scans


def trajectory_recommit_required(
    center_blocked: bool,
    preceding_clear_scans: int,
    clear_required_scans: int,
    handoff_active: bool = False,
    cross_track_m: Optional[float] = None,
    maximum_cross_track_m: Optional[float] = None,
) -> bool:
    """Recognize a new obstacle instead of extending an old recovery episode."""
    if preceding_clear_scans < 0:
        raise ValueError('preceding clear scans must be non-negative')
    if clear_required_scans < 1:
        raise ValueError('clear required scans must be positive')
    if maximum_cross_track_m is not None and not handoff_active:
        if (
            not math.isfinite(maximum_cross_track_m)
            or maximum_cross_track_m <= 0.0
        ):
            raise ValueError('maximum recommit cross track must be positive')
        if cross_track_m is None or not math.isfinite(cross_track_m):
            return False
    return bool(
        center_blocked
        and (
            handoff_active
            or (
                preceding_clear_scans >= clear_required_scans
                and (
                    maximum_cross_track_m is None
                    or abs(cross_track_m) <= maximum_cross_track_m
                )
            )
        )
    )


def trajectory_side_lock_release_permitted(
    center_blocked: bool,
    committed_side: int,
    cross_track_m: float,
    minimum_lateral_shift_m: float,
) -> bool:
    """Permit return steering after a real lateral shift around an obstacle.

    A side lock must survive the first few scans of an avoidance manoeuvre,
    even when the inside side corridor or the centre corridor happens to look
    empty.  A temporarily clear corridor is not proof that the vehicle body
    has passed the obstacle.  Odometry must first show that the configured
    lateral shift was actually achieved.
    """
    if committed_side not in (LEFT, RIGHT):
        raise ValueError('committed side must be LEFT or RIGHT')
    if not math.isfinite(cross_track_m):
        raise ValueError('cross track must be finite')
    if (
        not math.isfinite(minimum_lateral_shift_m)
        or minimum_lateral_shift_m <= 0.0
    ):
        raise ValueError('minimum lateral shift must be positive')
    signed_shift = committed_side * cross_track_m
    return bool(signed_shift >= minimum_lateral_shift_m)


def measure_path_alignment(pose: Pose2D, origin: Point2D, yaw_rad: float):
    """Measure signed cross-track and heading errors to a reference line."""
    if not all(math.isfinite(value) for value in (
        pose.x_m,
        pose.y_m,
        pose.yaw_rad,
        origin.x_m,
        origin.y_m,
        yaw_rad,
    )):
        raise ValueError('path alignment inputs must be finite')
    delta_x = pose.x_m - origin.x_m
    delta_y = pose.y_m - origin.y_m
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    return PathAlignment(
        progress_m=cosine * delta_x + sine * delta_y,
        cross_track_m=-sine * delta_x + cosine * delta_y,
        heading_error_rad=normalize_angle_rad(pose.yaw_rad - yaw_rad),
    )


def measure_polyline_alignment(
    pose: Pose2D,
    points: Sequence[Point2D],
    previous_segment_index: int = 0,
    search_ahead: int = 80,
    search_ahead_distance_m: Optional[float] = None,
):
    """Find alignment to the nearest not-yet-passed polyline segment.

    Searching only forward from ``previous_segment_index`` prevents a closed
    route from snapping back to a geometrically nearby segment from an earlier
    part of the lap.
    """
    if len(points) < 2:
        raise ValueError('polyline needs at least two points')
    if search_ahead < 1:
        raise ValueError('search_ahead must be positive')
    if (
        search_ahead_distance_m is not None
        and (
            not math.isfinite(search_ahead_distance_m)
            or search_ahead_distance_m <= 0.0
        )
    ):
        raise ValueError('search_ahead_distance_m must be positive')
    values = (pose.x_m, pose.y_m, pose.yaw_rad)
    if not all(math.isfinite(value) for value in values):
        raise ValueError('pose must be finite')
    first_index = max(
        0,
        min(int(previous_segment_index), len(points) - 2),
    )
    stop_index = min(len(points) - 1, first_index + search_ahead)
    best = None
    cumulative = 0.0
    distance_from_first = 0.0
    for index in range(len(points) - 1):
        first = points[index]
        second = points[index + 1]
        delta_x = second.x_m - first.x_m
        delta_y = second.y_m - first.y_m
        segment_length = math.hypot(delta_x, delta_y)
        if segment_length <= 1e-9:
            continue
        within_distance = (
            search_ahead_distance_m is None
            or distance_from_first <= search_ahead_distance_m
        )
        if first_index <= index < stop_index and within_distance:
            projection = (
                (pose.x_m - first.x_m) * delta_x
                + (pose.y_m - first.y_m) * delta_y
            ) / (segment_length * segment_length)
            ratio = max(0.0, min(1.0, projection))
            closest_x = first.x_m + ratio * delta_x
            closest_y = first.y_m + ratio * delta_y
            distance_squared = (
                (pose.x_m - closest_x) ** 2
                + (pose.y_m - closest_y) ** 2
            )
            candidate = (
                distance_squared,
                index,
                ratio,
                cumulative,
                segment_length,
                closest_x,
                closest_y,
                math.atan2(delta_y, delta_x),
            )
            if best is None or candidate[0] < best[0]:
                best = candidate
        cumulative += segment_length
        if index >= first_index:
            distance_from_first += segment_length
    if best is None:
        raise ValueError('polyline has no non-zero segment in search window')
    _, index, ratio, progress_before, segment_length, closest_x, closest_y, yaw = best
    cross_track = (
        -math.sin(yaw) * (pose.x_m - closest_x)
        + math.cos(yaw) * (pose.y_m - closest_y)
    )
    return PolylineAlignment(
        progress_m=progress_before + ratio * segment_length,
        cross_track_m=cross_track,
        heading_error_rad=normalize_angle_rad(pose.yaw_rad - yaw),
        segment_index=index,
        segment_ratio=ratio,
    )


def polyline_lookahead_target(
    points: Sequence[Point2D],
    alignment: PolylineAlignment,
    lookahead_m: float,
):
    """Return a point ``lookahead_m`` forward from a polyline projection."""
    if len(points) < 2:
        raise ValueError('polyline needs at least two points')
    if not math.isfinite(lookahead_m) or lookahead_m <= 0.0:
        raise ValueError('lookahead_m must be positive')
    index = max(0, min(alignment.segment_index, len(points) - 2))
    first = points[index]
    second = points[index + 1]
    ratio = max(0.0, min(1.0, alignment.segment_ratio))
    cursor = Point2D(
        first.x_m + ratio * (second.x_m - first.x_m),
        first.y_m + ratio * (second.y_m - first.y_m),
    )
    remaining = lookahead_m
    while index < len(points) - 1:
        endpoint = points[index + 1]
        segment_remaining = math.hypot(
            endpoint.x_m - cursor.x_m,
            endpoint.y_m - cursor.y_m,
        )
        if segment_remaining >= remaining and segment_remaining > 1e-9:
            target_ratio = remaining / segment_remaining
            return Point2D(
                cursor.x_m + target_ratio * (endpoint.x_m - cursor.x_m),
                cursor.y_m + target_ratio * (endpoint.y_m - cursor.y_m),
            )
        remaining -= segment_remaining
        index += 1
        cursor = points[index]
    return points[-1]


def _evaluate_swept_vehicle_footprint(
    xyz_points,
    curvature_per_m: float,
    minimum_x_m: float,
    stop_path_m: float,
    clear_path_m: float,
    half_width_m: float,
    vehicle_front_m: float,
    vehicle_rear_m: float,
    minimum_z_m: float,
    maximum_z_m: float,
    sample_spacing_m: float = 0.25,
    ego_half_width_m: float = 0.0,
) -> tuple[ForwardObstacleObservation, np.ndarray]:
    """Count returns touched by the complete vehicle body along an arc.

    Unlike a centre-line tube, every sampled pose uses an oriented rectangle
    extending from ``-vehicle_rear_m`` to ``vehicle_front_m``.  This captures
    the outside front and inside rear corners during tight Ackermann turns.
    ``minimum_x_m`` continues to reject roof-LiDAR returns from the ego hood.
    """
    points = np.asarray(xyz_points)
    if points.size == 0:
        points = np.empty((0, 3), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError('xyz_points must have shape (N, >=3)')
    values = (
        curvature_per_m,
        minimum_x_m,
        stop_path_m,
        clear_path_m,
        half_width_m,
        vehicle_front_m,
        vehicle_rear_m,
        minimum_z_m,
        maximum_z_m,
        sample_spacing_m,
        ego_half_width_m,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError('swept-footprint inputs must be finite')
    if minimum_x_m < 0.0:
        raise ValueError('minimum_x_m must be non-negative')
    if not 0.0 < stop_path_m < clear_path_m:
        raise ValueError('path limits must satisfy 0 < stop < clear')
    if min(
        half_width_m,
        vehicle_front_m,
        vehicle_rear_m,
        sample_spacing_m,
    ) <= 0.0:
        raise ValueError('vehicle footprint dimensions must be positive')
    if ego_half_width_m < 0.0:
        raise ValueError('ego exclusion width must be non-negative')
    if minimum_z_m >= maximum_z_m:
        raise ValueError('minimum_z_m must be below maximum_z_m')
    if points.shape[0] == 0:
        return (
            ForwardObstacleObservation(0, 0, math.inf),
            np.empty((0, 3), dtype=np.float32),
        )

    xyz = points[:, :3].astype(np.float64, copy=False)
    spatial_candidate = (
        np.isfinite(xyz).all(axis=1)
        & (xyz[:, 0] >= -vehicle_rear_m - half_width_m)
        & (xyz[:, 0] <= clear_path_m + vehicle_front_m)
        & (np.abs(xyz[:, 1]) <= clear_path_m + half_width_m)
        & (xyz[:, 2] >= minimum_z_m)
        & (xyz[:, 2] <= maximum_z_m)
    )
    if ego_half_width_m > 0.0:
        # Reject only returns from the ego body. Points beside the doors or
        # rear quarter remain available to catch side-swipe collisions.
        ego_body = (
            (xyz[:, 0] >= -vehicle_rear_m)
            & (xyz[:, 0] <= minimum_x_m)
            & (np.abs(xyz[:, 1]) <= ego_half_width_m)
        )
        candidate = spatial_candidate & ~ego_body
    else:
        # Backward-compatible behaviour for callers without an ego width.
        candidate = spatial_candidate & (xyz[:, 0] >= minimum_x_m)
    xy = xyz[candidate, :2]
    if xy.shape[0] == 0:
        return (
            ForwardObstacleObservation(0, 0, math.inf),
            np.empty((0, 3), dtype=np.float32),
        )

    samples = sample_constant_curvature_path(
        curvature_per_m,
        clear_path_m,
        sample_spacing_m,
    )
    sample_progress = np.linspace(0.0, clear_path_m, samples.shape[0])
    first_collision_progress = np.full(xy.shape[0], math.inf)
    for start in range(0, xy.shape[0], 4096):
        end = min(start + 4096, xy.shape[0])
        delta_x = xy[start:end, None, 0] - samples[None, :, 0]
        delta_y = xy[start:end, None, 1] - samples[None, :, 1]
        cosine = np.cos(samples[None, :, 2])
        sine = np.sin(samples[None, :, 2])
        longitudinal = cosine * delta_x + sine * delta_y
        lateral = -sine * delta_x + cosine * delta_y
        inside = (
            (longitudinal >= -vehicle_rear_m)
            & (longitudinal <= vehicle_front_m)
            & (np.abs(lateral) <= half_width_m)
        )
        any_inside = np.any(inside, axis=1)
        first_index = np.argmax(inside, axis=1)
        first_collision_progress[start:end][any_inside] = (
            sample_progress[first_index[any_inside]]
        )

    clear_mask = np.isfinite(first_collision_progress)
    stop_mask = clear_mask & (first_collision_progress <= stop_path_m)
    stop_count = int(np.count_nonzero(stop_mask))
    clear_count = int(np.count_nonzero(clear_mask))
    nearest = (
        float(np.min(first_collision_progress[clear_mask]))
        if clear_count else math.inf
    )
    observation = ForwardObstacleObservation(stop_count, clear_count, nearest)
    candidate_xyz = xyz[candidate, :3]
    stop_points = candidate_xyz[stop_mask].astype(np.float32, copy=True)
    return observation, stop_points


def observe_swept_vehicle_footprint(
    xyz_points,
    curvature_per_m: float,
    minimum_x_m: float,
    stop_path_m: float,
    clear_path_m: float,
    half_width_m: float,
    vehicle_front_m: float,
    vehicle_rear_m: float,
    minimum_z_m: float,
    maximum_z_m: float,
    sample_spacing_m: float = 0.25,
    ego_half_width_m: float = 0.0,
) -> ForwardObstacleObservation:
    """Return the legacy count-only swept-footprint observation."""
    observation, _ = _evaluate_swept_vehicle_footprint(
        xyz_points,
        curvature_per_m,
        minimum_x_m,
        stop_path_m,
        clear_path_m,
        half_width_m,
        vehicle_front_m,
        vehicle_rear_m,
        minimum_z_m,
        maximum_z_m,
        sample_spacing_m,
        ego_half_width_m,
    )
    return observation


def observe_commanded_trajectory(
    xyz_points,
    linear_speed_mps: float,
    yaw_rate_rps: float,
    previous_curvature_per_m: float,
    maximum_curvature_per_m: float,
    curvature_uncertainty_per_m: float,
    minimum_x_m: float,
    stop_path_m: float,
    clear_path_m: float,
    half_width_m: float,
    extra_width_m: float,
    vehicle_front_m: float,
    vehicle_rear_m: float,
    vehicle_width_m: float,
    minimum_z_m: float,
    maximum_z_m: float,
    sample_spacing_m: float,
) -> CommandedTrajectoryObservation:
    """Observe the Ackermann command and a small steering-lag envelope.

    This is shared by local planning and the final safety gate so both nodes
    interpret the commanded vehicle trajectory with identical geometry.
    """
    values = (
        linear_speed_mps,
        yaw_rate_rps,
        previous_curvature_per_m,
        maximum_curvature_per_m,
        curvature_uncertainty_per_m,
        extra_width_m,
        vehicle_width_m,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError('commanded trajectory inputs must be finite')
    if maximum_curvature_per_m <= 0.0:
        raise ValueError('maximum curvature must be positive')
    if curvature_uncertainty_per_m < 0.0 or extra_width_m < 0.0:
        raise ValueError('trajectory uncertainty must be non-negative')
    if vehicle_width_m <= 0.0:
        raise ValueError('vehicle width must be positive')

    if abs(linear_speed_mps) > 0.05:
        raw_curvature = yaw_rate_rps / abs(linear_speed_mps)
    else:
        raw_curvature = 0.0
    commanded_curvature = max(
        -maximum_curvature_per_m,
        min(maximum_curvature_per_m, raw_curvature),
    )
    previous_curvature = max(
        -maximum_curvature_per_m,
        min(maximum_curvature_per_m, previous_curvature_per_m),
    )
    # Exclude the complete *current* padded ego footprint, not only the
    # unpadded body width.  A roof LiDAR can see the hood, mirrors and body
    # edges slightly outside the nominal vehicle width.  Keeping those
    # returns makes the first swept sample report an obstacle at progress
    # zero and permanently deadlocks the safety gate.  The exclusion ends at
    # ``minimum_x_m``; obstacles beyond the ego nose are still evaluated by
    # every future swept footprint sample.
    ego_exclusion_half_width_m = max(
        0.5 * vehicle_width_m,
        half_width_m,
    )
    nominal, nominal_stop_points = _evaluate_swept_vehicle_footprint(
        xyz_points,
        commanded_curvature,
        minimum_x_m,
        stop_path_m,
        clear_path_m,
        half_width_m,
        vehicle_front_m,
        vehicle_rear_m,
        minimum_z_m,
        maximum_z_m,
        sample_spacing_m,
        ego_half_width_m=ego_exclusion_half_width_m,
    )
    curvatures = {
        commanded_curvature,
        previous_curvature,
        max(
            -maximum_curvature_per_m,
            commanded_curvature - curvature_uncertainty_per_m,
        ),
        min(
            maximum_curvature_per_m,
            commanded_curvature + curvature_uncertainty_per_m,
        ),
    }
    observations = [
        observe_swept_vehicle_footprint(
            xyz_points,
            curvature,
            minimum_x_m,
            stop_path_m,
            clear_path_m,
            half_width_m + extra_width_m,
            vehicle_front_m,
            vehicle_rear_m,
            minimum_z_m,
            maximum_z_m,
            sample_spacing_m,
            ego_half_width_m=ego_exclusion_half_width_m,
        )
        for curvature in curvatures
    ]
    envelope = max(
        observations,
        key=lambda value: (
            value.stop_point_count,
            value.clear_point_count,
            -value.nearest_distance_m,
        ),
    )
    return CommandedTrajectoryObservation(
        nominal=nominal,
        envelope=envelope,
        commanded_curvature_per_m=commanded_curvature,
        tested_curvature_count=len(observations),
        nominal_stop_points=nominal_stop_points,
    )


def observe_curved_corridor(
    xyz_points,
    curvature_per_m: float,
    minimum_path_m: float,
    stop_path_m: float,
    clear_path_m: float,
    half_width_m: float,
    minimum_z_m: float,
    maximum_z_m: float,
    sample_spacing_m: float = 0.35,
) -> ForwardObstacleObservation:
    """Count returns in the swept corridor of a constant-curvature path."""
    points = np.asarray(xyz_points)
    if points.size == 0:
        points = np.empty((0, 3), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError('xyz_points must have shape (N, >=3)')
    values = (
        curvature_per_m,
        minimum_path_m,
        stop_path_m,
        clear_path_m,
        half_width_m,
        minimum_z_m,
        maximum_z_m,
        sample_spacing_m,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError('curved corridor inputs must be finite')
    if minimum_path_m < 0.0:
        raise ValueError('minimum path distance must be non-negative')
    if not minimum_path_m < stop_path_m < clear_path_m:
        raise ValueError('path limits must satisfy minimum < stop < clear')
    if half_width_m <= 0.0 or sample_spacing_m <= 0.0:
        raise ValueError('corridor width and spacing must be positive')
    if minimum_z_m >= maximum_z_m:
        raise ValueError('minimum_z_m must be below maximum_z_m')
    if points.shape[0] == 0:
        return ForwardObstacleObservation(0, 0, math.inf)

    xyz = points[:, :3].astype(np.float64, copy=False)
    finite_height = (
        np.isfinite(xyz).all(axis=1)
        & (xyz[:, 2] >= minimum_z_m)
        & (xyz[:, 2] <= maximum_z_m)
    )
    xy = xyz[finite_height, :2]
    if xy.shape[0] == 0:
        return ForwardObstacleObservation(0, 0, math.inf)

    samples = sample_constant_curvature_path(
        curvature_per_m,
        clear_path_m,
        sample_spacing_m,
    )
    sample_xy = samples[:, :2]
    sample_distance = np.linspace(0.0, clear_path_m, samples.shape[0])
    minimum_distance_squared = np.full(xy.shape[0], math.inf)
    closest_path_distance = np.zeros(xy.shape[0], dtype=np.float64)
    # Chunking avoids an N-points by N-samples allocation for a full scan.
    for start in range(0, xy.shape[0], 4096):
        end = min(start + 4096, xy.shape[0])
        delta = xy[start:end, None, :] - sample_xy[None, :, :]
        distance_squared = np.sum(delta * delta, axis=2)
        closest_index = np.argmin(distance_squared, axis=1)
        minimum_distance_squared[start:end] = distance_squared[
            np.arange(end - start), closest_index
        ]
        closest_path_distance[start:end] = sample_distance[closest_index]

    inside = minimum_distance_squared <= half_width_m ** 2
    relevant = inside & (closest_path_distance >= minimum_path_m)
    stop_mask = relevant & (closest_path_distance <= stop_path_m)
    clear_mask = relevant & (closest_path_distance <= clear_path_m)
    stop_count = int(np.count_nonzero(stop_mask))
    clear_count = int(np.count_nonzero(clear_mask))
    nearest = (
        float(np.min(closest_path_distance[clear_mask]))
        if clear_count else math.inf
    )
    return ForwardObstacleObservation(stop_count, clear_count, nearest)


def select_collision_free_trajectory(
    xyz_points,
    desired_curvature_per_m: float,
    previous_curvature_per_m: Optional[float],
    maximum_curvature_per_m: float,
    candidate_count: int,
    horizon_m: float,
    sample_spacing_m: float,
    minimum_path_m: float,
    corridor_half_width_m: float,
    proximity_margin_m: float,
    minimum_obstacle_points: int,
    minimum_z_m: float,
    maximum_z_m: float,
    preferred_side: int = LEFT,
    goal_weight: float = 3.0,
    switch_weight: float = 1.0,
    curvature_weight: float = 0.15,
    proximity_weight: float = 0.6,
    voxel_size_m: float = 0.0,
    vehicle_front_m: float = 0.0,
    vehicle_rear_m: float = 0.0,
    ego_half_width_m: float = 0.0,
    opposite_goal_weight: float = 0.0,
    allowed_side: Optional[int] = None,
    validation_clear_path_m: Optional[float] = None,
    validation_sample_spacing_m: Optional[float] = None,
    validation_minimum_obstacle_points: Optional[int] = None,
    excluded_curvatures: Sequence[float] = (),
    excluded_curvature_tolerance_per_m: float = 0.0,
) -> Optional[LocalTrajectoryPlan]:
    """Select a safe arc while preferring goal progress and smooth steering."""
    if maximum_curvature_per_m <= 0.0:
        raise ValueError('maximum trajectory curvature must be positive')
    if candidate_count < 3 or candidate_count % 2 == 0:
        raise ValueError('trajectory candidate count must be odd and >= 3')
    if horizon_m <= minimum_path_m + sample_spacing_m:
        raise ValueError('trajectory horizon is too short')
    if corridor_half_width_m <= 0.0 or proximity_margin_m < 0.0:
        raise ValueError('trajectory clearance dimensions are invalid')
    if minimum_obstacle_points < 1:
        raise ValueError('minimum obstacle points must be positive')
    if not math.isfinite(voxel_size_m) or voxel_size_m < 0.0:
        raise ValueError('trajectory voxel size must be non-negative')
    if preferred_side not in (LEFT, RIGHT):
        raise ValueError('preferred side must be LEFT or RIGHT')
    if allowed_side not in (None, LEFT, RIGHT):
        raise ValueError('allowed side must be left, right, or unrestricted')
    if validation_clear_path_m is not None:
        if not math.isfinite(validation_clear_path_m):
            raise ValueError('validation clear path must be finite')
        if validation_clear_path_m <= minimum_path_m:
            raise ValueError('validation clear path is too short')
        if validation_sample_spacing_m is None:
            validation_sample_spacing_m = sample_spacing_m
        if (
            not math.isfinite(validation_sample_spacing_m)
            or validation_sample_spacing_m <= 0.0
        ):
            raise ValueError('validation sample spacing must be positive')
        if validation_minimum_obstacle_points is None:
            validation_minimum_obstacle_points = minimum_obstacle_points
        if validation_minimum_obstacle_points < 1:
            raise ValueError('validation obstacle points must be positive')
    if (
        not math.isfinite(excluded_curvature_tolerance_per_m)
        or excluded_curvature_tolerance_per_m < 0.0
    ):
        raise ValueError('excluded curvature tolerance is invalid')
    excluded = tuple(float(value) for value in excluded_curvatures)
    if not all(math.isfinite(value) for value in excluded):
        raise ValueError('excluded curvatures must be finite')
    weights = (
        goal_weight,
        switch_weight,
        curvature_weight,
        proximity_weight,
        opposite_goal_weight,
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in weights):
        raise ValueError('trajectory weights must be finite and non-negative')

    desired = max(
        -maximum_curvature_per_m,
        min(maximum_curvature_per_m, desired_curvature_per_m),
    )
    previous = (
        desired if previous_curvature_per_m is None
        else max(
            -maximum_curvature_per_m,
            min(maximum_curvature_per_m, previous_curvature_per_m),
        )
    )
    curvatures = np.linspace(
        -maximum_curvature_per_m,
        maximum_curvature_per_m,
        candidate_count,
    )
    raw_points = np.asarray(xyz_points)
    if raw_points.size == 0:
        raw_points = np.empty((0, 3), dtype=np.float32)
    if raw_points.ndim != 2 or raw_points.shape[1] < 3:
        raise ValueError('xyz_points must have shape (N, >=3)')
    raw_points = raw_points[:, :3].astype(np.float32, copy=False)
    finite = np.isfinite(raw_points).all(axis=1)
    spatial_margin = corridor_half_width_m + proximity_margin_m
    forward_margin = max(spatial_margin, vehicle_front_m)
    rear_margin = max(spatial_margin, vehicle_rear_m)
    relevant = (
        finite
        & (raw_points[:, 0] >= -rear_margin)
        & (raw_points[:, 0] <= horizon_m + forward_margin)
        & (np.abs(raw_points[:, 1]) <= horizon_m + spatial_margin)
        & (raw_points[:, 2] >= minimum_z_m)
        & (raw_points[:, 2] <= maximum_z_m)
    )
    validation_points = raw_points[relevant]
    planning_points = validation_points
    if voxel_size_m > 0.0 and planning_points.shape[0] > 0:
        voxel = np.floor(
            planning_points / voxel_size_m
        ).astype(np.int32)
        _unique, unique_indices = np.unique(voxel, axis=0, return_index=True)
        planning_points = planning_points[np.sort(unique_indices)]
    best = None
    valid_count = 0
    footprint_enabled = vehicle_front_m > 0.0 and vehicle_rear_m > 0.0
    for curvature in curvatures:
        if any(
            abs(float(curvature) - rejected)
            <= excluded_curvature_tolerance_per_m
            for rejected in excluded
        ):
            continue
        if (
            allowed_side == LEFT and curvature < -1e-6
            or allowed_side == RIGHT and curvature > 1e-6
        ):
            continue
        observer = (
            observe_swept_vehicle_footprint
            if footprint_enabled else observe_curved_corridor
        )
        footprint_arguments = (
            (vehicle_front_m, vehicle_rear_m) if footprint_enabled else ()
        )
        footprint_keywords = (
            {'ego_half_width_m': ego_half_width_m}
            if footprint_enabled else {}
        )
        hard = observer(
            planning_points,
            float(curvature),
            minimum_path_m,
            horizon_m - sample_spacing_m,
            horizon_m,
            corridor_half_width_m,
            *footprint_arguments,
            minimum_z_m,
            maximum_z_m,
            sample_spacing_m,
            **footprint_keywords,
        )
        if hard.clear_point_count >= minimum_obstacle_points:
            continue
        safety_validation = ForwardObstacleObservation(0, 0, math.inf)
        if validation_clear_path_m is not None:
            validation_stop_path_m = max(
                minimum_path_m + 1e-3,
                validation_clear_path_m - validation_sample_spacing_m,
            )
            safety_validation = observer(
                validation_points,
                float(curvature),
                minimum_path_m,
                validation_stop_path_m,
                validation_clear_path_m,
                corridor_half_width_m,
                *footprint_arguments,
                minimum_z_m,
                maximum_z_m,
                validation_sample_spacing_m,
                **footprint_keywords,
            )
            if (
                safety_validation.clear_point_count
                >= validation_minimum_obstacle_points
            ):
                continue
        valid_count += 1
        proximity = observer(
            planning_points,
            float(curvature),
            minimum_path_m,
            horizon_m - sample_spacing_m,
            horizon_m,
            corridor_half_width_m + proximity_margin_m,
            *footprint_arguments,
            minimum_z_m,
            maximum_z_m,
            sample_spacing_m,
            **footprint_keywords,
        )
        scale = maximum_curvature_per_m
        # A hard cap made every close path look equally bad once it contained
        # 20 nearby voxels.  log1p keeps the score bounded in practice while
        # still preferring the path with materially more side clearance.
        proximity_ratio = math.log1p(
            proximity.clear_point_count / float(minimum_obstacle_points)
        )
        side_penalty = 0.0
        if abs(curvature) > 1e-6 and int(math.copysign(1, curvature)) != preferred_side:
            side_penalty = 0.02
        opposite_goal_penalty = 0.0
        if (
            abs(desired) > 0.02
            and abs(curvature) > 1e-6
            and desired * curvature < 0.0
        ):
            opposite_goal_penalty = opposite_goal_weight
        score = (
            goal_weight * abs(curvature - desired) / scale
            + switch_weight * abs(curvature - previous) / scale
            + curvature_weight * abs(curvature) / scale
            + proximity_weight * proximity_ratio
            + side_penalty
            + opposite_goal_penalty
        )
        candidate = (
            score,
            abs(curvature - desired),
            abs(curvature),
            float(curvature),
            hard,
            safety_validation,
            proximity,
        )
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    if best is None:
        return None

    (
        score,
        _goal_delta,
        _magnitude,
        curvature,
        hard,
        safety_validation,
        proximity,
    ) = best
    path = sample_constant_curvature_path(
        curvature,
        horizon_m,
        sample_spacing_m,
    )
    return LocalTrajectoryPlan(
        curvature_per_m=curvature,
        desired_curvature_per_m=desired,
        collision_point_count=hard.clear_point_count,
        safety_validation_point_count=(
            safety_validation.clear_point_count
        ),
        proximity_point_count=proximity.clear_point_count,
        nearest_path_m=proximity.nearest_distance_m,
        score=score,
        valid_candidate_count=valid_count,
        candidate_count=candidate_count,
        path_points=tuple(
            Point2D(float(point[0]), float(point[1])) for point in path
        ),
    )


def choose_feasible_trajectory_side(
    left_plan: Optional[LocalTrajectoryPlan],
    right_plan: Optional[LocalTrajectoryPlan],
    preferred_side: int = LEFT,
    left_corridor_points: Optional[int] = None,
    right_corridor_points: Optional[int] = None,
    corridor_advantage_points: int = 0,
) -> Optional[int]:
    """Choose a side from paths already proven feasible by swept geometry.

    Side selection previously used separate rectangular corridors.  That could
    commit to a visually open side even when every Ackermann arc on that side
    clipped an obstacle.  Comparing the best validated plan from each side
    keeps direction choice and collision checking on the same geometry.
    """
    if preferred_side not in (LEFT, RIGHT):
        raise ValueError('preferred_side must be LEFT or RIGHT')
    if corridor_advantage_points < 0:
        raise ValueError('corridor advantage points must be non-negative')
    corridor_counts_available = (
        left_corridor_points is not None
        and right_corridor_points is not None
    )
    if corridor_counts_available and (
        left_corridor_points < 0 or right_corridor_points < 0
    ):
        raise ValueError('corridor point counts must be non-negative')
    if left_plan is None and right_plan is None:
        return None
    if right_plan is None:
        return LEFT
    if left_plan is None:
        return RIGHT

    # Exact swept-footprint validation remains authoritative.  A broad
    # left/right S-corridor is used only when both best arc families have the
    # same raw safety and proximity evidence.  This resolves a genuine tie
    # such as left=50/right=0 points without allowing a rectangular corridor
    # to overrule a materially safer Ackermann arc.
    left_exact_key = (
        left_plan.safety_validation_point_count,
        left_plan.proximity_point_count,
    )
    right_exact_key = (
        right_plan.safety_validation_point_count,
        right_plan.proximity_point_count,
    )
    if left_exact_key < right_exact_key:
        return LEFT
    if right_exact_key < left_exact_key:
        return RIGHT
    if (
        corridor_counts_available
        and abs(left_corridor_points - right_corridor_points)
        >= corridor_advantage_points
        and left_corridor_points != right_corridor_points
    ):
        return (
            LEFT
            if left_corridor_points < right_corridor_points
            else RIGHT
        )

    def clearance_key(plan):
        nearest = plan.nearest_path_m
        finite_nearest = nearest if math.isfinite(nearest) else math.inf
        return (
            -finite_nearest,
            -plan.valid_candidate_count,
            plan.score,
        )

    left_key = clearance_key(left_plan)
    right_key = clearance_key(right_plan)
    if left_key < right_key:
        return LEFT
    if right_key < left_key:
        return RIGHT
    return preferred_side


def associate_progress_interval(
    progress_values,
    candidate_mask,
    minimum_progress_m: float,
    maximum_progress_m: float,
    association_gap_m: float,
    maximum_iterations: int = 4,
):
    """Expand one tracked obstacle interval through connected LiDAR returns."""
    progress = np.asarray(progress_values, dtype=np.float64)
    candidates = np.asarray(candidate_mask, dtype=bool)
    if progress.ndim != 1 or candidates.shape != progress.shape:
        raise ValueError('progress and candidate mask must be equal 1D arrays')
    values = (
        minimum_progress_m,
        maximum_progress_m,
        association_gap_m,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError('tracked interval values must be finite')
    if maximum_progress_m < minimum_progress_m:
        raise ValueError('tracked interval bounds are reversed')
    if association_gap_m <= 0.0 or maximum_iterations < 1:
        raise ValueError('association controls must be positive')

    minimum = minimum_progress_m
    maximum = maximum_progress_m
    eligible = candidates & np.isfinite(progress)
    associated = np.zeros(progress.shape, dtype=bool)
    for _ in range(maximum_iterations):
        associated = (
            eligible
            & (progress >= minimum - association_gap_m)
            & (progress <= maximum + association_gap_m)
        )
        if not np.any(associated):
            break
        new_minimum = min(minimum, float(np.min(progress[associated])))
        new_maximum = max(maximum, float(np.max(progress[associated])))
        if (
            abs(new_minimum - minimum) < 1e-3
            and abs(new_maximum - maximum) < 1e-3
        ):
            break
        minimum = new_minimum
        maximum = new_maximum

    associated = (
        eligible
        & (progress >= minimum - association_gap_m)
        & (progress <= maximum + association_gap_m)
    )
    return minimum, maximum, associated


def clamp_tracked_progress_interval(
    minimum_progress_m: float,
    maximum_progress_m: float,
    seed_progress_m: float,
    maximum_backward_extent_m: float,
    maximum_forward_extent_m: float,
):
    """Keep one tracked obstacle from absorbing connected city geometry."""
    values = (
        minimum_progress_m,
        maximum_progress_m,
        seed_progress_m,
        maximum_backward_extent_m,
        maximum_forward_extent_m,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError('tracked progress interval inputs must be finite')
    if minimum_progress_m > maximum_progress_m:
        raise ValueError('tracked progress interval is reversed')
    if maximum_backward_extent_m <= 0.0 or maximum_forward_extent_m <= 0.0:
        raise ValueError('tracked obstacle extents must be positive')
    lower_bound = seed_progress_m - maximum_backward_extent_m
    upper_bound = seed_progress_m + maximum_forward_extent_m
    return (
        max(lower_bound, minimum_progress_m),
        min(upper_bound, maximum_progress_m),
    )


def choose_avoidance_side(
    left: ForwardObstacleObservation,
    right: ForwardObstacleObservation,
    minimum_obstacle_points: int,
    preferred_side: int = LEFT,
) -> Optional[int]:
    """Choose a clear side, preferring fewer returns then a fixed tie-break."""
    if minimum_obstacle_points < 1:
        raise ValueError('minimum_obstacle_points must be at least one')
    if preferred_side not in (LEFT, RIGHT):
        raise ValueError('preferred_side must be LEFT or RIGHT')
    left_open = left.clear_point_count < minimum_obstacle_points
    right_open = right.clear_point_count < minimum_obstacle_points
    if not left_open and not right_open:
        return None
    if left_open and not right_open:
        return LEFT
    if right_open and not left_open:
        return RIGHT
    if left.clear_point_count < right.clear_point_count:
        return LEFT
    if right.clear_point_count < left.clear_point_count:
        return RIGHT
    return preferred_side


def obstacle_requires_detour(
    obstacle_nearest_m: float,
    goal_distance_m: float,
    goal_stop_distance_m: float,
    margin_m: float,
) -> bool:
    """Return whether an obstacle lies before the intended goal stop point."""
    values = (
        obstacle_nearest_m,
        goal_distance_m,
        goal_stop_distance_m,
        margin_m,
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError(
            'goal guard distances must be finite and non-negative'
        )
    remaining_travel = max(0.0, goal_distance_m - goal_stop_distance_m)
    return obstacle_nearest_m <= remaining_travel + margin_m


def compute_trajectory_planning_horizon(
    configured_horizon_m: float,
    minimum_horizon_m: float,
    goal_distance_m: Optional[float],
    goal_stop_distance_m: float,
    goal_obstacle_margin_m: float,
    terminal_goal: bool,
) -> float:
    """Keep full look-ahead between waypoints and clip only at route end.

    The published GNSS goal is normally an intermediate waypoint. Shortening
    the LiDAR horizon for that point can hide a wall which remains visible to
    the downstream safety node. A terminal goal may shorten the horizon, but
    never below ``minimum_horizon_m``.
    """
    values = (
        configured_horizon_m,
        minimum_horizon_m,
        goal_stop_distance_m,
        goal_obstacle_margin_m,
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError('trajectory horizon distances must be non-negative')
    if configured_horizon_m <= 0.0:
        raise ValueError('configured trajectory horizon must be positive')
    if minimum_horizon_m <= 0.0:
        raise ValueError('minimum trajectory horizon must be positive')
    if minimum_horizon_m > configured_horizon_m:
        raise ValueError(
            'minimum trajectory horizon cannot exceed configured horizon'
        )
    if not terminal_goal or goal_distance_m is None:
        return configured_horizon_m
    if not math.isfinite(goal_distance_m) or goal_distance_m < 0.0:
        raise ValueError('goal distance must be finite and non-negative')
    remaining_travel = max(0.0, goal_distance_m - goal_stop_distance_m)
    return min(
        configured_horizon_m,
        max(
            minimum_horizon_m,
            remaining_travel + goal_obstacle_margin_m,
        ),
    )


def update_lateral_shift_confirmation(
    actual_lateral_m: float,
    target_lateral_m: float,
    tolerance_m: float,
    previous_count: int,
    required_count: int,
):
    """Debounce confirmation that the requested lateral shift was reached."""
    values = (actual_lateral_m, target_lateral_m, tolerance_m)
    if not all(math.isfinite(value) for value in values):
        raise ValueError('lateral confirmation distances must be finite')
    if target_lateral_m <= 0.0:
        raise ValueError('target lateral distance must be positive')
    if not 0.0 <= tolerance_m < target_lateral_m:
        raise ValueError('lateral confirmation tolerance is invalid')
    if previous_count < 0 or required_count < 1:
        raise ValueError('lateral confirmation counts are invalid')
    if actual_lateral_m >= target_lateral_m - tolerance_m:
        count = previous_count + 1
    else:
        count = 0
    return count, count >= required_count


def local_to_world(origin: Pose2D, forward_m: float, left_m: float) -> Point2D:
    """Transform a vehicle-entry-frame point into the odometry frame."""
    cosine = math.cos(origin.yaw_rad)
    sine = math.sin(origin.yaw_rad)
    return Point2D(
        origin.x_m + cosine * forward_m - sine * left_m,
        origin.y_m + sine * forward_m + cosine * left_m,
    )


def world_to_local(origin: Pose2D, point: Point2D) -> Point2D:
    """Transform an odometry-frame point into an entry-pose local frame."""
    delta_x = point.x_m - origin.x_m
    delta_y = point.y_m - origin.y_m
    cosine = math.cos(origin.yaw_rad)
    sine = math.sin(origin.yaw_rad)
    return Point2D(
        cosine * delta_x + sine * delta_y,
        -sine * delta_x + cosine * delta_y,
    )


def quintic_smoothstep(ratio: float) -> float:
    """Return a zero-slope, zero-curvature transition in ``[0, 1]``."""
    clipped = max(0.0, min(1.0, ratio))
    return clipped ** 3 * (
        clipped * (clipped * 6.0 - 15.0) + 10.0
    )


def detour_lateral_offset(
    forward_m,
    side: int,
    lateral_offset_m: float,
    shift_forward_m: float,
):
    """Return the S-curve lateral center at one or more forward positions."""
    if side not in (LEFT, RIGHT):
        raise ValueError('side must be LEFT or RIGHT')
    if lateral_offset_m <= 0.0 or shift_forward_m <= 0.0:
        raise ValueError('detour offset and shift distance must be positive')
    forward = np.asarray(forward_m, dtype=np.float64)
    ratio = np.clip(forward / shift_forward_m, 0.0, 1.0)
    smooth = ratio ** 3 * (ratio * (ratio * 6.0 - 15.0) + 10.0)
    result = side * lateral_offset_m * smooth
    return float(result) if result.ndim == 0 else result


def observe_detour_corridor(
    xyz_points,
    side: int,
    minimum_x_m: float,
    maximum_x_m: float,
    lateral_offset_m: float,
    shift_forward_m: float,
    corridor_half_width_m: float,
    vehicle_front_m: float,
    minimum_z_m: float,
    maximum_z_m: float,
) -> ForwardObstacleObservation:
    """Measure obstacles inside the swept vehicle-width S-curve corridor."""
    points = np.asarray(xyz_points)
    if points.size == 0:
        points = np.empty((0, 3), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError('xyz_points must have shape (N, >=3)')
    values = (
        minimum_x_m,
        maximum_x_m,
        lateral_offset_m,
        shift_forward_m,
        corridor_half_width_m,
        vehicle_front_m,
        minimum_z_m,
        maximum_z_m,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError('detour corridor limits must be finite')
    if side not in (LEFT, RIGHT):
        raise ValueError('side must be LEFT or RIGHT')
    if minimum_x_m < 0.0 or maximum_x_m <= minimum_x_m:
        raise ValueError('detour corridor x limits are invalid')
    if min(
        lateral_offset_m,
        shift_forward_m,
        corridor_half_width_m,
        vehicle_front_m,
    ) <= 0.0:
        raise ValueError('detour corridor dimensions must be positive')
    if minimum_z_m >= maximum_z_m:
        raise ValueError('minimum_z_m must be below maximum_z_m')
    if points.shape[0] == 0:
        return ForwardObstacleObservation(0, 0, math.inf)

    xyz = points[:, :3].astype(np.float64, copy=False)
    finite = np.isfinite(xyz).all(axis=1)
    candidate = (
        finite
        & (xyz[:, 0] >= minimum_x_m)
        & (xyz[:, 0] <= maximum_x_m)
        & (xyz[:, 2] >= minimum_z_m)
        & (xyz[:, 2] <= maximum_z_m)
    )
    if not np.any(candidate):
        return ForwardObstacleObservation(0, 0, math.inf)

    filtered = xyz[candidate]
    vehicle_center_forward = np.maximum(
        0.0,
        filtered[:, 0] - vehicle_front_m,
    )
    path_center_y = detour_lateral_offset(
        vehicle_center_forward,
        side,
        lateral_offset_m,
        shift_forward_m,
    )
    occupied = (
        np.abs(filtered[:, 1] - path_center_y)
        <= corridor_half_width_m
    )
    count = int(np.count_nonzero(occupied))
    nearest = (
        float(np.min(filtered[occupied, 0])) if count else math.inf
    )
    return ForwardObstacleObservation(count, count, nearest)


def observe_inner_side_passage(
    xyz_points,
    avoidance_side: int,
    vehicle_rear_m: float,
    rear_clearance_m: float,
    minimum_lateral_m: float,
    maximum_lateral_m: float,
    rear_detection_m: float,
    forward_detection_m: float,
    minimum_z_m: float,
    maximum_z_m: float,
) -> PassageObservation:
    """Count obstacle points not yet past and safely behind the vehicle.

    Once the vehicle shifts left, the avoided obstacle is expected on its
    right, and vice versa.  The signed inside lateral coordinate makes the
    same filter work for both manoeuvre directions.  A point is considered
    safely behind only after it has moved behind the rear bumper plus the
    requested clearance.
    """
    points = np.asarray(xyz_points)
    if points.size == 0:
        points = np.empty((0, 3), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError('xyz_points must have shape (N, >=3)')
    values = (
        vehicle_rear_m,
        rear_clearance_m,
        minimum_lateral_m,
        maximum_lateral_m,
        rear_detection_m,
        forward_detection_m,
        minimum_z_m,
        maximum_z_m,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError('passage observation limits must be finite')
    if avoidance_side not in (LEFT, RIGHT):
        raise ValueError('avoidance_side must be LEFT or RIGHT')
    if min(vehicle_rear_m, rear_detection_m, forward_detection_m) <= 0.0:
        raise ValueError('passage longitudinal dimensions must be positive')
    if rear_clearance_m < 0.0:
        raise ValueError('rear clearance must be non-negative')
    if minimum_lateral_m < 0.0 or maximum_lateral_m <= minimum_lateral_m:
        raise ValueError('passage lateral limits are invalid')
    if minimum_z_m >= maximum_z_m:
        raise ValueError('minimum_z_m must be below maximum_z_m')
    if points.shape[0] == 0:
        return PassageObservation(0, 0)

    xyz = points[:, :3].astype(np.float64, copy=False)
    finite = np.isfinite(xyz).all(axis=1)
    inside_lateral = -avoidance_side * xyz[:, 1]
    relevant = (
        finite
        & (xyz[:, 0] >= -rear_detection_m)
        & (xyz[:, 0] <= forward_detection_m)
        & (inside_lateral >= minimum_lateral_m)
        & (inside_lateral <= maximum_lateral_m)
        & (xyz[:, 2] >= minimum_z_m)
        & (xyz[:, 2] <= maximum_z_m)
    )
    if not np.any(relevant):
        return PassageObservation(0, 0)

    rear_boundary_m = -(vehicle_rear_m + rear_clearance_m)
    relevant_x = xyz[relevant, 0]
    not_passed = int(np.count_nonzero(relevant_x >= rear_boundary_m))
    behind = int(np.count_nonzero(relevant_x < rear_boundary_m))
    return PassageObservation(not_passed, behind)


def command_to_point(
    current: Pose2D,
    target: Point2D,
    speed_mps: float,
    heading_kp: float,
    maximum_yaw_rate_rps: float,
    minimum_heading_speed_ratio: float,
) -> LocalCommand:
    """Generate a forward-only command toward one odometry-frame point."""
    values = (
        current.x_m,
        current.y_m,
        current.yaw_rad,
        target.x_m,
        target.y_m,
        speed_mps,
        heading_kp,
        maximum_yaw_rate_rps,
        minimum_heading_speed_ratio,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError('local command inputs must be finite')
    if speed_mps < 0.0 or heading_kp < 0.0 or maximum_yaw_rate_rps < 0.0:
        raise ValueError('speed, gain, and yaw limit must be non-negative')
    if not 0.0 <= minimum_heading_speed_ratio <= 1.0:
        raise ValueError('minimum heading speed ratio must be in [0, 1]')

    delta_x = target.x_m - current.x_m
    delta_y = target.y_m - current.y_m
    distance = math.hypot(delta_x, delta_y)
    desired_yaw = math.atan2(delta_y, delta_x)
    heading_error = normalize_angle_rad(desired_yaw - current.yaw_rad)
    heading_ratio = max(
        minimum_heading_speed_ratio,
        max(0.0, math.cos(abs(heading_error))),
    )
    yaw_rate = max(
        -maximum_yaw_rate_rps,
        min(maximum_yaw_rate_rps, heading_kp * heading_error),
    )
    return LocalCommand(
        speed_mps * heading_ratio,
        yaw_rate,
        heading_error,
        distance,
    )


def pure_pursuit_command(
    current: Pose2D,
    target: Point2D,
    speed_mps: float,
    maximum_yaw_rate_rps: float,
    minimum_heading_speed_ratio: float,
    maximum_curvature_per_m: float = math.inf,
) -> LocalCommand:
    """Track a forward look-ahead point using pure-pursuit curvature."""
    values = (
        current.x_m,
        current.y_m,
        current.yaw_rad,
        target.x_m,
        target.y_m,
        speed_mps,
        maximum_yaw_rate_rps,
        minimum_heading_speed_ratio,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError('pure-pursuit inputs must be finite')
    if speed_mps < 0.0 or maximum_yaw_rate_rps < 0.0:
        raise ValueError('speed and yaw limit must be non-negative')
    if not 0.0 <= minimum_heading_speed_ratio <= 1.0:
        raise ValueError('minimum heading speed ratio must be in [0, 1]')
    if maximum_curvature_per_m <= 0.0:
        raise ValueError('maximum curvature must be positive')

    local_target = world_to_local(current, target)
    distance_squared = (
        local_target.x_m ** 2 + local_target.y_m ** 2
    )
    distance = math.sqrt(distance_squared)
    heading_error = math.atan2(local_target.y_m, local_target.x_m)
    heading_ratio = max(
        minimum_heading_speed_ratio,
        max(0.0, math.cos(abs(heading_error))),
    )
    output_speed = speed_mps * heading_ratio
    if distance_squared <= 1e-9:
        yaw_rate = 0.0
    else:
        curvature = 2.0 * local_target.y_m / distance_squared
        curvature = max(
            -maximum_curvature_per_m,
            min(maximum_curvature_per_m, curvature),
        )
        yaw_rate = output_speed * curvature
    yaw_rate = max(
        -maximum_yaw_rate_rps,
        min(maximum_yaw_rate_rps, yaw_rate),
    )
    return LocalCommand(
        output_speed,
        yaw_rate,
        heading_error,
        distance,
    )
