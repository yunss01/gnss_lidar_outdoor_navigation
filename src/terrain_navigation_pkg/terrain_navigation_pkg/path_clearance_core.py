"""Geometry helpers for validating a swept vehicle footprint along a path."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PathPose2D:
    """Planar vehicle pose in the path frame."""

    x: float
    y: float
    yaw: float
    distance: float = 0.0


@dataclass(frozen=True)
class PathClearanceResult:
    """Minimum obstacle clearance measured from the unpadded vehicle body."""

    minimum_clearance_m: float
    minimum_pose_index: int
    minimum_obstacle_index: int
    hard_valid: bool
    preferred_valid: bool
    checked_pose_count: int
    checked_length_m: float


@dataclass(frozen=True)
class PathTrimResult:
    """Path suffix beginning at the position nearest to the vehicle."""

    poses: tuple
    nearest_distance_m: float
    start_segment_index: int


def trim_path_to_nearest_position(poses, vehicle_x, vehicle_y):
    """Discard the completed path prefix and project onto the nearest segment.

    Nav2 may retain a plan whose first pose is behind the moving vehicle.  A
    clearance validator must inspect the path ahead of the vehicle, not the
    already-traversed prefix.  The projected first pose also avoids a jump of
    up to one planner sample when the vehicle lies between two path poses.
    """
    poses = list(poses)
    vehicle_x = float(vehicle_x)
    vehicle_y = float(vehicle_y)
    if not math.isfinite(vehicle_x) or not math.isfinite(vehicle_y):
        raise ValueError('vehicle position must be finite')
    if not poses:
        return PathTrimResult(tuple(), math.inf, -1)
    if len(poses) == 1:
        pose = poses[0]
        distance = math.hypot(
            float(pose.x) - vehicle_x,
            float(pose.y) - vehicle_y,
        )
        return PathTrimResult((pose,), distance, 0)

    best_distance_sq = math.inf
    best_segment_index = -1
    best_projection = None
    for segment_index, (start, end) in enumerate(
        zip(poses[:-1], poses[1:])
    ):
        start_x = float(start.x)
        start_y = float(start.y)
        delta_x = float(end.x) - start_x
        delta_y = float(end.y) - start_y
        length_sq = delta_x * delta_x + delta_y * delta_y
        if length_sq <= 1.0e-12:
            fraction = 0.0
        else:
            fraction = (
                (vehicle_x - start_x) * delta_x
                + (vehicle_y - start_y) * delta_y
            ) / length_sq
            fraction = min(1.0, max(0.0, fraction))
        projected_x = start_x + fraction * delta_x
        projected_y = start_y + fraction * delta_y
        distance_sq = (
            (projected_x - vehicle_x) ** 2
            + (projected_y - vehicle_y) ** 2
        )
        if distance_sq < best_distance_sq:
            yaw = (
                math.atan2(delta_y, delta_x)
                if length_sq > 1.0e-12
                else float(start.yaw)
            )
            best_distance_sq = distance_sq
            best_segment_index = segment_index
            best_projection = PathPose2D(
                projected_x, projected_y, yaw, 0.0
            )

    if best_projection is None:
        pose = poses[0]
        distance = math.hypot(
            float(pose.x) - vehicle_x,
            float(pose.y) - vehicle_y,
        )
        return PathTrimResult((pose,), distance, 0)

    trimmed = [best_projection]
    for pose in poses[best_segment_index + 1:]:
        if math.hypot(
            float(pose.x) - float(trimmed[-1].x),
            float(pose.y) - float(trimmed[-1].y),
        ) > 1.0e-6:
            trimmed.append(pose)
    return PathTrimResult(
        tuple(trimmed),
        math.sqrt(best_distance_sq),
        best_segment_index,
    )


def sample_path(poses, spacing_m, maximum_length_m):
    """Interpolate a polyline using bounded spacing and tangent heading."""
    if spacing_m <= 0.0:
        raise ValueError('spacing_m must be positive')
    if maximum_length_m <= 0.0:
        raise ValueError('maximum_length_m must be positive')
    poses = list(poses)
    if not poses:
        return []
    if len(poses) == 1:
        pose = poses[0]
        return [
            PathPose2D(
                float(pose.x), float(pose.y), float(pose.yaw), 0.0
            )
        ]

    sampled = []
    cumulative = 0.0
    previous_yaw = float(poses[0].yaw)
    for start, end in zip(poses[:-1], poses[1:]):
        delta_x = float(end.x) - float(start.x)
        delta_y = float(end.y) - float(start.y)
        segment_length = math.hypot(delta_x, delta_y)
        if segment_length <= 1.0e-6:
            continue
        segment_yaw = math.atan2(delta_y, delta_x)
        previous_yaw = segment_yaw
        step_count = max(1, int(math.ceil(segment_length / spacing_m)))
        for step in range(step_count):
            offset = segment_length * step / step_count
            path_distance = cumulative + offset
            if path_distance > maximum_length_m + 1.0e-9:
                return sampled
            fraction = offset / segment_length
            candidate = PathPose2D(
                float(start.x) + fraction * delta_x,
                float(start.y) + fraction * delta_y,
                segment_yaw,
                path_distance,
            )
            sufficiently_far = (
                not sampled
                or path_distance - sampled[-1].distance >= spacing_m * 0.5
            )
            if sufficiently_far:
                sampled.append(candidate)
        cumulative += segment_length
        if cumulative >= maximum_length_m:
            break

    terminal_distance = min(cumulative, maximum_length_m)
    if cumulative <= maximum_length_m + 1.0e-9:
        terminal = poses[-1]
        terminal_pose = PathPose2D(
            float(terminal.x),
            float(terminal.y),
            previous_yaw,
            terminal_distance,
        )
        if not sampled or terminal_distance - sampled[-1].distance > 1.0e-6:
            sampled.append(terminal_pose)
    return sampled


def evaluate_path_clearance(
    sampled_poses,
    obstacle_xy,
    vehicle_front_m,
    vehicle_rear_m,
    vehicle_half_width_m,
    hard_clearance_m,
    preferred_clearance_m,
):
    """Measure point-to-body clearance for every sampled path footprint.

    Clearance is measured from the physical, unpadded rectangular body.  A
    point inside the body has zero clearance.  The hard and preferred margins
    are therefore explicit and independent of Nav2 costmap implementation.
    """
    poses = list(sampled_poses)
    points = np.asarray(obstacle_xy, dtype=np.float64)
    if vehicle_front_m <= 0.0 or vehicle_rear_m <= 0.0:
        raise ValueError('vehicle longitudinal extents must be positive')
    if vehicle_half_width_m <= 0.0:
        raise ValueError('vehicle_half_width_m must be positive')
    if hard_clearance_m < 0.0:
        raise ValueError('hard_clearance_m must be non-negative')
    if preferred_clearance_m < hard_clearance_m:
        raise ValueError('preferred clearance must be >= hard clearance')
    if points.size == 0:
        points = np.empty((0, 2), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError('obstacle_xy must have shape (N, 2)')

    if not poses or points.shape[0] == 0:
        return PathClearanceResult(
            math.inf,
            -1,
            -1,
            True,
            True,
            len(poses),
            float(poses[-1].distance) if poses else 0.0,
        )

    best_clearance = math.inf
    best_pose_index = -1
    best_obstacle_index = -1
    for pose_index, pose in enumerate(poses):
        cos_yaw = math.cos(float(pose.yaw))
        sin_yaw = math.sin(float(pose.yaw))
        delta_x = points[:, 0] - float(pose.x)
        delta_y = points[:, 1] - float(pose.y)
        local_x = cos_yaw * delta_x + sin_yaw * delta_y
        local_y = -sin_yaw * delta_x + cos_yaw * delta_y

        outside_x = np.maximum(
            np.maximum(local_x - vehicle_front_m, -vehicle_rear_m - local_x),
            0.0,
        )
        outside_y = np.maximum(np.abs(local_y) - vehicle_half_width_m, 0.0)
        clearance = np.hypot(outside_x, outside_y)
        obstacle_index = int(np.argmin(clearance))
        candidate = float(clearance[obstacle_index])
        if candidate < best_clearance:
            best_clearance = candidate
            best_pose_index = pose_index
            best_obstacle_index = obstacle_index

    return PathClearanceResult(
        best_clearance,
        best_pose_index,
        best_obstacle_index,
        best_clearance >= hard_clearance_m,
        best_clearance >= preferred_clearance_m,
        len(poses),
        float(poses[-1].distance),
    )
