"""Smooth route generation and forward-only path tracking helpers.

The generated reference path stays within the configured arrival radius of
every supplied waypoint.  A forward-only Pure Pursuit tracker then turns that
reference into curvature commands limited by the physical Ackermann steering
geometry.  Keeping this module independent from ROS makes the geometry
deterministic and unit-testable.
"""

from dataclasses import dataclass
import math
from typing import List, Sequence

from .navigation_core import EnuPoint, normalize_angle_rad


@dataclass(frozen=True)
class PathTrackingGeometry:
    """Geometry needed to turn a path target into a vehicle command."""

    nearest_index: int
    target_index: int
    remaining_distance_m: float
    final_distance_m: float
    nearest_distance_m: float
    target_distance_m: float
    heading_error_rad: float
    curvature_per_m: float


def _distance(first: EnuPoint, second: EnuPoint) -> float:
    return math.hypot(
        second.east_m - first.east_m,
        second.north_m - first.north_m,
    )


def _remove_duplicate_points(
    points: Sequence[EnuPoint],
    minimum_separation_m: float = 0.05,
) -> List[EnuPoint]:
    result: List[EnuPoint] = []
    for point in points:
        values = (point.east_m, point.north_m, point.up_m)
        if not all(math.isfinite(value) for value in values):
            raise ValueError('route points must be finite')
        if not result or _distance(result[-1], point) >= minimum_separation_m:
            result.append(point)
    return result


def generate_smooth_route(
    control_points: Sequence[EnuPoint],
    start_yaw_rad: float,
    spacing_m: float = 0.5,
    tangent_scale: float = 0.5,
    maximum_waypoint_deviation_m: float = 2.0,
) -> List[EnuPoint]:
    """Round a waypoint polyline with tangent quadratic Bezier corners.

    A smooth curve which passes exactly through a sharp corner must briefly
    overshoot outside the polyline.  For outdoor navigation that can leave the
    intended corridor, so this function instead cuts each corner on its inside.
    The nearest part of the curve remains within
    ``maximum_waypoint_deviation_m`` of the waypoint, preserving the goal
    manager's arrival test.  ``start_yaw_rad`` is validated here; departure
    heading mismatch is handled by the forward-only Ackermann tracker.
    """
    if not math.isfinite(spacing_m) or spacing_m <= 0.0:
        raise ValueError('spacing_m must be positive')
    if not math.isfinite(start_yaw_rad):
        raise ValueError('start_yaw_rad must be finite')
    if not 0.0 < tangent_scale <= 1.0:
        raise ValueError('tangent_scale must be in (0, 1]')
    if maximum_waypoint_deviation_m <= 0.0:
        raise ValueError('maximum waypoint deviation must be positive')
    points = _remove_duplicate_points(control_points)
    if len(points) < 2:
        raise ValueError('a route needs at least two distinct points')

    def append_line(result, first, second):
        distance = _distance(first, second)
        count = max(1, int(math.ceil(distance / spacing_m)))
        for sample_index in range(1, count + 1):
            ratio = sample_index / count
            result.append(EnuPoint(
                first.east_m + (second.east_m - first.east_m) * ratio,
                first.north_m + (second.north_m - first.north_m) * ratio,
                first.up_m + (second.up_m - first.up_m) * ratio,
            ))

    sampled: List[EnuPoint] = [points[0]]
    cursor = points[0]
    for index in range(1, len(points) - 1):
        corner = points[index]
        following = points[index + 1]
        incoming_length = _distance(points[index - 1], corner)
        outgoing_length = _distance(corner, following)
        incoming_unit = (
            (corner.east_m - points[index - 1].east_m) / incoming_length,
            (corner.north_m - points[index - 1].north_m) / incoming_length,
        )
        outgoing_unit = (
            (following.east_m - corner.east_m) / outgoing_length,
            (following.north_m - corner.north_m) / outgoing_length,
        )
        dot_product = max(
            -1.0,
            min(
                1.0,
                incoming_unit[0] * outgoing_unit[0]
                + incoming_unit[1] * outgoing_unit[1],
            ),
        )
        turn_angle = math.acos(dot_product)
        if turn_angle < math.radians(3.0):
            append_line(sampled, cursor, corner)
            cursor = corner
            continue

        turn_sine_half = math.sin(0.5 * turn_angle)
        deviation_limited_trim = (
            2.0 * maximum_waypoint_deviation_m
            / max(turn_sine_half, 1e-6)
        )
        trim_distance = min(
            tangent_scale * min(incoming_length, outgoing_length),
            deviation_limited_trim,
            0.49 * incoming_length,
            0.49 * outgoing_length,
        )
        entry = EnuPoint(
            corner.east_m - incoming_unit[0] * trim_distance,
            corner.north_m - incoming_unit[1] * trim_distance,
            corner.up_m,
        )
        exit_point = EnuPoint(
            corner.east_m + outgoing_unit[0] * trim_distance,
            corner.north_m + outgoing_unit[1] * trim_distance,
            corner.up_m,
        )
        append_line(sampled, cursor, entry)
        curve_count = max(
            2,
            int(math.ceil(2.0 * trim_distance / spacing_m)),
        )
        for sample_index in range(1, curve_count + 1):
            parameter = sample_index / curve_count
            inverse = 1.0 - parameter
            sampled.append(EnuPoint(
                inverse * inverse * entry.east_m
                + 2.0 * inverse * parameter * corner.east_m
                + parameter * parameter * exit_point.east_m,
                inverse * inverse * entry.north_m
                + 2.0 * inverse * parameter * corner.north_m
                + parameter * parameter * exit_point.north_m,
                inverse * inverse * entry.up_m
                + 2.0 * inverse * parameter * corner.up_m
                + parameter * parameter * exit_point.up_m,
            ))
        cursor = exit_point
    append_line(sampled, cursor, points[-1])
    return sampled


def cumulative_path_lengths(points: Sequence[EnuPoint]) -> List[float]:
    """Return cumulative planar distance for every path point."""
    if not points:
        raise ValueError('path cannot be empty')
    lengths = [0.0]
    for index in range(1, len(points)):
        lengths.append(lengths[-1] + _distance(points[index - 1], points[index]))
    return lengths


def nearest_path_index(
    points: Sequence[EnuPoint],
    current: EnuPoint,
    previous_index: int = 0,
    search_ahead: int = 400,
    maximum_index: int = None,
) -> int:
    """Find the nearest point without allowing route progress to go backward."""
    if not points:
        raise ValueError('path cannot be empty')
    if search_ahead < 1:
        raise ValueError('search_ahead must be positive')
    previous_index = max(0, min(int(previous_index), len(points) - 1))
    stop_index = min(len(points), previous_index + search_ahead + 1)
    if maximum_index is not None:
        maximum_index = max(
            previous_index,
            min(int(maximum_index), len(points) - 1),
        )
        stop_index = min(stop_index, maximum_index + 1)
    return min(
        range(previous_index, stop_index),
        key=lambda index: (
            points[index].east_m - current.east_m
        ) ** 2 + (
            points[index].north_m - current.north_m
        ) ** 2,
    )


def compute_path_tracking_geometry(
    points: Sequence[EnuPoint],
    cumulative_lengths: Sequence[float],
    current: EnuPoint,
    current_yaw_rad: float,
    previous_index: int,
    lookahead_distance_m: float,
    maximum_curvature_per_m: float,
    search_ahead: int = 400,
    maximum_progress_distance_m: float = None,
) -> PathTrackingGeometry:
    """Compute a curvature-limited Pure Pursuit target on ``points``."""
    if len(points) != len(cumulative_lengths) or not points:
        raise ValueError(
            'path and cumulative lengths must be non-empty and equal'
        )
    if lookahead_distance_m <= 0.0:
        raise ValueError('lookahead distance must be positive')
    if maximum_curvature_per_m <= 0.0:
        raise ValueError('maximum curvature must be positive')
    maximum_index = None
    if maximum_progress_distance_m is not None:
        if not math.isfinite(maximum_progress_distance_m):
            raise ValueError('maximum progress distance must be finite')
        maximum_progress_distance_m = max(0.0, maximum_progress_distance_m)
        maximum_index = 0
        while (
            maximum_index + 1 < len(cumulative_lengths)
            and cumulative_lengths[maximum_index + 1]
            <= maximum_progress_distance_m
        ):
            maximum_index += 1
    nearest_index = nearest_path_index(
        points,
        current,
        previous_index,
        search_ahead,
        maximum_index,
    )
    target_length = cumulative_lengths[nearest_index] + lookahead_distance_m
    target_index = nearest_index
    while (
        target_index + 1 < len(points)
        and cumulative_lengths[target_index] < target_length
    ):
        target_index += 1

    target = points[target_index]
    delta_east = target.east_m - current.east_m
    delta_north = target.north_m - current.north_m
    target_distance = math.hypot(delta_east, delta_north)
    desired_yaw = math.atan2(delta_north, delta_east)
    heading_error = normalize_angle_rad(desired_yaw - current_yaw_rad)
    lateral_left = (
        -math.sin(current_yaw_rad) * delta_east
        + math.cos(current_yaw_rad) * delta_north
    )
    forward = (
        math.cos(current_yaw_rad) * delta_east
        + math.sin(current_yaw_rad) * delta_north
    )
    if target_distance <= 1e-6:
        curvature = 0.0
    elif forward <= 0.0:
        # Pure Pursuit is forward-only.  When the path is behind, command a
        # physical maximum-radius U-turn instead of an in-place rotation.
        turn_sign = 1.0 if heading_error >= 0.0 else -1.0
        curvature = turn_sign * maximum_curvature_per_m
    else:
        curvature = 2.0 * lateral_left / (target_distance * target_distance)
        curvature = max(
            -maximum_curvature_per_m,
            min(maximum_curvature_per_m, curvature),
        )
    return PathTrackingGeometry(
        nearest_index=nearest_index,
        target_index=target_index,
        remaining_distance_m=max(
            0.0,
            cumulative_lengths[-1] - cumulative_lengths[nearest_index],
        ),
        final_distance_m=_distance(current, points[-1]),
        nearest_distance_m=_distance(current, points[nearest_index]),
        target_distance_m=target_distance,
        heading_error_rad=heading_error,
        curvature_per_m=curvature,
    )
