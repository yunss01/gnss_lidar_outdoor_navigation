"""Validated GNSS waypoint routes and deterministic sequence state."""

from dataclasses import dataclass
import json
import math
from typing import List, Optional, Tuple

from .navigation_core import GeodeticPoint, validate_geodetic


@dataclass(frozen=True)
class WaypointRoute:
    waypoints: List[GeodeticPoint]
    loop: bool = False
    start: bool = False
    rolling_horizon: bool = False


def route_tangent_headings(
    points_xy: List[Tuple[float, float]],
    closed_tolerance_m: float = 0.25,
) -> List[Optional[float]]:
    """Compute the intended forward tangent at each mission waypoint.

    Open-route corners use the bisector of their incoming and outgoing unit
    directions. A route whose final point repeats the first is treated as a
    closed lap, giving both copies of that point one continuous tangent. A
    single point has no route-defined heading.
    """
    if not points_xy:
        return []
    points = [
        (float(x_value), float(y_value))
        for x_value, y_value in points_xy
    ]
    if not all(
        math.isfinite(value)
        for point in points
        for value in point
    ):
        raise ValueError('route waypoint coordinates must be finite')
    if len(points) == 1:
        return [None]

    closed = (
        len(points) >= 4
        and math.hypot(
            points[-1][0] - points[0][0],
            points[-1][1] - points[0][1],
        ) <= float(closed_tolerance_m)
    )
    unique_points = points[:-1] if closed else points

    def direction(first, second):
        dx_value = second[0] - first[0]
        dy_value = second[1] - first[1]
        length = math.hypot(dx_value, dy_value)
        if length <= 1.0e-6:
            return None
        return dx_value / length, dy_value / length

    def combined_heading(incoming, outgoing):
        if incoming is None:
            vector = outgoing
        elif outgoing is None:
            vector = incoming
        else:
            summed_x = incoming[0] + outgoing[0]
            summed_y = incoming[1] + outgoing[1]
            if math.hypot(summed_x, summed_y) <= 1.0e-6:
                # A 180-degree cusp has no unique bisector. Prefer the exit
                # direction so a forward-only vehicle does not turn back.
                vector = outgoing
            else:
                vector = summed_x, summed_y
        if vector is None:
            return None
        return math.atan2(vector[1], vector[0])

    headings = []
    if closed:
        point_count = len(unique_points)
        for index, point in enumerate(unique_points):
            incoming = direction(unique_points[index - 1], point)
            outgoing = direction(
                point, unique_points[(index + 1) % point_count]
            )
            headings.append(combined_heading(incoming, outgoing))
        headings.append(headings[0])
        return headings

    for index, point in enumerate(unique_points):
        incoming = (
            direction(unique_points[index - 1], point)
            if index > 0 else None
        )
        outgoing = (
            direction(point, unique_points[index + 1])
            if index + 1 < len(unique_points) else None
        )
        headings.append(combined_heading(incoming, outgoing))
    return headings


def parse_waypoint_route_json(payload: str) -> WaypointRoute:
    """Parse the standard-string route contract used by UI clients."""
    try:
        data = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError('waypoint route is not valid JSON') from error
    if not isinstance(data, dict):
        raise ValueError('waypoint route root must be an object')
    raw_waypoints = data.get('waypoints')
    if not isinstance(raw_waypoints, list) or not raw_waypoints:
        raise ValueError('waypoint route must contain at least one waypoint')
    waypoints = []
    for index, item in enumerate(raw_waypoints):
        if not isinstance(item, dict):
            raise ValueError('waypoint {} must be an object'.format(index))
        try:
            point = GeodeticPoint(
                float(item['latitude']),
                float(item['longitude']),
                float(item.get('altitude', 0.0)),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                'waypoint {} has invalid coordinates'.format(index)
            ) from error
        validate_geodetic(point)
        waypoints.append(point)
    return WaypointRoute(
        waypoints=waypoints,
        loop=bool(data.get('loop', False)),
        start=bool(data.get('start', False)),
        rolling_horizon=bool(data.get('rolling_horizon', False)),
    )


class WaypointSequence:
    """Advance only on a false-to-true arrival cycle for each waypoint."""

    def __init__(self):
        self.clear()

    def clear(self) -> None:
        self.waypoints: List[GeodeticPoint] = []
        self.loop = False
        self.active = False
        self.completed = False
        self.index: Optional[int] = None
        self.arrival_armed = False

    def load(self, route: WaypointRoute) -> None:
        self.waypoints = list(route.waypoints)
        self.loop = bool(route.loop)
        self.active = False
        self.completed = False
        self.index = None
        self.arrival_armed = False

    def start(self) -> int:
        if not self.waypoints:
            raise ValueError('cannot start an empty waypoint route')
        self.active = True
        self.completed = False
        self.index = 0
        self.arrival_armed = False
        return self.index

    def observe_goal_reached(
        self,
        reached: bool,
        defer_final_completion: bool = False,
    ) -> Tuple[str, Optional[int]]:
        if not self.active or self.index is None:
            return 'none', None
        if not reached:
            self.arrival_armed = True
            return 'none', None
        if not self.arrival_armed:
            return 'none', None
        self.arrival_armed = False
        next_index = self.index + 1
        if next_index < len(self.waypoints):
            self.index = next_index
            return 'publish', self.index
        if self.loop:
            self.index = 0
            return 'publish', self.index
        if defer_final_completion:
            return 'await_confirmation', self.index
        return self.confirm_final_completion()

    def synchronize_remaining_poses(
        self,
        remaining: int,
    ) -> Tuple[str, Optional[int]]:
        """Advance intermediate UI state from Nav2 route feedback."""
        if not self.active or self.index is None or not self.waypoints:
            return 'none', None
        remaining = int(remaining)
        if remaining < 0 or remaining > len(self.waypoints):
            return 'none', None
        desired_index = min(
            len(self.waypoints) - 1,
            len(self.waypoints) - remaining,
        )
        if desired_index <= self.index:
            return 'none', None
        self.index = desired_index
        self.arrival_armed = False
        return 'publish', self.index

    def confirm_final_completion(self) -> Tuple[str, Optional[int]]:
        """Complete a non-looping route after the motion action succeeds."""
        if (
            not self.active
            or self.index is None
            or self.loop
            or self.index != len(self.waypoints) - 1
        ):
            return 'none', None
        self.active = False
        self.completed = True
        return 'complete', self.index
