"""Load human-authored WGS84 mission routes from YAML."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import List

import yaml

from .navigation_core import GeodeticPoint, validate_geodetic
from .waypoint_route_core import WaypointRoute


@dataclass(frozen=True)
class MissionWaypoint:
    """A named WGS84 waypoint used only for mission-file diagnostics."""

    name: str
    point: GeodeticPoint


@dataclass(frozen=True)
class MissionRoute:
    """Validated mission metadata and the standard runtime route."""

    name: str
    coordinate_frame: str
    named_waypoints: List[MissionWaypoint]
    route: WaypointRoute
    temporary_waypoints_enabled: bool = False
    temporary_waypoint_spacing_m: float = 25.0


def _require_bool(data, key, default=False):
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ValueError('{} must be true or false'.format(key))
    return value


def parse_mission_route_yaml(payload: str) -> MissionRoute:
    """Parse and validate the mission-route YAML contract."""
    try:
        data = yaml.safe_load(payload)
    except yaml.YAMLError as error:
        raise ValueError('mission route is not valid YAML') from error
    if not isinstance(data, dict):
        raise ValueError('mission route root must be an object')

    route_name = data.get('route_name', 'unnamed_route')
    if not isinstance(route_name, str) or not route_name.strip():
        raise ValueError('route_name must be a non-empty string')
    route_name = route_name.strip()

    coordinate_frame = data.get('coordinate_frame', 'wgs84')
    if not isinstance(coordinate_frame, str):
        raise ValueError('coordinate_frame must be a string')
    coordinate_frame = coordinate_frame.strip().lower()
    if coordinate_frame != 'wgs84':
        raise ValueError('coordinate_frame must be wgs84')

    raw_waypoints = data.get('waypoints')
    if not isinstance(raw_waypoints, list) or len(raw_waypoints) < 2:
        raise ValueError('mission route must contain at least two waypoints')

    named_waypoints = []
    for index, item in enumerate(raw_waypoints):
        if not isinstance(item, dict):
            raise ValueError('waypoint {} must be an object'.format(index + 1))
        waypoint_name = item.get('name', 'wp{}'.format(index + 1))
        if not isinstance(waypoint_name, str) or not waypoint_name.strip():
            raise ValueError(
                'waypoint {} name must be a non-empty string'.format(index + 1)
            )
        try:
            point = GeodeticPoint(
                float(item['latitude']),
                float(item['longitude']),
                float(item.get('altitude', 0.0)),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                'waypoint {} has invalid coordinates'.format(index + 1)
            ) from error
        validate_geodetic(point)
        named_waypoints.append(
            MissionWaypoint(waypoint_name.strip(), point)
        )

    temporary = data.get('temporary_waypoints', {})
    if temporary is None:
        temporary = {}
    if not isinstance(temporary, dict):
        raise ValueError('temporary_waypoints must be an object')
    temporary_enabled = _require_bool(temporary, 'enabled', False)
    try:
        temporary_spacing_m = float(temporary.get('spacing_m', 25.0))
    except (TypeError, ValueError) as error:
        raise ValueError(
            'temporary_waypoints.spacing_m must be numeric'
        ) from error
    if temporary_spacing_m <= 0.0:
        raise ValueError('temporary_waypoints.spacing_m must be positive')

    route = WaypointRoute(
        waypoints=[item.point for item in named_waypoints],
        loop=_require_bool(data, 'closed_loop', False),
        start=_require_bool(data, 'auto_start', False),
        rolling_horizon=_require_bool(data, 'rolling_horizon', True),
    )
    return MissionRoute(
        name=route_name,
        coordinate_frame=coordinate_frame,
        named_waypoints=named_waypoints,
        route=route,
        temporary_waypoints_enabled=temporary_enabled,
        temporary_waypoint_spacing_m=temporary_spacing_m,
    )


def load_mission_route_file(path: str) -> MissionRoute:
    """Read a UTF-8 YAML file and return a validated mission route."""
    route_path = Path(path).expanduser()
    if not route_path.is_file():
        raise ValueError('mission route file does not exist: {}'.format(path))
    try:
        payload = route_path.read_text(encoding='utf-8')
    except OSError as error:
        raise ValueError(
            'could not read mission route file: {}'.format(path)
        ) from error
    return parse_mission_route_yaml(payload)


def mission_route_to_json(
    mission: MissionRoute,
    force_start: bool = False,
) -> str:
    """Convert YAML mission data to the existing runtime JSON contract."""
    route = mission.route
    payload = {
        'waypoints': [
            {
                'latitude': point.latitude_deg,
                'longitude': point.longitude_deg,
                'altitude': point.altitude_m,
            }
            for point in route.waypoints
        ],
        'loop': bool(route.loop),
        'start': bool(route.start or force_start),
        'rolling_horizon': bool(route.rolling_horizon),
    }
    return json.dumps(payload, separators=(',', ':'))
