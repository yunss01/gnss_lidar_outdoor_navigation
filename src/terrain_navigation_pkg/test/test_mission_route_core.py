import json

import pytest

from terrain_navigation_pkg.mission_route_core import mission_route_to_json
from terrain_navigation_pkg.mission_route_core import parse_mission_route_yaml
from terrain_navigation_pkg.waypoint_route_core import parse_waypoint_route_json


VALID_ROUTE = """
route_name: lab_to_gate
coordinate_frame: wgs84
closed_loop: false
auto_start: false
rolling_horizon: true
waypoints:
  - name: start
    latitude: 37.1
    longitude: 127.1
    altitude: 12.0
  - name: goal
    latitude: 37.2
    longitude: 127.2
    altitude: 13.0
temporary_waypoints:
  enabled: false
  spacing_m: 25.0
"""


def test_parse_and_convert_preserves_standard_route_contract():
    mission = parse_mission_route_yaml(VALID_ROUTE)
    assert mission.name == 'lab_to_gate'
    assert [item.name for item in mission.named_waypoints] == [
        'start', 'goal'
    ]
    route = parse_waypoint_route_json(mission_route_to_json(mission))
    assert len(route.waypoints) == 2
    assert route.waypoints[0].latitude_deg == pytest.approx(37.1)
    assert not route.start
    assert not route.loop
    assert route.rolling_horizon


def test_force_start_changes_message_without_mutating_yaml_route():
    mission = parse_mission_route_yaml(VALID_ROUTE)
    payload = json.loads(mission_route_to_json(mission, force_start=True))
    assert payload['start'] is True
    assert mission.route.start is False


@pytest.mark.parametrize(
    'replacement',
    [
        ('coordinate_frame: wgs84', 'coordinate_frame: map'),
        ('latitude: 37.1', 'latitude: 137.1'),
        ('spacing_m: 25.0', 'spacing_m: 0.0'),
        ('waypoints:', 'waypoints: []\nunused:'),
    ],
)
def test_invalid_missions_are_rejected(replacement):
    old, new = replacement
    with pytest.raises(ValueError):
        parse_mission_route_yaml(VALID_ROUTE.replace(old, new, 1))
