import json
import math

import pytest

from terrain_navigation_pkg.waypoint_route_core import (
    parse_waypoint_route_json,
)
from terrain_navigation_pkg.waypoint_route_core import route_tangent_headings
from terrain_navigation_pkg.waypoint_route_core import WaypointSequence


def _route(loop=False, start=True, rolling_horizon=False):
    return json.dumps({
        'waypoints': [
            {'latitude': 37.0, 'longitude': 127.0, 'altitude': 10.0},
            {'latitude': 37.0001, 'longitude': 127.0001, 'altitude': 11.0},
        ],
        'loop': loop,
        'start': start,
        'rolling_horizon': rolling_horizon,
    })


def test_parse_waypoint_route_json_preserves_coordinates_and_flags():
    route = parse_waypoint_route_json(
        _route(loop=True, start=True, rolling_horizon=True)
    )

    assert len(route.waypoints) == 2
    assert route.waypoints[1].latitude_deg == pytest.approx(37.0001)
    assert route.waypoints[1].longitude_deg == pytest.approx(127.0001)
    assert route.waypoints[1].altitude_m == pytest.approx(11.0)
    assert route.loop
    assert route.start
    assert route.rolling_horizon


def test_route_defaults_to_complete_preview_for_f9_compatibility():
    route = parse_waypoint_route_json(_route())

    assert not route.rolling_horizon


@pytest.mark.parametrize(
    'payload',
    [
        'not json',
        '{}',
        '{"waypoints": []}',
        '{"waypoints": [{"latitude": 91, "longitude": 0}]}',
    ],
)
def test_parse_waypoint_route_json_rejects_invalid_routes(payload):
    with pytest.raises(ValueError):
        parse_waypoint_route_json(payload)


def test_sequence_requires_departure_before_accepting_arrival():
    sequence = WaypointSequence()
    sequence.load(parse_waypoint_route_json(_route()))

    assert sequence.start() == 0
    assert sequence.observe_goal_reached(True) == ('none', None)
    assert sequence.observe_goal_reached(False) == ('none', None)
    assert sequence.observe_goal_reached(True) == ('publish', 1)
    assert sequence.observe_goal_reached(False) == ('none', None)
    assert sequence.observe_goal_reached(True) == ('complete', 1)
    assert sequence.completed
    assert not sequence.active


def test_looping_sequence_returns_to_first_waypoint():
    sequence = WaypointSequence()
    sequence.load(parse_waypoint_route_json(_route(loop=True)))
    sequence.start()

    sequence.observe_goal_reached(False)
    assert sequence.observe_goal_reached(True) == ('publish', 1)
    sequence.observe_goal_reached(False)
    assert sequence.observe_goal_reached(True) == ('publish', 0)
    assert sequence.active
    assert not sequence.completed


def test_nav2_remaining_feedback_advances_intermediate_ui_index():
    sequence = WaypointSequence()
    sequence.load(parse_waypoint_route_json(_route()))
    sequence.start()

    assert sequence.synchronize_remaining_poses(2) == ('none', None)
    assert sequence.synchronize_remaining_poses(1) == ('publish', 1)
    assert sequence.index == 1
    assert sequence.synchronize_remaining_poses(2) == ('none', None)


def test_nav2_remaining_feedback_does_not_complete_final_route():
    sequence = WaypointSequence()
    sequence.load(parse_waypoint_route_json(_route()))
    sequence.start()

    assert sequence.synchronize_remaining_poses(0) == ('publish', 1)
    assert sequence.active
    assert not sequence.completed


def test_final_completion_can_wait_for_external_confirmation():
    sequence = WaypointSequence()
    sequence.load(parse_waypoint_route_json(_route()))
    sequence.start()

    sequence.observe_goal_reached(False)
    assert sequence.observe_goal_reached(True) == ('publish', 1)
    sequence.observe_goal_reached(False)
    assert sequence.observe_goal_reached(
        True,
        defer_final_completion=True,
    ) == ('await_confirmation', 1)
    assert sequence.active
    assert not sequence.completed

    assert sequence.confirm_final_completion() == ('complete', 1)
    assert not sequence.active
    assert sequence.completed


def test_open_route_tangents_point_through_corner_and_into_final_point():
    headings = route_tangent_headings([
        (0.0, 0.0),
        (10.0, 0.0),
        (10.0, 10.0),
    ])

    assert headings[0] == pytest.approx(0.0)
    assert headings[1] == pytest.approx(math.radians(45.0))
    assert headings[2] == pytest.approx(math.radians(90.0))


def test_closed_route_repeated_start_has_continuous_matching_tangent():
    headings = route_tangent_headings([
        (0.0, 0.0),
        (10.0, 0.0),
        (10.0, 10.0),
        (0.0, 10.0),
        (0.0, 0.0),
    ])

    assert headings[0] == pytest.approx(math.radians(-45.0))
    assert headings[-1] == pytest.approx(headings[0])
    assert headings[1] == pytest.approx(math.radians(45.0))
