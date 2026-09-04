import csv
import json

from terrain_navigation_pkg.evaluate_navigation_runs import (
    _declared_evaluation_variant,
)
from terrain_navigation_pkg.navigation_evaluation_core import (
    evaluate_session,
    mission_polyline_length,
    summarize_runs,
)


def test_recorder_variant_can_be_checked_before_registration(tmp_path):
    session = tmp_path / 'session_20260831_120000_000000'
    session.mkdir()
    (session / 'metadata.json').write_text(json.dumps({
        'evaluation_variant': 'b0_fixed_ground',
    }))

    assert _declared_evaluation_variant(session) == 'b0_fixed_ground'


def test_mission_polyline_length_uses_gnss_waypoints():
    route = {'waypoints': [
        {'latitude': 37.0, 'longitude': 127.0, 'altitude': 10.0},
        {'latitude': 37.0, 'longitude': 127.00001, 'altitude': 10.0},
    ]}
    assert 0.8 < mission_polyline_length(json.dumps(route)) < 1.0


def test_evaluate_session_computes_success_stop_and_collision(tmp_path):
    session = tmp_path / 'session_20260828_120000_000000'
    session.mkdir()
    metadata = {
        'session': session.name,
        'started_at': '2026-08-28T12:00:00+09:00',
        'ended_at': '2026-08-28T12:00:03+09:00',
        'result': 'completed',
        'route_index': 2,
        'route_size': 2,
        'route_json': json.dumps({'waypoints': [
            {'latitude': 37.0, 'longitude': 127.0, 'altitude': 0.0},
            {'latitude': 37.0, 'longitude': 127.00001, 'altitude': 0.0},
        ]}),
        'collision_event_count': 1,
    }
    (session / 'metadata.json').write_text(json.dumps(metadata))
    fields = [
        'ros_time_s', 'vehicle_x', 'vehicle_y', 'speed_mps',
        'safety_state', 'path_hard_valid', 'nav2_plan_points',
        'path_clearance_status', 'route_index', 'route_size',
    ]
    rows = [
        ['0', '0', '0', '0', 'clear', '1', '10',
         '{"minimum_clearance_m": 1.2}', '0', '2'],
        ['1', '0', '0', '0', 'obstacle_stop', '0', '10',
         '{"minimum_clearance_m": 0.8}', '1', '2'],
        ['2', '1', '0', '1', 'clear', '1', '10',
         '{"minimum_clearance_m": 1.1}', '2', '2'],
    ]
    with (session / 'frames.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)
    result = evaluate_session(session, 'proposed', 'F9')
    assert result['success'] == 1
    assert result['collision'] == 1
    assert result['path_length_m'] == 1.0
    assert result['total_stop_s'] == 1.0
    assert result['obstacle_stop_s'] == 1.0
    assert result['invalid_path_s'] == 1.0
    assert result['minimum_clearance_m'] == 0.8


def test_summary_counts_all_attempts_but_times_successes_only():
    base = {
        'variant': 'proposed', 'scenario': 'F10', 'collision': 0,
        'path_length_m': 10, 'path_length_ratio': 1.0,
        'total_stop_s': 2, 'max_stop_s': 1, 'obstacle_stop_s': 0,
        'invalid_path_s': 0, 'minimum_clearance_m': 1.0,
    }
    success = dict(base, success=1, duration_s=20)
    failure = dict(base, success=0, duration_s=100)
    result = summarize_runs([success, failure])[0]
    assert result['success_rate'] == 0.5
    assert result['duration_s_mean'] == 20.0


def test_incomplete_current_four_of_four_is_three_quarters_complete(tmp_path):
    session = tmp_path / 'session_20260828_130000_000000'
    session.mkdir()
    metadata = {
        'session': session.name,
        'started_at': '2026-08-28T13:00:00+09:00',
        'ended_at': '2026-08-28T13:01:00+09:00',
        'result': 'incomplete',
        'route_index': 4,
        'route_size': 4,
    }
    (session / 'metadata.json').write_text(json.dumps(metadata))

    result = evaluate_session(session, 'proposed', 'F9')

    assert result['success'] == 0
    assert result['completion_ratio'] == 0.75
