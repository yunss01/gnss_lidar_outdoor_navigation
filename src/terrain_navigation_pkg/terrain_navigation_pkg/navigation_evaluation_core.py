"""Offline metrics for F9/F10 navigation recorder sessions."""

from collections import defaultdict
import csv
from datetime import datetime
import json
import math
from pathlib import Path


RUN_FIELDS = [
    'session', 'variant', 'scenario', 'result', 'success', 'collision',
    'collision_event_count', 'duration_s', 'path_length_m',
    'mission_polyline_m', 'path_length_ratio', 'mean_speed_mps',
    'max_speed_mps', 'total_stop_s', 'max_stop_s', 'stop_event_count',
    'obstacle_stop_s', 'obstacle_stop_event_count', 'invalid_path_s',
    'invalid_path_event_count', 'minimum_clearance_m', 'route_index',
    'route_size', 'completion_ratio', 'samples', 'dropped_samples',
    'started_at', 'ended_at', 'session_path',
]


def _float(value, default=math.nan):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _bool(value):
    return str(value).strip().lower() in {'1', 'true', 'yes'}


def _seconds_between(start, end):
    try:
        return max(0.0, (datetime.fromisoformat(end) -
                         datetime.fromisoformat(start)).total_seconds())
    except (TypeError, ValueError):
        return math.nan


def _haversine_m(first, second):
    radius = 6371008.8
    lat1 = math.radians(float(first['latitude']))
    lat2 = math.radians(float(second['latitude']))
    dlat = lat2 - lat1
    dlon = math.radians(
        float(second['longitude']) - float(first['longitude'])
    )
    value = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    horizontal = 2.0 * radius * math.asin(min(1.0, math.sqrt(value)))
    dz = float(second.get('altitude', 0.0)) - float(
        first.get('altitude', 0.0)
    )
    return math.hypot(horizontal, dz)


def mission_polyline_length(route_json):
    try:
        route = json.loads(route_json) if isinstance(route_json, str) else route_json
        points = route.get('waypoints', [])
    except (AttributeError, TypeError, json.JSONDecodeError):
        return math.nan
    if len(points) < 2:
        return 0.0
    return sum(_haversine_m(a, b) for a, b in zip(points, points[1:]))


def _interval_metrics(rows, stop_speed_mps=0.05, maximum_gap_s=1.0):
    result = defaultdict(float)
    result.update({
        'path_length_m': 0.0,
        'total_stop_s': 0.0,
        'max_stop_s': 0.0,
        'stop_event_count': 0,
        'obstacle_stop_s': 0.0,
        'obstacle_stop_event_count': 0,
        'invalid_path_s': 0.0,
        'invalid_path_event_count': 0,
        'minimum_clearance_m': math.nan,
    })
    if not rows:
        return result

    stop_run = obstacle_run = invalid_run = 0.0
    stop_active = obstacle_active = invalid_active = False
    clearances = []
    speeds = []
    previous = rows[0]
    for current in rows:
        speed = abs(_float(current.get('speed_mps'), 0.0))
        speeds.append(speed)
        status_text = current.get('path_clearance_status', '')
        try:
            clearance = json.loads(status_text) if status_text else {}
        except json.JSONDecodeError:
            clearance = {}
        value = _float(clearance.get('minimum_clearance_m'))
        if math.isfinite(value) and value >= 0.0:
            clearances.append(value)

        if current is rows[0]:
            continue
        dt = _float(current.get('ros_time_s')) - _float(
            previous.get('ros_time_s')
        )
        if not math.isfinite(dt) or dt <= 0.0 or dt > maximum_gap_s:
            dt = 0.0
        dx = _float(current.get('vehicle_x')) - _float(
            previous.get('vehicle_x')
        )
        dy = _float(current.get('vehicle_y')) - _float(
            previous.get('vehicle_y')
        )
        step = math.hypot(dx, dy)
        if math.isfinite(step) and step <= 5.0:
            result['path_length_m'] += step

        is_stop = speed <= stop_speed_mps
        is_obstacle = current.get('safety_state', '') == 'obstacle_stop'
        has_plan = _int(current.get('nav2_plan_points')) > 0
        if 'hard_valid' in clearance:
            hard_valid = bool(clearance['hard_valid'])
        else:
            hard_valid = _bool(current.get('path_hard_valid'))
        is_invalid = has_plan and not hard_valid

        if is_stop:
            if not stop_active:
                result['stop_event_count'] += 1
            stop_active = True
            stop_run += dt
            result['total_stop_s'] += dt
            result['max_stop_s'] = max(result['max_stop_s'], stop_run)
        else:
            stop_active = False
            stop_run = 0.0
        if is_obstacle:
            if not obstacle_active:
                result['obstacle_stop_event_count'] += 1
            obstacle_active = True
            obstacle_run += dt
            result['obstacle_stop_s'] += dt
        else:
            obstacle_active = False
            obstacle_run = 0.0
        if is_invalid:
            if not invalid_active:
                result['invalid_path_event_count'] += 1
            invalid_active = True
            invalid_run += dt
            result['invalid_path_s'] += dt
        else:
            invalid_active = False
            invalid_run = 0.0
        previous = current

    result['mean_speed_mps'] = sum(speeds) / len(speeds)
    result['max_speed_mps'] = max(speeds)
    if clearances:
        result['minimum_clearance_m'] = min(clearances)
    return result


def evaluate_session(session_path, variant, scenario):
    """Return one flat, CSV-ready dictionary for a recorder session."""
    session_path = Path(session_path)
    metadata_path = session_path / 'metadata.json'
    frames_path = session_path / 'frames.csv'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    rows = []
    if frames_path.exists():
        with frames_path.open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))

    metrics = _interval_metrics(rows)
    route_size = _int(metadata.get('route_size'))
    route_index = _int(metadata.get('route_index'))
    if rows:
        route_size = max(route_size, _int(rows[-1].get('route_size')))
        route_index = max(route_index, _int(rows[-1].get('route_index')))
    result_text = str(metadata.get('result', 'incomplete'))
    success = result_text == 'completed' and (
        route_size <= 0 or route_index >= route_size
    )
    # route_index denotes the target currently being pursued, not the number
    # already completed.  Therefore an interrupted "4/4" run has completed
    # only three waypoints; only an explicit completed result earns 4/4.
    completed_waypoints = route_index if success else max(0, route_index - 1)
    if route_size > 0:
        completed_waypoints = min(route_size, completed_waypoints)
    states = {row.get('safety_state', '') for row in rows}
    collision_count = _int(metadata.get('collision_event_count'))
    collision = collision_count > 0 or 'collision_stop' in states
    if collision and collision_count == 0:
        collision_count = 1

    duration = _seconds_between(
        metadata.get('started_at'), metadata.get('ended_at')
    )
    if not math.isfinite(duration) and len(rows) >= 2:
        duration = max(
            0.0,
            _float(rows[-1].get('ros_time_s'))
            - _float(rows[0].get('ros_time_s')),
        )
    mission_length = mission_polyline_length(metadata.get('route_json', ''))
    ratio = math.nan
    if math.isfinite(mission_length) and mission_length > 0.0:
        ratio = metrics['path_length_m'] / mission_length

    output = {
        'session': metadata.get('session', session_path.name),
        'variant': variant,
        'scenario': scenario.upper(),
        'result': result_text,
        'success': int(success),
        'collision': int(collision),
        'collision_event_count': collision_count,
        'duration_s': duration,
        'mission_polyline_m': mission_length,
        'path_length_ratio': ratio,
        'route_index': route_index,
        'route_size': route_size,
        'completion_ratio': (
            completed_waypoints / route_size if route_size else math.nan
        ),
        'samples': len(rows),
        'dropped_samples': _int(metadata.get('dropped_samples')),
        'started_at': metadata.get('started_at', ''),
        'ended_at': metadata.get('ended_at', ''),
        'session_path': str(session_path.resolve()),
    }
    output.update(metrics)
    return {key: output.get(key, '') for key in RUN_FIELDS}


def summarize_runs(runs):
    """Group run dictionaries into paper-table summary rows."""
    groups = defaultdict(list)
    for run in runs:
        groups[(run['variant'], run['scenario'])].append(run)
    summaries = []
    numeric = [
        'duration_s', 'path_length_m', 'path_length_ratio',
        'total_stop_s', 'max_stop_s', 'obstacle_stop_s',
        'invalid_path_s', 'minimum_clearance_m',
    ]
    for (variant, scenario), items in sorted(groups.items()):
        successes = [item for item in items if _bool(item['success'])]
        row = {
            'variant': variant,
            'scenario': scenario,
            'runs': len(items),
            'successes': len(successes),
            'success_rate': len(successes) / len(items),
            'collision_runs': sum(_bool(item['collision']) for item in items),
            'collision_rate': sum(_bool(item['collision']) for item in items) / len(items),
        }
        for field in numeric:
            # Time/path efficiency are meaningful primarily for completed runs.
            source = successes if successes else items
            values = [_float(item[field]) for item in source]
            values = [value for value in values if math.isfinite(value)]
            row[field + '_mean'] = sum(values) / len(values) if values else math.nan
            if len(values) > 1:
                mean = row[field + '_mean']
                row[field + '_std'] = math.sqrt(
                    sum((value - mean) ** 2 for value in values) /
                    (len(values) - 1)
                )
            else:
                row[field + '_std'] = 0.0 if values else math.nan
        summaries.append(row)
    return summaries
