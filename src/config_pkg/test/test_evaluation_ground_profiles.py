"""Checks for reproducible ICCE ground-processing ablations."""

from pathlib import Path

import yaml


CONFIGURATION_KEYS = {
    'ground_relative_low_obstacles_enabled',
    'local_ground_enabled',
    'local_ground_plane_enabled',
}
NODES = {
    'lidar_obstacle_filter_node',
    'lidar_emergency_stop_node',
}
EXPECTED = {
    'evaluation_proposed.yaml': (True, True, True),
    'evaluation_b0_fixed_ground.yaml': (False, False, False),
    'evaluation_b1_plane_ground.yaml': (True, False, False),
}


def _read_yaml(path):
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def test_ground_profiles_change_only_the_intended_parameters():
    config_dir = Path(__file__).resolve().parents[1] / 'config'
    for filename, values in EXPECTED.items():
        profile = _read_yaml(config_dir / filename)
        assert set(profile) == NODES
        expected = dict(zip(sorted(CONFIGURATION_KEYS), values))
        for node in NODES:
            parameters = profile[node]['ros__parameters']
            assert set(parameters) == CONFIGURATION_KEYS
            assert {
                key: parameters[key] for key in sorted(parameters)
            } == expected


def test_proposed_profile_matches_the_frozen_common_configuration():
    config_dir = Path(__file__).resolve().parents[1] / 'config'
    common = _read_yaml(config_dir / 'params.yaml')
    proposed = _read_yaml(config_dir / 'evaluation_proposed.yaml')
    for node in NODES:
        common_parameters = common[node]['ros__parameters']
        profile_parameters = proposed[node]['ros__parameters']
        for key, value in profile_parameters.items():
            assert common_parameters[key] is value


def test_nav2_launch_applies_one_profile_to_both_ground_consumers():
    launch_path = (
        Path(__file__).resolve().parents[2]
        / 'launch_pkg'
        / 'launch'
        / 'terrain_navigation_nav2.launch.py'
    )
    source = launch_path.read_text(encoding='utf-8')
    assert "'evaluation_variant'" in source
    assert "'evaluation_profile_file'" in source
    assert source.count(
        'parameters=[common_parameters, evaluation_parameters]'
    ) == 1
    emergency_parameters = source.split(
        "executable='lidar_emergency_stop_node'", 1
    )[1].split('condition=IfCondition', 1)[0]
    assert 'evaluation_parameters' in emergency_parameters
    assert "'evaluation_variant': LaunchConfiguration(" in source
