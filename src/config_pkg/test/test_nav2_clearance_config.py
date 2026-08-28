"""Regression checks for consistent Nav2 collision envelopes."""

from pathlib import Path
import xml.etree.ElementTree as ET

import yaml


def test_global_and_local_hard_footprints_match():
    """Do not let a safe local pose become an invalid planning start pose."""
    config_path = (
        Path(__file__).resolve().parents[1]
        / 'config'
        / 'nav2_outdoor_params.yaml'
    )
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))

    local = config['local_costmap']['local_costmap']['ros__parameters']
    global_ = config['global_costmap']['global_costmap']['ros__parameters']
    planner = config['planner_server']['ros__parameters']['GridBased']

    assert local['footprint'] == global_['footprint']
    assert local['footprint_padding'] == global_['footprint_padding'] == 0.35
    assert global_['inflation_layer']['inflation_radius'] == 2.5
    assert planner['motion_model_for_search'] == 'DUBIN'
    assert planner['minimum_turning_radius'] == 4.1


def test_through_pose_tree_keeps_hybrid_planner_for_nonrolling_routes():
    """Any remaining ThroughPoses use must preserve Ackermann kinematics."""
    config_dir = Path(__file__).resolve().parents[1] / 'config'
    tree = ET.parse(
        config_dir / 'nav2_ackermann_through_poses_replanning.xml'
    )
    planner_nodes = tree.findall('.//ComputePathThroughPoses')
    passed_goal_nodes = tree.findall('.//RemovePassedGoals')

    assert len(planner_nodes) == 1
    assert planner_nodes[0].attrib['planner_id'] == 'GridBased'
    assert len(passed_goal_nodes) == 1
    assert float(passed_goal_nodes[0].attrib['radius']) == 1.35


def test_nav2_launch_selects_single_pose_hybrid_tree_for_rolling_routes():
    """F9/F10 segments must use Hybrid ComputePathToPose, not ThroughPoses."""
    launch_path = (
        Path(__file__).resolve().parents[2]
        / 'launch_pkg'
        / 'launch'
        / 'terrain_navigation_nav2.launch.py'
    )
    source = launch_path.read_text(encoding='utf-8')

    rolling_block = source.split(
        'rolling_behavior_tree = PathJoinSubstitution([', 1
    )[1].split('])', 1)[0]
    assert 'nav2_ackermann_replanning.xml' in rolling_block
    assert 'nav2_ackermann_through_poses_replanning.xml' not in rolling_block


def test_bridge_sends_rolling_segments_as_navigate_to_pose_actions():
    """Passed WPs must leave the action, while Hybrid planning stays active."""
    node_path = (
        Path(__file__).resolve().parents[2]
        / 'terrain_navigation_pkg'
        / 'terrain_navigation_pkg'
        / 'nav2_goal_bridge_node.py'
    )
    source = node_path.read_text(encoding='utf-8')
    rolling_send = source.split('if is_rolling_segment:', 1)[1].split(
        'if has_route:', 1
    )[0]

    assert 'NavigateToPose.Goal()' in rolling_send
    assert 'NavigateThroughPoses.Goal()' not in rolling_send
    assert "goal_mode='rolling_segment'" in rolling_send
    assert 'action_goal.behavior_tree = self.rolling_behavior_tree' in (
        rolling_send
    )
    assert 'self._queue_rolling_segment(next_start)' in source


def test_far_proposed_profile_keeps_valid_active_path_sticky():
    """Paper/proposed mode must not replace a valid action for path length."""
    config_path = Path(__file__).resolve().parents[1] / 'config' / 'params.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    parameters = config['far_nav2_guide_node']['ros__parameters']

    assert parameters['active_replanning_enabled'] is True
    assert parameters['active_replan_allow_valid_path_optimization'] is False
    assert parameters['path_efficiency_max_retries'] == 6
    assert parameters['path_efficiency_retry_min_lookahead_m'] == 5.0
    assert parameters['path_efficiency_blocked_retry_s'] == 5.0


def test_rviz_footprint_labels_match_configured_padding():
    rviz_path = (
        Path(__file__).resolve().parents[2]
        / 'launch_pkg'
        / 'rviz'
        / 'terrain_navigation.rviz'
    )
    source = rviz_path.read_text(encoding='utf-8')

    assert 'LOCAL hard footprint (+0.35m, red)' in source
    assert 'GLOBAL plan footprint (+0.35m, green)' in source
    assert 'GLOBAL plan footprint (+1.0m, green)' not in source
