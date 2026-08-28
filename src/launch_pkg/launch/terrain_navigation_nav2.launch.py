"""Preview or drive mapless outdoor navigation with Nav2 and 3D LiDAR."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_share = FindPackageShare('config_pkg')
    common_parameters = PathJoinSubstitution([
        config_share,
        'config',
        'params.yaml',
    ])
    nav2_parameters = PathJoinSubstitution([
        config_share,
        'config',
        'nav2_outdoor_params.yaml',
    ])
    rolling_behavior_tree = PathJoinSubstitution([
        config_share,
        'config',
        'nav2_ackermann_replanning.xml',
    ])
    nav_to_pose_behavior_tree = PathJoinSubstitution([
        config_share,
        'config',
        'nav2_ackermann_replanning.xml',
    ])
    nav_through_poses_behavior_tree = PathJoinSubstitution([
        config_share,
        'config',
        'nav2_ackermann_through_poses_replanning.xml',
    ])
    return LaunchDescription([
        DeclareLaunchArgument(
            'drive_enabled',
            default_value='false',
            description=(
                'Pass Nav2 commands through the independent LiDAR safety gate'
            ),
        ),
        DeclareLaunchArgument(
            'guide_mode',
            default_value='direct',
            description=(
                "'direct' keeps the proven F9/F10 bridge; 'far' uses an "
                'online long-range guide and short Smac Hybrid segments'
            ),
        ),
        DeclareLaunchArgument('goal_enabled', default_value='false'),
        DeclareLaunchArgument('goal_latitude', default_value='0.0'),
        DeclareLaunchArgument('goal_longitude', default_value='0.0'),
        DeclareLaunchArgument('goal_altitude', default_value='0.0'),
        DeclareLaunchArgument(
            'start_navigation_visualization',
            default_value='true',
        ),
        DeclareLaunchArgument(
            'mission_route_enabled',
            default_value='false',
            description='Load a YAML mission route with this Nav2 stack',
        ),
        DeclareLaunchArgument(
            'mission_route_file',
            default_value='',
            description='Absolute path to a WGS84 mission-route YAML file',
        ),
        DeclareLaunchArgument(
            'mission_route_start',
            default_value='false',
            description='Start the YAML route after loading it',
        ),
        DeclareLaunchArgument(
            'record_learning_data',
            default_value='true',
            description=(
                'Record subscriber-only LiDAR/navigation training samples '
                'while F9/F10 is active'
            ),
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='gnss_goal_manager_node',
            name='gnss_goal_manager_node',
            output='screen',
            parameters=[
                common_parameters,
                {
                    'goal_enabled': ParameterValue(
                        LaunchConfiguration('goal_enabled'),
                        value_type=bool,
                    ),
                    'goal_latitude': ParameterValue(
                        LaunchConfiguration('goal_latitude'),
                        value_type=float,
                    ),
                    'goal_longitude': ParameterValue(
                        LaunchConfiguration('goal_longitude'),
                        value_type=float,
                    ),
                    'goal_altitude': ParameterValue(
                        LaunchConfiguration('goal_altitude'),
                        value_type=float,
                    ),
                },
            ],
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='gnss_waypoint_manager_node',
            name='gnss_waypoint_manager_node',
            output='screen',
            parameters=[
                common_parameters,
                {
                    'require_nav2_success_for_final_completion': (
                        ParameterValue(
                            PythonExpression([
                                "'",
                                LaunchConfiguration('guide_mode'),
                                "' != 'far'",
                            ]),
                            value_type=bool,
                        )
                    ),
                },
            ],
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='lidar_obstacle_filter_node',
            name='lidar_obstacle_filter_node',
            output='screen',
            parameters=[common_parameters],
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='nav2_goal_bridge_node',
            name='nav2_goal_bridge_node',
            output='screen',
            parameters=[
                common_parameters,
                {'rolling_behavior_tree': rolling_behavior_tree},
            ],
            condition=IfCondition(PythonExpression([
                "'",
                LaunchConfiguration('guide_mode'),
                "' == 'direct'",
            ])),
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='far_nav2_guide_node',
            name='far_nav2_guide_node',
            output='screen',
            parameters=[
                common_parameters,
                {'behavior_tree': rolling_behavior_tree},
            ],
            condition=IfCondition(PythonExpression([
                "'",
                LaunchConfiguration('guide_mode'),
                "' == 'far'",
            ])),
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='path_clearance_validator_node',
            name='path_clearance_validator_node',
            output='screen',
            parameters=[common_parameters],
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='path_validity_gate_node',
            name='path_validity_gate_node',
            output='screen',
            parameters=[common_parameters],
            condition=IfCondition(LaunchConfiguration('drive_enabled')),
        ),
        # Explicit Nav2 processes are used instead of including the standard
        # launch file. This makes the command boundary unambiguous:
        # controller -> /nav2/cmd_vel_raw -> smoother -> /cmd_vel_nav2.
        # Nothing reaches CARLA's /cmd_vel unless drive_enabled starts the
        # independent emergency-stop node below.
        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[nav2_parameters],
            remappings=[('cmd_vel', '/nav2/cmd_vel_raw')],
        ),
        Node(
            package='nav2_smoother',
            executable='smoother_server',
            name='smoother_server',
            output='screen',
            parameters=[nav2_parameters],
        ),
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[nav2_parameters],
        ),
        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            name='behavior_server',
            output='screen',
            parameters=[nav2_parameters],
            remappings=[('cmd_vel', '/nav2/behavior_cmd_vel')],
        ),
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[
                nav2_parameters,
                {
                    'default_nav_to_pose_bt_xml': nav_to_pose_behavior_tree,
                    'default_nav_through_poses_bt_xml': (
                        nav_through_poses_behavior_tree
                    ),
                },
            ],
        ),
        Node(
            package='nav2_waypoint_follower',
            executable='waypoint_follower',
            name='waypoint_follower',
            output='screen',
            parameters=[nav2_parameters],
        ),
        Node(
            package='nav2_velocity_smoother',
            executable='velocity_smoother',
            name='velocity_smoother',
            output='screen',
            parameters=[nav2_parameters],
            remappings=[
                ('cmd_vel', '/nav2/cmd_vel_raw'),
                ('cmd_vel_smoothed', '/cmd_vel_nav2'),
            ],
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'autostart': True,
                'node_names': [
                    'controller_server',
                    'smoother_server',
                    'planner_server',
                    'behavior_server',
                    'bt_navigator',
                    'waypoint_follower',
                    'velocity_smoother',
                ],
            }],
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='lidar_emergency_stop_node',
            name='lidar_emergency_stop_node',
            output='screen',
            parameters=[
                common_parameters,
                {
                    'input_command_topic': '/cmd_vel_path_validated',
                    'output_command_topic': '/cmd_vel',
                },
            ],
            condition=IfCondition(LaunchConfiguration('drive_enabled')),
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='navigation_visualization_node',
            name='navigation_visualization_node',
            output='screen',
            parameters=[common_parameters],
            condition=IfCondition(
                LaunchConfiguration('start_navigation_visualization')
            ),
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='mission_route_loader_node',
            name='mission_route_loader_node',
            output='screen',
            parameters=[{
                'route_file': LaunchConfiguration('mission_route_file'),
                'start_route': ParameterValue(
                    LaunchConfiguration('mission_route_start'),
                    value_type=bool,
                ),
            }],
            condition=IfCondition(
                LaunchConfiguration('mission_route_enabled')
            ),
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='navigation_learning_recorder_node',
            name='navigation_learning_recorder_node',
            output='screen',
            parameters=[common_parameters],
            condition=IfCondition(
                LaunchConfiguration('record_learning_data')
            ),
        ),
    ])
