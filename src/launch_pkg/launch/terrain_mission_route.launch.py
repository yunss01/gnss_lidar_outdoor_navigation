"""Publish a YAML mission route to an already running navigation stack."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'route_file',
            description='Absolute path to a WGS84 mission-route YAML file',
        ),
        DeclareLaunchArgument(
            'start_route',
            default_value='false',
            description=(
                'Start immediately even when YAML auto_start is false'
            ),
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='mission_route_loader_node',
            name='mission_route_loader_node',
            output='screen',
            parameters=[{
                'route_file': LaunchConfiguration('route_file'),
                'start_route': ParameterValue(
                    LaunchConfiguration('start_route'),
                    value_type=bool,
                ),
            }],
        ),
    ])
