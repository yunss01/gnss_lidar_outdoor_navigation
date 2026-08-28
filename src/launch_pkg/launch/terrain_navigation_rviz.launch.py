"""Open RViz with the saved terrain-navigation diagnostic layout."""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    rviz_config = PathJoinSubstitution([
        FindPackageShare('launch_pkg'),
        'rviz',
        'terrain_navigation.rviz',
    ])
    return LaunchDescription([
        Node(
            package='rviz2',
            executable='rviz2',
            name='terrain_navigation_rviz',
            output='screen',
            arguments=['-d', rviz_config],
        ),
    ])
