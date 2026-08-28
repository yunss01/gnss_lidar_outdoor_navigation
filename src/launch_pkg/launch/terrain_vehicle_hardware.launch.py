"""Start the Arduino backend for the terrain-navigation vehicle."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    parameters = PathJoinSubstitution([
        FindPackageShare('config_pkg'),
        'config',
        'params.yaml',
    ])
    return LaunchDescription([
        DeclareLaunchArgument('serial_port', default_value='/dev/ttyACM0'),
        DeclareLaunchArgument('dry_run', default_value='false'),
        Node(
            package='vehicle_interface_pkg',
            executable='arduino_vehicle_interface_node',
            name='arduino_vehicle_interface_node',
            output='screen',
            parameters=[
                parameters,
                {
                    'serial_port': LaunchConfiguration('serial_port'),
                    'dry_run': ParameterValue(
                        LaunchConfiguration('dry_run'),
                        value_type=bool,
                    ),
                },
            ],
        ),
    ])
