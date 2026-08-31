from setuptools import find_packages, setup


package_name = 'terrain_navigation_pkg'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        (
            'share/' + package_name,
            [
                'package.xml', 'README.md', 'PLANNING_DESIGN.md',
                'EVALUATION.md',
            ],
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sukja',
    maintainer_email='sukja@todo.todo',
    description='GNSS-guided terrain-aware navigation nodes',
    license='GPL-3.0-only',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gnss_goal_manager_node = '
            'terrain_navigation_pkg.gnss_goal_manager_node:main',
            'gnss_waypoint_manager_node = '
            'terrain_navigation_pkg.gnss_waypoint_manager_node:main',
            'terrain_mapping_node = '
            'terrain_navigation_pkg.terrain_mapping_node:main',
            'gps_go_to_goal_controller_node = '
            'terrain_navigation_pkg.gps_go_to_goal_controller_node:main',
            'lidar_emergency_stop_node = '
            'terrain_navigation_pkg.lidar_emergency_stop_node:main',
            'local_avoidance_node = '
            'terrain_navigation_pkg.local_avoidance_node:main',
            'navigation_visualization_node = '
            'terrain_navigation_pkg.navigation_visualization_node:main',
            'nav2_goal_bridge_node = '
            'terrain_navigation_pkg.nav2_goal_bridge_node:main',
            'lidar_obstacle_filter_node = '
            'terrain_navigation_pkg.lidar_obstacle_filter_node:main',
            'path_clearance_validator_node = '
            'terrain_navigation_pkg.path_clearance_validator_node:main',
            'path_validity_gate_node = '
            'terrain_navigation_pkg.path_validity_gate_node:main',
            'mission_route_loader_node = '
            'terrain_navigation_pkg.mission_route_loader_node:main',
            'far_nav2_guide_node = '
            'terrain_navigation_pkg.far_nav2_guide_node:main',
            'navigation_learning_recorder_node = '
            'terrain_navigation_pkg.navigation_learning_recorder_node:main',
            'evaluate_navigation_runs = '
            'terrain_navigation_pkg.evaluate_navigation_runs:main',
        ],
    },
)
