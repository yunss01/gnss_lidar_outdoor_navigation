from setuptools import find_packages, setup


package_name = 'vehicle_interface_pkg'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml', 'README.md']),
        (
            'share/' + package_name + '/firmware/terrain_vehicle_controller',
            ['terrain_vehicle_controller.ino'],
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sukja',
    maintainer_email='sukja@todo.todo',
    description='Arduino hardware interface for terrain navigation',
    license='GPL-3.0-only',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'arduino_vehicle_interface_node = '
            'vehicle_interface_pkg.arduino_vehicle_interface_node:main',
        ],
    },
)
