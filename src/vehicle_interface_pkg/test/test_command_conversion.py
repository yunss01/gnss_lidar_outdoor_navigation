import math

from vehicle_interface_pkg.command_conversion import encode_serial_command
from vehicle_interface_pkg.command_conversion import velocity_to_arduino_command


COMMON = {
    'wheelbase_m': 1.0,
    'maximum_steering_angle_rad': math.radians(30.0),
    'maximum_steering_step': 7,
    'maximum_speed_mps': 1.0,
    'maximum_pwm': 100,
}


def test_stop_maps_to_zero_command():
    command = velocity_to_arduino_command(0.0, 2.0, **COMMON)
    assert command.steering == 0
    assert command.left_pwm == 0
    assert command.right_pwm == 0


def test_straight_speed_maps_to_equal_pwm():
    command = velocity_to_arduino_command(0.5, 0.0, **COMMON)
    assert command.steering == 0
    assert command.left_pwm == 50
    assert command.right_pwm == 50


def test_left_yaw_maps_to_positive_steering():
    command = velocity_to_arduino_command(1.0, 0.3, **COMMON)
    assert command.steering > 0
    assert command.left_pwm == 100
    assert command.right_pwm == 100


def test_limits_and_protocol():
    command = velocity_to_arduino_command(5.0, -20.0, **COMMON)
    assert command.steering == -7
    assert command.left_pwm == 100
    assert command.right_pwm == 100
    assert encode_serial_command(command) == 's-7l100r100\n'

