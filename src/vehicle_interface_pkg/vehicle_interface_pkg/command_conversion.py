"""Convert ROS velocity commands into the educational vehicle protocol."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ArduinoCommand:
    """Discrete steering and signed rear-motor PWM command."""

    steering: int
    left_pwm: int
    right_pwm: int


def clamp(value, lower, upper):
    """Return ``value`` limited to the closed interval."""
    return max(lower, min(upper, value))


def velocity_to_arduino_command(
    linear_mps,
    angular_rps,
    *,
    wheelbase_m,
    maximum_steering_angle_rad,
    maximum_steering_step,
    maximum_speed_mps,
    maximum_pwm,
    steering_direction=1.0,
    motor_direction=1.0,
):
    """Convert a bicycle-model ``Twist`` into the three Arduino integers.

    ``angular_rps`` is the ROS yaw-rate command, not a steering angle.  The
    bicycle relation ``yaw_rate = speed * tan(steering) / wheelbase`` is used
    before the physical wheel angle is quantized to the Arduino step range.
    Both rear motors initially receive the same PWM, matching the H-Mobility
    educational vehicle.  Per-wheel calibration can be added after bench
    measurements are available.
    """
    if wheelbase_m <= 0.0:
        raise ValueError('wheelbase_m must be positive')
    if maximum_steering_angle_rad <= 0.0:
        raise ValueError('maximum_steering_angle_rad must be positive')
    if maximum_steering_step <= 0:
        raise ValueError('maximum_steering_step must be positive')
    if maximum_speed_mps <= 0.0:
        raise ValueError('maximum_speed_mps must be positive')
    if maximum_pwm <= 0 or maximum_pwm > 255:
        raise ValueError('maximum_pwm must be in 1..255')

    speed = clamp(float(linear_mps), -maximum_speed_mps, maximum_speed_mps)
    if abs(speed) < 1.0e-4:
        return ArduinoCommand(0, 0, 0)

    steering_angle = math.atan(wheelbase_m * float(angular_rps) / speed)
    steering_angle = clamp(
        steering_angle,
        -maximum_steering_angle_rad,
        maximum_steering_angle_rad,
    )
    steering_ratio = steering_angle / maximum_steering_angle_rad
    steering = round(
        steering_direction * steering_ratio * maximum_steering_step
    )
    steering = int(clamp(
        steering,
        -maximum_steering_step,
        maximum_steering_step,
    ))

    pwm = round(motor_direction * speed / maximum_speed_mps * maximum_pwm)
    pwm = int(clamp(pwm, -maximum_pwm, maximum_pwm))
    return ArduinoCommand(steering, pwm, pwm)


def encode_serial_command(command):
    """Encode the legacy Arduino line protocol."""
    return (
        f's{command.steering}l{command.left_pwm}'
        f'r{command.right_pwm}\n'
    )

