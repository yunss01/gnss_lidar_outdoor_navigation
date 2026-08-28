# Terrain vehicle Arduino interface

This package is the real-vehicle backend for `terrain_nav_ws`.

## Data path

```text
Nav2 -> safety-gated /cmd_vel -> arduino_vehicle_interface_node
     -> USB serial -> Arduino Mega 2560 -> three motor drivers
```

The Arduino sketch is firmware, not a ROS node. Therefore the first hardware
version needs one additional ROS node and one Arduino sketch.

## Reference pin assignment

The initial assignment follows the 2026 H-Mobility training material:

| Device | Arduino Mega pin |
|---|---|
| Steering motor driver IN1 / IN2 | D2 / D3 |
| Right rear motor driver IN1 / IN2 | D4 / D5 |
| Left rear motor driver IN1 / IN2 | D6 / D7 |
| Steering potentiometer OUT | A2 |
| Steering potentiometer power | 5V / GND |

Change these constants at the top of `terrain_vehicle_controller.ino` if the
physical wiring changes.

## Build and dry-run

```bash
cd ~/terrain_nav_ws
colcon build --symlink-install --packages-select \
  interfaces_pkg config_pkg vehicle_interface_pkg launch_pkg
source install/setup.bash

ros2 launch launch_pkg terrain_vehicle_hardware.launch.py dry_run:=true
```

The dry-run receives `/cmd_vel` and publishes the converted integer command on
`/vehicle/arduino_command`, but does not open a serial device.

## Real hardware

Upload `terrain_vehicle_controller.ino` with the Arduino IDE, then run:

```bash
ros2 launch launch_pkg terrain_vehicle_hardware.launch.py \
  serial_port:=/dev/ttyACM0
```

Before driving, measure and update:

- actual wheelbase and maximum front-wheel steering angle in `params.yaml`;
- potentiometer values at the mechanical left and right stops in the sketch;
- steering and motor direction signs;
- the loaded relationship between PWM and vehicle speed.

The current first version intentionally does not contain a serial-command
watchdog or hardware emergency-stop logic. Those safety functions must be
implemented and bench-tested before unrestricted real-vehicle testing.
