"""Block stale controller commands while the active Nav2 path is invalid."""

import csv
from datetime import datetime
from pathlib import Path
import time

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from rclpy.qos import ReliabilityPolicy
from std_msgs.msg import Bool, String

from .path_validity_gate_core import PathValidityGateState


class PathValidityGateNode(Node):
    """Pass RPP commands only while `/plan` remains hard-valid."""

    def __init__(self):
        super().__init__('path_validity_gate_node')
        defaults = {
            'input_command_topic': '/cmd_vel_nav2',
            'output_command_topic': '/cmd_vel_path_validated',
            'hard_valid_topic': '/navigation/path_clearance/hard_valid',
            'blocked_topic': '/navigation/path_validity_gate/blocked',
            'status_topic': '/navigation/path_validity_gate/status',
            'invalid_hold_s': 0.75,
            'valid_release_s': 0.75,
            'validity_timeout_s': 1.5,
            'control_rate_hz': 20.0,
            'log_directory': '~/terrain_nav_data/logs/path_validity_gate',
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        control_rate_hz = float(
            self.get_parameter('control_rate_hz').value
        )
        if control_rate_hz <= 0.0:
            raise ValueError('control_rate_hz must be positive')
        self.gate = PathValidityGateState(
            float(self.get_parameter('invalid_hold_s').value),
            float(self.get_parameter('valid_release_s').value),
            float(self.get_parameter('validity_timeout_s').value),
        )

        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.command_publisher = self.create_publisher(
            Twist,
            str(self.get_parameter('output_command_topic').value),
            10,
        )
        self.blocked_publisher = self.create_publisher(
            Bool,
            str(self.get_parameter('blocked_topic').value),
            latched_qos,
        )
        self.status_publisher = self.create_publisher(
            String,
            str(self.get_parameter('status_topic').value),
            latched_qos,
        )
        self.create_subscription(
            Twist,
            str(self.get_parameter('input_command_topic').value),
            self._on_command,
            10,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter('hard_valid_topic').value),
            self._on_hard_valid,
            10,
        )
        self.latest_command = Twist()
        self.last_snapshot_key = None
        self.log_stream, self.log_writer, log_path = self._create_logger(
            str(self.get_parameter('log_directory').value)
        )
        self.create_timer(1.0 / control_rate_hz, self._on_timer)
        self._apply_snapshot(self.gate.evaluate(time.monotonic()))
        self.get_logger().info(
            'Path-validity gate ready: {} -> {}; invalid {:.2f} s, '
            'release {:.2f} s, timeout {:.2f} s; CSV={}'.format(
                self.get_parameter('input_command_topic').value,
                self.get_parameter('output_command_topic').value,
                self.gate.invalid_hold_s,
                self.gate.valid_release_s,
                self.gate.validity_timeout_s,
                log_path,
            )
        )

    @staticmethod
    def _create_logger(directory):
        root = Path(directory).expanduser()
        stamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')
        run_directory = root / ('run_' + stamp)
        run_directory.mkdir(parents=True, exist_ok=True)
        path = run_directory / 'path_validity_gate.csv'
        stream = path.open('w', newline='', encoding='utf-8')
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                'wall_time_iso',
                'reason',
                'blocked',
                'latest_validity',
                'validity_age_s',
                'invalid_duration_s',
                'valid_duration_s',
                'input_speed_mps',
                'input_yaw_rate_rps',
            ],
        )
        writer.writeheader()
        stream.flush()
        return stream, writer, path

    def _on_command(self, message):
        self.latest_command = message
        snapshot = self.gate.evaluate(time.monotonic())
        if snapshot.blocked:
            self.command_publisher.publish(Twist())
        else:
            self.command_publisher.publish(message)
        self._apply_snapshot(snapshot)

    def _on_hard_valid(self, message):
        snapshot = self.gate.update(bool(message.data), time.monotonic())
        self._apply_snapshot(snapshot)

    def _on_timer(self):
        snapshot = self.gate.evaluate(time.monotonic())
        if snapshot.blocked:
            # Keep the downstream watchdog explicitly stopped even if RPP
            # pauses command publication during a planner recovery.
            self.command_publisher.publish(Twist())
        self._apply_snapshot(snapshot)

    def _apply_snapshot(self, snapshot):
        key = (snapshot.blocked, snapshot.reason, snapshot.validity)
        if key == self.last_snapshot_key:
            return
        self.last_snapshot_key = key
        self.blocked_publisher.publish(Bool(data=snapshot.blocked))
        self.status_publisher.publish(String(data=snapshot.reason))
        self.log_writer.writerow({
            'wall_time_iso': datetime.now().astimezone().isoformat(
                timespec='milliseconds'
            ),
            'reason': snapshot.reason,
            'blocked': int(snapshot.blocked),
            'latest_validity': (
                '' if snapshot.validity is None else int(snapshot.validity)
            ),
            'validity_age_s': snapshot.validity_age_s,
            'invalid_duration_s': snapshot.invalid_duration_s,
            'valid_duration_s': snapshot.valid_duration_s,
            'input_speed_mps': self.latest_command.linear.x,
            'input_yaw_rate_rps': self.latest_command.angular.z,
        })
        self.log_stream.flush()
        message = (
            'Path-validity gate: {} (blocked={}, validity={}, age={:.2f} s)'
            .format(
                snapshot.reason,
                snapshot.blocked,
                snapshot.validity,
                snapshot.validity_age_s,
            )
        )
        # rclpy keys a log call by its Python call site and rejects changing
        # severity at the same call site.  Keep INFO and WARN on distinct
        # source lines so a blocked -> released transition cannot crash the
        # gate with "Logger severity cannot be changed between calls".
        if snapshot.blocked:
            self.get_logger().warning(message)
        else:
            self.get_logger().info(message)

    def destroy_node(self):
        if hasattr(self, 'log_stream') and not self.log_stream.closed:
            self.log_stream.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PathValidityGateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Publish the final stop only while the ROS context is still valid.
        # This avoids the noisy RCLError traceback that can otherwise appear
        # when Ctrl+C has already started tearing down rclpy.
        if rclpy.ok():
            node.command_publisher.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
