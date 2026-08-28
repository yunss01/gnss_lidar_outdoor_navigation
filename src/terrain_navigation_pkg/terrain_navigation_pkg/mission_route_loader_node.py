"""Publish a YAML mission through the existing waypoint-route interface."""

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from rclpy.qos import ReliabilityPolicy
from std_msgs.msg import String

from .mission_route_core import load_mission_route_file
from .mission_route_core import mission_route_to_json


class MissionRouteLoaderNode(Node):
    """Load one mission file, validate it, and publish it once when ready."""

    def __init__(self):
        super().__init__('mission_route_loader_node')
        self.declare_parameter('route_file', '')
        self.declare_parameter('route_topic', '/navigation/waypoint_route')
        self.declare_parameter(
            'status_topic', '/navigation/mission_route_status'
        )
        self.declare_parameter('start_route', False)
        self.declare_parameter('publish_delay_s', 1.0)
        self.declare_parameter('subscriber_wait_timeout_s', 10.0)

        route_file = str(self.get_parameter('route_file').value).strip()
        if not route_file:
            raise ValueError('route_file parameter must not be empty')
        self.mission = load_mission_route_file(route_file)
        self.force_start = bool(self.get_parameter('start_route').value)
        self.route_payload = mission_route_to_json(
            self.mission,
            force_start=self.force_start,
        )

        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.route_publisher = self.create_publisher(
            String,
            str(self.get_parameter('route_topic').value),
            latched_qos,
        )
        self.status_publisher = self.create_publisher(
            String,
            str(self.get_parameter('status_topic').value),
            latched_qos,
        )
        self.publish_delay_s = max(
            0.0, float(self.get_parameter('publish_delay_s').value)
        )
        self.wait_timeout_s = max(
            self.publish_delay_s,
            float(self.get_parameter('subscriber_wait_timeout_s').value),
        )
        self.start_time = self.get_clock().now()
        self.published = False
        self.timer = self.create_timer(0.1, self._try_publish)
        self._publish_status('loaded')
        self.get_logger().info(
            'Mission route loaded: name={} waypoints={} file={}'.format(
                self.mission.name,
                len(self.mission.route.waypoints),
                route_file,
            )
        )
        if self.mission.temporary_waypoints_enabled:
            self.get_logger().warning(
                'temporary_waypoints is enabled in YAML, but automatic '
                'subdivision is not implemented yet; original mission '
                'waypoints will be published'
            )

    def _publish_status(self, state):
        message = String()
        message.data = '{}:{}:{}'.format(
            state,
            self.mission.name,
            len(self.mission.route.waypoints),
        )
        self.status_publisher.publish(message)

    def _try_publish(self):
        if self.published:
            return
        elapsed = (
            self.get_clock().now() - self.start_time
        ).nanoseconds * 1.0e-9
        if elapsed < self.publish_delay_s:
            return
        subscribers = self.route_publisher.get_subscription_count()
        if subscribers < 1 and elapsed < self.wait_timeout_s:
            return
        if subscribers < 1:
            self.get_logger().warning(
                'No waypoint-route subscriber appeared within {:.1f} s; '
                'publishing a latched route anyway'.format(self.wait_timeout_s)
            )
        message = String()
        message.data = self.route_payload
        self.route_publisher.publish(message)
        self.published = True
        self.timer.cancel()
        started = self.mission.route.start or self.force_start
        self._publish_status('started' if started else 'published')
        self.get_logger().info(
            'Mission route published: name={} waypoints={} start={} '
            'rolling_horizon={} loop={}'.format(
                self.mission.name,
                len(self.mission.route.waypoints),
                started,
                self.mission.route.rolling_horizon,
                self.mission.route.loop,
            )
        )


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = MissionRouteLoaderNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ValueError as error:
        if node is not None:
            node.get_logger().error(str(error))
        else:
            print('Mission route loader error: {}'.format(error))
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
