"""Publish odom-frame reference and driven paths for RViz diagnostics."""

import math

from geometry_msgs.msg import Point32, PointStamped, PolygonStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from rclpy.qos import ReliabilityPolicy
from std_msgs.msg import String


class NavigationVisualizationNode(Node):
    """Keep visualization-only paths out of the navigation control loop."""

    def __init__(self):
        super().__init__('navigation_visualization_node')
        self.declare_parameter('odometry_topic', '/vehicle/odometry')
        self.declare_parameter(
            'current_local_topic', '/navigation/current_local'
        )
        self.declare_parameter(
            'smoothed_path_topic', '/navigation/smoothed_path'
        )
        self.declare_parameter(
            'smoothed_path_odom_topic',
            '/navigation/smoothed_path_odom',
        )
        self.declare_parameter(
            'driven_path_topic', '/navigation/driven_path'
        )
        self.declare_parameter(
            'vehicle_footprint_topic',
            '/navigation/vehicle_footprint',
        )
        self.declare_parameter(
            'route_topic', '/navigation/waypoint_route'
        )
        self.declare_parameter('output_frame', 'odom')
        self.declare_parameter('vehicle_front_m', 2.4)
        self.declare_parameter('vehicle_rear_m', 2.5)
        self.declare_parameter('vehicle_half_width_m', 1.0)
        self.declare_parameter('minimum_path_spacing_m', 0.15)
        self.declare_parameter('maximum_path_points', 5000)
        self.declare_parameter('maximum_odometry_step_m', 5.0)
        self.declare_parameter('publish_rate_hz', 5.0)

        self.output_frame = str(self.get_parameter('output_frame').value)
        self.minimum_path_spacing_m = float(
            self.get_parameter('minimum_path_spacing_m').value
        )
        self.maximum_path_points = int(
            self.get_parameter('maximum_path_points').value
        )
        self.maximum_odometry_step_m = float(
            self.get_parameter('maximum_odometry_step_m').value
        )
        self.vehicle_front_m = float(
            self.get_parameter('vehicle_front_m').value
        )
        self.vehicle_rear_m = float(
            self.get_parameter('vehicle_rear_m').value
        )
        self.vehicle_half_width_m = float(
            self.get_parameter('vehicle_half_width_m').value
        )
        publish_rate_hz = float(
            self.get_parameter('publish_rate_hz').value
        )
        if not self.output_frame:
            raise ValueError('visualization output frame must not be empty')
        if self.minimum_path_spacing_m <= 0.0:
            raise ValueError('minimum visualization path spacing must be positive')
        if self.maximum_path_points < 2:
            raise ValueError('maximum visualization path points must be >= 2')
        if self.maximum_odometry_step_m <= 0.0:
            raise ValueError('maximum visualization odometry step must be positive')
        if self.vehicle_front_m <= 0.0 or self.vehicle_rear_m <= 0.0:
            raise ValueError('vehicle footprint length values must be positive')
        if self.vehicle_half_width_m <= 0.0:
            raise ValueError('vehicle footprint half width must be positive')
        if publish_rate_hz <= 0.0:
            raise ValueError('visualization publish rate must be positive')

        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.smoothed_path_odom_publisher = self.create_publisher(
            Path,
            str(self.get_parameter('smoothed_path_odom_topic').value),
            latched_qos,
        )
        self.driven_path_publisher = self.create_publisher(
            Path,
            str(self.get_parameter('driven_path_topic').value),
            latched_qos,
        )
        self.vehicle_footprint_publisher = self.create_publisher(
            PolygonStamped,
            str(self.get_parameter('vehicle_footprint_topic').value),
            latched_qos,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter('odometry_topic').value),
            self._on_odometry,
            10,
        )
        self.create_subscription(
            PointStamped,
            str(self.get_parameter('current_local_topic').value),
            self._on_current_local,
            10,
        )
        self.create_subscription(
            Path,
            str(self.get_parameter('smoothed_path_topic').value),
            self._on_smoothed_path,
            latched_qos,
        )
        self.create_subscription(
            String,
            str(self.get_parameter('route_topic').value),
            self._on_route,
            latched_qos,
        )

        self.latest_odometry = None
        self.current_local = None
        self.map_to_odom_offset = None
        self.smoothed_path = None
        self.driven_poses = []
        self.driven_path_dirty = True
        self.create_timer(1.0 / publish_rate_hz, self._publish_driven_path)
        self.get_logger().info(
            'Navigation visualization ready: global={}, driven={}, frame={}'
            .format(
                self.get_parameter('smoothed_path_odom_topic').value,
                self.get_parameter('driven_path_topic').value,
                self.output_frame,
            )
        )

    def _on_route(self, _message):
        self.driven_poses = []
        self.driven_path_dirty = True
        self._publish_driven_path()

    def _on_current_local(self, message):
        values = (
            float(message.point.x),
            float(message.point.y),
            float(message.point.z),
        )
        if not all(math.isfinite(value) for value in values):
            return
        self.current_local = values
        self._try_establish_offset()

    def _on_smoothed_path(self, message):
        self.smoothed_path = message
        self._publish_smoothed_path_odom()

    def _try_establish_offset(self):
        if (
            self.map_to_odom_offset is not None
            or self.latest_odometry is None
            or self.current_local is None
        ):
            return
        position = self.latest_odometry.pose.pose.position
        self.map_to_odom_offset = (
            float(position.x) - self.current_local[0],
            float(position.y) - self.current_local[1],
            float(position.z) - self.current_local[2],
        )
        self._publish_smoothed_path_odom()
        self.get_logger().info(
            'Visualization map->odom translation fixed at '
            '({:.2f}, {:.2f}, {:.2f}) m'.format(*self.map_to_odom_offset)
        )

    def _publish_smoothed_path_odom(self):
        if self.smoothed_path is None:
            return
        source_frame = str(self.smoothed_path.header.frame_id)
        if source_frame == self.output_frame:
            offset = (0.0, 0.0, 0.0)
        elif source_frame == 'map' and self.map_to_odom_offset is not None:
            offset = self.map_to_odom_offset
        else:
            return
        output = Path()
        output.header.stamp = self.get_clock().now().to_msg()
        output.header.frame_id = self.output_frame
        for source_pose in self.smoothed_path.poses:
            pose = PoseStamped()
            pose.header = output.header
            pose.pose.position.x = source_pose.pose.position.x + offset[0]
            pose.pose.position.y = source_pose.pose.position.y + offset[1]
            pose.pose.position.z = source_pose.pose.position.z + offset[2]
            pose.pose.orientation = source_pose.pose.orientation
            output.poses.append(pose)
        self.smoothed_path_odom_publisher.publish(output)

    def _on_odometry(self, message):
        self.latest_odometry = message
        self._publish_vehicle_footprint(message)
        self._try_establish_offset()
        position = message.pose.pose.position
        values = (float(position.x), float(position.y), float(position.z))
        if not all(math.isfinite(value) for value in values):
            return
        if self.driven_poses:
            previous = self.driven_poses[-1].pose.position
            step = math.sqrt(
                (values[0] - previous.x) ** 2
                + (values[1] - previous.y) ** 2
                + (values[2] - previous.z) ** 2
            )
            if step > self.maximum_odometry_step_m:
                self.get_logger().warning(
                    'Odometry jumped {:.2f} m; resetting driven path'.format(
                        step
                    )
                )
                self.driven_poses = []
            elif step < self.minimum_path_spacing_m:
                return
        pose = PoseStamped()
        pose.header.stamp = message.header.stamp
        pose.header.frame_id = self.output_frame
        pose.pose.position.x = values[0]
        pose.pose.position.y = values[1]
        pose.pose.position.z = values[2]
        pose.pose.orientation = message.pose.pose.orientation
        self.driven_poses.append(pose)
        if len(self.driven_poses) > self.maximum_path_points:
            self.driven_poses = self.driven_poses[-self.maximum_path_points:]
        self.driven_path_dirty = True

    def _publish_vehicle_footprint(self, odometry):
        """Publish the unpadded physical body outline in the odom frame."""
        position = odometry.pose.pose.position
        orientation = odometry.pose.pose.orientation
        values = (
            float(position.x),
            float(position.y),
            float(position.z),
            float(orientation.x),
            float(orientation.y),
            float(orientation.z),
            float(orientation.w),
        )
        if not all(math.isfinite(value) for value in values):
            return

        yaw = math.atan2(
            2.0 * (
                orientation.w * orientation.z
                + orientation.x * orientation.y
            ),
            1.0 - 2.0 * (
                orientation.y * orientation.y
                + orientation.z * orientation.z
            ),
        )
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        body_points = (
            (self.vehicle_front_m, self.vehicle_half_width_m),
            (self.vehicle_front_m, -self.vehicle_half_width_m),
            (-self.vehicle_rear_m, -self.vehicle_half_width_m),
            (-self.vehicle_rear_m, self.vehicle_half_width_m),
        )
        footprint = PolygonStamped()
        footprint.header.stamp = odometry.header.stamp
        footprint.header.frame_id = self.output_frame
        for local_x, local_y in body_points:
            point = Point32()
            point.x = float(
                position.x + cos_yaw * local_x - sin_yaw * local_y
            )
            point.y = float(
                position.y + sin_yaw * local_x + cos_yaw * local_y
            )
            point.z = float(position.z + 0.08)
            footprint.polygon.points.append(point)
        self.vehicle_footprint_publisher.publish(footprint)

    def _publish_driven_path(self):
        if not self.driven_path_dirty:
            return
        message = Path()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.output_frame
        message.poses = list(self.driven_poses)
        self.driven_path_publisher.publish(message)
        self.driven_path_dirty = False


def main(args=None):
    rclpy.init(args=args)
    node = NavigationVisualizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
