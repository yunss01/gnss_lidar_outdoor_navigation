"""Sequence a JSON list of WGS84 waypoints through the GNSS goal manager."""

import math

from geometry_msgs.msg import Vector3Stamped
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import Bool, String, UInt32

from .navigation_core import geodetic_to_enu
from .waypoint_route_core import parse_waypoint_route_json
from .waypoint_route_core import route_tangent_headings
from .waypoint_route_core import WaypointSequence


class GnssWaypointManagerNode(Node):
    """Publish one GNSS goal at a time and advance after confirmed arrival."""

    def __init__(self):
        super().__init__('gnss_waypoint_manager_node')
        self.declare_parameter('route_topic', '/navigation/waypoint_route')
        self.declare_parameter('goal_topic', '/navigation/goal_gnss')
        self.declare_parameter(
            'goal_tangent_topic', '/navigation/goal_tangent'
        )
        self.declare_parameter(
            'goal_reached_topic', '/navigation/goal_reached'
        )
        self.declare_parameter(
            'nav2_status_topic', '/navigation/nav2_status'
        )
        self.declare_parameter(
            'nav2_route_remaining_topic',
            '/navigation/nav2_route_poses_remaining',
        )
        self.declare_parameter(
            'require_nav2_success_for_final_completion', False
        )
        self.declare_parameter(
            'use_gnss_arrival_for_rolling_progress', True
        )
        self.declare_parameter('status_topic', '/navigation/route_status')
        self.declare_parameter('index_topic', '/navigation/route_index')
        self.declare_parameter('size_topic', '/navigation/route_size')

        reliable_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.goal_publisher = self.create_publisher(
            NavSatFix,
            str(self.get_parameter('goal_topic').value),
            reliable_qos,
        )
        self.goal_tangent_publisher = self.create_publisher(
            Vector3Stamped,
            str(self.get_parameter('goal_tangent_topic').value),
            latched_qos,
        )
        self.status_publisher = self.create_publisher(
            String,
            str(self.get_parameter('status_topic').value),
            latched_qos,
        )
        self.index_publisher = self.create_publisher(
            UInt32,
            str(self.get_parameter('index_topic').value),
            latched_qos,
        )
        self.size_publisher = self.create_publisher(
            UInt32,
            str(self.get_parameter('size_topic').value),
            latched_qos,
        )
        self.create_subscription(
            String,
            str(self.get_parameter('route_topic').value),
            self._on_route,
            latched_qos,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter('goal_reached_topic').value),
            self._on_goal_reached,
            reliable_qos,
        )
        self.create_subscription(
            UInt32,
            str(self.get_parameter('nav2_route_remaining_topic').value),
            self._on_nav2_route_remaining,
            reliable_qos,
        )
        self.require_nav2_success = bool(
            self.get_parameter(
                'require_nav2_success_for_final_completion'
            ).value
        )
        self.use_gnss_arrival_for_rolling_progress = bool(
            self.get_parameter(
                'use_gnss_arrival_for_rolling_progress'
            ).value
        )
        if self.require_nav2_success:
            self.create_subscription(
                String,
                str(self.get_parameter('nav2_status_topic').value),
                self._on_nav2_status,
                latched_qos,
            )
        self.sequence = WaypointSequence()
        self.route_tangent_headings = []
        self._pending_goal_index = None
        self._goal_was_published = False
        self._final_arrival_pending = False
        self._nav2_success_for_current_goal = False
        self._nav2_controls_intermediate_progress = False
        self.create_timer(0.1, self._try_publish_pending_goal)
        self._publish_state('idle')
        self.get_logger().info(
            'GNSS waypoint manager ready: route={} goal={}'.format(
                self.get_parameter('route_topic').value,
                self.get_parameter('goal_topic').value,
            )
        )

    def _publish_state(self, status):
        status_message = String()
        status_message.data = status
        self.status_publisher.publish(status_message)
        index_message = UInt32()
        index_message.data = (
            self.sequence.index + 1
            if self.sequence.index is not None else 0
        )
        self.index_publisher.publish(index_message)
        size_message = UInt32()
        size_message.data = len(self.sequence.waypoints)
        self.size_publisher.publish(size_message)

    def _request_goal(self, index):
        self.sequence.arrival_armed = False
        self._goal_was_published = False
        self._final_arrival_pending = False
        self._nav2_success_for_current_goal = False
        self._pending_goal_index = index
        self._publish_state('waiting_for_goal_manager')
        self._try_publish_pending_goal()

    def _try_publish_pending_goal(self):
        if self._pending_goal_index is None:
            return
        if self.goal_publisher.get_subscription_count() < 1:
            return
        index = self._pending_goal_index
        self._pending_goal_index = None
        point = self.sequence.waypoints[index]
        tangent_heading = (
            self.route_tangent_headings[index]
            if index < len(self.route_tangent_headings) else None
        )
        tangent_message = Vector3Stamped()
        tangent_message.header.stamp = self.get_clock().now().to_msg()
        tangent_message.header.frame_id = 'enu'
        if tangent_heading is not None:
            tangent_message.vector.x = math.cos(tangent_heading)
            tangent_message.vector.y = math.sin(tangent_heading)
        # Retain a 1-based mission index for diagnostics and future message
        # association. The route tangent itself is the unit x/y vector.
        tangent_message.vector.z = float(index + 1)
        self.goal_tangent_publisher.publish(tangent_message)
        message = NavSatFix()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = 'wgs84'
        message.status.status = NavSatStatus.STATUS_FIX
        message.status.service = NavSatStatus.SERVICE_GPS
        message.latitude = point.latitude_deg
        message.longitude = point.longitude_deg
        message.altitude = point.altitude_m
        message.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN
        self.goal_publisher.publish(message)
        self._goal_was_published = True
        self._publish_state('navigating')
        self.get_logger().info(
            'Route waypoint {}/{}: latitude={:.9f}, longitude={:.9f}, '
            'altitude={:.3f} m, tangent={}'.format(
                index + 1,
                len(self.sequence.waypoints),
                point.latitude_deg,
                point.longitude_deg,
                point.altitude_m,
                (
                    '{:.1f} deg'.format(math.degrees(tangent_heading))
                    if tangent_heading is not None else 'undefined'
                ),
            )
        )

    def _on_route(self, message):
        try:
            route = parse_waypoint_route_json(message.data)
        except ValueError as error:
            self.get_logger().error(
                'Rejected waypoint route: {}'.format(error)
            )
            self._publish_state('invalid_route')
            return
        self.sequence.load(route)
        origin = route.waypoints[0]
        route_xy = []
        for point in route.waypoints:
            local = geodetic_to_enu(point, origin)
            route_xy.append((local.east_m, local.north_m))
        self.route_tangent_headings = route_tangent_headings(route_xy)
        self._nav2_controls_intermediate_progress = bool(
            route.rolling_horizon
        )
        self._final_arrival_pending = False
        self._nav2_success_for_current_goal = False
        self._publish_state('ready')
        self.get_logger().info(
            'Waypoint route loaded: count={} loop={} start={}'.format(
                len(route.waypoints), route.loop, route.start
            )
        )
        if route.start:
            self._request_goal(self.sequence.start())

    def _on_goal_reached(self, message):
        if not self._goal_was_published:
            return
        rolling_intermediate = bool(
            self._nav2_controls_intermediate_progress
            and self.sequence.index is not None
            and self.sequence.index < len(self.sequence.waypoints) - 1
        )
        if (
            rolling_intermediate
            and not self.use_gnss_arrival_for_rolling_progress
        ):
            return
        action, index = self.sequence.observe_goal_reached(
            bool(message.data),
            defer_final_completion=self.require_nav2_success,
        )
        if action == 'publish':
            if rolling_intermediate:
                # RemovePassedGoals and the bridge's geometric passage test
                # remain the primary rolling-route progress sources.  A
                # debounced GNSS arrival is an independent fallback for the
                # case where the vehicle enters the configured arrival circle
                # but Nav2 keeps the intermediate pose in its action feedback.
                # Publishing the next GNSS goal cannot replace the active
                # NavigateThroughPoses action while the bridge is in route
                # mode; it only latches mission/UI progress correctly.
                self.get_logger().info(
                    'Rolling waypoint progress synchronized from debounced '
                    'GNSS arrival: next waypoint {}/{}'.format(
                        index + 1,
                        len(self.sequence.waypoints),
                    )
                )
            self._request_goal(index)
        elif action == 'await_confirmation':
            self._goal_was_published = False
            self._final_arrival_pending = True
            self._publish_state('finishing')
            self.get_logger().info(
                'Final waypoint entered GNSS arrival radius; '
                'waiting for Nav2 success'
            )
            if self._nav2_success_for_current_goal:
                self._complete_final_route()
        elif action == 'complete':
            self._announce_route_complete()

    def _on_nav2_route_remaining(self, message):
        if not self.sequence.active or self.sequence.index is None:
            return
        route_size = len(self.sequence.waypoints)
        remaining = int(message.data)
        if route_size < 1 or remaining < 0 or remaining > route_size:
            return
        # Nav2 RemovePassedGoals is the authority for intermediate progress.
        # This also handles a route that starts inside WP1's GNSS arrival
        # radius, where the false-to-true arrival guard intentionally cannot
        # arm before the vehicle leaves the point.
        action, desired_index = self.sequence.synchronize_remaining_poses(
            remaining
        )
        if action != 'publish':
            return
        self.get_logger().info(
            'Route progress synchronized from Nav2: waypoint {}/{} '
            '({} pose(s) remaining)'.format(
                desired_index + 1,
                route_size,
                remaining,
            )
        )
        self._request_goal(desired_index)

    def _on_nav2_status(self, message):
        status = str(message.data)
        if status in ('goal_sent', 'navigating'):
            self._nav2_success_for_current_goal = False
            return
        if status == 'nav2_result_4':
            self._nav2_success_for_current_goal = True
            if self._final_arrival_pending:
                self._complete_final_route()
            return
        if self._final_arrival_pending and status in (
            'nav2_result_5',
            'nav2_result_6',
            'goal_rejected',
            'result_failed',
        ):
            self._publish_state('final_nav2_failed')
            self.get_logger().warning(
                'Final waypoint reached GNSS radius but Nav2 ended with {}'
                .format(status)
            )

    def _complete_final_route(self):
        action, _ = self.sequence.confirm_final_completion()
        if action != 'complete':
            return
        self._final_arrival_pending = False
        self._announce_route_complete()

    def _announce_route_complete(self):
        self._goal_was_published = False
        self._publish_state('completed')
        self.get_logger().info(
            'Waypoint route complete: {} waypoint(s)'.format(
                len(self.sequence.waypoints)
            )
        )


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GnssWaypointManagerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
