"""Publish Ackermann commands toward a GNSS goal or smooth waypoint route."""

import math
import time

from geometry_msgs.msg import PointStamped, PoseStamped, Twist, Vector3Stamped
from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool, Float32, String, UInt32

from .navigation_core import compute_go_to_goal_command, EnuPoint
from .navigation_core import geodetic_to_enu, GeodeticPoint, GoToGoalCommand
from .route_path_core import compute_path_tracking_geometry
from .route_path_core import cumulative_path_lengths, generate_smooth_route
from .waypoint_route_core import parse_waypoint_route_json


def _yaw_from_quaternion(quaternion) -> float:
    """Return ROS ENU yaw from a geometry_msgs quaternion."""
    sin_yaw = 2.0 * (
        quaternion.w * quaternion.z
        + quaternion.x * quaternion.y
    )
    cos_yaw = 1.0 - 2.0 * (
        quaternion.y * quaternion.y
        + quaternion.z * quaternion.z
    )
    return math.atan2(sin_yaw, cos_yaw)


class GpsGoToGoalControllerNode(Node):
    """Convert GNSS goal guidance into conservative ``/cmd_vel`` output."""

    def __init__(self):
        super().__init__('gps_go_to_goal_controller_node')
        self.declare_parameter('goal_vector_topic', '/navigation/goal_vector')
        self.declare_parameter(
            'distance_topic', '/navigation/distance_to_goal'
        )
        self.declare_parameter('goal_reached_topic', '/navigation/goal_reached')
        self.declare_parameter('status_topic', '/navigation/status')
        self.declare_parameter('odometry_topic', '/vehicle/odometry')
        self.declare_parameter('command_topic', '/cmd_vel_navigation')
        self.declare_parameter('route_topic', '/navigation/waypoint_route')
        self.declare_parameter('route_status_topic', '/navigation/route_status')
        self.declare_parameter('route_index_topic', '/navigation/route_index')
        self.declare_parameter('origin_topic', '/navigation/origin_gnss')
        self.declare_parameter(
            'current_local_topic', '/navigation/current_local'
        )
        self.declare_parameter(
            'smoothed_path_topic', '/navigation/smoothed_path'
        )
        self.declare_parameter('smooth_route_enabled', True)
        self.declare_parameter('route_path_spacing_m', 0.5)
        self.declare_parameter('route_tangent_scale', 0.5)
        self.declare_parameter('route_maximum_waypoint_deviation_m', 2.0)
        self.declare_parameter('route_lookahead_distance_m', 5.0)
        self.declare_parameter('route_nearest_search_ahead', 80)
        self.declare_parameter('route_progress_distance_ratio', 1.5)
        self.declare_parameter('route_progress_lead_m', 2.0)
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('input_timeout_s', 1.0)
        self.declare_parameter('maximum_speed_mps', 3.0)
        self.declare_parameter('minimum_speed_mps', 0.5)
        self.declare_parameter('slowdown_distance_m', 12.0)
        self.declare_parameter('stop_distance_m', 3.0)
        self.declare_parameter('heading_kp', 1.0)
        self.declare_parameter('maximum_yaw_rate_rps', 0.5)
        self.declare_parameter('minimum_heading_speed_ratio', 0.2)
        self.declare_parameter('wheelbase_m', 2.85)
        self.declare_parameter('maximum_steering_angle_deg', 35.0)
        self.declare_parameter('recovery_heading_threshold_deg', 75.0)
        self.declare_parameter('recovery_minimum_speed_mps', 0.8)
        self.declare_parameter('arrival_maximum_curvature_per_m', 0.10)
        self.declare_parameter('terminal_log_period_s', 2.0)

        self.publish_rate_hz = float(
            self.get_parameter('publish_rate_hz').value
        )
        self.input_timeout_s = float(
            self.get_parameter('input_timeout_s').value
        )
        self.maximum_speed_mps = float(
            self.get_parameter('maximum_speed_mps').value
        )
        self.minimum_speed_mps = float(
            self.get_parameter('minimum_speed_mps').value
        )
        self.slowdown_distance_m = float(
            self.get_parameter('slowdown_distance_m').value
        )
        self.stop_distance_m = float(
            self.get_parameter('stop_distance_m').value
        )
        self.heading_kp = float(
            self.get_parameter('heading_kp').value
        )
        self.maximum_yaw_rate_rps = float(
            self.get_parameter('maximum_yaw_rate_rps').value
        )
        self.minimum_heading_speed_ratio = float(
            self.get_parameter('minimum_heading_speed_ratio').value
        )
        self.wheelbase_m = float(self.get_parameter('wheelbase_m').value)
        self.maximum_steering_angle_rad = math.radians(float(
            self.get_parameter('maximum_steering_angle_deg').value
        ))
        self.recovery_heading_threshold_rad = math.radians(float(
            self.get_parameter('recovery_heading_threshold_deg').value
        ))
        self.recovery_minimum_speed_mps = float(
            self.get_parameter('recovery_minimum_speed_mps').value
        )
        self.arrival_maximum_curvature_per_m = float(
            self.get_parameter('arrival_maximum_curvature_per_m').value
        )
        self.smooth_route_enabled = bool(
            self.get_parameter('smooth_route_enabled').value
        )
        self.route_path_spacing_m = float(
            self.get_parameter('route_path_spacing_m').value
        )
        self.route_tangent_scale = float(
            self.get_parameter('route_tangent_scale').value
        )
        self.route_maximum_waypoint_deviation_m = float(
            self.get_parameter(
                'route_maximum_waypoint_deviation_m'
            ).value
        )
        self.route_lookahead_distance_m = float(
            self.get_parameter('route_lookahead_distance_m').value
        )
        self.route_nearest_search_ahead = int(
            self.get_parameter('route_nearest_search_ahead').value
        )
        self.route_progress_distance_ratio = float(
            self.get_parameter('route_progress_distance_ratio').value
        )
        self.route_progress_lead_m = float(
            self.get_parameter('route_progress_lead_m').value
        )
        if self.wheelbase_m <= 0.0:
            raise ValueError('wheelbase_m must be positive')
        if not 0.0 < self.maximum_steering_angle_rad < 0.5 * math.pi:
            raise ValueError('maximum steering angle must be in (0, 90) deg')
        physical_maximum_curvature = (
            math.tan(self.maximum_steering_angle_rad) / self.wheelbase_m
        )
        if not (
            0.0 < self.arrival_maximum_curvature_per_m
            <= physical_maximum_curvature
        ):
            raise ValueError('arrival curvature limit is invalid')
        self.physical_maximum_curvature_per_m = physical_maximum_curvature
        if self.route_path_spacing_m <= 0.0:
            raise ValueError('route_path_spacing_m must be positive')
        if not 0.0 < self.route_tangent_scale <= 1.0:
            raise ValueError('route_tangent_scale must be in (0, 1]')
        if self.route_maximum_waypoint_deviation_m <= 0.0:
            raise ValueError(
                'route_maximum_waypoint_deviation_m must be positive'
            )
        if self.route_lookahead_distance_m <= 0.0:
            raise ValueError('route_lookahead_distance_m must be positive')
        if self.route_nearest_search_ahead < 1:
            raise ValueError('route_nearest_search_ahead must be positive')
        if self.route_progress_distance_ratio < 1.0:
            raise ValueError(
                'route_progress_distance_ratio must be at least 1.0'
            )
        if self.route_progress_lead_m < 0.0:
            raise ValueError('route_progress_lead_m cannot be negative')
        self.terminal_log_period_s = float(
            self.get_parameter('terminal_log_period_s').value
        )
        if self.publish_rate_hz <= 0.0:
            raise ValueError('publish_rate_hz must be positive')
        if self.input_timeout_s <= 0.0:
            raise ValueError('input_timeout_s must be positive')

        self.goal_east_m = None
        self.goal_north_m = None
        self.distance_m = None
        self.current_yaw_rad = None
        self.goal_reached = False
        self.navigation_status = 'waiting_for_position'
        self.goal_wall_time = None
        self.distance_wall_time = None
        self.odometry_wall_time = None
        self.last_log_wall_time = None
        self.last_state = None
        self.origin = None
        self.current_local = None
        self.current_local_wall_time = None
        self.route_waypoints = []
        self.route_started = False
        self.route_status = 'idle'
        self.route_active_index = 0
        self.smoothed_path = []
        self.smoothed_path_lengths = []
        self.route_waypoint_path_indices = []
        self.route_nearest_index = 0
        self.route_travelled_distance_m = 0.0
        self.route_last_local = None

        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.command_publisher = self.create_publisher(
            Twist,
            str(self.get_parameter('command_topic').value),
            10,
        )
        self.smoothed_path_publisher = self.create_publisher(
            Path,
            str(self.get_parameter('smoothed_path_topic').value),
            latched_qos,
        )
        self.create_subscription(
            Vector3Stamped,
            str(self.get_parameter('goal_vector_topic').value),
            self._on_goal_vector,
            10,
        )
        self.create_subscription(
            Float32,
            str(self.get_parameter('distance_topic').value),
            self._on_distance,
            10,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter('goal_reached_topic').value),
            self._on_goal_reached,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter('status_topic').value),
            self._on_status,
            10,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter('odometry_topic').value),
            self._on_odometry,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter('route_topic').value),
            self._on_route,
            latched_qos,
        )
        self.create_subscription(
            String,
            str(self.get_parameter('route_status_topic').value),
            self._on_route_status,
            latched_qos,
        )
        self.create_subscription(
            UInt32,
            str(self.get_parameter('route_index_topic').value),
            self._on_route_index,
            latched_qos,
        )
        self.create_subscription(
            NavSatFix,
            str(self.get_parameter('origin_topic').value),
            self._on_origin,
            latched_qos,
        )
        self.create_subscription(
            PointStamped,
            str(self.get_parameter('current_local_topic').value),
            self._on_current_local,
            10,
        )
        self.timer = self.create_timer(
            1.0 / self.publish_rate_hz,
            self._publish_command,
        )
        self.get_logger().info(
            'GPS go-to-goal controller ready: speed <= {:.2f} m/s, '
            'stop at {:.1f} m, minimum turn radius={:.2f} m, command={}, '
            'smooth route={}'
            .format(
                self.maximum_speed_mps,
                self.stop_distance_m,
                self.wheelbase_m / math.tan(
                    self.maximum_steering_angle_rad
                ),
                self.get_parameter('command_topic').value,
                self.smooth_route_enabled,
            )
        )

    def _on_goal_vector(self, message):
        self.goal_east_m = float(message.vector.x)
        self.goal_north_m = float(message.vector.y)
        self.goal_wall_time = time.monotonic()

    def _on_distance(self, message):
        self.distance_m = float(message.data)
        self.distance_wall_time = time.monotonic()

    def _on_goal_reached(self, message):
        self.goal_reached = bool(message.data)

    def _on_status(self, message):
        self.navigation_status = str(message.data)
        # A direct F6 goal may be sent after a completed waypoint route.  In
        # that case return to the original single-goal controller instead of
        # letting the stale completed route keep the vehicle stopped.
        if (
            self.route_status == 'completed'
            and self.navigation_status == 'navigating'
        ):
            self.route_started = False

    def _on_odometry(self, message):
        self.current_yaw_rad = _yaw_from_quaternion(
            message.pose.pose.orientation
        )
        self.odometry_wall_time = time.monotonic()
        self._try_build_smoothed_path()

    def _on_origin(self, message):
        values = (message.latitude, message.longitude, message.altitude)
        if not all(math.isfinite(float(value)) for value in values):
            self.get_logger().error('Ignored non-finite GNSS origin')
            return
        self.origin = GeodeticPoint(
            float(message.latitude),
            float(message.longitude),
            float(message.altitude),
        )
        self._try_build_smoothed_path()

    def _on_current_local(self, message):
        values = (message.point.x, message.point.y, message.point.z)
        if not all(math.isfinite(float(value)) for value in values):
            return
        current_local = EnuPoint(
            float(message.point.x),
            float(message.point.y),
            float(message.point.z),
        )
        if self.smoothed_path and self.route_last_local is not None:
            self.route_travelled_distance_m += math.hypot(
                current_local.east_m - self.route_last_local.east_m,
                current_local.north_m - self.route_last_local.north_m,
            )
        self.current_local = current_local
        if self.smoothed_path:
            self.route_last_local = current_local
        self.current_local_wall_time = time.monotonic()
        self._try_build_smoothed_path()

    def _on_route_status(self, message):
        self.route_status = str(message.data)

    def _on_route_index(self, message):
        self.route_active_index = int(message.data)

    def _on_route(self, message):
        try:
            route = parse_waypoint_route_json(message.data)
        except ValueError as error:
            self.get_logger().error(
                'Cannot smooth invalid waypoint route: {}'.format(error)
            )
            self.route_started = False
            self.route_active_index = 0
            self.route_waypoints = []
            self.smoothed_path = []
            self.smoothed_path_lengths = []
            self.route_waypoint_path_indices = []
            return
        self.route_waypoints = list(route.waypoints)
        self.route_started = bool(route.start and self.smooth_route_enabled)
        self.route_active_index = 1 if self.route_started else 0
        self.smoothed_path = []
        self.smoothed_path_lengths = []
        self.route_waypoint_path_indices = []
        self.route_nearest_index = 0
        self.route_travelled_distance_m = 0.0
        self.route_last_local = None
        if route.loop and self.route_started:
            self.get_logger().warning(
                'Smooth route currently follows one waypoint-list pass; '
                'repeat the start command for another loop'
            )
        self._try_build_smoothed_path()

    def _try_build_smoothed_path(self):
        if (
            not self.route_started
            or self.smoothed_path
            or not self.route_waypoints
            or self.origin is None
            or self.current_local is None
            or self.current_yaw_rad is None
        ):
            return
        try:
            route_points = [self.current_local]
            route_points.extend(
                geodetic_to_enu(waypoint, self.origin)
                for waypoint in self.route_waypoints
            )
            self.smoothed_path = generate_smooth_route(
                route_points,
                self.current_yaw_rad,
                self.route_path_spacing_m,
                self.route_tangent_scale,
                self.route_maximum_waypoint_deviation_m,
            )
            self.smoothed_path_lengths = cumulative_path_lengths(
                self.smoothed_path
            )
            previous_index = 0
            for waypoint_point in route_points[1:]:
                previous_index = min(
                    range(previous_index, len(self.smoothed_path)),
                    key=lambda index: math.hypot(
                        self.smoothed_path[index].east_m
                        - waypoint_point.east_m,
                        self.smoothed_path[index].north_m
                        - waypoint_point.north_m,
                    ),
                )
                self.route_waypoint_path_indices.append(previous_index)
        except ValueError as error:
            self.route_started = False
            self.get_logger().error(
                'Cannot build smooth waypoint path: {}'.format(error)
            )
            return
        self.route_nearest_index = 0
        self.route_travelled_distance_m = 0.0
        self.route_last_local = self.current_local
        self._publish_smoothed_path()
        self.get_logger().info(
            'Smooth Ackermann route ready: waypoints={}, path_points={}, '
            'length={:.1f} m, lookahead={:.1f} m'.format(
                len(self.route_waypoints),
                len(self.smoothed_path),
                self.smoothed_path_lengths[-1],
                self.route_lookahead_distance_m,
            )
        )

    def _publish_smoothed_path(self):
        message = Path()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = 'map'
        for index, point in enumerate(self.smoothed_path):
            if index + 1 < len(self.smoothed_path):
                next_point = self.smoothed_path[index + 1]
            else:
                next_point = self.smoothed_path[index - 1]
            yaw = math.atan2(
                next_point.north_m - point.north_m,
                next_point.east_m - point.east_m,
            )
            if index + 1 == len(self.smoothed_path):
                yaw += math.pi
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x = point.east_m
            pose.pose.position.y = point.north_m
            pose.pose.position.z = point.up_m
            pose.pose.orientation.z = math.sin(0.5 * yaw)
            pose.pose.orientation.w = math.cos(0.5 * yaw)
            message.poses.append(pose)
        self.smoothed_path_publisher.publish(message)

    def _inputs_are_fresh(self):
        now = time.monotonic()
        update_times = (
            self.goal_wall_time,
            self.distance_wall_time,
            self.odometry_wall_time,
        )
        return all(
            update_time is not None
            and now - update_time <= self.input_timeout_s
            for update_time in update_times
        )

    def _route_inputs_are_fresh(self):
        now = time.monotonic()
        return all(
            update_time is not None
            and now - update_time <= self.input_timeout_s
            for update_time in (
                self.current_local_wall_time,
                self.odometry_wall_time,
            )
        )

    def _log(self, state, command=None, distance_m=None, path_text=''):
        now = time.monotonic()
        state_changed = state != self.last_state
        period_elapsed = (
            self.last_log_wall_time is None
            or self.terminal_log_period_s == 0.0
            or now - self.last_log_wall_time >= self.terminal_log_period_s
        )
        if not state_changed and not period_elapsed:
            return
        if command is None:
            self.get_logger().info('control_state={} command=STOP'.format(state))
        else:
            self.get_logger().info(
                'control_state={} distance={:.2f} m heading_error={:.1f} deg '
                'speed={:.2f} m/s yaw_rate={:.2f} rad/s{}'.format(
                    state,
                    self.distance_m if distance_m is None else distance_m,
                    math.degrees(command.heading_error_rad),
                    command.speed_mps,
                    command.yaw_rate_rps,
                    path_text,
                )
            )
        self.last_state = state
        self.last_log_wall_time = now

    def _publish_stop(self, state):
        self.command_publisher.publish(Twist())
        self._log(state)

    def _route_mode_is_active(self):
        return bool(
            self.route_started
            and self.smoothed_path
            and self.route_status in (
                'waiting_for_goal_manager',
                'navigating',
                'completed',
            )
        )

    def _compute_route_command(self):
        maximum_progress_distance_m = min(
            self.smoothed_path_lengths[-1],
            self.route_progress_lead_m
            + self.route_progress_distance_ratio
            * self.route_travelled_distance_m,
        )
        if (
            self.route_active_index > 0
            and self.route_active_index <= len(self.route_waypoint_path_indices)
        ):
            active_waypoint_path_index = self.route_waypoint_path_indices[
                self.route_active_index - 1
            ]
            active_waypoint_progress_m = self.smoothed_path_lengths[
                active_waypoint_path_index
            ]
            maximum_progress_distance_m = min(
                maximum_progress_distance_m,
                active_waypoint_progress_m + self.route_progress_lead_m,
            )
        geometry = compute_path_tracking_geometry(
            self.smoothed_path,
            self.smoothed_path_lengths,
            self.current_local,
            self.current_yaw_rad,
            self.route_nearest_index,
            self.route_lookahead_distance_m,
            self.physical_maximum_curvature_per_m,
            self.route_nearest_search_ahead,
            maximum_progress_distance_m,
        )
        self.route_nearest_index = geometry.nearest_index

        effective_remaining_distance_m = max(
            geometry.remaining_distance_m,
            geometry.final_distance_m,
        )
        if effective_remaining_distance_m <= self.stop_distance_m:
            speed = 0.0
        else:
            distance_ratio = min(
                1.0,
                max(
                    0.0,
                    (
                        effective_remaining_distance_m
                        - self.stop_distance_m
                    ) / (
                        self.slowdown_distance_m - self.stop_distance_m
                    ),
                ),
            )
            distance_speed = (
                self.minimum_speed_mps
                + (self.maximum_speed_mps - self.minimum_speed_mps)
                * distance_ratio
            )
            curvature_ratio = min(
                1.0,
                abs(geometry.curvature_per_m)
                / self.physical_maximum_curvature_per_m,
            )
            turning_speed_ratio = max(
                self.minimum_heading_speed_ratio,
                math.sqrt(max(0.0, 1.0 - curvature_ratio)),
            )
            speed = distance_speed * turning_speed_ratio
            if (
                abs(geometry.heading_error_rad)
                >= self.recovery_heading_threshold_rad
            ):
                speed = max(speed, self.recovery_minimum_speed_mps)

        curvature = geometry.curvature_per_m
        if effective_remaining_distance_m < self.slowdown_distance_m:
            curvature = max(
                -self.arrival_maximum_curvature_per_m,
                min(self.arrival_maximum_curvature_per_m, curvature),
            )
        yaw_rate = max(
            -self.maximum_yaw_rate_rps,
            min(self.maximum_yaw_rate_rps, speed * curvature),
        )
        return (
            GoToGoalCommand(
                speed,
                yaw_rate,
                geometry.heading_error_rad,
            ),
            geometry,
            maximum_progress_distance_m,
            effective_remaining_distance_m,
        )

    def _publish_route_command(self):
        if self.route_status == 'completed':
            self._publish_stop('route_completed')
            return
        if not self._route_inputs_are_fresh():
            self._publish_stop('route_input_stale')
            return
        try:
            (
                command,
                geometry,
                maximum_progress_distance_m,
                effective_remaining_distance_m,
            ) = self._compute_route_command()
        except ValueError as error:
            self.get_logger().error('Invalid route input: {}'.format(error))
            self._publish_stop('invalid_route_input')
            return
        message = Twist()
        message.linear.x = command.speed_mps
        message.angular.z = command.yaw_rate_rps
        self.command_publisher.publish(message)
        if command.speed_mps == 0.0:
            state = 'route_waiting_for_final_arrival'
        elif (
            abs(command.heading_error_rad)
            >= self.recovery_heading_threshold_rad
        ):
            state = 'route_ackermann_recovery'
        elif effective_remaining_distance_m < self.slowdown_distance_m:
            state = 'route_arriving'
        else:
            state = 'route_following'
        self._log(
            state,
            command,
            effective_remaining_distance_m,
            ' path={}/{} target={} path_remaining={:.2f} '
            'final={:.2f} travelled={:.2f} budget={:.2f} route_wp={}/{}'.format(
                geometry.nearest_index,
                len(self.smoothed_path) - 1,
                geometry.target_index,
                geometry.remaining_distance_m,
                geometry.final_distance_m,
                self.route_travelled_distance_m,
                maximum_progress_distance_m,
                self.route_active_index,
                len(self.route_waypoint_path_indices),
            ),
        )

    def _publish_command(self):
        if self._route_mode_is_active():
            self._publish_route_command()
            return
        if self.goal_reached or self.navigation_status == 'goal_reached':
            self._publish_stop('goal_reached')
            return
        if self.navigation_status != 'navigating':
            self._publish_stop(self.navigation_status)
            return
        if not self._inputs_are_fresh():
            self._publish_stop('input_stale')
            return

        try:
            command = compute_go_to_goal_command(
                self.goal_east_m,
                self.goal_north_m,
                self.current_yaw_rad,
                self.distance_m,
                self.maximum_speed_mps,
                self.minimum_speed_mps,
                self.slowdown_distance_m,
                self.stop_distance_m,
                self.heading_kp,
                self.maximum_yaw_rate_rps,
                self.minimum_heading_speed_ratio,
                self.wheelbase_m,
                self.maximum_steering_angle_rad,
                self.recovery_heading_threshold_rad,
                self.recovery_minimum_speed_mps,
                self.arrival_maximum_curvature_per_m,
            )
        except ValueError as error:
            self.get_logger().error('Invalid navigation input: {}'.format(error))
            self._publish_stop('invalid_input')
            return

        message = Twist()
        message.linear.x = command.speed_mps
        message.angular.z = command.yaw_rate_rps
        self.command_publisher.publish(message)
        if (
            abs(command.heading_error_rad)
            >= self.recovery_heading_threshold_rad
        ):
            state = 'ackermann_recovery'
        elif self.distance_m < self.slowdown_distance_m:
            state = 'arriving'
        else:
            state = 'driving'
        self._log(state, command)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GpsGoToGoalControllerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            if rclpy.ok():
                node.command_publisher.publish(Twist())
            try:
                node.destroy_node()
            except Exception:
                # A launch SIGINT may invalidate the context before this
                # process reaches its local cleanup block.
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
