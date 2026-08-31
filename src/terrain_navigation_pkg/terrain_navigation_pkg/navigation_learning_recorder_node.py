#!/usr/bin/env python3
"""
Record synchronized navigation observations without affecting control.

The node is intentionally subscriber-only.  It captures the current LiDAR,
costmaps, plans, teacher commands and mission state while F9/F10 is active,
then writes compressed samples from a background thread.
"""

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import queue
import threading
import time

from geometry_msgs.msg import PointStamped, PoseStamped, Twist, Vector3Stamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path as PathMessage
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Float32, String, UInt32

from .navigation_learning_recorder_core import (
    BevGeometry,
    build_lidar_bev,
    occupancy_grid_to_vehicle_bev,
    quaternion_to_yaw,
    world_to_vehicle_xy,
)


CSV_FIELDS = [
    'sample_id', 'wall_time_iso', 'ros_time_s', 'cloud_stamp_s',
    'cloud_age_s', 'odom_age_s', 'file', 'route_status', 'route_index',
    'route_size', 'vehicle_x', 'vehicle_y', 'vehicle_z', 'vehicle_yaw',
    'speed_mps', 'goal_vehicle_x', 'goal_vehicle_y', 'goal_vehicle_z',
    'teacher_subgoal_x', 'teacher_subgoal_y', 'nav2_cmd_speed',
    'nav2_cmd_yaw_rate', 'output_cmd_speed', 'output_cmd_yaw_rate',
    'safety_state', 'safety_obstacle_points', 'path_hard_valid',
    'nav2_status', 'far_guide_status', 'path_clearance_status',
    'nav2_plan_points', 'far_guide_points', 'raw_lidar_points',
    'collision_event_count', 'collision_max_intensity',
]


def _stamp_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def _json_write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    temporary.replace(path)


class NavigationLearningRecorder(Node):
    """Create one learning-data session for each active F9/F10 mission."""

    def __init__(self):
        super().__init__('navigation_learning_recorder_node')
        self._declare_parameters()
        self.enabled = bool(self.get_parameter('enabled').value)
        self.output_root = Path(
            str(self.get_parameter('output_directory').value)
        ).expanduser()
        self.record_only_active = bool(
            self.get_parameter('record_only_when_route_active').value
        )
        self.maximum_cloud_age = float(
            self.get_parameter('maximum_cloud_age_s').value
        )
        self.maximum_odom_age = float(
            self.get_parameter('maximum_odometry_age_s').value
        )
        self.save_raw_points = bool(
            self.get_parameter('save_raw_points').value
        )
        self.save_sample_files = bool(
            self.get_parameter('save_sample_files').value
        )
        self.geometry = BevGeometry(
            x_min_m=float(self.get_parameter('bev_x_min_m').value),
            x_max_m=float(self.get_parameter('bev_x_max_m').value),
            y_min_m=float(self.get_parameter('bev_y_min_m').value),
            y_max_m=float(self.get_parameter('bev_y_max_m').value),
            resolution_m=float(self.get_parameter('bev_resolution_m').value),
            z_min_m=float(self.get_parameter('bev_z_min_m').value),
            z_max_m=float(self.get_parameter('bev_z_max_m').value),
        )

        self._lock = threading.Lock()
        self._latest = {}
        self._latest_wall = {}
        self._route_json = ''
        self._route_status = 'idle'
        self._route_index = 0
        self._route_size = 0
        self._active_session = None
        self._sample_sequence = 0
        self._last_cloud_key = None
        self._queued_samples = 0
        self._dropped_samples = 0
        self._collision_event_count = 0
        self._collision_max_intensity = 0.0

        queue_size = int(self.get_parameter('writer_queue_size').value)
        self._writer_queue = queue.Queue(maxsize=max(4, queue_size))
        self._writer_errors = queue.Queue()
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name='navigation-learning-writer',
            daemon=True,
        )
        self._writer_thread.start()

        self._create_subscriptions()
        rate = max(0.2, float(self.get_parameter('record_rate_hz').value))
        self.create_timer(1.0 / rate, self._capture)
        self.create_timer(1.0, self._report_writer_errors)
        self.get_logger().info(
            'Navigation learning recorder ready: enabled=%s, rate=%.1f Hz, '
            'BEV=%dx%d, output=%s' % (
                self.enabled, rate, self.geometry.width,
                self.geometry.height, self.output_root,
            )
        )

    def _declare_parameters(self):
        defaults = {
            'enabled': True,
            'output_directory': '~/terrain_nav_data/learning/raw',
            'record_rate_hz': 5.0,
            'record_only_when_route_active': True,
            'maximum_cloud_age_s': 0.7,
            'maximum_odometry_age_s': 0.7,
            'writer_queue_size': 64,
            'save_raw_points': True,
            'save_sample_files': True,
            'bev_x_min_m': -10.0,
            'bev_x_max_m': 30.0,
            'bev_y_min_m': -20.0,
            'bev_y_max_m': 20.0,
            'bev_z_min_m': -2.0,
            'bev_z_max_m': 3.0,
            'bev_resolution_m': 0.25,
            'point_cloud_topic': '/lidar/points',
            'odometry_topic': '/vehicle/odometry',
            'route_topic': '/navigation/waypoint_route',
            'route_status_topic': '/navigation/route_status',
            'route_index_topic': '/navigation/route_index',
            'route_size_topic': '/navigation/route_size',
            'goal_local_topic': '/navigation/goal_local',
            'current_local_topic': '/navigation/current_local',
            'goal_vector_topic': '/navigation/goal_vector',
            'far_subgoal_topic': '/navigation/far_subgoal',
            'far_guide_path_topic': '/navigation/far_guide_path',
            'nav2_plan_topic': '/plan',
            'local_costmap_topic': '/local_costmap/costmap',
            'global_costmap_topic': '/global_costmap/costmap',
            'nav2_command_topic': '/cmd_vel_nav2',
            'output_command_topic': '/cmd_vel',
            'safety_state_topic': '/safety/state',
            'safety_obstacle_points_topic': '/safety/obstacle_points',
            'path_hard_valid_topic': '/navigation/path_clearance/hard_valid',
            'path_clearance_status_topic': '/navigation/path_clearance/status',
            'nav2_status_topic': '/navigation/nav2_status',
            'far_guide_status_topic': '/navigation/far_guide_status',
            'collision_topic': '/vehicle/collision',
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _create_subscriptions(self):
        latched = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        reliable = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        def topic(name):
            return str(self.get_parameter(name).value)

        self.create_subscription(
            PointCloud2, topic('point_cloud_topic'),
            lambda msg: self._remember('cloud', msg), qos_profile_sensor_data,
        )
        self.create_subscription(
            Odometry, topic('odometry_topic'),
            lambda msg: self._remember('odom', msg), qos_profile_sensor_data,
        )
        self.create_subscription(
            String, topic('route_topic'), self._on_route, latched
        )
        self.create_subscription(
            String, topic('route_status_topic'), self._on_route_status, latched
        )
        self.create_subscription(
            UInt32, topic('route_index_topic'), self._on_route_index, latched
        )
        self.create_subscription(
            UInt32, topic('route_size_topic'), self._on_route_size, latched
        )
        self.create_subscription(
            Float32, topic('collision_topic'), self._on_collision, reliable
        )
        for message_type, parameter, key, qos in [
            (PointStamped, 'goal_local_topic', 'goal_local', latched),
            (PointStamped, 'current_local_topic', 'current_local', latched),
            (Vector3Stamped, 'goal_vector_topic', 'goal_vector', reliable),
            (PoseStamped, 'far_subgoal_topic', 'far_subgoal', latched),
            (PathMessage, 'far_guide_path_topic', 'far_guide_path', reliable),
            (PathMessage, 'nav2_plan_topic', 'nav2_plan', reliable),
            (OccupancyGrid, 'local_costmap_topic', 'local_costmap', latched),
            (OccupancyGrid, 'global_costmap_topic', 'global_costmap', latched),
            (Twist, 'nav2_command_topic', 'nav2_command', reliable),
            (Twist, 'output_command_topic', 'output_command', reliable),
            (String, 'safety_state_topic', 'safety_state', reliable),
            (UInt32, 'safety_obstacle_points_topic', 'safety_points', reliable),
            (Bool, 'path_hard_valid_topic', 'path_hard_valid', latched),
            (String, 'path_clearance_status_topic', 'clearance_status', reliable),
            (String, 'nav2_status_topic', 'nav2_status', reliable),
            (String, 'far_guide_status_topic', 'far_guide_status', reliable),
        ]:
            self.create_subscription(
                message_type, topic(parameter),
                lambda msg, item=key: self._remember(item, msg), qos,
            )

    def _remember(self, key, message):
        with self._lock:
            self._latest[key] = message
            self._latest_wall[key] = time.monotonic()

    def _on_route(self, message):
        with self._lock:
            self._route_json = message.data

    def _on_route_index(self, message):
        with self._lock:
            self._route_index = int(message.data)

    def _on_route_size(self, message):
        with self._lock:
            self._route_size = int(message.data)

    def _on_collision(self, message):
        intensity = max(0.0, float(message.data))
        if intensity <= 0.0:
            return
        with self._lock:
            if self._active_session is not None:
                self._collision_event_count += 1
                self._collision_max_intensity = max(
                    self._collision_max_intensity, intensity
                )

    def _on_route_status(self, message):
        status = message.data.strip().lower()
        with self._lock:
            previous = self._route_status
            self._route_status = status
            active = self._active_session is not None
        if not self.enabled:
            return
        if status == 'navigating' and not active:
            self._start_session()
        elif active and status in {
            'completed', 'final_nav2_failed', 'invalid_route',
        }:
            self._close_session(status)
        elif active and status in {'idle', 'ready'} and previous not in {
            'idle', 'ready', 'waiting_for_goal_manager',
        }:
            self._close_session('interrupted_' + status)

    def _start_session(self):
        now = datetime.now(timezone.utc).astimezone()
        session_name = now.strftime('session_%Y%m%d_%H%M%S_%f')
        session_path = self.output_root / session_name
        with self._lock:
            if self._active_session is not None:
                return
            self._active_session = session_path
            self._sample_sequence = 0
            self._last_cloud_key = None
            self._queued_samples = 0
            self._dropped_samples = 0
            self._collision_event_count = 0
            self._collision_max_intensity = 0.0
            route_json = self._route_json
            route_size = self._route_size
        metadata = {
            'schema_version': 1,
            'session': session_name,
            'started_at': now.isoformat(),
            'purpose': (
                'local-navigation policy learning' if self.save_sample_files
                else 'navigation performance evaluation'
            ),
            'control_effect': 'none; subscriber-only recorder',
            'route_json': route_json,
            'route_size': route_size,
            'bev': {
                'x_min_m': self.geometry.x_min_m,
                'x_max_m': self.geometry.x_max_m,
                'y_min_m': self.geometry.y_min_m,
                'y_max_m': self.geometry.y_max_m,
                'z_min_m': self.geometry.z_min_m,
                'z_max_m': self.geometry.z_max_m,
                'resolution_m': self.geometry.resolution_m,
                'height': self.geometry.height,
                'width': self.geometry.width,
                'lidar_channels': [
                    'occupancy', 'log_density', 'maximum_height',
                    'height_span',
                ],
                'orientation': 'row 0 forward, column 0 vehicle left',
            },
        }
        self._writer_queue.put(('start', session_path, metadata))
        self.get_logger().info('Learning session started: %s' % session_path)

    def _close_session(self, result):
        with self._lock:
            path = self._active_session
            if path is None:
                return
            summary = {
                'result': result,
                'route_status': self._route_status,
                'route_index': self._route_index,
                'route_size': self._route_size,
                'route_json': self._route_json,
                'queued_samples': self._queued_samples,
                'dropped_samples': self._dropped_samples,
                'collision_event_count': self._collision_event_count,
                'collision_max_intensity': self._collision_max_intensity,
                'ended_at': datetime.now(timezone.utc).astimezone().isoformat(),
            }
            self._active_session = None
        self._writer_queue.put(('close', path, summary))
        self.get_logger().info(
            'Learning session queued for close: result=%s, samples=%d, '
            'dropped=%d' % (
                result, summary['queued_samples'], summary['dropped_samples']
            )
        )

    def _capture(self):
        if not self.enabled:
            return
        now_wall = time.monotonic()
        with self._lock:
            session = self._active_session
            if session is None and self.record_only_active:
                return
            cloud = self._latest.get('cloud')
            odom = self._latest.get('odom')
            cloud_age = now_wall - self._latest_wall.get('cloud', -math.inf)
            odom_age = now_wall - self._latest_wall.get('odom', -math.inf)
            if cloud is None or odom is None:
                return
            if cloud_age > self.maximum_cloud_age or odom_age > self.maximum_odom_age:
                return
            cloud_key = (
                cloud.header.stamp.sec, cloud.header.stamp.nanosec,
                cloud.width, cloud.row_step,
            )
            if cloud_key == self._last_cloud_key:
                return
            self._last_cloud_key = cloud_key
            self._sample_sequence += 1
            sample_id = self._sample_sequence
            snapshot = dict(self._latest)
            snapshot_wall = dict(self._latest_wall)
            context = {
                'session': session,
                'sample_id': sample_id,
                'wall_time_iso': datetime.now(
                    timezone.utc
                ).astimezone().isoformat(),
                'ros_time_s': self.get_clock().now().nanoseconds * 1.0e-9,
                'cloud_age_s': cloud_age,
                'odom_age_s': odom_age,
                'route_status': self._route_status,
                'route_index': self._route_index,
                'route_size': self._route_size,
                'route_json': self._route_json,
                'snapshot_wall': snapshot_wall,
                'collision_event_count': self._collision_event_count,
                'collision_max_intensity': self._collision_max_intensity,
            }
        try:
            self._writer_queue.put_nowait(('sample', snapshot, context))
            with self._lock:
                self._queued_samples += 1
        except queue.Full:
            with self._lock:
                self._dropped_samples += 1

    @staticmethod
    def _cloud_xyz(message):
        try:
            values = point_cloud2.read_points_numpy(
                message, field_names=['x', 'y', 'z'], skip_nans=True
            )
            array = np.asarray(values)
            if array.dtype.names:
                return np.column_stack([
                    array['x'], array['y'], array['z']
                ]).astype(np.float32, copy=False)
            return np.asarray(array, dtype=np.float32).reshape((-1, 3))
        except (AssertionError, ValueError, TypeError):
            values = point_cloud2.read_points(
                message, field_names=['x', 'y', 'z'], skip_nans=True
            )
            array = np.asarray(values)
            if array.dtype.names:
                return np.column_stack([
                    array['x'], array['y'], array['z']
                ]).astype(np.float32, copy=False)
            return np.asarray(array, dtype=np.float32).reshape((-1, 3))

    @staticmethod
    def _path_vehicle(message, pose):
        if message is None or not message.poses:
            return np.empty((0, 2), dtype=np.float32)
        points = np.asarray([
            [item.pose.position.x, item.pose.position.y]
            for item in message.poses
        ], dtype=np.float32)
        return world_to_vehicle_xy(points, pose[0], pose[1], pose[3])

    def _costmap_vehicle(self, message, pose):
        if message is None or message.info.width == 0:
            return np.full(
                (self.geometry.height, self.geometry.width), -1, dtype=np.int8
            )
        grid = np.asarray(message.data, dtype=np.int16).reshape(
            (message.info.height, message.info.width)
        )
        origin = message.info.origin
        yaw = quaternion_to_yaw(
            origin.orientation.x, origin.orientation.y,
            origin.orientation.z, origin.orientation.w,
        )
        return occupancy_grid_to_vehicle_bev(
            grid, message.info.resolution,
            origin.position.x, origin.position.y, yaw,
            pose[0], pose[1], pose[3], self.geometry,
        )

    def _process_sample(self, snapshot, context):
        cloud = snapshot['cloud']
        odom = snapshot['odom']
        points = self._cloud_xyz(cloud)
        keep = (
            np.isfinite(points).all(axis=1)
            & (points[:, 0] >= self.geometry.x_min_m)
            & (points[:, 0] < self.geometry.x_max_m)
            & (points[:, 1] >= self.geometry.y_min_m)
            & (points[:, 1] < self.geometry.y_max_m)
            & (points[:, 2] >= self.geometry.z_min_m)
            & (points[:, 2] <= self.geometry.z_max_m)
        )
        points = points[keep].astype(np.float32, copy=False)
        lidar_bev = None
        if self.save_sample_files:
            lidar_bev = build_lidar_bev(points, self.geometry)

        position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        yaw = quaternion_to_yaw(
            orientation.x, orientation.y, orientation.z, orientation.w
        )
        pose = (position.x, position.y, position.z, yaw)
        plan = self._path_vehicle(snapshot.get('nav2_plan'), pose)
        guide = self._path_vehicle(snapshot.get('far_guide_path'), pose)
        local_costmap = global_costmap = None
        if self.save_sample_files:
            local_costmap = self._costmap_vehicle(
                snapshot.get('local_costmap'), pose
            )
            global_costmap = self._costmap_vehicle(
                snapshot.get('global_costmap'), pose
            )

        goal = np.full(3, np.nan, dtype=np.float32)
        vector = snapshot.get('goal_vector')
        if vector is not None:
            cosine = math.cos(yaw)
            sine = math.sin(yaw)
            goal[0] = cosine * vector.vector.x + sine * vector.vector.y
            goal[1] = -sine * vector.vector.x + cosine * vector.vector.y
            goal[2] = vector.vector.z
        else:
            goal_local = snapshot.get('goal_local')
            current_local = snapshot.get('current_local')
            if goal_local is not None and current_local is not None:
                delta = np.asarray([[
                    goal_local.point.x - current_local.point.x,
                    goal_local.point.y - current_local.point.y,
                ]], dtype=np.float32)
                goal[:2] = world_to_vehicle_xy(delta, 0.0, 0.0, yaw)[0]
                goal[2] = goal_local.point.z - current_local.point.z

        subgoal = np.full(2, np.nan, dtype=np.float32)
        subgoal_message = snapshot.get('far_subgoal')
        if subgoal_message is not None:
            subgoal[:] = world_to_vehicle_xy(np.asarray([[
                subgoal_message.pose.position.x,
                subgoal_message.pose.position.y,
            ]]), pose[0], pose[1], pose[3])[0]

        nav2_command = snapshot.get('nav2_command') or Twist()
        output_command = snapshot.get('output_command') or Twist()
        twist = odom.twist.twist
        sample_name = 'sample_%06d.npz' % context['sample_id']
        sample_path = context['session'] / 'samples' / sample_name
        arrays = {
            'lidar_bev': (
                lidar_bev.astype(np.float16) if lidar_bev is not None
                else np.empty((0,), dtype=np.float16)
            ),
            'lidar_points_xyz': (
                points if self.save_raw_points
                else np.empty((0, 3), dtype=np.float32)
            ),
            'local_costmap_bev': local_costmap,
            'global_costmap_bev': global_costmap,
            'nav2_plan_vehicle_xy': plan,
            'far_guide_path_vehicle_xy': guide,
            'teacher_subgoal_vehicle_xy': subgoal,
            'goal_vehicle_xyz': goal,
            'vehicle_pose_odom_xyzyaw': np.asarray(pose, dtype=np.float64),
            'vehicle_twist_xyz_rpy': np.asarray([
                twist.linear.x, twist.linear.y, twist.linear.z,
                twist.angular.x, twist.angular.y, twist.angular.z,
            ], dtype=np.float32),
            'cmd_vel_nav2': np.asarray([
                nav2_command.linear.x, nav2_command.angular.z
            ], dtype=np.float32),
            'cmd_vel_output': np.asarray([
                output_command.linear.x, output_command.angular.z
            ], dtype=np.float32),
        }
        if self.save_sample_files:
            np.savez_compressed(sample_path, **arrays)
        speed = math.sqrt(
            twist.linear.x ** 2 + twist.linear.y ** 2 + twist.linear.z ** 2
        )

        def text_value(key, default=''):
            message = snapshot.get(key)
            return message.data if message is not None else default

        row = {
            'sample_id': context['sample_id'],
            'wall_time_iso': context['wall_time_iso'],
            'ros_time_s': '%.9f' % context['ros_time_s'],
            'cloud_stamp_s': '%.9f' % _stamp_seconds(cloud.header.stamp),
            'cloud_age_s': '%.4f' % context['cloud_age_s'],
            'odom_age_s': '%.4f' % context['odom_age_s'],
            'file': ('samples/' + sample_name) if self.save_sample_files else '',
            'route_status': context['route_status'],
            'route_index': context['route_index'],
            'route_size': context['route_size'],
            'vehicle_x': '%.6f' % pose[0],
            'vehicle_y': '%.6f' % pose[1],
            'vehicle_z': '%.6f' % pose[2],
            'vehicle_yaw': '%.7f' % pose[3],
            'speed_mps': '%.5f' % speed,
            'goal_vehicle_x': '%.5f' % goal[0],
            'goal_vehicle_y': '%.5f' % goal[1],
            'goal_vehicle_z': '%.5f' % goal[2],
            'teacher_subgoal_x': '%.5f' % subgoal[0],
            'teacher_subgoal_y': '%.5f' % subgoal[1],
            'nav2_cmd_speed': '%.5f' % nav2_command.linear.x,
            'nav2_cmd_yaw_rate': '%.5f' % nav2_command.angular.z,
            'output_cmd_speed': '%.5f' % output_command.linear.x,
            'output_cmd_yaw_rate': '%.5f' % output_command.angular.z,
            'safety_state': text_value('safety_state'),
            'safety_obstacle_points': int(
                getattr(snapshot.get('safety_points'), 'data', 0)
            ),
            'path_hard_valid': int(bool(
                getattr(snapshot.get('path_hard_valid'), 'data', False)
            )),
            'nav2_status': text_value('nav2_status'),
            'far_guide_status': text_value('far_guide_status'),
            'path_clearance_status': text_value('clearance_status'),
            'nav2_plan_points': int(plan.shape[0]),
            'far_guide_points': int(guide.shape[0]),
            'raw_lidar_points': int(points.shape[0]),
            'collision_event_count': context['collision_event_count'],
            'collision_max_intensity': '%.5f' % context[
                'collision_max_intensity'
            ],
        }
        return row

    def _writer_loop(self):
        sessions = {}
        while True:
            item = self._writer_queue.get()
            try:
                if item is None:
                    break
                action = item[0]
                if action == 'start':
                    _, path, metadata = item
                    path.mkdir(parents=True, exist_ok=True)
                    if self.save_sample_files:
                        (path / 'samples').mkdir(parents=True, exist_ok=True)
                    _json_write(path / 'metadata.json', metadata)
                    stream = (path / 'frames.csv').open(
                        'w', newline='', encoding='utf-8'
                    )
                    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
                    writer.writeheader()
                    stream.flush()
                    sessions[path] = {
                        'metadata': metadata, 'stream': stream,
                        'writer': writer, 'written': 0,
                    }
                elif action == 'sample':
                    _, snapshot, context = item
                    state = sessions.get(context['session'])
                    if state is None:
                        raise RuntimeError('sample arrived before session start')
                    row = self._process_sample(snapshot, context)
                    state['writer'].writerow(row)
                    state['stream'].flush()
                    state['written'] += 1
                elif action == 'close':
                    _, path, summary = item
                    state = sessions.pop(path, None)
                    if state is not None:
                        state['metadata'].update(summary)
                        state['metadata']['written_samples'] = state['written']
                        _json_write(path / 'metadata.json', state['metadata'])
                        state['stream'].close()
            except Exception as error:  # keep later sessions recordable
                self._writer_errors.put(repr(error))
            finally:
                self._writer_queue.task_done()
        for state in sessions.values():
            state['stream'].close()

    def _report_writer_errors(self):
        while True:
            try:
                error = self._writer_errors.get_nowait()
            except queue.Empty:
                return
            self.get_logger().error('Learning recorder writer error: ' + error)

    def shutdown(self):
        if self._active_session is not None:
            self._close_session('node_shutdown')
        try:
            self._writer_queue.put(None, timeout=2.0)
            self._writer_thread.join(timeout=30.0)
        except Exception as error:
            self.get_logger().error('Recorder shutdown error: %r' % error)


def main(args=None):
    rclpy.init(args=args)
    node = NavigationLearningRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
