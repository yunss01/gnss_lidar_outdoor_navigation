"""Validate the Nav2 global path against the latest 3D LiDAR obstacles."""

import csv
import json
import math
import os
from pathlib import Path as FilePath

from geometry_msgs.msg import Point
from nav_msgs.msg import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Bool, Float32, String
import tf2_ros
from visualization_msgs.msg import Marker

from .path_clearance_core import evaluate_path_clearance
from .path_clearance_core import PathPose2D
from .path_clearance_core import sample_path
from .path_clearance_core import trim_path_to_nearest_position
from .terrain_mapping_core import quaternion_rotation_matrix


def _yaw_from_quaternion(quaternion):
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y ** 2 + quaternion.z ** 2),
    )


class PathClearanceValidatorNode(Node):
    """Publish explicit hard/preferred clearance diagnostics for `/plan`."""

    def __init__(self):
        super().__init__('path_clearance_validator_node')
        defaults = {
            'path_topic': '/plan',
            'cloud_topic': '/lidar/nav2_obstacles',
            'hard_valid_topic': '/navigation/path_clearance/hard_valid',
            'preferred_valid_topic': (
                '/navigation/path_clearance/preferred_valid'
            ),
            'minimum_clearance_topic': '/navigation/path_clearance/minimum',
            'status_topic': '/navigation/path_clearance/status',
            'marker_topic': '/navigation/path_clearance/marker',
            'base_frame': 'base_link',
            'vehicle_front_m': 2.4,
            'vehicle_rear_m': 2.5,
            'vehicle_half_width_m': 1.0,
            'hard_clearance_m': 0.35,
            'preferred_clearance_m': 0.75,
            'path_sample_spacing_m': 0.25,
            'validation_horizon_m': 30.0,
            'validation_rate_hz': 2.0,
            'transform_timeout_s': 0.20,
            'log_directory': '~/terrain_nav_data/logs/path_clearance',
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.vehicle_front_m = float(
            self.get_parameter('vehicle_front_m').value
        )
        self.vehicle_rear_m = float(
            self.get_parameter('vehicle_rear_m').value
        )
        self.vehicle_half_width_m = float(
            self.get_parameter('vehicle_half_width_m').value
        )
        self.hard_clearance_m = float(
            self.get_parameter('hard_clearance_m').value
        )
        self.preferred_clearance_m = float(
            self.get_parameter('preferred_clearance_m').value
        )
        self.path_sample_spacing_m = float(
            self.get_parameter('path_sample_spacing_m').value
        )
        self.validation_horizon_m = float(
            self.get_parameter('validation_horizon_m').value
        )
        validation_rate_hz = float(
            self.get_parameter('validation_rate_hz').value
        )
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.transform_timeout_s = float(
            self.get_parameter('transform_timeout_s').value
        )
        if self.preferred_clearance_m < self.hard_clearance_m:
            raise ValueError('preferred clearance must be >= hard clearance')
        if validation_rate_hz <= 0.0:
            raise ValueError('validation_rate_hz must be positive')
        if self.transform_timeout_s < 0.0:
            raise ValueError('transform_timeout_s cannot be negative')
        if not self.base_frame:
            raise ValueError('base_frame cannot be empty')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.latest_path = None
        self.latest_cloud = None
        self.validation_sequence = 0
        self.last_tf_warning_ns = 0

        self.hard_publisher = self.create_publisher(
            Bool, str(self.get_parameter('hard_valid_topic').value), 10
        )
        self.preferred_publisher = self.create_publisher(
            Bool, str(self.get_parameter('preferred_valid_topic').value), 10
        )
        self.clearance_publisher = self.create_publisher(
            Float32,
            str(self.get_parameter('minimum_clearance_topic').value),
            10,
        )
        self.status_publisher = self.create_publisher(
            String, str(self.get_parameter('status_topic').value), 10
        )
        self.marker_publisher = self.create_publisher(
            Marker, str(self.get_parameter('marker_topic').value), 10
        )
        self.create_subscription(
            Path,
            str(self.get_parameter('path_topic').value),
            self._on_path,
            10,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter('cloud_topic').value),
            self._on_cloud,
            qos_profile_sensor_data,
        )
        self.create_timer(1.0 / validation_rate_hz, self._validate)
        self._open_log()
        self.get_logger().info(
            'Path clearance validator ready: hard={:.2f} m, '
            'preferred={:.2f} m, horizon={:.1f} m'.format(
                self.hard_clearance_m,
                self.preferred_clearance_m,
                self.validation_horizon_m,
            )
        )

    def _open_log(self):
        root = FilePath(os.path.expanduser(
            str(self.get_parameter('log_directory').value)
        ))
        timestamp = self.get_clock().now().nanoseconds
        run_directory = root / 'run_{}'.format(timestamp)
        run_directory.mkdir(parents=True, exist_ok=True)
        self.log_file = (run_directory / 'path_clearance.csv').open(
            'w', newline='', encoding='utf-8'
        )
        self.log_writer = csv.DictWriter(
            self.log_file,
            fieldnames=[
                'ros_time_s', 'sequence', 'path_poses', 'cloud_points',
                'checked_poses', 'checked_length_m', 'minimum_clearance_m',
                'vehicle_to_path_m', 'path_start_segment_index',
                'minimum_path_distance_m', 'minimum_path_x_m',
                'minimum_path_y_m', 'minimum_obstacle_x_m',
                'minimum_obstacle_y_m', 'path_stamp_sec',
                'path_stamp_nanosec', 'cloud_stamp_sec',
                'cloud_stamp_nanosec',
                'hard_valid', 'preferred_valid', 'path_frame', 'cloud_frame',
            ],
        )
        self.log_writer.writeheader()
        self.log_file.flush()
        self.get_logger().info('Path clearance CSV: {}'.format(
            run_directory / 'path_clearance.csv'
        ))

    def _on_path(self, message):
        self.latest_path = message

    def _on_cloud(self, message):
        self.latest_cloud = message

    @staticmethod
    def _float32_view(message, name):
        field = next(
            (field for field in message.fields if field.name == name),
            None,
        )
        if field is None or field.datatype != PointField.FLOAT32:
            raise ValueError('{} FLOAT32 field is required'.format(name))
        count = int(message.width) * int(message.height)
        endian = '>f4' if message.is_bigendian else '<f4'
        return np.ndarray(
            shape=(count,), dtype=np.dtype(endian), buffer=message.data,
            offset=int(field.offset), strides=(int(message.point_step),),
        )

    def _measurement_time(self, message):
        stamp = message.header.stamp
        if int(stamp.sec) == 0 and int(stamp.nanosec) == 0:
            return Time()
        return Time.from_msg(stamp)

    def _lookup_transform(self, target_frame, source_frame, stamp):
        return self.tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            stamp,
            timeout=Duration(seconds=self.transform_timeout_s),
        )

    def _cloud_xy_in_path_frame(self, cloud, target_frame, stamp):
        transform = self.tf_buffer.lookup_transform(
            target_frame,
            cloud.header.frame_id,
            stamp,
            timeout=Duration(seconds=self.transform_timeout_s),
        )
        x_values = self._float32_view(cloud, 'x')
        y_values = self._float32_view(cloud, 'y')
        z_values = self._float32_view(cloud, 'z')
        finite = (
            np.isfinite(x_values)
            & np.isfinite(y_values)
            & np.isfinite(z_values)
        )
        source = np.column_stack((
            x_values[finite], y_values[finite], z_values[finite]
        )).astype(np.float64, copy=False)
        rotation = quaternion_rotation_matrix(
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        )
        translation = np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ])
        transformed = source @ rotation.T + translation
        return transformed[:, :2]

    def _vehicle_xy_in_path_frame(self, target_frame, stamp):
        transform = self._lookup_transform(
            target_frame, self.base_frame, stamp
        )
        return (
            float(transform.transform.translation.x),
            float(transform.transform.translation.y),
        )

    def _path_poses(self, message):
        return [
            PathPose2D(
                float(item.pose.position.x),
                float(item.pose.position.y),
                _yaw_from_quaternion(item.pose.orientation),
            )
            for item in message.poses
        ]

    def _delete_diagnostic_markers(self, header):
        for marker_id in range(1, 5):
            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = header.frame_id
            marker.ns = 'path_clearance'
            marker.id = marker_id
            marker.action = Marker.DELETE
            self.marker_publisher.publish(marker)

    @staticmethod
    def _footprint_points(pose, front, rear, half_width, z):
        cos_yaw = math.cos(float(pose.yaw))
        sin_yaw = math.sin(float(pose.yaw))
        local_corners = (
            (front, half_width),
            (front, -half_width),
            (-rear, -half_width),
            (-rear, half_width),
            (front, half_width),
        )
        return [
            Point(
                x=float(pose.x) + cos_yaw * x - sin_yaw * y,
                y=float(pose.y) + sin_yaw * x + cos_yaw * y,
                z=z,
            )
            for x, y in local_corners
        ]

    def _publish_marker(self, header, sampled, obstacle_xy, result):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = header.frame_id
        marker.ns = 'path_clearance'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.30
        marker.color.a = 1.0
        if not result.hard_valid:
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.05, 0.05
        elif not result.preferred_valid:
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.65, 0.0
        else:
            marker.color.r, marker.color.g, marker.color.b = 0.1, 1.0, 0.35
        marker.points = [
            Point(x=pose.x, y=pose.y, z=0.65) for pose in sampled
        ]
        self.marker_publisher.publish(marker)

        pose_index = int(result.minimum_pose_index)
        obstacle_index = int(result.minimum_obstacle_index)
        if (
            pose_index < 0
            or obstacle_index < 0
            or pose_index >= len(sampled)
            or obstacle_index >= obstacle_xy.shape[0]
        ):
            self._delete_diagnostic_markers(header)
            return

        minimum_pose = sampled[pose_index]
        obstacle_x = float(obstacle_xy[obstacle_index, 0])
        obstacle_y = float(obstacle_xy[obstacle_index, 1])
        invalid = not result.hard_valid

        footprint = Marker()
        footprint.header = marker.header
        footprint.ns = marker.ns
        footprint.id = 1
        footprint.type = Marker.LINE_STRIP
        footprint.action = Marker.ADD
        footprint.pose.orientation.w = 1.0
        footprint.scale.x = 0.12
        footprint.color.a = 1.0
        footprint.color.r = 1.0
        footprint.color.g = 0.1 if invalid else 0.75
        footprint.color.b = 0.8 if invalid else 0.0
        footprint.points = self._footprint_points(
            minimum_pose,
            self.vehicle_front_m + self.hard_clearance_m,
            self.vehicle_rear_m + self.hard_clearance_m,
            self.vehicle_half_width_m + self.hard_clearance_m,
            0.78,
        )
        self.marker_publisher.publish(footprint)

        obstacle = Marker()
        obstacle.header = marker.header
        obstacle.ns = marker.ns
        obstacle.id = 2
        obstacle.type = Marker.SPHERE
        obstacle.action = Marker.ADD
        obstacle.pose.position.x = obstacle_x
        obstacle.pose.position.y = obstacle_y
        obstacle.pose.position.z = 0.88
        obstacle.pose.orientation.w = 1.0
        obstacle.scale.x = 0.50
        obstacle.scale.y = 0.50
        obstacle.scale.z = 0.50
        obstacle.color.a = 1.0
        obstacle.color.r = 1.0
        obstacle.color.g = 0.0
        obstacle.color.b = 0.85
        self.marker_publisher.publish(obstacle)

        connector = Marker()
        connector.header = marker.header
        connector.ns = marker.ns
        connector.id = 3
        connector.type = Marker.LINE_LIST
        connector.action = Marker.ADD
        connector.pose.orientation.w = 1.0
        connector.scale.x = 0.08
        connector.color.a = 1.0
        connector.color.r = 1.0
        connector.color.g = 1.0
        connector.color.b = 0.0
        connector.points = [
            Point(x=minimum_pose.x, y=minimum_pose.y, z=0.90),
            Point(x=obstacle_x, y=obstacle_y, z=0.90),
        ]
        self.marker_publisher.publish(connector)

        label = Marker()
        label.header = marker.header
        label.ns = marker.ns
        label.id = 4
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = float(minimum_pose.x)
        label.pose.position.y = float(minimum_pose.y)
        label.pose.position.z = 1.55
        label.pose.orientation.w = 1.0
        label.scale.z = 0.55
        label.color.a = 1.0
        label.color.r = 1.0
        label.color.g = 1.0
        label.color.b = 1.0
        label.text = 'clear={:.2f}m  s={:.1f}m'.format(
            result.minimum_clearance_m,
            minimum_pose.distance,
        )
        self.marker_publisher.publish(label)

    def _validate(self):
        path = self.latest_path
        cloud = self.latest_cloud
        if (
            path is None
            or cloud is None
            or not path.poses
            or not path.header.frame_id
        ):
            return
        measurement_time = self._measurement_time(cloud)
        try:
            obstacle_xy = self._cloud_xy_in_path_frame(
                cloud, path.header.frame_id, measurement_time
            )
            vehicle_x, vehicle_y = self._vehicle_xy_in_path_frame(
                path.header.frame_id, measurement_time
            )
        except (ValueError, tf2_ros.TransformException) as error:
            now_ns = self.get_clock().now().nanoseconds
            if now_ns - self.last_tf_warning_ns > 5_000_000_000:
                self.get_logger().warning(
                    'Path clearance transform failed: {}'.format(error)
                )
                self.last_tf_warning_ns = now_ns
            return
        trimmed = trim_path_to_nearest_position(
            self._path_poses(path), vehicle_x, vehicle_y
        )
        sampled = sample_path(
            trimmed.poses,
            self.path_sample_spacing_m,
            self.validation_horizon_m,
        )
        result = evaluate_path_clearance(
            sampled,
            obstacle_xy,
            self.vehicle_front_m,
            self.vehicle_rear_m,
            self.vehicle_half_width_m,
            self.hard_clearance_m,
            self.preferred_clearance_m,
        )
        self.validation_sequence += 1
        minimum = result.minimum_clearance_m
        minimum_pose = (
            sampled[result.minimum_pose_index]
            if 0 <= result.minimum_pose_index < len(sampled) else None
        )
        minimum_obstacle = (
            obstacle_xy[result.minimum_obstacle_index]
            if 0 <= result.minimum_obstacle_index < obstacle_xy.shape[0]
            else None
        )
        self.hard_publisher.publish(Bool(data=result.hard_valid))
        self.preferred_publisher.publish(Bool(data=result.preferred_valid))
        self.clearance_publisher.publish(Float32(
            data=float(minimum if math.isfinite(minimum) else -1.0)
        ))
        status = {
            'sequence': self.validation_sequence,
            'minimum_clearance_m': minimum if math.isfinite(minimum) else None,
            'hard_clearance_m': self.hard_clearance_m,
            'preferred_clearance_m': self.preferred_clearance_m,
            'hard_valid': result.hard_valid,
            'preferred_valid': result.preferred_valid,
            'checked_length_m': result.checked_length_m,
            'checked_pose_count': result.checked_pose_count,
            'vehicle_to_path_m': trimmed.nearest_distance_m,
            'path_start_segment_index': trimmed.start_segment_index,
            'minimum_path_distance_m': (
                minimum_pose.distance if minimum_pose is not None else None
            ),
            'minimum_path_x_m': (
                minimum_pose.x if minimum_pose is not None else None
            ),
            'minimum_path_y_m': (
                minimum_pose.y if minimum_pose is not None else None
            ),
            'minimum_obstacle_x_m': (
                float(minimum_obstacle[0])
                if minimum_obstacle is not None else None
            ),
            'minimum_obstacle_y_m': (
                float(minimum_obstacle[1])
                if minimum_obstacle is not None else None
            ),
        }
        self.status_publisher.publish(String(data=json.dumps(status)))
        self._publish_marker(path.header, sampled, obstacle_xy, result)
        self.log_writer.writerow({
            'ros_time_s': self.get_clock().now().nanoseconds / 1.0e9,
            'sequence': self.validation_sequence,
            'path_poses': len(path.poses),
            'cloud_points': int(obstacle_xy.shape[0]),
            'checked_poses': result.checked_pose_count,
            'checked_length_m': '{:.3f}'.format(result.checked_length_m),
            'minimum_clearance_m': (
                '{:.3f}'.format(minimum) if math.isfinite(minimum) else 'inf'
            ),
            'vehicle_to_path_m': '{:.3f}'.format(
                trimmed.nearest_distance_m
            ),
            'path_start_segment_index': trimmed.start_segment_index,
            'minimum_path_distance_m': (
                '{:.3f}'.format(minimum_pose.distance)
                if minimum_pose is not None else ''
            ),
            'minimum_path_x_m': (
                '{:.3f}'.format(minimum_pose.x)
                if minimum_pose is not None else ''
            ),
            'minimum_path_y_m': (
                '{:.3f}'.format(minimum_pose.y)
                if minimum_pose is not None else ''
            ),
            'minimum_obstacle_x_m': (
                '{:.3f}'.format(float(minimum_obstacle[0]))
                if minimum_obstacle is not None else ''
            ),
            'minimum_obstacle_y_m': (
                '{:.3f}'.format(float(minimum_obstacle[1]))
                if minimum_obstacle is not None else ''
            ),
            'path_stamp_sec': int(path.header.stamp.sec),
            'path_stamp_nanosec': int(path.header.stamp.nanosec),
            'cloud_stamp_sec': int(cloud.header.stamp.sec),
            'cloud_stamp_nanosec': int(cloud.header.stamp.nanosec),
            'hard_valid': int(result.hard_valid),
            'preferred_valid': int(result.preferred_valid),
            'path_frame': path.header.frame_id,
            'cloud_frame': cloud.header.frame_id,
        })
        self.log_file.flush()
        if self.validation_sequence == 1 or self.validation_sequence % 10 == 0:
            label = 'PREFERRED' if result.preferred_valid else (
                'HARD-ONLY' if result.hard_valid else 'INVALID'
            )
            clearance_text = (
                'inf'
                if not math.isfinite(minimum)
                else '{:.2f}'.format(minimum)
            )
            self.get_logger().info(
                'Path clearance #{}: {} m [{}], checked {:.1f} m'.format(
                    self.validation_sequence, clearance_text, label,
                    result.checked_length_m,
                )
            )

    def destroy_node(self):
        if hasattr(self, 'log_file') and not self.log_file.closed:
            self.log_file.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PathClearanceValidatorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
