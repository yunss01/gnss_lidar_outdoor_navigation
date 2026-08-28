"""Publish a compact obstacle-only PointCloud2 for Nav2 costmaps."""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import UInt32

from .ground_obstacle_core import extract_ground_relative_obstacles


class LidarObstacleFilterNode(Node):
    """Filter ground/sky points and voxel-downsample the CARLA 3D scan."""

    def __init__(self):
        super().__init__('lidar_obstacle_filter_node')
        self.declare_parameter('input_topic', '/lidar/points')
        self.declare_parameter('output_topic', '/lidar/nav2_obstacles')
        self.declare_parameter(
            'point_count_topic', '/lidar/nav2_obstacle_count'
        )
        self.declare_parameter('minimum_z_m', -1.4)
        self.declare_parameter('maximum_z_m', 1.0)
        self.declare_parameter('minimum_range_m', 1.5)
        self.declare_parameter('maximum_range_m', 45.0)
        self.declare_parameter('voxel_size_m', 0.20)
        self.declare_parameter('remove_ego_body_points', True)
        self.declare_parameter('ego_front_m', 2.55)
        self.declare_parameter('ego_rear_m', 2.65)
        self.declare_parameter('ego_half_width_m', 1.15)
        self.declare_parameter('ground_relative_low_obstacles_enabled', True)
        self.declare_parameter('ground_fit_minimum_range_m', 2.0)
        self.declare_parameter('ground_fit_maximum_range_m', 20.0)
        self.declare_parameter('ground_fit_minimum_z_m', -2.5)
        self.declare_parameter('ground_fit_maximum_z_m', -0.8)
        self.declare_parameter('ground_seed_quantile', 0.35)
        self.declare_parameter('ground_inlier_tolerance_m', 0.06)
        self.declare_parameter('minimum_ground_points', 80)
        self.declare_parameter('low_obstacle_minimum_height_m', 0.07)
        self.declare_parameter('low_obstacle_maximum_height_m', 0.45)
        self.declare_parameter('local_ground_enabled', True)
        self.declare_parameter('local_ground_resolution_m', 0.75)
        self.declare_parameter('local_ground_radius_m', 1.50)
        self.declare_parameter('local_ground_quantile', 0.25)
        self.declare_parameter('local_ground_plane_enabled', True)
        self.declare_parameter(
            'local_ground_plane_residual_tolerance_m', 0.055
        )
        self.declare_parameter(
            'clearing_output_topic', '/lidar/nav2_clearing'
        )
        self.declare_parameter('clearing_voxel_size_m', 0.50)

        self.minimum_z_m = float(self.get_parameter('minimum_z_m').value)
        self.maximum_z_m = float(self.get_parameter('maximum_z_m').value)
        self.minimum_range_m = float(
            self.get_parameter('minimum_range_m').value
        )
        self.maximum_range_m = float(
            self.get_parameter('maximum_range_m').value
        )
        self.voxel_size_m = float(self.get_parameter('voxel_size_m').value)
        self.remove_ego_body_points = bool(
            self.get_parameter('remove_ego_body_points').value
        )
        self.ego_front_m = float(self.get_parameter('ego_front_m').value)
        self.ego_rear_m = float(self.get_parameter('ego_rear_m').value)
        self.ego_half_width_m = float(
            self.get_parameter('ego_half_width_m').value
        )
        self.ground_filter_parameters = {
            'enabled': bool(self.get_parameter(
                'ground_relative_low_obstacles_enabled'
            ).value),
            'ground_fit_minimum_range_m': float(self.get_parameter(
                'ground_fit_minimum_range_m'
            ).value),
            'ground_fit_maximum_range_m': float(self.get_parameter(
                'ground_fit_maximum_range_m'
            ).value),
            'ground_fit_minimum_z_m': float(self.get_parameter(
                'ground_fit_minimum_z_m'
            ).value),
            'ground_fit_maximum_z_m': float(self.get_parameter(
                'ground_fit_maximum_z_m'
            ).value),
            'ground_seed_quantile': float(self.get_parameter(
                'ground_seed_quantile'
            ).value),
            'ground_inlier_tolerance_m': float(self.get_parameter(
                'ground_inlier_tolerance_m'
            ).value),
            'minimum_ground_points': int(self.get_parameter(
                'minimum_ground_points'
            ).value),
            'low_obstacle_minimum_height_m': float(self.get_parameter(
                'low_obstacle_minimum_height_m'
            ).value),
            'low_obstacle_maximum_height_m': float(self.get_parameter(
                'low_obstacle_maximum_height_m'
            ).value),
            'local_ground_enabled': bool(self.get_parameter(
                'local_ground_enabled'
            ).value),
            'local_ground_resolution_m': float(self.get_parameter(
                'local_ground_resolution_m'
            ).value),
            'local_ground_radius_m': float(self.get_parameter(
                'local_ground_radius_m'
            ).value),
            'local_ground_quantile': float(self.get_parameter(
                'local_ground_quantile'
            ).value),
            'local_ground_plane_enabled': bool(self.get_parameter(
                'local_ground_plane_enabled'
            ).value),
            'local_ground_plane_residual_tolerance_m': float(
                self.get_parameter(
                    'local_ground_plane_residual_tolerance_m'
                ).value
            ),
        }
        self.clearing_voxel_size_m = float(
            self.get_parameter('clearing_voxel_size_m').value
        )
        extract_ground_relative_obstacles(
            [],
            fixed_minimum_z_m=self.minimum_z_m,
            fixed_maximum_z_m=self.maximum_z_m,
            **self.ground_filter_parameters,
        )
        if self.minimum_z_m >= self.maximum_z_m:
            raise ValueError('minimum_z_m must be less than maximum_z_m')
        if not 0.0 <= self.minimum_range_m < self.maximum_range_m:
            raise ValueError('LiDAR range limits are invalid')
        if self.voxel_size_m <= 0.0:
            raise ValueError('voxel_size_m must be positive')
        if self.clearing_voxel_size_m <= 0.0:
            raise ValueError('clearing_voxel_size_m must be positive')
        if (
            self.ego_front_m <= 0.0
            or self.ego_rear_m <= 0.0
            or self.ego_half_width_m <= 0.0
        ):
            raise ValueError('ego body extents must be positive')

        self.publisher = self.create_publisher(
            PointCloud2,
            str(self.get_parameter('output_topic').value),
            qos_profile_sensor_data,
        )
        self.clearing_publisher = self.create_publisher(
            PointCloud2,
            str(self.get_parameter('clearing_output_topic').value),
            qos_profile_sensor_data,
        )
        self.count_publisher = self.create_publisher(
            UInt32,
            str(self.get_parameter('point_count_topic').value),
            10,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter('input_topic').value),
            self._on_cloud,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            (
                'Nav2 LiDAR filter ready: z={:.2f}..{:.2f} m, '
                'range={:.1f}..{:.1f} m, voxel={:.2f} m, ego_filter={}, '
                'low_obstacles={}, local_plane={}, clearing_voxel={:.2f} m'
            ).format(
                self.minimum_z_m,
                self.maximum_z_m,
                self.minimum_range_m,
                self.maximum_range_m,
                self.voxel_size_m,
                self.remove_ego_body_points,
                self.ground_filter_parameters['enabled'],
                self.ground_filter_parameters['local_ground_plane_enabled'],
                self.clearing_voxel_size_m,
            )
        )

    @staticmethod
    def _float32_view(message, name):
        field = next(
            (item for item in message.fields if item.name == name), None
        )
        if field is None or field.datatype != PointField.FLOAT32:
            raise ValueError('{} FLOAT32 field is required'.format(name))
        endian = '>f4' if message.is_bigendian else '<f4'
        count = int(message.width) * int(message.height)
        return np.ndarray(
            shape=(count,),
            dtype=np.dtype(endian),
            buffer=message.data,
            offset=int(field.offset),
            strides=(int(message.point_step),),
        )

    def _on_cloud(self, message):
        count = int(message.width) * int(message.height)
        if count == 0 or message.point_step <= 0:
            return
        try:
            x = self._float32_view(message, 'x')
            y = self._float32_view(message, 'y')
            z = self._float32_view(message, 'z')
        except ValueError as error:
            self.get_logger().error(str(error))
            return
        range_sq = x * x + y * y
        ground_result = extract_ground_relative_obstacles(
            np.column_stack((x, y, z)),
            fixed_minimum_z_m=self.minimum_z_m,
            fixed_maximum_z_m=self.maximum_z_m,
            **self.ground_filter_parameters,
        )
        mask = (
            ground_result.obstacle_mask
            & (range_sq >= self.minimum_range_m ** 2)
            & (range_sq <= self.maximum_range_m ** 2)
        )
        if self.remove_ego_body_points:
            # A circular minimum range of 1.5 m does not remove returns from
            # the Lincoln hood and rear body, which extend about 2.5 m from
            # base_link.  Keep self returns out of both Nav2 costmaps and the
            # path-clearance validator.  The independent safety node still
            # consumes the unfiltered /lidar/points cloud.
            ego_body = (
                (x >= -self.ego_rear_m)
                & (x <= self.ego_front_m)
                & (np.abs(y) <= self.ego_half_width_m)
            )
            mask &= ~ego_body
        selected = np.flatnonzero(mask)
        if selected.size:
            xyz = np.column_stack((x[selected], y[selected], z[selected]))
            keys = np.floor(xyz / self.voxel_size_m).astype(np.int32)
            _, unique_indices = np.unique(keys, axis=0, return_index=True)
            selected = selected[np.sort(unique_indices)]

        records = np.frombuffer(message.data, dtype=np.uint8).reshape(
            count, int(message.point_step)
        )
        output_records = np.ascontiguousarray(records[selected])
        output = PointCloud2()
        output.header = message.header
        output.height = 1
        output.width = int(selected.size)
        output.fields = message.fields
        output.is_bigendian = message.is_bigendian
        output.point_step = message.point_step
        output.row_step = output.point_step * output.width
        output.is_dense = True
        output.data = output_records.tobytes()
        self.publisher.publish(output)

        # Obstacle-only clouds are correct for marking, but are insufficient
        # for clearing: when a transient ground false-positive disappears from
        # the filter, its ray disappears too and the old costmap cell lingers.
        # Publish a sparse copy of the complete scan as a clearing-only source.
        # Nav2 uses these endpoints solely to raytrace free space; they never
        # create occupied cells.
        clearing_mask = (
            np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            & (range_sq >= self.minimum_range_m ** 2)
            & (range_sq <= self.maximum_range_m ** 2)
        )
        clearing_selected = np.flatnonzero(clearing_mask)
        if clearing_selected.size:
            clearing_xyz = np.column_stack((
                x[clearing_selected],
                y[clearing_selected],
                z[clearing_selected],
            ))
            clearing_keys = np.floor(
                clearing_xyz / self.clearing_voxel_size_m
            ).astype(np.int32)
            _, clearing_unique = np.unique(
                clearing_keys, axis=0, return_index=True
            )
            clearing_selected = clearing_selected[
                np.sort(clearing_unique)
            ]
        clearing_records = np.ascontiguousarray(
            records[clearing_selected]
        )
        clearing_output = PointCloud2()
        clearing_output.header = message.header
        clearing_output.height = 1
        clearing_output.width = int(clearing_selected.size)
        clearing_output.fields = message.fields
        clearing_output.is_bigendian = message.is_bigendian
        clearing_output.point_step = message.point_step
        clearing_output.row_step = (
            clearing_output.point_step * clearing_output.width
        )
        clearing_output.is_dense = True
        clearing_output.data = clearing_records.tobytes()
        self.clearing_publisher.publish(clearing_output)

        count_message = UInt32()
        count_message.data = output.width
        self.count_publisher.publish(count_message)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = LidarObstacleFilterNode()
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
