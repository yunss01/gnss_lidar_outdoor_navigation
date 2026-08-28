"""Fuse CARLA 3D LiDAR scans into a bounded odom-frame terrain map."""

from collections import deque
import math

import numpy as np
from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
import tf2_ros

from .terrain_mapping_core import RollingVoxelMap
from .terrain_mapping_core import build_terrain_layers
from .terrain_mapping_core import filter_sensor_points
from .terrain_mapping_core import quaternion_rotation_matrix
from .terrain_mapping_core import transform_xyzi


def _timestamp_seconds(stamp):
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


def _point_cloud_message(points, header):
    points = np.asarray(points, dtype=np.float32)
    message = PointCloud2()
    message.header = header
    message.height = 1
    message.width = int(points.shape[0])
    message.fields = [
        PointField(
            name='x', offset=0, datatype=PointField.FLOAT32, count=1
        ),
        PointField(
            name='y', offset=4, datatype=PointField.FLOAT32, count=1
        ),
        PointField(
            name='z', offset=8, datatype=PointField.FLOAT32, count=1
        ),
        PointField(
            name='intensity', offset=12,
            datatype=PointField.FLOAT32, count=1,
        ),
    ]
    message.is_bigendian = False
    message.point_step = 16
    message.row_step = message.point_step * message.width
    message.is_dense = bool(
        points.shape[0] == 0 or np.all(np.isfinite(points))
    )
    message.data = np.ascontiguousarray(points).tobytes()
    return message


def _occupancy_grid_message(layer, terrain_layers, header):
    message = OccupancyGrid()
    message.header = header
    message.info.map_load_time = header.stamp
    message.info.resolution = terrain_layers.resolution_m
    message.info.width = int(layer.shape[1])
    message.info.height = int(layer.shape[0])
    message.info.origin.position.x = terrain_layers.origin_x_m
    message.info.origin.position.y = terrain_layers.origin_y_m
    message.info.origin.orientation.w = 1.0
    message.data = layer.astype(np.int8, copy=False).ravel().tolist()
    return message


class TerrainMappingNode(Node):
    """Create a rolling voxel map and terrain cost layers."""

    def __init__(self):
        super().__init__('terrain_mapping_node')
        self.declare_parameter('input_cloud_topic', '/lidar/points')
        self.declare_parameter('map_points_topic', '/terrain/map_points')
        self.declare_parameter(
            'slope_costmap_topic', '/terrain/slope_costmap'
        )
        self.declare_parameter(
            'traversability_costmap_topic',
            '/terrain/traversability_costmap',
        )
        self.declare_parameter('map_frame', 'odom')
        self.declare_parameter('voxel_size_m', 0.20)
        self.declare_parameter('rolling_radius_m', 40.0)
        self.declare_parameter('maximum_voxel_age_s', 0.0)
        self.declare_parameter('map_resolution_m', 0.25)
        self.declare_parameter('map_publish_rate_hz', 1.0)
        self.declare_parameter('minimum_range_m', 1.0)
        self.declare_parameter('maximum_range_m', 50.0)
        self.declare_parameter('ego_min_x_m', -2.8)
        self.declare_parameter('ego_max_x_m', 2.8)
        self.declare_parameter('ego_half_width_m', 1.25)
        self.declare_parameter('minimum_points_per_cell', 1)
        self.declare_parameter('safe_slope_deg', 8.0)
        self.declare_parameter('lethal_slope_deg', 25.0)
        self.declare_parameter('safe_step_height_m', 0.10)
        self.declare_parameter('lethal_step_height_m', 0.35)
        self.declare_parameter('pending_cloud_limit', 20)
        self.declare_parameter('terminal_log_period_s', 2.0)

        self.map_frame = str(self.get_parameter('map_frame').value)
        self.voxel_size_m = float(
            self.get_parameter('voxel_size_m').value
        )
        self.rolling_radius_m = float(
            self.get_parameter('rolling_radius_m').value
        )
        self.maximum_voxel_age_s = float(
            self.get_parameter('maximum_voxel_age_s').value
        )
        self.map_resolution_m = float(
            self.get_parameter('map_resolution_m').value
        )
        self.minimum_range_m = float(
            self.get_parameter('minimum_range_m').value
        )
        self.maximum_range_m = float(
            self.get_parameter('maximum_range_m').value
        )
        self.ego_min_x_m = float(
            self.get_parameter('ego_min_x_m').value
        )
        self.ego_max_x_m = float(
            self.get_parameter('ego_max_x_m').value
        )
        self.ego_half_width_m = float(
            self.get_parameter('ego_half_width_m').value
        )
        self.minimum_points_per_cell = int(
            self.get_parameter('minimum_points_per_cell').value
        )
        self.safe_slope_deg = float(
            self.get_parameter('safe_slope_deg').value
        )
        self.lethal_slope_deg = float(
            self.get_parameter('lethal_slope_deg').value
        )
        self.safe_step_height_m = float(
            self.get_parameter('safe_step_height_m').value
        )
        self.lethal_step_height_m = float(
            self.get_parameter('lethal_step_height_m').value
        )
        self.terminal_log_period_s = float(
            self.get_parameter('terminal_log_period_s').value
        )
        publish_rate_hz = float(
            self.get_parameter('map_publish_rate_hz').value
        )
        pending_limit = int(
            self.get_parameter('pending_cloud_limit').value
        )
        if self.rolling_radius_m <= self.voxel_size_m:
            raise ValueError('rolling_radius_m must exceed voxel_size_m')
        if publish_rate_hz <= 0.0:
            raise ValueError('map_publish_rate_hz must be positive')
        if pending_limit < 1:
            raise ValueError('pending_cloud_limit must be positive')

        self.voxel_map = RollingVoxelMap(self.voxel_size_m)
        self.pending_clouds = deque(maxlen=pending_limit)
        self.latest_center_xy = None
        self.latest_stamp = None
        self.processed_clouds = 0
        self.rejected_clouds = 0
        self.last_log_time = None

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.map_points_publisher = self.create_publisher(
            PointCloud2,
            str(self.get_parameter('map_points_topic').value),
            map_qos,
        )
        self.slope_publisher = self.create_publisher(
            OccupancyGrid,
            str(self.get_parameter('slope_costmap_topic').value),
            map_qos,
        )
        self.traversability_publisher = self.create_publisher(
            OccupancyGrid,
            str(
                self.get_parameter(
                    'traversability_costmap_topic'
                ).value
            ),
            map_qos,
        )

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=False
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter('input_cloud_topic').value),
            self._cloud_callback,
            sensor_qos,
        )
        self.create_timer(0.02, self._process_pending_clouds)
        self.create_timer(1.0 / publish_rate_hz, self._publish_map)
        self.get_logger().info(
            'Terrain mapper ready: input={} frame={} radius={:.1f} m '
            'voxel={:.2f} m grid={:.2f} m'.format(
                self.get_parameter('input_cloud_topic').value,
                self.map_frame,
                self.rolling_radius_m,
                self.voxel_size_m,
                self.map_resolution_m,
            )
        )

    def _cloud_callback(self, message):
        if len(self.pending_clouds) == self.pending_clouds.maxlen:
            self.rejected_clouds += 1
        self.pending_clouds.append(message)

    def _lookup_transform(self, message):
        return self.tf_buffer.lookup_transform(
            self.map_frame,
            message.header.frame_id,
            Time.from_msg(message.header.stamp),
            timeout=Duration(seconds=0.0),
        )

    def _process_pending_clouds(self):
        processed_this_tick = 0
        while self.pending_clouds and processed_this_tick < 2:
            message = self.pending_clouds[0]
            try:
                transform = self._lookup_transform(message)
            except tf2_ros.TransformException:
                return
            self.pending_clouds.popleft()
            try:
                self._integrate_cloud(message, transform)
            except (AssertionError, ValueError) as error:
                self.rejected_clouds += 1
                self.get_logger().warning(
                    'Rejected PointCloud2: {}'.format(error)
                )
            processed_this_tick += 1

    def _integrate_cloud(self, message, transform):
        structured = point_cloud2.read_points(
            message,
            field_names=['x', 'y', 'z', 'intensity'],
            skip_nans=False,
        )
        points = np.column_stack([
            structured[field_name]
            for field_name in ('x', 'y', 'z', 'intensity')
        ]).astype(np.float32, copy=False)
        points = filter_sensor_points(
            points,
            self.minimum_range_m,
            self.maximum_range_m,
            self.ego_min_x_m,
            self.ego_max_x_m,
            self.ego_half_width_m,
        )

        rotation_message = transform.transform.rotation
        rotation = quaternion_rotation_matrix(
            rotation_message.x,
            rotation_message.y,
            rotation_message.z,
            rotation_message.w,
        )
        translation_message = transform.transform.translation
        translation = np.asarray([
            translation_message.x,
            translation_message.y,
            translation_message.z,
        ])
        map_points = transform_xyzi(points, rotation, translation)
        timestamp_s = _timestamp_seconds(message.header.stamp)
        self.voxel_map.update(map_points, timestamp_s)
        self.latest_center_xy = (
            float(translation[0]), float(translation[1])
        )
        self.latest_stamp = message.header.stamp
        self.processed_clouds += 1

    def _publish_map(self):
        if self.latest_center_xy is None or self.latest_stamp is None:
            return
        timestamp_s = _timestamp_seconds(self.latest_stamp)
        self.voxel_map.prune(
            self.latest_center_xy,
            self.rolling_radius_m,
            timestamp_s,
            self.maximum_voxel_age_s,
        )
        points = self.voxel_map.points()
        header = Header()
        header.stamp = self.latest_stamp
        header.frame_id = self.map_frame
        self.map_points_publisher.publish(
            _point_cloud_message(points, header)
        )

        terrain_layers = build_terrain_layers(
            points,
            self.latest_center_xy,
            2.0 * self.rolling_radius_m,
            self.map_resolution_m,
            self.minimum_points_per_cell,
            self.safe_slope_deg,
            self.lethal_slope_deg,
            self.safe_step_height_m,
            self.lethal_step_height_m,
        )
        slope_cost = np.full(
            terrain_layers.slope_deg.shape, -1, dtype=np.int8
        )
        slope_known = np.isfinite(terrain_layers.slope_deg)
        slope_cost[slope_known] = np.rint(np.clip(
            terrain_layers.slope_deg[slope_known]
            / self.lethal_slope_deg * 100.0,
            0.0,
            100.0,
        )).astype(np.int8)
        self.slope_publisher.publish(
            _occupancy_grid_message(
                slope_cost, terrain_layers, header
            )
        )
        self.traversability_publisher.publish(
            _occupancy_grid_message(
                terrain_layers.traversability_cost,
                terrain_layers,
                header,
            )
        )

        now = self.get_clock().now().nanoseconds * 1e-9
        if (
                self.last_log_time is None
                or now - self.last_log_time
                >= self.terminal_log_period_s):
            known_cells = int(np.count_nonzero(
                terrain_layers.traversability_cost >= 0
            ))
            lethal_cells = int(np.count_nonzero(
                terrain_layers.traversability_cost >= 100
            ))
            self.get_logger().info(
                'Terrain map: clouds={} voxels={} known_cells={} '
                'lethal_cells={} pending={} rejected={}'.format(
                    self.processed_clouds,
                    len(self.voxel_map),
                    known_cells,
                    lethal_cells,
                    len(self.pending_clouds),
                    self.rejected_clouds,
                )
            )
            self.last_log_time = now


def main(args=None):
    rclpy.init(args=args)
    node = TerrainMappingNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
