"""Pure helpers for vehicle-centric navigation learning records."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class BevGeometry:
    """Describe a forward-up, left-up vehicle-centric BEV image."""

    x_min_m: float = -10.0
    x_max_m: float = 30.0
    y_min_m: float = -20.0
    y_max_m: float = 20.0
    resolution_m: float = 0.25
    z_min_m: float = -2.0
    z_max_m: float = 3.0

    def __post_init__(self):
        if self.x_max_m <= self.x_min_m:
            raise ValueError('x_max_m must exceed x_min_m')
        if self.y_max_m <= self.y_min_m:
            raise ValueError('y_max_m must exceed y_min_m')
        if self.z_max_m <= self.z_min_m:
            raise ValueError('z_max_m must exceed z_min_m')
        if self.resolution_m <= 0.0:
            raise ValueError('resolution_m must be positive')

    @property
    def height(self):
        return int(round(
            (self.x_max_m - self.x_min_m) / self.resolution_m
        ))

    @property
    def width(self):
        return int(round(
            (self.y_max_m - self.y_min_m) / self.resolution_m
        ))


def quaternion_to_yaw(x, y, z, w):
    """Return ROS quaternion yaw in radians."""
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def world_to_vehicle_xy(points_xy, vehicle_x, vehicle_y, vehicle_yaw):
    """Transform odom/world XY points into x-forward, y-left coordinates."""
    points = np.asarray(points_xy, dtype=np.float32).reshape((-1, 2))
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    dx = points[:, 0] - float(vehicle_x)
    dy = points[:, 1] - float(vehicle_y)
    cosine = math.cos(float(vehicle_yaw))
    sine = math.sin(float(vehicle_yaw))
    result = np.empty_like(points, dtype=np.float32)
    result[:, 0] = cosine * dx + sine * dy
    result[:, 1] = -sine * dx + cosine * dy
    return result


def build_lidar_bev(points_xyz, geometry):
    """
    Build occupancy, log-density, maximum-height and height-span layers.

    The returned image is ``[4, H, W]``. Row zero is the far-forward edge,
    and column zero is the vehicle-left edge. Values are float32 so callers
    can choose their own storage or normalization precision.
    """
    points = np.asarray(points_xyz, dtype=np.float32).reshape((-1, 3))
    height = geometry.height
    width = geometry.width
    bev = np.zeros((4, height, width), dtype=np.float32)
    if points.size == 0:
        return bev

    finite = np.isfinite(points).all(axis=1)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    keep = (
        finite
        & (x >= geometry.x_min_m)
        & (x < geometry.x_max_m)
        & (y >= geometry.y_min_m)
        & (y < geometry.y_max_m)
        & (z >= geometry.z_min_m)
        & (z <= geometry.z_max_m)
    )
    if not np.any(keep):
        return bev
    x = x[keep]
    y = y[keep]
    z = z[keep]

    rows = np.floor(
        (geometry.x_max_m - x) / geometry.resolution_m
    ).astype(np.int32)
    columns = np.floor(
        (geometry.y_max_m - y) / geometry.resolution_m
    ).astype(np.int32)
    rows = np.clip(rows, 0, height - 1)
    columns = np.clip(columns, 0, width - 1)
    flat = rows * width + columns

    counts = np.zeros(height * width, dtype=np.uint16)
    np.add.at(counts, flat, 1)
    minimum = np.full(height * width, np.inf, dtype=np.float32)
    maximum = np.full(height * width, -np.inf, dtype=np.float32)
    np.minimum.at(minimum, flat, z)
    np.maximum.at(maximum, flat, z)
    occupied = counts > 0

    bev[0] = occupied.reshape((height, width)).astype(np.float32)
    density = np.log1p(counts.astype(np.float32)) / math.log(32.0)
    bev[1] = np.clip(density, 0.0, 1.0).reshape((height, width))
    maximum[~occupied] = 0.0
    minimum[~occupied] = 0.0
    bev[2] = maximum.reshape((height, width))
    bev[3] = (maximum - minimum).reshape((height, width))
    return bev


def occupancy_grid_to_vehicle_bev(
    grid,
    grid_resolution,
    grid_origin_x,
    grid_origin_y,
    grid_origin_yaw,
    vehicle_x,
    vehicle_y,
    vehicle_yaw,
    geometry,
):
    """Nearest-neighbour sample an OccupancyGrid into the LiDAR BEV frame."""
    source = np.asarray(grid, dtype=np.int16)
    if source.ndim != 2:
        raise ValueError('grid must be a 2-D array')
    if grid_resolution <= 0.0:
        raise ValueError('grid_resolution must be positive')

    rows = np.arange(geometry.height, dtype=np.float32)
    columns = np.arange(geometry.width, dtype=np.float32)
    forward = geometry.x_max_m - (
        rows + 0.5
    ) * geometry.resolution_m
    left = geometry.y_max_m - (
        columns + 0.5
    ) * geometry.resolution_m
    forward_grid, left_grid = np.meshgrid(forward, left, indexing='ij')

    vehicle_cosine = math.cos(float(vehicle_yaw))
    vehicle_sine = math.sin(float(vehicle_yaw))
    world_x = (
        float(vehicle_x)
        + vehicle_cosine * forward_grid
        - vehicle_sine * left_grid
    )
    world_y = (
        float(vehicle_y)
        + vehicle_sine * forward_grid
        + vehicle_cosine * left_grid
    )

    dx = world_x - float(grid_origin_x)
    dy = world_y - float(grid_origin_y)
    origin_cosine = math.cos(float(grid_origin_yaw))
    origin_sine = math.sin(float(grid_origin_yaw))
    grid_x = origin_cosine * dx + origin_sine * dy
    grid_y = -origin_sine * dx + origin_cosine * dy
    source_columns = np.floor(grid_x / grid_resolution).astype(np.int32)
    source_rows = np.floor(grid_y / grid_resolution).astype(np.int32)

    output = np.full(
        (geometry.height, geometry.width), -1, dtype=np.int8
    )
    valid = (
        (source_rows >= 0)
        & (source_rows < source.shape[0])
        & (source_columns >= 0)
        & (source_columns < source.shape[1])
    )
    output[valid] = source[
        source_rows[valid], source_columns[valid]
    ].astype(np.int8)
    return output
