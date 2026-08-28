"""Pure NumPy helpers for rolling 3D terrain mapping."""

from dataclasses import dataclass
import math
from typing import Dict, Tuple

import numpy as np


def quaternion_rotation_matrix(x_value, y_value, z_value, w_value):
    """Return a 3x3 rotation matrix for a normalized ROS quaternion."""
    quaternion = np.asarray(
        [x_value, y_value, z_value, w_value], dtype=np.float64
    )
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError('quaternion must be finite and non-zero')
    x_value, y_value, z_value, w_value = quaternion / norm
    return np.asarray([
        [
            1.0 - 2.0 * (y_value ** 2 + z_value ** 2),
            2.0 * (x_value * y_value - z_value * w_value),
            2.0 * (x_value * z_value + y_value * w_value),
        ],
        [
            2.0 * (x_value * y_value + z_value * w_value),
            1.0 - 2.0 * (x_value ** 2 + z_value ** 2),
            2.0 * (y_value * z_value - x_value * w_value),
        ],
        [
            2.0 * (x_value * z_value - y_value * w_value),
            2.0 * (y_value * z_value + x_value * w_value),
            1.0 - 2.0 * (x_value ** 2 + y_value ** 2),
        ],
    ])


def transform_xyzi(points, rotation, translation):
    """Transform Nx4 XYZI points while preserving intensity."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError('points must have shape (N, 4)')
    transformed = np.empty_like(points)
    transformed[:, :3] = (
        np.matmul(points[:, :3], np.asarray(rotation).T)
        + np.asarray(translation)
    )
    transformed[:, 3] = points[:, 3]
    return transformed


def filter_sensor_points(
    points,
    minimum_range_m,
    maximum_range_m,
    ego_min_x_m,
    ego_max_x_m,
    ego_half_width_m,
):
    """Remove invalid/range-limited points and returns from the ego vehicle."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError('points must have shape (N, 4)')
    if minimum_range_m < 0.0 or maximum_range_m <= minimum_range_m:
        raise ValueError('invalid range limits')
    finite = np.all(np.isfinite(points), axis=1)
    squared_range = np.sum(points[:, :3] ** 2, axis=1)
    in_range = (
        (squared_range >= minimum_range_m ** 2)
        & (squared_range <= maximum_range_m ** 2)
    )
    on_vehicle = (
        (points[:, 0] >= ego_min_x_m)
        & (points[:, 0] <= ego_max_x_m)
        & (np.abs(points[:, 1]) <= ego_half_width_m)
    )
    return points[finite & in_range & ~on_vehicle]


@dataclass
class VoxelObservation:
    """One fused voxel representative."""

    mean: np.ndarray
    count: int
    last_update_s: float


class RollingVoxelMap:
    """Bounded voxel dictionary with running-mean fusion."""

    def __init__(self, voxel_size_m):
        if voxel_size_m <= 0.0:
            raise ValueError('voxel_size_m must be positive')
        self.voxel_size_m = float(voxel_size_m)
        self._voxels: Dict[Tuple[int, int, int], VoxelObservation] = {}

    def __len__(self):
        return len(self._voxels)

    def clear(self):
        """Remove every voxel."""
        self._voxels.clear()

    def update(self, points, timestamp_s):
        """Voxelize one cloud and fuse it into the persistent dictionary."""
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 4:
            raise ValueError('points must have shape (N, 4)')
        if points.shape[0] == 0:
            return 0
        if not math.isfinite(timestamp_s):
            raise ValueError('timestamp_s must be finite')

        keys = np.floor(
            points[:, :3] / self.voxel_size_m
        ).astype(np.int32)
        unique_keys, inverse, counts = np.unique(
            keys, axis=0, return_inverse=True, return_counts=True
        )
        sums = np.column_stack([
            np.bincount(
                inverse, weights=points[:, column],
                minlength=unique_keys.shape[0],
            )
            for column in range(4)
        ])
        means = sums / counts[:, None]

        for key_array, mean, count in zip(unique_keys, means, counts):
            key = tuple(int(value) for value in key_array)
            previous = self._voxels.get(key)
            if previous is None:
                self._voxels[key] = VoxelObservation(
                    mean.astype(np.float32),
                    int(count),
                    float(timestamp_s),
                )
                continue
            combined_count = previous.count + int(count)
            combined_mean = (
                previous.mean * previous.count + mean * int(count)
            ) / combined_count
            self._voxels[key] = VoxelObservation(
                combined_mean.astype(np.float32),
                combined_count,
                float(timestamp_s),
            )
        return unique_keys.shape[0]

    def prune(self, center_xy, radius_m, timestamp_s, maximum_age_s=0.0):
        """Drop voxels outside the rolling radius or optional age limit."""
        if radius_m <= 0.0:
            raise ValueError('radius_m must be positive')
        center_x, center_y = (float(value) for value in center_xy)
        radius_squared = radius_m ** 2
        stale_keys = []
        for key, observation in self._voxels.items():
            voxel_x = (key[0] + 0.5) * self.voxel_size_m
            voxel_y = (key[1] + 0.5) * self.voxel_size_m
            outside = (
                (voxel_x - center_x) ** 2
                + (voxel_y - center_y) ** 2
                > radius_squared
            )
            too_old = (
                maximum_age_s > 0.0
                and timestamp_s - observation.last_update_s > maximum_age_s
            )
            if outside or too_old:
                stale_keys.append(key)
        for key in stale_keys:
            del self._voxels[key]
        return len(stale_keys)

    def points(self):
        """Return fused XYZI representatives."""
        if not self._voxels:
            return np.empty((0, 4), dtype=np.float32)
        return np.stack(
            [observation.mean for observation in self._voxels.values()]
        ).astype(np.float32, copy=False)


@dataclass(frozen=True)
class TerrainLayers:
    """Rolling grid products derived from fused 3D points."""

    origin_x_m: float
    origin_y_m: float
    resolution_m: float
    elevation_m: np.ndarray
    slope_deg: np.ndarray
    step_height_m: np.ndarray
    traversability_cost: np.ndarray


def _normalized_cost(values, safe_value, lethal_value):
    if lethal_value <= safe_value:
        raise ValueError('lethal threshold must exceed safe threshold')
    return np.clip(
        (values - safe_value) / (lethal_value - safe_value) * 100.0,
        0.0,
        100.0,
    )


def build_terrain_layers(
    points,
    center_xy,
    map_size_m,
    resolution_m,
    minimum_points_per_cell,
    safe_slope_deg,
    lethal_slope_deg,
    safe_step_height_m,
    lethal_step_height_m,
):
    """Project a fused 3D cloud into elevation and traversability layers."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError('points must have shape (N, 4)')
    if map_size_m <= 0.0 or resolution_m <= 0.0:
        raise ValueError('map size and resolution must be positive')
    if minimum_points_per_cell < 1:
        raise ValueError('minimum_points_per_cell must be positive')

    cell_count = int(math.ceil(map_size_m / resolution_m))
    center_x, center_y = (float(value) for value in center_xy)
    origin_x = (
        math.floor(
            (center_x - 0.5 * cell_count * resolution_m) / resolution_m
        ) * resolution_m
    )
    origin_y = (
        math.floor(
            (center_y - 0.5 * cell_count * resolution_m) / resolution_m
        ) * resolution_m
    )

    elevation_min = np.full(
        (cell_count, cell_count), np.inf, dtype=np.float32
    )
    elevation_max = np.full(
        (cell_count, cell_count), -np.inf, dtype=np.float32
    )
    counts = np.zeros((cell_count, cell_count), dtype=np.int32)

    if points.shape[0]:
        x_indices = np.floor(
            (points[:, 0] - origin_x) / resolution_m
        ).astype(np.int32)
        y_indices = np.floor(
            (points[:, 1] - origin_y) / resolution_m
        ).astype(np.int32)
        inside = (
            (x_indices >= 0)
            & (x_indices < cell_count)
            & (y_indices >= 0)
            & (y_indices < cell_count)
            & np.isfinite(points[:, 2])
        )
        x_indices = x_indices[inside]
        y_indices = y_indices[inside]
        heights = points[inside, 2]
        np.minimum.at(elevation_min, (y_indices, x_indices), heights)
        np.maximum.at(elevation_max, (y_indices, x_indices), heights)
        np.add.at(counts, (y_indices, x_indices), 1)

    observed = counts >= minimum_points_per_cell
    elevation = np.full_like(elevation_min, np.nan)
    elevation[observed] = elevation_min[observed]
    vertical_span = np.zeros_like(elevation_min)
    vertical_span[observed] = (
        elevation_max[observed] - elevation_min[observed]
    )
    maximum_neighbor_delta = np.zeros_like(elevation_min)
    maximum_neighbor_slope = np.zeros_like(elevation_min)
    neighbor_count = np.zeros_like(counts)

    for row_offset, column_offset in ((0, 1), (1, 0), (1, 1), (1, -1)):
        if column_offset >= 0:
            first_columns = slice(0, cell_count - column_offset)
            second_columns = slice(column_offset, cell_count)
        else:
            first_columns = slice(-column_offset, cell_count)
            second_columns = slice(0, cell_count + column_offset)
        first_rows = slice(0, cell_count - row_offset)
        second_rows = slice(row_offset, cell_count)

        first_valid = observed[first_rows, first_columns]
        second_valid = observed[second_rows, second_columns]
        pair_valid = first_valid & second_valid
        delta = np.abs(
            elevation[first_rows, first_columns]
            - elevation[second_rows, second_columns]
        )
        delta[~pair_valid] = 0.0
        distance = resolution_m * math.hypot(
            row_offset, column_offset
        )
        slope = np.degrees(np.arctan2(delta, distance))

        first_delta = maximum_neighbor_delta[first_rows, first_columns]
        second_delta = maximum_neighbor_delta[second_rows, second_columns]
        np.maximum(first_delta, delta, out=first_delta)
        np.maximum(second_delta, delta, out=second_delta)
        first_slope = maximum_neighbor_slope[first_rows, first_columns]
        second_slope = maximum_neighbor_slope[second_rows, second_columns]
        np.maximum(first_slope, slope, out=first_slope)
        np.maximum(second_slope, slope, out=second_slope)
        neighbor_count[first_rows, first_columns] += pair_valid
        neighbor_count[second_rows, second_columns] += pair_valid

    step_height = np.maximum(vertical_span, maximum_neighbor_delta)
    slope_deg = np.full_like(elevation_min, np.nan)
    slope_known = observed & (neighbor_count > 0)
    slope_deg[slope_known] = maximum_neighbor_slope[slope_known]

    cost = np.full((cell_count, cell_count), -1, dtype=np.int8)
    known = slope_known
    slope_cost = _normalized_cost(
        maximum_neighbor_slope, safe_slope_deg, lethal_slope_deg
    )
    step_cost = _normalized_cost(
        step_height, safe_step_height_m, lethal_step_height_m
    )
    combined = np.maximum(slope_cost, step_cost)
    cost[known] = np.rint(combined[known]).astype(np.int8)

    return TerrainLayers(
        origin_x,
        origin_y,
        float(resolution_m),
        elevation,
        slope_deg,
        step_height,
        cost,
    )
