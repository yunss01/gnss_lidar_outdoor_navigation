"""Ground-relative low-obstacle extraction for roof-mounted LiDAR scans."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroundObstacleResult:
    """Obstacle mask plus diagnostics for one point-cloud frame."""

    obstacle_mask: np.ndarray
    ground_plane: np.ndarray | None
    ground_point_count: int
    low_obstacle_point_count: int


def _validate_points(points_xyz):
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError('points_xyz must have shape (N, 3+)')
    return points[:, :3]


def extract_ground_relative_obstacles(
    points_xyz,
    *,
    fixed_minimum_z_m,
    fixed_maximum_z_m,
    enabled=True,
    ground_fit_minimum_range_m=2.0,
    ground_fit_maximum_range_m=20.0,
    ground_fit_minimum_z_m=-2.5,
    ground_fit_maximum_z_m=-0.8,
    ground_seed_quantile=0.35,
    ground_inlier_tolerance_m=0.06,
    minimum_ground_points=80,
    low_obstacle_minimum_height_m=0.07,
    low_obstacle_maximum_height_m=0.45,
    local_ground_enabled=False,
    local_ground_resolution_m=0.75,
    local_ground_radius_m=1.50,
    local_ground_quantile=0.25,
    local_ground_plane_enabled=True,
    local_ground_plane_residual_tolerance_m=0.055,
):
    """Return points representing ordinary or ground-relative obstacles.

    A roof LiDAR sees the road around ``z=-1.8 m`` on the full-size CARLA
    Lincoln.  The legacy fixed ``z >= -1.4 m`` crop therefore removes both
    the road and a 10--20 cm curb.  This function retains the old crop for
    ordinary objects, estimates the local road plane, and additionally keeps
    points sufficiently above that plane.  It never treats points below the
    fitted plane as obstacles.

    If a stable plane cannot be fitted, only the legacy fixed-height mask is
    returned.  This fail-closed-to-the-old-behaviour fallback avoids turning
    an uncertain fit into a cloud full of false obstacles.
    """

    points = _validate_points(points_xyz)
    if fixed_minimum_z_m >= fixed_maximum_z_m:
        raise ValueError('fixed z bounds are invalid')
    if not 0.0 <= ground_fit_minimum_range_m < ground_fit_maximum_range_m:
        raise ValueError('ground fit range bounds are invalid')
    if ground_fit_minimum_z_m >= ground_fit_maximum_z_m:
        raise ValueError('ground fit z bounds are invalid')
    if not 0.0 < ground_seed_quantile < 1.0:
        raise ValueError('ground_seed_quantile must be in (0, 1)')
    if ground_inlier_tolerance_m <= 0.0:
        raise ValueError('ground_inlier_tolerance_m must be positive')
    if minimum_ground_points < 3:
        raise ValueError('minimum_ground_points must be at least 3')
    if not 0.0 < low_obstacle_minimum_height_m < low_obstacle_maximum_height_m:
        raise ValueError('low-obstacle height bounds are invalid')
    if local_ground_resolution_m <= 0.0:
        raise ValueError('local_ground_resolution_m must be positive')
    if local_ground_radius_m < local_ground_resolution_m:
        raise ValueError(
            'local_ground_radius_m must be at least one grid resolution'
        )
    if not 0.0 < local_ground_quantile < 1.0:
        raise ValueError('local_ground_quantile must be in (0, 1)')
    if local_ground_plane_residual_tolerance_m <= 0.0:
        raise ValueError(
            'local_ground_plane_residual_tolerance_m must be positive'
        )

    if points.shape[0] == 0:
        return GroundObstacleResult(
            obstacle_mask=np.zeros(0, dtype=bool),
            ground_plane=None,
            ground_point_count=0,
            low_obstacle_point_count=0,
        )

    x, y, z = points.T
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    fixed_mask = (
        finite
        & (z >= float(fixed_minimum_z_m))
        & (z <= float(fixed_maximum_z_m))
    )
    if not enabled:
        return GroundObstacleResult(
            obstacle_mask=fixed_mask,
            ground_plane=None,
            ground_point_count=0,
            low_obstacle_point_count=0,
        )

    range_sq = x * x + y * y
    fit_candidates = (
        finite
        & (range_sq >= float(ground_fit_minimum_range_m) ** 2)
        & (range_sq <= float(ground_fit_maximum_range_m) ** 2)
        & (z >= float(ground_fit_minimum_z_m))
        & (z <= float(ground_fit_maximum_z_m))
    )
    candidate_indices = np.flatnonzero(fit_candidates)
    if candidate_indices.size < int(minimum_ground_points):
        return GroundObstacleResult(
            obstacle_mask=fixed_mask,
            ground_plane=None,
            ground_point_count=0,
            low_obstacle_point_count=0,
        )

    seed_limit = np.quantile(
        z[candidate_indices], float(ground_seed_quantile)
    )
    inliers = fit_candidates & (z <= seed_limit)
    coefficients = None
    for _ in range(4):
        if np.count_nonzero(inliers) < int(minimum_ground_points):
            coefficients = None
            break
        design = np.column_stack(
            (x[inliers], y[inliers], np.ones(np.count_nonzero(inliers)))
        )
        coefficients, *_ = np.linalg.lstsq(design, z[inliers], rcond=None)
        predicted = (
            coefficients[0] * x
            + coefficients[1] * y
            + coefficients[2]
        )
        residual = z - predicted
        inliers = fit_candidates & (
            np.abs(residual) <= float(ground_inlier_tolerance_m)
        )

    ground_point_count = int(np.count_nonzero(inliers))
    if coefficients is None or ground_point_count < int(minimum_ground_points):
        return GroundObstacleResult(
            obstacle_mask=fixed_mask,
            ground_plane=None,
            ground_point_count=ground_point_count,
            low_obstacle_point_count=0,
        )

    # Report and use a plane fitted to the final inlier set, not the inlier
    # set from the preceding refinement iteration.
    design = np.column_stack(
        (x[inliers], y[inliers], np.ones(ground_point_count))
    )
    coefficients, *_ = np.linalg.lstsq(design, z[inliers], rcond=None)

    predicted = coefficients[0] * x + coefficients[1] * y + coefficients[2]
    if local_ground_enabled:
        # A single plane over a 20 m intersection mistakes road crown, shallow
        # drainage slopes and small CARLA mesh seams for 7--15 cm obstacles.
        # The lower quartile in each cell represents road returns.  Fit a
        # small plane through neighbouring cell estimates so the prediction
        # follows the local road pitch and crown instead of assuming that the
        # whole neighbourhood is horizontal.  Positive residual rejection
        # prevents a curb top from pulling that road plane upwards.  Cells
        # without enough local support do not create extra low-obstacle
        # points; ordinary fixed-height obstacles remain active everywhere.
        resolution = float(local_ground_resolution_m)
        radius = float(local_ground_radius_m)
        grid = np.floor(points[:, :2] / resolution).astype(np.int32)
        candidate_grid = grid[fit_candidates]
        ground_cells, candidate_cell_ids = np.unique(
            candidate_grid, axis=0, return_inverse=True
        )
        candidate_z = z[fit_candidates]
        cell_ground = np.empty(ground_cells.shape[0], dtype=np.float64)
        for cell_id in range(ground_cells.shape[0]):
            cell_ground[cell_id] = float(np.quantile(
                candidate_z[candidate_cell_ids == cell_id],
                float(local_ground_quantile),
            ))
        cell_lookup = {
            tuple(cell): cell_id
            for cell_id, cell in enumerate(ground_cells)
        }
        step_count = int(np.ceil(radius / resolution))
        offsets = [
            (dx, dy)
            for dx in range(-step_count, step_count + 1)
            for dy in range(-step_count, step_count + 1)
            if (dx * resolution) ** 2 + (dy * resolution) ** 2
            <= radius ** 2 + 1.0e-9
        ]
        cell_centres = (
            ground_cells.astype(np.float64) + 0.5
        ) * resolution
        local_models = np.full((ground_cells.shape[0], 3), np.nan)
        local_heights = np.full(ground_cells.shape[0], np.nan)
        for cell_id, cell in enumerate(ground_cells):
            cx, cy = int(cell[0]), int(cell[1])
            neighbour_ids = [
                cell_lookup[(cx + dx, cy + dy)]
                for dx, dy in offsets
                if (cx + dx, cy + dy) in cell_lookup
            ]
            if not neighbour_ids:
                continue
            neighbour_ids = np.asarray(neighbour_ids, dtype=np.int32)
            if local_ground_plane_enabled and neighbour_ids.size >= 4:
                design = np.column_stack((
                    cell_centres[neighbour_ids],
                    np.ones(neighbour_ids.size),
                ))
                keep = np.ones(neighbour_ids.size, dtype=bool)
                local_model = None
                for _ in range(3):
                    if np.count_nonzero(keep) < 4:
                        local_model = None
                        break
                    local_model, *_ = np.linalg.lstsq(
                        design[keep], cell_ground[neighbour_ids][keep],
                        rcond=None,
                    )
                    residual = (
                        cell_ground[neighbour_ids] - design @ local_model
                    )
                    # Retain the locally lower continuous surface.  An abrupt
                    # positive step (curb/sidewalk top) must remain an obstacle
                    # rather than becoming part of the fitted road surface.
                    keep = residual <= float(
                        local_ground_plane_residual_tolerance_m
                    )
                if local_model is not None:
                    local_models[cell_id] = local_model
                    continue
            if not local_ground_plane_enabled:
                local_heights[cell_id] = float(np.median(
                    cell_ground[neighbour_ids]
                ))

        point_cell_ids = np.fromiter(
            (cell_lookup.get(tuple(cell), -1) for cell in grid),
            dtype=np.int32,
            count=grid.shape[0],
        )
        supported = point_cell_ids >= 0
        local_prediction = np.full(points.shape[0], np.nan)
        supported_point_ids = np.flatnonzero(supported)
        supported_cell_ids = point_cell_ids[supported]
        point_models = local_models[supported_cell_ids]
        plane_supported = np.isfinite(point_models[:, 0])
        if np.any(plane_supported):
            point_ids = supported_point_ids[plane_supported]
            models = point_models[plane_supported]
            local_prediction[point_ids] = (
                models[:, 0] * x[point_ids]
                + models[:, 1] * y[point_ids]
                + models[:, 2]
            )
        height_supported = (
            ~plane_supported
            & np.isfinite(local_heights[supported_cell_ids])
        )
        if np.any(height_supported):
            point_ids = supported_point_ids[height_supported]
            local_prediction[point_ids] = local_heights[
                supported_cell_ids[height_supported]
            ]
        # In local mode an unsupported cell must not fall back to the
        # scan-wide plane: that fallback is precisely what turns a curved road
        # boundary into a phantom 7--15 cm obstacle.  NaN produces no *extra*
        # low obstacle, while fixed-height wall/vehicle detection is unchanged.
        predicted = local_prediction
    height_above_ground = z - predicted
    # The fixed crop is expressed in the roof-mounted sensor frame.  On an
    # uphill road, perfectly valid pavement can rise above fixed_minimum_z_m
    # and the legacy ``fixed_mask | low_obstacle_mask`` logic then marks it as
    # an obstacle even after the ground model identified it correctly.  Remove
    # only returns that agree with the estimated road surface.  A curb, wall or
    # vehicle remains because its returns sit above this tight inlier band.
    ground_surface_mask = (
        finite
        & np.isfinite(predicted)
        & (np.abs(height_above_ground) <= float(ground_inlier_tolerance_m))
    )
    fixed_mask = fixed_mask & ~ground_surface_mask
    low_obstacle_mask = (
        finite
        & (z < float(fixed_minimum_z_m))
        & (z <= float(fixed_maximum_z_m))
        & (height_above_ground >= float(low_obstacle_minimum_height_m))
        & (height_above_ground <= float(low_obstacle_maximum_height_m))
    )
    return GroundObstacleResult(
        obstacle_mask=fixed_mask | low_obstacle_mask,
        ground_plane=np.asarray(coefficients, dtype=np.float64),
        ground_point_count=ground_point_count,
        low_obstacle_point_count=int(np.count_nonzero(low_obstacle_mask)),
    )
