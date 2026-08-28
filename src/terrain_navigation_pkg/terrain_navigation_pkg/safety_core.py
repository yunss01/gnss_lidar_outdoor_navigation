"""Pure geometry and state helpers for LiDAR emergency stopping."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ForwardObstacleObservation:
    """Point counts inside the stop and hysteresis corridors."""

    stop_point_count: int
    clear_point_count: int
    nearest_distance_m: float


def observe_forward_corridor(
    xyz_points,
    minimum_x_m: float,
    stop_distance_m: float,
    clear_distance_m: float,
    half_width_m: float,
    minimum_z_m: float,
    maximum_z_m: float,
    center_y_m: float = 0.0,
) -> ForwardObstacleObservation:
    """Count obstacle returns in a LiDAR-frame forward corridor.

    The expected frame follows ROS FLU: x forward, y left, z up.
    """
    points = np.asarray(xyz_points)
    if points.size == 0:
        points = np.empty((0, 3), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError('xyz_points must have shape (N, >=3)')
    limits = (
        minimum_x_m,
        stop_distance_m,
        clear_distance_m,
        half_width_m,
        minimum_z_m,
        maximum_z_m,
        center_y_m,
    )
    if not all(math.isfinite(value) for value in limits):
        raise ValueError('corridor limits must be finite')
    if minimum_x_m < 0.0:
        raise ValueError('minimum_x_m must be non-negative')
    if not minimum_x_m < stop_distance_m < clear_distance_m:
        raise ValueError(
            'distances must satisfy minimum_x < stop < clear'
        )
    if half_width_m <= 0.0:
        raise ValueError('half_width_m must be positive')
    if minimum_z_m >= maximum_z_m:
        raise ValueError('minimum_z_m must be below maximum_z_m')
    if points.shape[0] == 0:
        return ForwardObstacleObservation(0, 0, math.inf)

    xyz = points[:, :3].astype(np.float32, copy=False)
    finite = np.isfinite(xyz).all(axis=1)
    base_mask = (
        finite
        & (xyz[:, 0] >= minimum_x_m)
        & (np.abs(xyz[:, 1] - center_y_m) <= half_width_m)
        & (xyz[:, 2] >= minimum_z_m)
        & (xyz[:, 2] <= maximum_z_m)
    )
    stop_mask = base_mask & (xyz[:, 0] <= stop_distance_m)
    clear_mask = base_mask & (xyz[:, 0] <= clear_distance_m)
    stop_count = int(np.count_nonzero(stop_mask))
    clear_count = int(np.count_nonzero(clear_mask))
    nearest = (
        float(np.min(xyz[clear_mask, 0]))
        if clear_count else math.inf
    )
    return ForwardObstacleObservation(stop_count, clear_count, nearest)


class EmergencyStopHysteresis:
    """Stop immediately, but require repeated clear scans to resume."""

    def __init__(self, minimum_points: int, clear_required_scans: int):
        if minimum_points < 1:
            raise ValueError('minimum_points must be at least one')
        if clear_required_scans < 1:
            raise ValueError('clear_required_scans must be at least one')
        self.minimum_points = int(minimum_points)
        self.clear_required_scans = int(clear_required_scans)
        self.stopped = False
        self.clear_scan_count = 0

    def update(
        self,
        observation: ForwardObstacleObservation,
        allow_clear_progress: bool = True,
    ) -> bool:
        """Update and return the obstacle-stop state."""
        if not self.stopped:
            if observation.stop_point_count >= self.minimum_points:
                self.stopped = True
                self.clear_scan_count = 0
            return self.stopped

        if observation.clear_point_count >= self.minimum_points:
            self.clear_scan_count = 0
            return True

        # A newly received steering command can be checked immediately against
        # the latest cloud, but it must not make the stop hysteresis count the
        # same LiDAR scan more than once.  Only a genuinely new clear scan may
        # advance the release counter.
        if not allow_clear_progress:
            return True

        self.clear_scan_count += 1
        if self.clear_scan_count >= self.clear_required_scans:
            self.stopped = False
            self.clear_scan_count = 0
        return self.stopped


class CollisionStopLatch:
    """Hold a collision stop until an explicit reset is received."""

    def __init__(self, minimum_intensity: float = 1.0):
        if not math.isfinite(minimum_intensity) or minimum_intensity < 0.0:
            raise ValueError('minimum collision intensity must be finite')
        self.minimum_intensity = float(minimum_intensity)
        self.latched = False

    def observe(self, intensity: float) -> bool:
        """Latch on a finite collision event at or above the threshold."""
        if not math.isfinite(intensity):
            return self.latched
        if intensity >= self.minimum_intensity:
            self.latched = True
        return self.latched

    def reset(self) -> None:
        self.latched = False


class RecoveryCurvatureHold:
    """Keep a validated avoidance curve through command chatter."""

    def __init__(
        self,
        minimum_curvature: float,
        candidate_max_age_s: float,
        active_max_duration_s: float,
        smoothing_alpha: float,
    ):
        values = (
            minimum_curvature,
            candidate_max_age_s,
            active_max_duration_s,
            smoothing_alpha,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError('recovery hold parameters must be finite')
        if minimum_curvature < 0.0:
            raise ValueError('minimum recovery curvature is invalid')
        if candidate_max_age_s <= 0.0 or active_max_duration_s <= 0.0:
            raise ValueError('recovery hold durations must be positive')
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError('recovery smoothing alpha must be in (0, 1]')
        self.minimum_curvature = float(minimum_curvature)
        self.candidate_max_age_s = float(candidate_max_age_s)
        self.active_max_duration_s = float(active_max_duration_s)
        self.smoothing_alpha = float(smoothing_alpha)
        self.candidate_curvature = None
        self.candidate_time = None
        self.active_curvature = None
        self.active_start_time = None
        self.active_refresh_time = None

    @staticmethod
    def _same_direction(first, second):
        return first * second > 0.0

    def observe_safe(self, curvature: float, now: float) -> None:
        """Remember a safe non-straight candidate from the local controller."""
        if not math.isfinite(curvature) or not math.isfinite(now):
            return
        if abs(curvature) < self.minimum_curvature:
            return
        curvature = float(curvature)
        now = float(now)
        if (
            self.active_curvature is not None
            and not self._same_direction(self.active_curvature, curvature)
        ):
            # A recovery side is committed until the stop state clears. An
            # opposite MPPI sample must not replace it and recreate chatter.
            return
        self.candidate_curvature = curvature
        self.candidate_time = now
        if (
            self.active_curvature is not None
            and self._same_direction(self.active_curvature, curvature)
        ):
            alpha = self.smoothing_alpha
            self.active_curvature = (
                (1.0 - alpha) * self.active_curvature + alpha * curvature
            )
            self.active_refresh_time = now

    def activate_or_current(self, now: float):
        """Return active curvature, activating a recent candidate if needed."""
        if not math.isfinite(now):
            return None
        now = float(now)
        if self.active_curvature is not None:
            expired = (
                now - self.active_start_time > self.active_max_duration_s
                or now - self.active_refresh_time > self.candidate_max_age_s
            )
            if expired:
                self.reset_active()
                return None
            return self.active_curvature
        if (
            self.candidate_curvature is None
            or now - self.candidate_time > self.candidate_max_age_s
        ):
            return None
        self.active_curvature = self.candidate_curvature
        self.active_start_time = now
        self.active_refresh_time = self.candidate_time
        return self.active_curvature

    def invalidate(self) -> None:
        """Discard a held curve that is no longer collision-free."""
        self.candidate_curvature = None
        self.candidate_time = None
        self.reset_active()

    def reset_active(self) -> None:
        self.active_curvature = None
        self.active_start_time = None
        self.active_refresh_time = None
