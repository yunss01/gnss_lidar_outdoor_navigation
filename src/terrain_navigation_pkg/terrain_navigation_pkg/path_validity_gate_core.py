"""Debounced fail-safe state for blocking commands on an invalid path."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PathValidityGateSnapshot:
    """One deterministic view of the path-validity gate state."""

    blocked: bool
    reason: str
    validity: object
    validity_age_s: float
    invalid_duration_s: float
    valid_duration_s: float


class PathValidityGateState:
    """Debounce path validity and fail closed when updates become stale."""

    def __init__(
        self,
        invalid_hold_s,
        valid_release_s,
        validity_timeout_s,
    ):
        values = (invalid_hold_s, valid_release_s, validity_timeout_s)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError('path-validity timing values must be finite')
        if invalid_hold_s < 0.0:
            raise ValueError('invalid_hold_s cannot be negative')
        if valid_release_s < 0.0:
            raise ValueError('valid_release_s cannot be negative')
        if validity_timeout_s <= 0.0:
            raise ValueError('validity_timeout_s must be positive')

        self.invalid_hold_s = float(invalid_hold_s)
        self.valid_release_s = float(valid_release_s)
        self.validity_timeout_s = float(validity_timeout_s)
        self.blocked = True
        self.latest_validity = None
        self.latest_update_s = None
        self.invalid_since_s = None
        self.valid_since_s = None

    @staticmethod
    def _elapsed(now_s, since_s):
        if since_s is None:
            return 0.0
        return max(0.0, float(now_s) - float(since_s))

    def update(self, valid, now_s):
        """Accept one validity observation and return the current snapshot."""
        now_s = float(now_s)
        if not math.isfinite(now_s):
            raise ValueError('now_s must be finite')
        valid = bool(valid)
        self.latest_validity = valid
        self.latest_update_s = now_s
        if valid:
            self.invalid_since_s = None
            if self.valid_since_s is None:
                self.valid_since_s = now_s
        else:
            self.valid_since_s = None
            if self.invalid_since_s is None:
                self.invalid_since_s = now_s
        return self.evaluate(now_s)

    def evaluate(self, now_s):
        """Advance debounce timers and return a fail-safe gate snapshot."""
        now_s = float(now_s)
        if not math.isfinite(now_s):
            raise ValueError('now_s must be finite')

        if self.latest_update_s is None:
            self.blocked = True
            return PathValidityGateSnapshot(
                True, 'waiting_for_path_validity', None, math.inf, 0.0, 0.0
            )

        validity_age_s = self._elapsed(now_s, self.latest_update_s)
        invalid_duration_s = self._elapsed(now_s, self.invalid_since_s)
        valid_duration_s = self._elapsed(now_s, self.valid_since_s)
        if validity_age_s > self.validity_timeout_s:
            self.blocked = True
            return PathValidityGateSnapshot(
                True,
                'path_validity_stale',
                self.latest_validity,
                validity_age_s,
                invalid_duration_s,
                valid_duration_s,
            )

        if self.latest_validity:
            if self.blocked and valid_duration_s >= self.valid_release_s:
                self.blocked = False
            reason = 'path_valid' if not self.blocked else 'validating_path'
        else:
            if (
                not self.blocked
                and invalid_duration_s >= self.invalid_hold_s
            ):
                self.blocked = True
            reason = 'path_invalid' if self.blocked else 'invalid_pending'

        return PathValidityGateSnapshot(
            self.blocked,
            reason,
            self.latest_validity,
            validity_age_s,
            invalid_duration_s,
            valid_duration_s,
        )
