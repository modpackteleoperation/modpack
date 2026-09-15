from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import numpy as np
from rby1.pose_utils import lerp_value, slerp_quaternion, normalize_quaternion


def _normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    """Backward-compatible alias for pose_utils.normalize_quaternion."""
    return normalize_quaternion(quat)


@dataclass
class JointTargets:
    """Thread-safe shared joint targets and current qpos snapshot for IK."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    duration: float = 0.0
    timestamp: float = 0.0

    left_qpos_start: Optional[np.ndarray] = None
    left_width_start: Optional[float] = None
    left_qpos: Optional[np.ndarray] = None
    left_width: Optional[float] = None

    right_qpos_start: Optional[np.ndarray] = None
    right_width_start: Optional[float] = None
    right_qpos: Optional[np.ndarray] = None
    right_width: Optional[float] = None

    head_qpos_start: Optional[np.ndarray] = None
    head_qpos: Optional[np.ndarray] = None  # [pan, tilt] radians (head_0, head_1)

    def set_targets(
        self,
        left_qpos: np.ndarray,
        right_qpos: np.ndarray,
        left_width: Optional[float] = None,
        right_width: Optional[float] = None,
        head_qpos: Optional[np.ndarray] = None,
        duration: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        with self.lock:
            now = time.monotonic() if timestamp is None else float(timestamp)

            # Store previous targets for interpolation
            prev_left_qpos = self.left_qpos.copy() if self.left_qpos is not None else None
            prev_left_width = self.left_width

            prev_right_qpos = self.right_qpos.copy() if self.right_qpos is not None else None
            prev_right_width = self.right_width

            prev_head_qpos = self.head_qpos.copy() if self.head_qpos is not None else None

            self.left_qpos_start = prev_left_qpos if prev_left_qpos is not None else left_qpos.copy()
            self.left_width_start = prev_left_width if prev_left_width is not None else (None if left_width is None else float(left_width))

            self.right_qpos_start = prev_right_qpos if prev_right_qpos is not None else right_qpos.copy()
            self.right_width_start = prev_right_width if prev_right_width is not None else (None if right_width is None else float(right_width))

            self.head_qpos_start = prev_head_qpos if prev_head_qpos is not None else (head_qpos.copy() if head_qpos is not None else None)

            # Set new targets
            self.left_qpos = left_qpos.copy()
            self.right_qpos = right_qpos.copy()
            self.left_width = None if left_width is None else float(left_width)
            self.right_width = None if right_width is None else float(right_width)
            if head_qpos is not None:
                self.head_qpos = head_qpos.copy()

            duration_value = 0.0 if duration is None else max(0.0, float(duration))
            self.duration = duration_value
            self.timestamp = now

    def get_for_ik(self, use_interpolation: bool = False) -> Tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[float],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[float],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        if not use_interpolation:
            return self.get_target()

        # Return the linearly interpolated targets based on elapsed time since setting
        current_time = time.monotonic()
        with self.lock:
            duration = max(self.duration, 0.0)
            elapsed = max(0.0, current_time - self.timestamp)
            alpha = min(1.0, elapsed / duration) if duration > 0.0 else 1.0

            lt_p = lerp_value(self.left_qpos_start, self.left_qpos, alpha)
            lt_q = None
            lw = lerp_value(self.left_width_start, self.left_width, alpha)

            rt_p = lerp_value(self.right_qpos_start, self.right_qpos, alpha)
            rt_q = None
            rw = lerp_value(self.right_width_start, self.right_width, alpha)

            hp = lerp_value(self.head_qpos_start, self.head_qpos, alpha)
            hq = None

        return lt_p, lt_q, lw, rt_p, rt_q, rw, hp, hq

    def get_target(self) -> Tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[float],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[float],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        with self.lock:
            lt_p = None if self.left_qpos is None else self.left_qpos.copy()
            lt_q = None
            lw = self.left_width

            rt_p = None if self.right_qpos is None else self.right_qpos.copy()
            rt_q = None
            rw = self.right_width

            hp = None if self.head_qpos is None else self.head_qpos.copy()
            hq = None

        return lt_p, lt_q, lw, rt_p, rt_q, rw, hp, hq


__all__ = ["JointTargets", "_normalize_quaternion"]
