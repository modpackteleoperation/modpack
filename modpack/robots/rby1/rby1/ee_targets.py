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
class EETargets:
    """Thread-safe shared end-effector targets and current qpos snapshot for IK."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    duration: float = 0.0
    timestamp: float = 0.0

    left_pos_start: Optional[np.ndarray] = None
    left_quat_start: Optional[np.ndarray] = None
    left_width_start: Optional[float] = None
    left_pos: Optional[np.ndarray] = None
    left_quat: Optional[np.ndarray] = None
    left_width: Optional[float] = None

    right_pos_start: Optional[np.ndarray] = None
    right_quat_start: Optional[np.ndarray] = None
    right_width_start: Optional[float] = None
    right_pos: Optional[np.ndarray] = None
    right_quat: Optional[np.ndarray] = None
    right_width: Optional[float] = None

    head_pos_start: Optional[np.ndarray] = None
    head_quat_start: Optional[np.ndarray] = None
    head_pos: Optional[np.ndarray] = None
    head_quat: Optional[np.ndarray] = None

    def set_targets(
        self,
        left_pos: np.ndarray,
        left_quat: np.ndarray,
        right_pos: np.ndarray,
        right_quat: np.ndarray,
        left_width: Optional[float] = None,
        right_width: Optional[float] = None,
        head_pos: Optional[np.ndarray] = None,
        head_quat: Optional[np.ndarray] = None,
        duration: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        with self.lock:
            now = time.monotonic() if timestamp is None else float(timestamp)

            # Store previous targets for interpolation
            prev_left_pos = self.left_pos.copy() if self.left_pos is not None else None
            prev_left_quat = self.left_quat.copy() if self.left_quat is not None else None
            prev_left_width = self.left_width

            prev_right_pos = self.right_pos.copy() if self.right_pos is not None else None
            prev_right_quat = self.right_quat.copy() if self.right_quat is not None else None
            prev_right_width = self.right_width

            prev_head_pos = self.head_pos.copy() if self.head_pos is not None else None
            prev_head_quat = self.head_quat.copy() if self.head_quat is not None else None

            self.left_pos_start = prev_left_pos if prev_left_pos is not None else left_pos.copy()
            self.left_quat_start = prev_left_quat if prev_left_quat is not None else left_quat.copy()
            self.left_width_start = prev_left_width if prev_left_width is not None else (None if left_width is None else float(left_width))

            self.right_pos_start = prev_right_pos if prev_right_pos is not None else right_pos.copy()
            self.right_quat_start = prev_right_quat if prev_right_quat is not None else right_quat.copy()
            self.right_width_start = prev_right_width if prev_right_width is not None else (None if right_width is None else float(right_width))

            self.head_pos_start = prev_head_pos if prev_head_pos is not None else (None if head_pos is None else head_pos.copy())
            self.head_quat_start = prev_head_quat if prev_head_quat is not None else (None if head_quat is None else head_quat.copy())

            # Set new targets
            self.left_pos = left_pos.copy()
            self.left_quat = left_quat.copy()
            self.right_pos = right_pos.copy()
            self.right_quat = right_quat.copy()
            self.left_width = None if left_width is None else float(left_width)
            self.right_width = None if right_width is None else float(right_width)
            self.head_pos = None if head_pos is None else head_pos.copy()
            self.head_quat = None if head_quat is None else head_quat.copy()

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

            lt_p = lerp_value(self.left_pos_start, self.left_pos, alpha)
            lt_q = slerp_quaternion(self.left_quat_start, self.left_quat, alpha)
            lw = lerp_value(self.left_width_start, self.left_width, alpha)

            rt_p = lerp_value(self.right_pos_start, self.right_pos, alpha)
            rt_q = slerp_quaternion(self.right_quat_start, self.right_quat, alpha)
            rw = lerp_value(self.right_width_start, self.right_width, alpha)

            hp = lerp_value(self.head_pos_start, self.head_pos, alpha)
            hq = slerp_quaternion(self.head_quat_start, self.head_quat, alpha)

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
            lt_p = None if self.left_pos is None else self.left_pos.copy()
            lt_q = None if self.left_quat is None else self.left_quat.copy()
            lw = self.left_width

            rt_p = None if self.right_pos is None else self.right_pos.copy()
            rt_q = None if self.right_quat is None else self.right_quat.copy()
            rw = self.right_width

            hp = None if self.head_pos is None else self.head_pos.copy()
            hq = None if self.head_quat is None else self.head_quat.copy()

        return lt_p, lt_q, lw, rt_p, rt_q, rw, hp, hq


__all__ = ["EETargets", "_normalize_quaternion"]
