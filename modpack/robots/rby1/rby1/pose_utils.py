from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np


def lerp_value(
    start: Optional[Union[np.ndarray, float]],
    end: Optional[Union[np.ndarray, float]],
    alpha: float,
):
    if end is None:
        return None
    if start is None:
        if isinstance(end, np.ndarray):
            return end.copy()
        return float(end)
    alpha_clamped = max(0.0, min(1.0, alpha))
    if isinstance(end, np.ndarray):
        return (1.0 - alpha_clamped) * start + alpha_clamped * end
    return float((1.0 - alpha_clamped) * start + alpha_clamped * end)


def normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return quat / norm


def slerp_quaternion(
    start: Optional[np.ndarray],
    end: Optional[np.ndarray],
    alpha: float,
) -> Optional[np.ndarray]:
    if end is None:
        return None
    if start is None:
        return end.copy()

    start_norm = normalize_quaternion(start)
    end_norm = normalize_quaternion(end)

    dot = float(np.dot(start_norm, end_norm))
    if dot < 0.0:
        end_norm = -end_norm
        dot = -dot
    dot = max(-1.0, min(1.0, dot))

    if dot > 0.9995:
        result = start_norm + alpha * (end_norm - start_norm)
        return normalize_quaternion(result)

    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    if sin_theta_0 < 1e-6:
        return end_norm.copy()

    alpha_clamped = max(0.0, min(1.0, alpha))
    theta = theta_0 * alpha_clamped
    sin_theta = math.sin(theta)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    result = s0 * start_norm + s1 * end_norm
    return normalize_quaternion(result)


def quat_wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    """Convert a quaternion from wxyz to xyzw ordering."""
    return np.array([quat[1], quat[2], quat[3], quat[0]], dtype=float)


def quat_xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    """Convert a quaternion from xyzw to wxyz ordering."""
    return np.array([quat[3], quat[0], quat[1], quat[2]], dtype=float)


__all__ = [
    "lerp_value",
    "normalize_quaternion",
    "slerp_quaternion",
    "quat_wxyz_to_xyzw",
    "quat_xyzw_to_wxyz",
]
