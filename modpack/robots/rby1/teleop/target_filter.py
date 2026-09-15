from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from rby1.pose_utils import (
    lerp_value,
    normalize_quaternion,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
    slerp_quaternion,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WBC_CONFIG = PROJECT_ROOT / "config" / "wbc.yaml"
DEFAULT_TELEOP_VR_CONFIG = PROJECT_ROOT / "config" / "teleop_vr.yaml"
DEFAULT_TELEOP_IPHONE_CONFIG = PROJECT_ROOT / "config" / "teleop_iphone.yaml"


@dataclass
class TeleopFilterConfig:
    enabled: bool = True
    output_frequency_hz: float = 120.0
    max_translation_m_per_sec: float = 1.2
    max_rotation_deg_per_sec: float = 240.0

    @classmethod
    def from_mapping(cls, mapping: Optional[Mapping[str, Any]]) -> "TeleopFilterConfig":
        if mapping is None:
            return cls()
        return cls(
            enabled=bool(mapping.get("enabled", True)),
            output_frequency_hz=float(mapping.get("output_frequency_hz", 120.0)),
            max_translation_m_per_sec=float(mapping.get("max_translation_m_per_sec", 1.2)),
            max_rotation_deg_per_sec=float(mapping.get("max_rotation_deg_per_sec", 240.0)),
        )


def load_teleop_filter_config(config_path: Optional[Path] = None) -> TeleopFilterConfig:
    """Load teleop filter config, preferring teleop-specific YAML files."""
    candidate_paths = (
        [config_path]
        if config_path is not None
        else [DEFAULT_TELEOP_VR_CONFIG, DEFAULT_TELEOP_IPHONE_CONFIG, DEFAULT_WBC_CONFIG]
    )
    for path in candidate_paths:
        if path is None:
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
            if not isinstance(cfg, dict):
                continue
            return TeleopFilterConfig.from_mapping(cfg.get("teleop_filter"))
        except FileNotFoundError:
            continue
        except Exception:
            continue
    return TeleopFilterConfig()


@dataclass
class _FilterState:
    pos: np.ndarray
    quat: np.ndarray
    timestamp: float


class PoseFilter:
    """Low-pass filter with per-step clamp for teleop targets."""

    def __init__(self, config: Optional[TeleopFilterConfig] = None) -> None:
        self.config = config if config is not None else TeleopFilterConfig()
        self._state: Dict[str, _FilterState] = {}
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._state.clear()

    def filter(
        self,
        label: str,
        pos: Optional[np.ndarray],
        quat: Optional[np.ndarray],
        timestamp: Optional[float] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if pos is None or quat is None:
            return pos, quat
        now = time.monotonic() if timestamp is None else float(timestamp)
        pos_arr = np.asarray(pos, dtype=float)
        quat_arr = normalize_quaternion(np.asarray(quat, dtype=float))

        if not self.config.enabled:
            with self._lock:
                self._state[label] = _FilterState(pos=pos_arr.copy(), quat=quat_arr.copy(), timestamp=now)
            return pos_arr, quat_arr

        with self._lock:
            state = self._state.get(label)
            if state is None:
                self._state[label] = _FilterState(pos=pos_arr.copy(), quat=quat_arr.copy(), timestamp=now)
                return pos_arr, quat_arr

            dt = max(now - state.timestamp, 1e-4)
            alpha = 1.0 - math.exp(-dt * max(self.config.output_frequency_hz, 1e-3))

            # Initial smoothing pass (lerp/slerp) to soften jitter.
            candidate_pos = lerp_value(state.pos, pos_arr, alpha)
            candidate_quat = slerp_quaternion(state.quat, quat_arr, alpha)
            if candidate_pos is None or candidate_quat is None:
                self._state[label] = _FilterState(pos=pos_arr.copy(), quat=quat_arr.copy(), timestamp=now)
                return pos_arr, quat_arr

            # Translation clamp.
            max_step = max(self.config.max_translation_m_per_sec, 0.0) * dt
            delta = candidate_pos - state.pos
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm > max_step > 0.0:
                delta = delta * (max_step / delta_norm)
                candidate_pos = state.pos + delta

            # Rotation clamp.
            max_angle = math.radians(max(self.config.max_rotation_deg_per_sec, 0.0)) * dt
            candidate_quat = self._clamp_rotation(state.quat, candidate_quat, max_angle)

            self._state[label] = _FilterState(
                pos=candidate_pos.copy(),
                quat=candidate_quat.copy(),
                timestamp=now,
            )
            return candidate_pos, candidate_quat

    def _clamp_rotation(
        self,
        prev_quat_wxyz: np.ndarray,
        target_quat_wxyz: np.ndarray,
        max_angle_rad: float,
    ) -> np.ndarray:
        prev_rot = R.from_quat(quat_wxyz_to_xyzw(prev_quat_wxyz))
        target_rot = R.from_quat(quat_wxyz_to_xyzw(target_quat_wxyz))
        relative = target_rot * prev_rot.inv()
        angle = float(relative.magnitude())
        if angle <= max_angle_rad or max_angle_rad <= 0.0:
            return target_quat_wxyz
        scale = max_angle_rad / max(angle, 1e-9)
        limited_rel = R.from_rotvec(relative.as_rotvec() * scale)
        limited_rot = limited_rel * prev_rot
        limited_quat_xyzw = limited_rot.as_quat()
        limited_quat_wxyz = quat_xyzw_to_wxyz(limited_quat_xyzw)
        return normalize_quaternion(limited_quat_wxyz)


__all__ = [
    "TeleopFilterConfig",
    "PoseFilter",
    "load_teleop_filter_config",
    "DEFAULT_TELEOP_VR_CONFIG",
    "DEFAULT_TELEOP_IPHONE_CONFIG",
    "DEFAULT_WBC_CONFIG",
]
