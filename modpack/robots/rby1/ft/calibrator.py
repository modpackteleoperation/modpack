"""Utilities for transforming and calibrating force/torque sensor readings."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class FTCalibrator:
    """Applies frame transforms, offsets, and gravity compensation to FT data."""

    def __init__(self, config_path: Path | None = None) -> None:
        cfg_path = config_path if config_path is not None else PROJECT_ROOT / "config" / "ft_sensor.yaml"
        data = self._load_config(cfg_path)
        if not data:
            raise RuntimeError(f"No FT calibration data found in {cfg_path}")
        self.params = data

    def _load_config(self, path: Path) -> Dict[str, Dict[str, np.ndarray]]:
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        data: Dict[str, Dict[str, np.ndarray]] = {}
        for arm in ("left", "right"):
            entry = cfg.get(arm)
            if entry is None:
                continue
            data[arm] = {
                "force_offset": np.array(
                    [entry["offset"]["fx"], entry["offset"]["fy"], entry["offset"]["fz"]], dtype=float
                ),
                "torque_offset": np.array(
                    [entry["offset"]["tx"], entry["offset"]["ty"], entry["offset"]["tz"]], dtype=float
                ),
                "gravity": np.array([entry["gravity"]["x"], entry["gravity"]["y"], entry["gravity"]["z"]], dtype=float),
                "com": np.array([entry["com"]["x"], entry["com"]["y"], entry["com"]["z"]], dtype=float),
                "force_transform": np.array(entry["force_transform"], dtype=float),
                "torque_transform": np.array(entry.get("torque_transform", entry["force_transform"]), dtype=float),
            }
        return data

    def calibrate(
        self,
        left_wrench_raw: np.ndarray,
        right_wrench_raw: np.ndarray,
        left_pose: np.ndarray,
        right_pose: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns calibrated left/right wrenches."""
        left_cal = self._calibrate_single("left", left_wrench_raw, left_pose)
        right_cal = self._calibrate_single("right", right_wrench_raw, right_pose)
        return left_cal, right_cal

    def _calibrate_single(self, arm: str, wrench_raw: np.ndarray, pose: np.ndarray) -> np.ndarray:
        params = self.params.get(arm)
        wrench_tool = self._transform_wrench(wrench_raw, params)
        rot = Rotation.from_quat(self._wxyz_to_xyzw(pose[3:])).as_matrix()
        g_tool = rot.T @ params["gravity"]
        force = wrench_tool[:3] + params["force_offset"] - g_tool
        torque = wrench_tool[3:] + params["torque_offset"] - np.cross(params["com"], g_tool)
        return np.concatenate([force, torque])

    def _transform_wrench(self, raw: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
        force = params["force_transform"] @ raw[:3]
        torque = params["torque_transform"] @ raw[3:]
        return np.concatenate([force, torque])

    @staticmethod
    def _wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
        return np.array([quat[1], quat[2], quat[3], quat[0]], dtype=float)
