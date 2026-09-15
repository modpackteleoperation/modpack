from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

DEMO_DIR = Path(__file__).resolve().parent


class TrajectoryRecorder:
    """Collect teleop targets and serialize them as a single-episode trajectory."""

    def __init__(self, session_name: str, enabled: bool = True, demo_dir: Path | None = None) -> None:
        self.session_name = session_name
        self._enabled = enabled
        self._demo_dir = demo_dir or DEMO_DIR
        self._trajectory_path = self._demo_dir / f"{self.session_name}.pkl" if enabled else None
        if self._trajectory_path is not None:
            self._demo_dir.mkdir(parents=True, exist_ok=True)
        self._target_timestamps: List[float] = []
        self._target_data: Dict[str, Dict[str, List]] = {}
        self._closed = False

    @staticmethod
    def _quat_wxyz_to_rotvec(quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=float)
        if quat.shape[-1] != 4 or np.linalg.norm(quat) < 1e-9:
            return np.zeros(3, dtype=float)
        quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=float)
        return Rotation.from_quat(quat_xyzw).as_rotvec()

    def log_target(self, target, timestamp: Optional[float] = None) -> None:
        """Record a target sample if recording is enabled."""
        if not self._enabled:
            return
        ts = float(time.time()) if timestamp is None else float(timestamp)
        self._target_timestamps.append(ts)
        self._append_target("left", target.left_pos, target.left_quat, target.left_width)
        self._append_target("right", target.right_pos, target.right_quat, target.right_width)
        self._append_target("head", target.head_pos, target.head_quat, None)

    def _append_target(
        self,
        side: str,
        position: Optional[np.ndarray],
        quaternion_wxyz: Optional[np.ndarray],
        gripper_width: Optional[float],
    ) -> None:
        if position is None or quaternion_wxyz is None:
            return
        data = self._target_data.setdefault(side, {"tcp_pose": [], "gripper_width": []})
        rotvec = self._quat_wxyz_to_rotvec(quaternion_wxyz)
        pose_sample = list(np.asarray(position, dtype=float)) + list(rotvec)
        data["tcp_pose"].append(pose_sample)
        width_value = 0.0 if gripper_width is None else float(gripper_width)
        data["gripper_width"].append(width_value)

    def close(self) -> None:
        if self._closed or not self._enabled:
            self._closed = True
            return
        if not self._target_timestamps:
            self._closed = True
            return
        if self._trajectory_path is None:
            self._closed = True
            return

        def build_gripper_payload(side: str) -> List[Dict]:
            data = self._target_data.get(side)
            if not data or not data["tcp_pose"]:
                return []
            tcp_pose = np.asarray(data["tcp_pose"], dtype=np.float32)
            gripper_width = np.asarray(data["gripper_width"], dtype=np.float32)
            demo_start = np.repeat(tcp_pose[[0]], tcp_pose.shape[0], axis=0)
            demo_end = np.repeat(tcp_pose[[-1]], tcp_pose.shape[0], axis=0)
            return [
                {
                    "tcp_pose": tcp_pose,
                    "gripper_width": gripper_width,
                    "demo_start_pose": demo_start,
                    "demo_end_pose": demo_end,
                }
            ]

        episode = {
            "tasks": [],
            "episode_name": self.session_name,
            "grippers_left": build_gripper_payload("left"),
            "grippers_right": build_gripper_payload("right"),
            "grippers_head": build_gripper_payload("head"),
            "cameras_left": [],
            "cameras_right": [],
            "cameras_head": [],
            "target_timestamps": np.asarray(self._target_timestamps, dtype=np.float64),
        }

        with self._trajectory_path.open("wb") as fh:
            pickle.dump([episode], fh)
        self._closed = True
