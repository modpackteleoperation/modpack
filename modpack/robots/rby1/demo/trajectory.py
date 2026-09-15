import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

@dataclass
class GripperTrajectory:
    """
    Container for gripper trajectory data.
    - tcp_pose (np.ndarray): (Length, 6): Poses are relative to the robot base frame
    - gripper_width (np.ndarray): (Length,): Values are between 0.01 ~ 0.08
    - demo_start_pose (np.ndarray): (Length, 6): demo_start_pose[i] = tcp_pose[0]
    - demo_end_pose (np.ndarray): (Length, 6): demo_end_pose[i] = tcp_pose[-1]
    """
    tcp_pose: np.ndarray
    gripper_width: np.ndarray
    demo_start_pose: np.ndarray
    demo_end_pose: np.ndarray


@dataclass
class CameraTrajectory:
    """
    Container for camera trajectory data.
    - main_video_path (str): path
    - depth_video_path (str): path
    - ultrawide_video_path (str): path
    - pose_idx_to_main_idx (np.ndarray): (Length,)
    - pose_idx_to_depth_idx (np.ndarray): (Length,)
    - pose_idx_to_ultrawide_idx (np.ndarray): (Length,)
    """
    main_video_path: str
    depth_video_path: str
    ultrawide_video_path: str
    pose_idx_to_main_idx: np.ndarray
    pose_idx_to_depth_idx: np.ndarray
    pose_idx_to_ultrawide_idx: np.ndarray

class Trajectory:
    def __init__(self, trajectory_data: Dict[str, Any]):
        """
        Container for a single demonstration episode.
        """
        # High-level metadata
        self.tasks: List[str] = trajectory_data.get("tasks", [])
        self.episode_name: str = trajectory_data.get("episode_name", "")

        # Gripper info
        self.grippers_left: GripperTrajectory = self._parse_list_to_dataclass(
            trajectory_data.get("grippers_left", []), GripperTrajectory
        )
        self.grippers_right: GripperTrajectory = self._parse_list_to_dataclass(
            trajectory_data.get("grippers_right", []), GripperTrajectory
        )
        self.grippers_head: GripperTrajectory = self._parse_list_to_dataclass(
            trajectory_data.get("grippers_head", []), GripperTrajectory
        )

        # Camera info
        self.cameras_left: CameraTrajectory = self._parse_list_to_dataclass(
            trajectory_data.get("cameras_left", []), CameraTrajectory
        )
        self.cameras_right: CameraTrajectory = self._parse_list_to_dataclass(
            trajectory_data.get("cameras_right", []), CameraTrajectory
        )
        self.cameras_head: CameraTrajectory = self._parse_list_to_dataclass(
            trajectory_data.get("cameras_head", []), CameraTrajectory
        )

    # Helper for list[dict] → dataclass
    def _parse_list_to_dataclass(self, data, cls):
        if isinstance(data, list) and len(data) > 0:
            return cls(**data[0])
        elif isinstance(data, dict):
            return cls(**data)
        return None
