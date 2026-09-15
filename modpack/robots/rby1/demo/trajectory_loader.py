from pathlib import Path
import pickle
from typing import Optional, Tuple

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from transforms3d import affines

from demo.trajectory import Trajectory, GripperTrajectory


class RigidTransform:
    """Lightweight SE(3) transform helper with Drake-compatible operations."""

    __slots__ = ("_matrix",)

    def __init__(self, rotation=None, translation=None):
        if translation is None:
            if rotation is None:
                self._matrix = np.eye(4, dtype=float)
            else:
                matrix = np.asarray(rotation, dtype=float)
                if matrix.shape != (4, 4):
                    raise ValueError("Provide (4,4) matrix or (3,3)+(3,) pair")
                self._matrix = matrix.copy()
        else:
            rot = np.asarray(rotation, dtype=float).reshape(3, 3)
            trans = np.asarray(translation, dtype=float).reshape(3)
            matrix = np.eye(4, dtype=float)
            matrix[:3, :3] = rot
            matrix[:3, 3] = trans
            self._matrix = matrix

    @classmethod
    def identity(cls):
        return cls()

    def copy(self):
        return RigidTransform(self._matrix)

    def as_matrix(self):
        return self._matrix.copy()

    def rotation_matrix(self):
        return self._matrix[:3, :3].copy()

    def translation(self):
        return self._matrix[:3, 3].copy()

    def inverse(self):
        rot = self._matrix[:3, :3]
        trans = self._matrix[:3, 3]
        rot_inv = rot.T
        trans_inv = -rot_inv @ trans
        return RigidTransform(rot_inv, trans_inv)

    def __matmul__(self, other):
        if not isinstance(other, RigidTransform):
            return NotImplemented
        return RigidTransform(self._matrix @ other._matrix)


def rpy_to_matrix(rpy):
    return Rotation.from_euler("xyz", np.asarray(rpy, dtype=float)).as_matrix()


MODEL_TO_FRAME = {
    "right_arm": "EE_BODY_R",
    "left_arm": "EE_BODY_L",
    "head": "link_head_2",
}

TCP_TO_MODEL_FRAME = {
    "head": RigidTransform(rpy_to_matrix([-np.pi / 2, 0.0, - np.pi / 2]), [0.04, 0.0, 0.0601]).inverse(),
    "left_arm": RigidTransform(rpy_to_matrix([np.pi, 0.0, 0.0]), [0.0, 0.0, -0.2]).inverse(),
    "right_arm": RigidTransform(rpy_to_matrix([0.0, np.pi, 0.0]), [0.0, 0.0, -0.2]).inverse(),
}


def _load_episode(path: Path, idx: int) -> Trajectory:
    with open(path, "rb") as f:
        plan = pickle.load(f)

    try:
        plan_episode = plan[idx]
    except Exception as exc:
        raise ValueError(f"Failed to get episode index {idx} from plan with {len(plan)} episodes: {exc}") from exc

    return Trajectory(plan_episode)


def _gripper_to_transforms(gripper: Optional[GripperTrajectory]) -> Tuple[Optional[list], Optional[np.ndarray]]:
    if gripper is None or getattr(gripper, "tcp_pose", None) is None:
        return None, None

    tcp_pose = np.asarray(gripper.tcp_pose)
    if tcp_pose.size == 0:
        return None, None

    eef_pose_matrix = [
        RigidTransform(
            affines.compose(
                pose[:3],
                Rotation.from_rotvec(pose[3:]).as_matrix(),
                np.ones(3),
            )
        )
        for pose in tcp_pose
    ]

    widths = None
    gripper_width = getattr(gripper, "gripper_width", None)
    if gripper_width is not None:
        width_arr = np.asarray(gripper_width, dtype=np.float32)
        if width_arr.size != len(eef_pose_matrix):
            raise ValueError("Pose and gripper width lengths do not match")
        width_arr = width_arr.reshape(len(eef_pose_matrix))
        widths = np.expand_dims(width_arr, axis=-1)
    return eef_pose_matrix, widths


def load_trajectory(traj_dir, client, index: int = 0, use_head: bool = True, align_mode: str = "relative"):
    traj_dir = Path(traj_dir)
    if not traj_dir.exists():
        raise FileNotFoundError(f"Missing: {traj_dir}")

    episode = _load_episode(traj_dir, idx=index)
    pose_data = {}
    gripper_data = {}

    for model_name, gripper in (
        ("left_arm", episode.grippers_left),
        ("right_arm", episode.grippers_right),
        ("head", episode.grippers_head),
    ):
        poses, widths = _gripper_to_transforms(gripper)
        if poses is not None:
            pose_data[model_name] = poses
        if widths is not None:
            gripper_data[model_name] = widths

    required_missing = [m for m in ("left_arm", "right_arm") if m not in pose_data]
    if required_missing:
        raise ValueError(f"Missing pose data for {', '.join(required_missing)} in episode {index}")

    model_names = ["left_arm", "right_arm"]
    if use_head and "head" in pose_data:
        model_names.append("head")

    T = min(len(pose_data[m]) for m in model_names)

    for m in model_names:
        if len(pose_data[m]) != T:
            raise ValueError(f"Pose data for {m} has length {len(pose_data[m])}, expected {T}")

    if align_mode == "relative":
        poses_all = [pose_data[m][0] @ TCP_TO_MODEL_FRAME[m] for m in model_names]

        snapshot = client.robot_state.load()
        if snapshot is None or not snapshot.is_valid:
            snapshot = client.wait_for_first_state()
        with client._model_lock:
            qpos = client._snapshot_to_qpos(snapshot)
            client.data.qpos[:] = qpos
            mujoco.mj_forward(client.model, client.data)
            robot_poses = []
            for m in model_names:
                body_id = mujoco.mj_name2id(client.model, mujoco.mjtObj.mjOBJ_BODY, MODEL_TO_FRAME[m])
                rot = client.data.xmat[body_id].reshape(3, 3)
                pos = client.data.xpos[body_id].copy()
                robot_poses.append(RigidTransform(rot, pos))

        T_transform_all = {key: robot_poses[i] @ poses_all[i].inverse() for i, key in enumerate(model_names)}

    elif align_mode == "none":
        T_transform_all = {key: RigidTransform.identity() for key in model_names}

    else:
        raise ValueError(f"Unsupported align_mode: {align_mode}")

    poses_list = []
    widths_list = []

    for t in range(T):
        poses = {}
        for m in model_names:
            T_model = pose_data[m][t] @ TCP_TO_MODEL_FRAME[m]
            T_aligned = T_transform_all[m] @ T_model
            poses[m] = T_aligned

        widths = {}
        for m in model_names:
            if m in gripper_data:
                widths[f"{m.replace('_arm', '')}_width"] = gripper_data[m][t]
        poses_list.append(poses)
        widths_list.append(widths)

    return poses_list, widths_list
