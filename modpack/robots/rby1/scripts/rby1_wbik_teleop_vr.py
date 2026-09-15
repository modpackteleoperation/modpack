"""Meta Quest teleop frontend that runs whole-body IK with a MuJoCo preview."""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np
import yaml
from loop_rate_limiters import RateLimiter

# Ensure project root is on sys.path regardless of current working directory.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rby1.ee_targets import EETargets
from rby1.whole_body_ik import RBY1WholeBodyIK
from teleop.teleop_vr import TeleopVR


DEFAULT_MODEL = Path(PROJECT_ROOT) / "model" / "rby1" / "rby1.xml"
DEFAULT_WBC_CONFIG = Path(PROJECT_ROOT) / "config" / "wbc.yaml"
DEFAULT_TELEOP_VR_CONFIG = Path(PROJECT_ROOT) / "config" / "teleop_vr.yaml"


@dataclass
class _KinematicsSnapshot:
    qpos: np.ndarray
    is_valid: bool = True


class _KinematicsWBCShim:
    """Minimal interface for TeleopVR when no controller thread is running."""

    def __init__(self, model_path: Path, initial_qpos: np.ndarray) -> None:
        self.model_path = model_path.as_posix()
        self._lock = threading.Lock()
        self._snapshot = _KinematicsSnapshot(qpos=initial_qpos.copy())

    def get_latest_robot_state(self) -> _KinematicsSnapshot:
        with self._lock:
            return self._snapshot

    @staticmethod
    def snapshot_to_qpos(snapshot: _KinematicsSnapshot) -> Optional[np.ndarray]:
        if snapshot is None or not getattr(snapshot, "is_valid", False):
            return None
        return snapshot.qpos.copy()

    def update_qpos(self, qpos: np.ndarray) -> None:
        with self._lock:
            self._snapshot = _KinematicsSnapshot(qpos=qpos.copy())


def _site_pose(model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> tuple[np.ndarray, np.ndarray]:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    pos = data.site_xpos[sid].copy()
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, data.site_xmat[sid])
    return pos, quat


def _get_gripper_indices(model: mujoco.MjModel) -> dict[str, int]:
    names = ["gripper_finger_l1", "gripper_finger_l2", "gripper_finger_r1", "gripper_finger_r2"]
    indices: dict[str, int] = {}
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        indices[name] = int(model.jnt_qposadr[jid])
    return indices


def _apply_gripper_widths(qpos: np.ndarray, gripper_indices: dict[str, int], left_width: float, right_width: float) -> None:
    l1 = gripper_indices["gripper_finger_l1"]
    l2 = gripper_indices["gripper_finger_l2"]
    r1 = gripper_indices["gripper_finger_r1"]
    r2 = gripper_indices["gripper_finger_r2"]
    qpos[l1] = -0.5 * left_width
    qpos[l2] = 0.5 * left_width
    qpos[r1] = -0.5 * right_width
    qpos[r2] = 0.5 * right_width


def _apply_initial_configuration(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ik_solver: RBY1WholeBodyIK,
    init_config: dict,
) -> None:
    """Mirror the initial joint pose loading from rby1_wbc."""

    def _to_array(values: Optional[list[float]]) -> Optional[np.ndarray]:
        if values is None:
            return None
        return np.asarray(values, dtype=float)

    init_pos = init_config.get("init_position", {}) if init_config else {}
    torso_targets = _to_array(init_pos.get("torso"))
    right_targets = _to_array(init_pos.get("right_arm"))
    left_targets = _to_array(init_pos.get("left_arm"))
    head_targets = _to_array(init_pos.get("head"))

    if all(
        target is None for target in (torso_targets, right_targets, left_targets, head_targets)
    ):
        return

    sol_qpos = data.qpos.copy()
    torso_indices = getattr(ik_solver, "torso_qpos_indices", [])
    right_indices = getattr(ik_solver, "right_arm_qpos_indices", [])
    left_indices = getattr(ik_solver, "left_arm_qpos_indices", [])
    head_indices = getattr(ik_solver, "head_qpos_indices", [])

    if torso_targets is not None:
        for idx, value in zip(torso_indices, torso_targets):
            sol_qpos[int(idx)] = float(value)
    if right_targets is not None:
        for idx, value in zip(right_indices, right_targets):
            sol_qpos[int(idx)] = float(value)
    if left_targets is not None:
        for idx, value in zip(left_indices, left_targets):
            sol_qpos[int(idx)] = float(value)
    if head_targets is not None:
        for idx, value in zip(head_indices, head_targets):
            sol_qpos[int(idx)] = float(value)

    data.qpos[:] = sol_qpos
    mujoco.mj_forward(model, data)


def _add_pose_marker(
    scene: mujoco.MjvScene,
    pos: Optional[np.ndarray],
    quat: Optional[np.ndarray],
    alpha: float,
    line_width: float,
) -> None:
    if scene is None or pos is None or quat is None or scene.ngeom >= scene.maxgeom:
        return
    pos_arr = np.asarray(pos, dtype=np.float64)
    quat_arr = np.asarray(quat, dtype=np.float64)
    axis_length = 0.15
    line_size = np.zeros(3, dtype=np.float64)
    identity_mat = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis_colors = (
        np.array([1.0, 0.0, 0.0, alpha], dtype=np.float32),
        np.array([0.0, 1.0, 0.0, alpha], dtype=np.float32),
        np.array([0.0, 0.0, 1.0, alpha], dtype=np.float32),
    )
    frame_mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(frame_mat, quat_arr)
    rot = frame_mat.reshape(3, 3)
    for axis_idx, axis_color in enumerate(axis_colors):
        if scene.ngeom >= scene.maxgeom:
            break
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            line_size,
            pos_arr,
            identity_mat,
            axis_color,
        )
        endpoint = pos_arr + rot[:, axis_idx] * axis_length
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            line_width,
            pos_arr,
            endpoint,
        )
        scene.ngeom += 1


def _rotation_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _rotation_y(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )


def _project_head_orientation(policy_quat: np.ndarray, head_site_base_rot: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if head_site_base_rot is None:
        return None
    quat_arr = np.asarray(policy_quat, dtype=np.float64)
    norm = np.linalg.norm(quat_arr)
    if norm < 1e-9:
        return None
    quat_arr = quat_arr / norm

    target_rot_flat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(target_rot_flat, quat_arr)
    target_rot = target_rot_flat.reshape(3, 3, order="F")

    relative_rot = head_site_base_rot.T @ target_rot

    rel_20 = float(np.clip(relative_rot[2, 0], -1.0, 1.0))
    pitch = -math.asin(rel_20)
    cos_pitch = math.cos(pitch)
    if abs(cos_pitch) < 1e-8:
        yaw = math.atan2(-relative_rot[0, 1], relative_rot[1, 1])
    else:
        yaw = math.atan2(relative_rot[1, 0], relative_rot[0, 0])

    projected_rot = head_site_base_rot @ _rotation_z(yaw) @ _rotation_y(pitch)
    projected_flat = projected_rot.reshape(9, order="F")
    projected_quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(projected_quat, projected_flat)
    return projected_quat


def _prepare_head_policy_pose(
    policy_pos: Optional[np.ndarray],
    policy_quat: Optional[np.ndarray],
    actual_pos: np.ndarray,
    head_site_base_rot: Optional[np.ndarray],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if policy_quat is None or head_site_base_rot is None:
        return None, None
    projected_quat = _project_head_orientation(policy_quat, head_site_base_rot)
    if projected_quat is None:
        return policy_pos, policy_quat
    return actual_pos.copy(), projected_quat


def _update_pose_markers(
    viewer,
    data: mujoco.MjData,
    site_ids: dict[str, int],
    target: Optional[EETargets],
    head_site_base_rot: Optional[np.ndarray],
) -> None:
    if viewer is None or viewer.user_scn is None:
        return
    scene = viewer.user_scn
    scene.ngeom = 0

    def actual_pose(label: str) -> tuple[np.ndarray, np.ndarray]:
        sid = site_ids[label]
        pos = data.site_xpos[sid].copy()
        mat = data.site_xmat[sid].copy()
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, mat)
        return pos, quat

    actual_alpha = 0.9
    actual_width = 3.0
    policy_alpha = 0.5
    policy_width = 1.5

    actual_left_pos, actual_left_quat = actual_pose("left")
    actual_right_pos, actual_right_quat = actual_pose("right")
    actual_head_pos, actual_head_quat = actual_pose("head")

    _add_pose_marker(scene, actual_left_pos, actual_left_quat, actual_alpha, actual_width)
    _add_pose_marker(scene, actual_right_pos, actual_right_quat, actual_alpha, actual_width)
    _add_pose_marker(scene, actual_head_pos, actual_head_quat, actual_alpha, actual_width)

    if target is None:
        return

    head_pos_viz, head_quat_viz = _prepare_head_policy_pose(
        target.head_pos,
        target.head_quat,
        actual_head_pos,
        head_site_base_rot,
    )

    _add_pose_marker(scene, target.left_pos, target.left_quat, policy_alpha, policy_width)
    _add_pose_marker(scene, target.right_pos, target.right_quat, policy_alpha, policy_width)
    _add_pose_marker(
        scene,
        head_pos_viz if head_pos_viz is not None else target.head_pos,
        head_quat_viz if head_quat_viz is not None else target.head_quat,
        policy_alpha,
        policy_width,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RBY1 whole-body IK in MuJoCo driven by the Meta Quest teleop frontend."
    )
    parser.add_argument(
        "--local_ip",
        required=True,
        help="Local Wi-Fi/LAN IP address where the Meta Quest headset sends controller poses.",
    )
    parser.add_argument(
        "--meta_quest_ip",
        required=True,
        help="IP address of the Meta Quest headset on the same network.",
    )
    parser.add_argument("--local_port", type=int, default=5005, help="Local UDP port to listen for headset packets.")
    parser.add_argument("--meta_quest_port", type=int, default=6000, help="Remote UDP port for headset discovery.")
    parser.add_argument("--ik-hz", type=float, default=100.0, help="Frequency for the IK integration loop.")
    parser.add_argument(
        "--target-hz",
        type=float,
        default=30.0,
        help="Rate at which new teleop targets are sampled and queued for IK interpolation.",
    )
    parser.add_argument(
        "--save-trajectory",
        action="store_true",
        help="Record the teleop target stream under demo/ for offline inspection.",
    )
    parser.add_argument("--headless", action="store_true", help="Skip launching the MuJoCo viewer.")
    args = parser.parse_args()

    if args.headless:
        os.environ.setdefault("MUJOCO_GL", "egl")

    model = mujoco.MjModel.from_xml_path(DEFAULT_MODEL.as_posix())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    ik_solver = RBY1WholeBodyIK()
    init_config: dict = {}
    if DEFAULT_WBC_CONFIG.exists():
        try:
            with DEFAULT_WBC_CONFIG.open("r", encoding="utf-8") as fh:
                init_config = yaml.safe_load(fh) or {}
        except Exception as exc:
            print(f"Warning: failed to load {DEFAULT_WBC_CONFIG}: {exc}")
    teleop_config: dict = {}
    if DEFAULT_TELEOP_VR_CONFIG.exists():
        try:
            with DEFAULT_TELEOP_VR_CONFIG.open("r", encoding="utf-8") as fh:
                teleop_config = yaml.safe_load(fh) or {}
        except Exception as exc:
            print(f"Warning: failed to load {DEFAULT_TELEOP_VR_CONFIG}: {exc}")
    if init_config:
        _apply_initial_configuration(model, data, ik_solver, init_config)
    gripper_indices = _get_gripper_indices(model)
    init_pos = init_config.get("init_position", {}) if init_config else {}
    gripper_targets = init_pos.get("grippers") if isinstance(init_pos, dict) else None
    if gripper_targets and len(gripper_targets) >= 2:
        left_gripper_width = float(gripper_targets[0])
        right_gripper_width = float(gripper_targets[1])
    else:
        left_gripper_width = right_gripper_width = 0.1
    _apply_gripper_widths(data.qpos, gripper_indices, left_gripper_width, right_gripper_width)
    mujoco.mj_forward(model, data)

    site_ids = {
        "left": model.site("end_effector_l").id,
        "right": model.site("end_effector_r").id,
        "head": model.site("head").id,
    }
    base_data = mujoco.MjData(model)
    mujoco.mj_forward(model, base_data)
    head_site_base_rot = base_data.site_xmat[site_ids["head"]].copy().reshape(3, 3, order="F")

    stub_wbc = _KinematicsWBCShim(model_path=DEFAULT_MODEL, initial_qpos=data.qpos.copy())
    teleop = TeleopVR(
        wbc=stub_wbc,
        local_ip=args.local_ip,
        meta_quest_ip=args.meta_quest_ip,
        local_port=args.local_port,
        meta_quest_port=args.meta_quest_port,
        save_trajectory=args.save_trajectory,
    )
    if not teleop.initialize():
        raise RuntimeError("Failed to initialize the Meta Quest teleop interface.")
    teleop.start()

    shared_targets = EETargets()
    latest_target: Optional[EETargets] = None

    ik_rate = RateLimiter(frequency=args.ik_hz, warn=False)
    target_interval = 1.0 / max(args.target_hz, 1e-3)
    next_target_time = time.monotonic()

    viewer = None
    if not args.headless:
        viewer = mujoco.viewer.launch_passive(
            model=model,
            data=data,
            show_left_ui=False,
            show_right_ui=False,
        )

    def _update_targets_from_teleop(duration: float) -> Optional[EETargets]:
        target = teleop.compute_target()
        if target is None:
            return None
        shared_targets.set_targets(
            duration=duration,
            left_pos=target.left_pos,
            left_quat=target.left_quat,
            right_pos=target.right_pos,
            right_quat=target.right_quat,
            left_width=target.left_width,
            right_width=target.right_width,
            head_pos=target.head_pos,
            head_quat=target.head_quat,
            timestamp=time.monotonic(),
        )
        return target

    def _tick_once() -> bool:
        nonlocal next_target_time, left_gripper_width, right_gripper_width, latest_target
        now = time.monotonic()
        if now >= next_target_time:
            tgt = _update_targets_from_teleop(duration=target_interval)
            if tgt is not None:
                latest_target = tgt
            next_target_time = now + target_interval

        (
            left_pos,
            left_quat,
            left_width,
            right_pos,
            right_quat,
            right_width,
            head_pos,
            head_quat,
        ) = shared_targets.get_target()

        if (
            left_pos is None
            or left_quat is None
            or right_pos is None
            or right_quat is None
        ):
            ik_rate.sleep()
            return True

        current_qpos = data.qpos.copy()
        if left_width is not None:
            left_gripper_width = float(left_width)
        if right_width is not None:
            right_gripper_width = float(right_width)

        sol_qpos, _sol_vel, success, _info = ik_solver.solve(
            left_target_pos=left_pos,
            left_target_quat=left_quat,
            right_target_pos=right_pos,
            right_target_quat=right_quat,
            head_target_pos=head_pos,
            head_target_quat=head_quat,
            current_qpos=current_qpos,
            dt=ik_rate.dt,
        )
        if not success:
            ik_rate.sleep()
            return True

        _apply_gripper_widths(sol_qpos, gripper_indices, left_gripper_width, right_gripper_width)
        data.qpos[:] = sol_qpos
        mujoco.mj_forward(model, data)
        stub_wbc.update_qpos(sol_qpos)

        if viewer is not None:
            mujoco.mj_camlight(model, data)
            _update_pose_markers(viewer, data, site_ids, latest_target, head_site_base_rot)
            viewer.sync()
        ik_rate.sleep()
        return True

    try:
        if viewer is None:
            while True:
                _tick_once()
        else:
            with viewer:
                while viewer.is_running():
                    if not _tick_once():
                        break
    except KeyboardInterrupt:
        pass
    finally:
        teleop.stop()
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
