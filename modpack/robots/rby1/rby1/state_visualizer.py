from __future__ import annotations

import math
import time
import sys
from typing import Dict, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np

def _quat_angle_error(target_quat: np.ndarray, actual_quat: np.ndarray) -> float:
    target = np.asarray(target_quat, dtype=np.float64)
    actual = np.asarray(actual_quat, dtype=np.float64)
    target_norm = np.linalg.norm(target)
    actual_norm = np.linalg.norm(actual)
    if target_norm < 1e-9 or actual_norm < 1e-9:
        return float("nan")
    target /= target_norm
    actual /= actual_norm
    dot = float(np.dot(target, actual))
    dot = max(-1.0, min(1.0, abs(dot)))
    return 2.0 * math.acos(dot)

class StateVisualizer:
    """Draw policy and actual end-effector poses in the MuJoCo viewer."""

    def __init__(self, model_path: str, initial_qpos: Optional[np.ndarray], print_errors: bool = False) -> None:
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        if initial_qpos is not None:
            self.apply_qpos(initial_qpos)

        self.site_ids = {
            "left": self.model.site("end_effector_l").id,
            "right": self.model.site("end_effector_r").id,
            "head": self.model.site("head").id,
        }
        base_data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, base_data)
        self.head_site_base_rot = (
            base_data.site_xmat[self.site_ids["head"]].copy().reshape(3, 3, order="F")
        )

        self.viewer = mujoco.viewer.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False,
        )
        mujoco.mjv_defaultFreeCamera(self.model, self.viewer.cam)
        self.print_errors_default = print_errors

    def init_mocap_targets(
        self,
        *,
        left_body: str = "ee_l_target",
        right_body: str = "ee_r_target",
        head_body: str = "head_target",
        left_site: str = "end_effector_l",
        right_site: str = "end_effector_r",
        head_site: str = "head",
    ) -> Dict[str, int]:
        """Initialize mocap bodies to the nominal EE poses and return their ids."""
        mocap_ids = {
            "left": self.model.body(left_body).mocapid[0],
            "right": self.model.body(right_body).mocapid[0],
            "head": self.model.body(head_body).mocapid[0],
        }

        left_pos = self.site_pos(left_site, self.data.qpos)
        right_pos = self.site_pos(right_site, self.data.qpos)
        head_pos, head_quat = self.site_pose(head_site, self.data.qpos)

        self.data.mocap_pos[mocap_ids["left"]] = left_pos
        self.data.mocap_pos[mocap_ids["right"]] = right_pos
        self.data.mocap_pos[mocap_ids["head"]] = head_pos

        l_q = self.data.xquat[self.model.body("EE_BODY_L").id].copy()
        r_q = self.data.xquat[self.model.body("EE_BODY_R").id].copy()
        self.data.mocap_quat[mocap_ids["left"]] = l_q
        self.data.mocap_quat[mocap_ids["right"]] = r_q
        self.data.mocap_quat[mocap_ids["head"]] = head_quat
        return mocap_ids

    def get_mocap_targets(self, mocap_ids: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (left_pos, left_quat, right_pos, right_quat, head_pos, head_quat) from mocap bodies."""
        def _get(mid: int) -> Tuple[np.ndarray, np.ndarray]:
            return (
                self.data.mocap_pos[mid].copy(),
                self.data.mocap_quat[mid].copy(),
            )

        left_pos, left_quat = _get(mocap_ids["left"])
        right_pos, right_quat = _get(mocap_ids["right"])
        head_pos, head_quat = _get(mocap_ids["head"])
        return left_pos, left_quat, right_pos, right_quat, head_pos, head_quat

    def render(
        self,
        actual_qpos: Optional[np.ndarray],
        policy_targets: Tuple[
            Optional[np.ndarray],
            Optional[np.ndarray],
            Optional[float],
            Optional[np.ndarray],
            Optional[np.ndarray],
            Optional[float],
            Optional[np.ndarray],
            Optional[np.ndarray],
        ]
    ) -> None:
        if self.viewer is None or self.viewer.user_scn is None:
            return

        self.apply_qpos(actual_qpos)

        lt_p, lt_q, lw, rt_p, rt_q, rw, head_p, head_q = policy_targets
        actual_left_pos, actual_left_quat = self._actual_pose("left")
        actual_right_pos, actual_right_quat = self._actual_pose("right")
        actual_head_pos, actual_head_quat = self._actual_pose("head")

        head_pos_viz, head_quat_viz = self._prepare_head_policy_pose(
            head_q, actual_head_pos
        )

        self._update_pose_markers(
            policy_left_pos=lt_p,
            policy_left_quat=lt_q,
            policy_right_pos=rt_p,
            policy_right_quat=rt_q,
            policy_head_pos=head_pos_viz,
            policy_head_quat=head_quat_viz,
            actual_left_pos=actual_left_pos,
            actual_left_quat=actual_left_quat,
            actual_right_pos=actual_right_pos,
            actual_right_quat=actual_right_quat,
            actual_head_pos=actual_head_pos,
            actual_head_quat=actual_head_quat,
        )

        if self.print_errors_default:
            self._print_errors(
                lt_p,
                lt_q,
                actual_left_pos,
                actual_left_quat,
                rt_p,
                rt_q,
                actual_right_pos,
                actual_right_quat,
                head_pos_viz,
                head_quat_viz,
                actual_head_pos,
                actual_head_quat,
            )

        mujoco.mj_camlight(self.model, self.data)
        self.viewer.sync()

    def apply_qpos(self, qpos) -> None:
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

    def site_pose(self, site_name: str, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        pos = self.data.site_xpos[sid].copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[sid])
        return pos, quat

    def site_pos(self, site_name: str, qpos: np.ndarray) -> np.ndarray:
        pos, _ = self.site_pose(site_name, qpos)
        return pos

    def _actual_pose(self, label: str) -> tuple[np.ndarray, np.ndarray]:
        sid = self.site_ids[label]
        pos = self.data.site_xpos[sid].copy()
        mat = self.data.site_xmat[sid].copy()
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, mat)
        return pos, quat

    def _prepare_head_policy_pose(
        self,
        policy_quat: Optional[np.ndarray],
        actual_pos: np.ndarray,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if policy_quat is None:
            return None, None
        projected_quat = self._project_head_orientation(policy_quat)
        if projected_quat is None:
            return None, None
        return actual_pos.copy(), projected_quat

    def _project_head_orientation(
        self, policy_quat: np.ndarray
    ) -> Optional[np.ndarray]:
        if self.head_site_base_rot is None:
            return np.asarray(policy_quat, dtype=np.float64).copy()

        quat_arr = np.asarray(policy_quat, dtype=np.float64)
        norm = np.linalg.norm(quat_arr)
        if norm < 1e-9:
            return None
        quat_arr = quat_arr / norm

        target_rot_flat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(target_rot_flat, quat_arr)
        target_rot = target_rot_flat.reshape(3, 3, order="F")

        relative_rot = self.head_site_base_rot.T @ target_rot

        rel_20 = float(np.clip(relative_rot[2, 0], -1.0, 1.0))
        pitch = -math.asin(rel_20)
        cos_pitch = math.cos(pitch)
        if abs(cos_pitch) < 1e-8:
            yaw = math.atan2(-relative_rot[0, 1], relative_rot[1, 1])
        else:
            yaw = math.atan2(relative_rot[1, 0], relative_rot[0, 0])

        projected_rot = (
            self.head_site_base_rot @ self._rotation_z(yaw) @ self._rotation_y(pitch)
        )
        projected_flat = projected_rot.reshape(9, order="F")
        projected_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(projected_quat, projected_flat)
        return projected_quat

    def _rotation_z(self, angle: float) -> np.ndarray:
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

    def _rotation_y(self, angle: float) -> np.ndarray:
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

    def _update_pose_markers(
        self,
        policy_left_pos: Optional[np.ndarray],
        policy_left_quat: Optional[np.ndarray],
        policy_right_pos: Optional[np.ndarray],
        policy_right_quat: Optional[np.ndarray],
        policy_head_pos: Optional[np.ndarray],
        policy_head_quat: Optional[np.ndarray],
        actual_left_pos: np.ndarray,
        actual_left_quat: np.ndarray,
        actual_right_pos: np.ndarray,
        actual_right_quat: np.ndarray,
        actual_head_pos: np.ndarray,
        actual_head_quat: np.ndarray,
    ) -> None:
        scene = self.viewer.user_scn
        scene.ngeom = 0
        policy_alpha = 0.5
        policy_width = 1.5
        actual_alpha = 0.9
        actual_width = 3.0

        self._add_pose_marker(
            scene, policy_left_pos, policy_left_quat, policy_alpha, policy_width
        )
        self._add_pose_marker(
            scene, policy_right_pos, policy_right_quat, policy_alpha, policy_width
        )
        self._add_pose_marker(
            scene, policy_head_pos, policy_head_quat, policy_alpha, policy_width
        )
        self._add_pose_marker(
            scene, actual_left_pos, actual_left_quat, actual_alpha, actual_width
        )
        self._add_pose_marker(
            scene, actual_right_pos, actual_right_quat, actual_alpha, actual_width
        )
        self._add_pose_marker(
            scene, actual_head_pos, actual_head_quat, actual_alpha, actual_width
        )

    def _add_pose_marker(
        self,
        scene: mujoco.MjvScene,
        pos: Optional[np.ndarray],
        quat: Optional[np.ndarray],
        alpha: float,
        line_width: float,
    ) -> None:
        if pos is None or quat is None or scene.ngeom >= scene.maxgeom:
            return

        pos_arr = np.asarray(pos, dtype=np.float64)
        quat_arr = np.asarray(quat, dtype=np.float64)

        axis_length = 0.15
        line_size = np.zeros(3, dtype=np.float64)
        identity_mat = np.array(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64
        )
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

    def _print_errors(
        self,
        policy_left_pos: Optional[np.ndarray],
        policy_left_quat: Optional[np.ndarray],
        actual_left_pos: np.ndarray,
        actual_left_quat: np.ndarray,
        policy_right_pos: Optional[np.ndarray],
        policy_right_quat: Optional[np.ndarray],
        actual_right_pos: np.ndarray,
        actual_right_quat: np.ndarray,
        policy_head_pos: Optional[np.ndarray],
        policy_head_quat: Optional[np.ndarray],
        actual_head_pos: np.ndarray,
        actual_head_quat: np.ndarray,
    ) -> None:
        left_err = self._format_ee_error(
            "L", policy_left_pos, policy_left_quat, actual_left_pos, actual_left_quat
        )
        right_err = self._format_ee_error(
            "R",
            policy_right_pos,
            policy_right_quat,
            actual_right_pos,
            actual_right_quat,
        )
        head_err = self._format_ee_error(
            "H",
            policy_head_pos,
            policy_head_quat,
            actual_head_pos,
            actual_head_quat,
            include_position=False,
        )

        errors = [err for err in (left_err, right_err, head_err) if err is not None]
        if errors:
            msg = f"[EE error] {' | '.join(errors)}"
            sys.stdout.write(f"\r{msg}\x1b[K")
            sys.stdout.flush()

    def _format_ee_error(
        self,
        label: str,
        target_pos: Optional[np.ndarray],
        target_quat: Optional[np.ndarray],
        actual_pos: np.ndarray,
        actual_quat: np.ndarray,
        include_position: bool = True,
    ) -> Optional[str]:
        if target_pos is None or target_quat is None:
            return None
        parts: list[str] = []
        if include_position:
            pos_err = np.asarray(target_pos, dtype=np.float64) - np.asarray(
                actual_pos, dtype=np.float64
            )
            pos_err_norm = float(np.linalg.norm(pos_err))
            parts.append(f"dpos={pos_err_norm:.4f}")
        quat_err = _quat_angle_error(target_quat, actual_quat)
        parts.append(f"dq={quat_err:.4f}")
        return f"{label} {' '.join(parts)}"


__all__ = ["StateVisualizer"]
