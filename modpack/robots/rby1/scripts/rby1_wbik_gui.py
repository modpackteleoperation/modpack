from pathlib import Path
import sys
import time
import math
from typing import Optional
import numpy as np
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

# Ensure project root is on sys.path regardless of current working directory.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rby1.whole_body_ik import RBY1WholeBodyIK

def main():
    # Viewer model/data (independent from IK's internal model)
    model = mujoco.MjModel.from_xml_path(PROJECT_ROOT + "/model/rby1/rby1_mocap.xml")
    data = mujoco.MjData(model)
    ik = RBY1WholeBodyIK()

    # Helper: FK using viewer model
    def site_pose(site_name: str, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        pos = data.site_xpos[sid].copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, data.site_xmat[sid])
        return pos, quat

    def site_pos(site_name: str, qpos: np.ndarray) -> np.ndarray:
        pos, _ = site_pose(site_name, qpos)
        return pos

    def _quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
        q1 = np.asarray(q1, dtype=float)
        q2 = np.asarray(q2, dtype=float)
        q1 = q1 / max(np.linalg.norm(q1), 1e-9)
        q2 = q2 / max(np.linalg.norm(q2), 1e-9)
        dot = float(np.clip(np.dot(q1, q2), -1.0, 1.0))
        angle_rad = 2.0 * math.acos(abs(dot))
        return math.degrees(angle_rad)

    def _passes_incremental_safety(
        current_qpos: np.ndarray,
        left_pos: np.ndarray,
        left_quat: np.ndarray,
        right_pos: np.ndarray,
        right_quat: np.ndarray,
        head_pos: Optional[np.ndarray],
        head_quat: Optional[np.ndarray],
    ) -> bool:
        # Hardcoded limits for the GUI-only WBIK preview
        MAX_POS_DELTA = 0.1  # meters
        MAX_ROT_DELTA_DEG = 20.0  # degrees

        cur_left_pos, cur_left_quat = site_pose("end_effector_l", current_qpos)
        cur_right_pos, cur_right_quat = site_pose("end_effector_r", current_qpos)

        left_delta = float(np.linalg.norm(left_pos - cur_left_pos))
        left_rot_delta = _quat_angle_deg(cur_left_quat, left_quat)
        if left_delta > MAX_POS_DELTA or left_rot_delta > MAX_ROT_DELTA_DEG:
            print(
                f"[wbik_gui] reject left target: dpos={left_delta:.3f}m"
                f" drot={left_rot_delta:.1f}deg (limits {MAX_POS_DELTA}m {MAX_ROT_DELTA_DEG}deg)"
            )
            return False

        right_delta = float(np.linalg.norm(right_pos - cur_right_pos))
        right_rot_delta = _quat_angle_deg(cur_right_quat, right_quat)
        if right_delta > MAX_POS_DELTA or right_rot_delta > MAX_ROT_DELTA_DEG:
            print(
                f"[wbik_gui] reject right target: dpos={right_delta:.3f}m"
                f" drot={right_rot_delta:.1f}deg (limits {MAX_POS_DELTA}m {MAX_ROT_DELTA_DEG}deg)"
            )
            return False

        return True

    # Initialize from model default
    mujoco.mj_forward(model, data)
    # Use IK's nominal pose as initial posture for both IK and viewer
    nominal_qpos = ik._get_nominal_posture(ik.data.qpos.copy())
    ik.data.qpos[:] = nominal_qpos
    mujoco.mj_forward(ik.model, ik.data)
    ik.configuration.update(q=ik.data.qpos)

    data.qpos[:] = nominal_qpos
    current_qpos = data.qpos.copy()
    prev_qpos = current_qpos.copy()

    # Get mocap ids for target bodies and initialize them at current EE poses
    ee_l_mid = model.body("ee_l_target").mocapid[0]
    ee_r_mid = model.body("ee_r_target").mocapid[0]
    head_mid = model.body("head_target").mocapid[0]

    left_nominal_pos = site_pos("end_effector_l", current_qpos)
    right_nominal_pos = site_pos("end_effector_r", current_qpos)
    head_nominal_pos, head_nominal_quat = site_pose("head", current_qpos)

    left_nominal_quat = data.xquat[model.body("EE_BODY_L").id].copy()
    right_nominal_quat = data.xquat[model.body("EE_BODY_R").id].copy()

    data.mocap_pos[ee_l_mid] = left_nominal_pos
    data.mocap_pos[ee_r_mid] = right_nominal_pos
    data.mocap_pos[head_mid] = head_nominal_pos
    data.mocap_quat[ee_l_mid] = left_nominal_quat
    data.mocap_quat[ee_r_mid] = right_nominal_quat
    data.mocap_quat[head_mid] = head_nominal_quat

    # Passive viewer loop
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        rate = RateLimiter(frequency=30.0, warn=False)

        while viewer.is_running():
            # Read targets from mocap spheres (drag with mouse in viewer)
            left_pos = data.mocap_pos[ee_l_mid].copy()
            left_quat = data.mocap_quat[ee_l_mid].copy()
            right_pos = data.mocap_pos[ee_r_mid].copy()
            right_quat = data.mocap_quat[ee_r_mid].copy()
            head_pos = data.mocap_pos[head_mid].copy()
            head_quat = data.mocap_quat[head_mid].copy()

            # Run whole-body IK from current viewer state
            current_qpos = data.qpos.copy()
            if not _passes_incremental_safety(
                current_qpos,
                left_pos,
                left_quat,
                right_pos,
                right_quat,
                head_pos,
                head_quat,
            ):
                print("[wbik_gui] resetting targets to current pose due to incremental safety limits")
                # Snap mocap targets back to current EE poses to keep the UI responsive.
                left_pos, left_quat = site_pose("end_effector_l", current_qpos)
                right_pos, right_quat = site_pose("end_effector_r", current_qpos)
                head_pos, head_quat = site_pose("head", current_qpos)
                data.mocap_pos[ee_l_mid] = left_pos
                data.mocap_pos[ee_r_mid] = right_pos
                data.mocap_pos[head_mid] = head_pos
                data.mocap_quat[ee_l_mid] = left_quat
                data.mocap_quat[ee_r_mid] = right_quat
                data.mocap_quat[head_mid] = head_quat
            ik_start = time.perf_counter()
            sol_qpos, sol_qvel, success, _info = ik.solve(
                left_target_pos=left_pos,
                left_target_quat=left_quat,
                right_target_pos=right_pos,
                right_target_quat=right_quat,
                head_target_pos=None,
                head_target_quat=None,
                current_qpos=current_qpos,
                dt=rate.dt,
            )
            ik_elapsed_ms = (time.perf_counter() - ik_start) * 1000.0
            # print(f"[wbik_gui] IK solve took {ik_elapsed_ms:.3f} ms")
            if not success:
                print(f"[wbc] IK failed: {_info}")

            # Apply solution to viewer
            data.qpos[:] = sol_qpos
            mujoco.mj_forward(model, data)
            mujoco.mj_camlight(model, data)

            # Render
            viewer.sync()
            rate.sleep()

if __name__ == "__main__":
    main()
