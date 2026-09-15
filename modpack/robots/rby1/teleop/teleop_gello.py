"""Gello teleoperation interface: subscribes to RMQ joint position topics and
produces JointTargets for the WBC trajectory loop."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import json
import math
import sys
import mujoco
import numpy as np
from robotmq.utils import deserialize, serialize
from scipy.spatial.transform import Rotation

MODPACK_ROOT = Path(__file__).resolve().parent.parent.parent.parent
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

PROJECT_ROOT = Path(MODPACK_ROOT) / "robots" / "rby1"

# def _add_modpack_root() -> None:
#     env_root = os.environ.get("MODPACK_ROOT")
#     candidates = [Path(env_root).expanduser()] if env_root else []

#     # rby1 lives at <modpack>/modpack/robots/rby1, so the repo root that
#     # holds the 'modpack' package is a fixed three levels up. Keep the older
#     # sibling-checkout guesses as fallbacks.
#     candidates.extend(
#         [
#             PROJECT_ROOT.parent.parent.parent,
#             PROJECT_ROOT.parent / "modpack",
#             PROJECT_ROOT,
#         ]
#     )

#     for candidate in candidates:
#         if (candidate / "modpack").is_dir():
#             candidate_str = str(candidate)
#             if candidate_str not in sys.path:
#                 sys.path.insert(0, candidate_str)
#             return

#     raise ModuleNotFoundError(
#         "Could not locate the 'modpack' package. "
#         "Set MODPACK_ROOT to the modpack repository root."
#     )


# _add_modpack_root()

from modpack.orchestration.message_formats import (
    BaseProprioMessage,
    MessageFactory,
    Topics,
    deserialize_neck_message,
    serialize_base_proprio_message,
    serialize_torque_message,
)

from rby1.base_targets import BaseTargets
from rby1.joint_targets import JointTargets
from demo.trajectory_recorder import TrajectoryRecorder
from .session_logger import SessionLogger, generate_session_name

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)-8s - %(message)s"
)


@dataclass
class GelloState:
    left_qpos: Optional[np.ndarray] = None
    right_qpos: Optional[np.ndarray] = None
    left_gripper: float = 0.0
    right_gripper: float = 0.0
    timestamp: float = 0.0
    controller_state: dict = field(default_factory=dict)
    head_qpos: Optional[np.ndarray] = None  # [pan, tilt] radians (head_0, head_1)


class TeleopGello:
    """Subscribes to ModPack RMQ topics and exposes joint targets for the WBC."""

    # Gripper position from gello is [0, 1]; convert to meters (0 = open = 0.1m)
    _GRIPPER_OPEN_M = 0.1

    def __init__(
        self,
        wbc,
        host: str,
        port: int,
        activation_host: str,
        activation_port: int,
        active_arms: Optional[List[str]] = None,
        joint_offsets: Optional[Dict[str, List[float]]] = None,
        joint_signs: Optional[Dict[str, List[float]]] = None,
        save_trajectory: bool = False,
        active_systems: Optional[Dict[str, bool]] = None,
        torque_feedback_enabled: bool = False,
        torque_feedback_gain: float = 1.0,
        torque_feedback_debug: bool = False,
        torque_feedback_debug_period_s: float = 2.0,
        urdf_path: str = "",
        bridge=None,
    ):
        self.wbc = wbc
        self.host = host
        self.port = port
        self._activation_host = activation_host or host
        self._activation_port = activation_port
        self.active_systems = active_systems or {"arms": True, "body": True}
        self._head_enabled: bool = self.active_systems.get("head", False)
        self._factory = MessageFactory()
        self._base_pub_enabled: bool = self.active_systems.get("base_pub", False)
        self._base_pub_seq: int = 0
        self.active_arms = active_arms or ["left", "right"]
        self.joint_offsets = joint_offsets or {arm: [0.0] * 7 for arm in self.active_arms}
        self.joint_signs = joint_signs or {arm: [1.0] * 7 for arm in self.active_arms}

        self._state_lock = threading.Lock()
        self.gello_state = GelloState()
        self._base_targets = BaseTargets()

        self._last_gripper_log_t: Dict[str, float] = {"left": 0.0, "right": 0.0}
        self._last_gripper_log_pos: Dict[str, Optional[float]] = {"left": None, "right": None}

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._bridge = bridge

        self.session_name = generate_session_name()
        self._session_logger = SessionLogger(self.session_name)
        self._trajectory_recorder = TrajectoryRecorder(self.session_name, enabled=save_trajectory)

        # Torque feedback: J^T * F_ee → per-joint torques sent back to Gello PC
        self._dyn_robot = None
        self._dyn_state_left = None
        self._dyn_state_right = None
        self._base_link_idx_left: int = 0
        self._base_link_idx_right: int = 0
        self._left_ee_idx: int = 0
        self._right_ee_idx: int = 0
        self._left_arm_cols: List[int] = []
        self._right_arm_cols: List[int] = []
        self._snap_to_dyn_qidx: Optional[np.ndarray] = None
        self._torque_feedback_requested = bool(torque_feedback_enabled)
        self._torque_feedback_gain = torque_feedback_gain
        self._torque_feedback_debug = bool(torque_feedback_debug)
        self._torque_feedback_debug_period_s = float(torque_feedback_debug_period_s)
        self._torque_diag_last_t = 0.0
        self._torque_gate_info_last_t = 0.0
        self._torque_gate_open_last: Optional[bool] = None
        self._torque_sent_count: Dict[str, int] = {"left": 0, "right": 0}
        self._torque_put_fail_count: Dict[str, int] = {"left": 0, "right": 0}
        self._torque_compute_fail_count: Dict[str, int] = {"left": 0, "right": 0}
        self._torque_other_fail_count: int = 0

        if torque_feedback_enabled:
            self._init_torque_feedback(urdf_path)

    def _init_torque_feedback(self, urdf_path: str) -> None:
        """Initialize RBY1 SDK dynamics model for J^T * F_ee torque projection."""
        rby_dyn = None
        try:
            import rby1_sdk.dynamics as rby_dyn  # type: ignore[assignment]
        except ImportError:
            sdk_py = str((PROJECT_ROOT / "rby1-sdk" / "python").resolve())
            if os.path.isdir(sdk_py) and sdk_py not in sys.path:
                sys.path.insert(0, sdk_py)
                try:
                    import rby1_sdk.dynamics as rby_dyn  # type: ignore[no-redef]
                except ImportError:
                    rby_dyn = None

        if rby_dyn is None:
            logging.warning("rby1_sdk.dynamics not available; torque feedback disabled.")
            return

        urdf_path = str(urdf_path).strip()
        if not urdf_path:
            raise ValueError(
                "torque_feedback_enabled is true but urdf_path is empty; "
                "set config key 'urdf_path' to something like "
                "'rby1-sdk/models/rby1m/urdf/model.urdf'."
            )

        urdf_path = os.path.expanduser(os.path.expandvars(urdf_path))
        p = Path(urdf_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        p = p.resolve()
        if not p.exists():
            raise FileNotFoundError(str(p))

        logging.info("[torque_feedback] loading URDF: %s", str(p))
        try:
            robot_config = rby_dyn.load_robot_from_urdf(str(p), "base")
        except Exception as exc:
            raise RuntimeError(f"Load URDF failed: {p}") from exc
        dyn_robot = rby_dyn.Robot(robot_config)

        # Joint names expected for RBY1-M URDF (WBC snapshot uses the same SDK ordering):
        #   wheel_fr, wheel_fl, wheel_rr, wheel_rl
        #   torso_0..5
        #   right_arm_0..6, left_arm_0..6
        #   head_0..1
        if hasattr(dyn_robot, "get_joint_names"):
            all_joint_names = list(dyn_robot.get_joint_names())
        else:
            joint_names_attr = getattr(dyn_robot, "joint_names", None)
            if joint_names_attr is None:
                raise AttributeError(
                    "Robot does not expose joint names (expected get_joint_names() or joint_names)."
                )
            all_joint_names = list(joint_names_attr() if callable(joint_names_attr) else joint_names_attr)

        # WBC snapshot joint ordering for RBY1-M (see control/rby1_wbc.py:_build_joint_mapping).
        snap_joint_names = [
            "wheel_fr",
            "wheel_fl",
            "wheel_rr",
            "wheel_rl",
            "torso_0",
            "torso_1",
            "torso_2",
            "torso_3",
            "torso_4",
            "torso_5",
            "right_arm_0",
            "right_arm_1",
            "right_arm_2",
            "right_arm_3",
            "right_arm_4",
            "right_arm_5",
            "right_arm_6",
            "left_arm_0",
            "left_arm_1",
            "left_arm_2",
            "left_arm_3",
            "left_arm_4",
            "left_arm_5",
            "left_arm_6",
            "head_0",
            "head_1",
        ]
        snap_name_to_idx = {name: i for i, name in enumerate(snap_joint_names)}
        try:
            self._snap_to_dyn_qidx = np.array([snap_name_to_idx[name] for name in all_joint_names], dtype=int)
        except KeyError as exc:
            missing = str(exc)
            raise KeyError(
                f"Dynamics joint name not found in expected WBC snapshot order: {missing}. "
                f"dyn_joint_names={all_joint_names}"
            ) from exc
        right_arm_joints = [f"right_arm_{i}" for i in range(7)]
        left_arm_joints = [f"left_arm_{i}" for i in range(7)]
        self._right_arm_cols = [all_joint_names.index(j) for j in right_arm_joints]
        self._left_arm_cols = [all_joint_names.index(j) for j in left_arm_joints]

        self._dyn_state_left = dyn_robot.make_state(["base", "ee_left"], all_joint_names)
        self._dyn_state_right = dyn_robot.make_state(["base", "ee_right"], all_joint_names)
        left_links = list(self._dyn_state_left.get_link_names())
        right_links = list(self._dyn_state_right.get_link_names())
        self._base_link_idx_left = left_links.index("base")
        self._left_ee_idx = left_links.index("ee_left")
        self._base_link_idx_right = right_links.index("base")
        self._right_ee_idx = right_links.index("ee_right")
        self._dyn_robot = dyn_robot
        logging.info(
            "Torque feedback initialized (URDF: %s). right_arm_cols=%s, left_arm_cols=%s",
            str(p),
            self._right_arm_cols,
            self._left_arm_cols,
        )

    def initialize(self) -> bool:
        self._stop_event.clear()
        logging.info("Using RobotBridge for RMQ communication at %s:%d", self.host, self.port)

        if self._torque_feedback_requested and self._torque_feedback_debug:
            logging.info(
                "[torque_feedback] requested=True dyn_model_ready=%s",
                self._dyn_robot is not None,
            )

        if self._dyn_robot is not None and self._torque_feedback_requested:
            logging.info("[torque_feedback] Torque feedback enabled — using bridge for publish.")

        self._thread = threading.Thread(
            target=self._subscriber_loop, name="gello-subscriber", daemon=True
        )
        self._thread.start()
        return True

    def _send_vp_activation(self, delay: float = 2.0) -> None:
        """Send VP activation after a short delay so vision_pro_process is ready."""
        def _send():
            time.sleep(delay)
            try:
                self._bridge.publish_activation_command("activate", target="vision_pro")
                logging.info("Sent VP activation via bridge")
            except Exception as exc:
                logging.warning("Failed to send VP activation: %s", exc)
        threading.Thread(target=_send, name="vp-activator", daemon=True).start()

    def start(self) -> None:
        if self._head_enabled:
            self._send_vp_activation()
        logging.info("Gello teleop running. Move the gello arms to begin.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._session_logger.close()
        self._trajectory_recorder.close()

    def _subscriber_loop(self) -> None:
        """Poll RMQ topics for each active arm and update GelloState."""
        TOPIC_MAP = {"left": "left", "right": "right"}
        _to = 1.5
        while not self._stop_event.is_set():
            controller_snapshot = None
            for arm in (self.active_arms if self.active_systems.get("arms", True) else []):
                topic = TOPIC_MAP[arm]
                raw_list = self._bridge.peek_topic(topic, timeout_s=_to)
                if not raw_list:
                    continue

                # Use the latest message only
                raw = raw_list[-1]
                try:
                    msg = deserialize(raw)
                except Exception as exc:
                    logging.warning("Failed to deserialize gello message: %s", exc)
                    continue

                joint_positions = msg.get("joint_positions")
                gripper_position = msg.get("gripper_position", 0.0)
                timestamp = msg.get("timestamp", time.time())

                if joint_positions is None or len(joint_positions) != 7:
                    logging.warning(
                        "Unexpected joint_positions length from arm '%s': %s", arm, joint_positions
                    )
                    continue

                self._maybe_log_gripper_input(arm, float(gripper_position))

                with self._state_lock:
                    prev_qpos = (
                        self.gello_state.left_qpos if arm == "left"
                        else self.gello_state.right_qpos
                    )
                qpos = self._apply_mapping(
                    arm, np.array(joint_positions, dtype=np.float64), prev_qpos
                )

                with self._state_lock:
                    if arm == "left":
                        self.gello_state.left_qpos = qpos
                        self.gello_state.left_gripper = float(gripper_position)
                    else:
                        self.gello_state.right_qpos = qpos
                        self.gello_state.right_gripper = float(gripper_position)
                    self.gello_state.timestamp = timestamp
                    self.gello_state.controller_state[arm] = {
                        "joint_positions": qpos.tolist(),
                        "gripper_position": float(gripper_position),
                        "timestamp": timestamp,
                    }
                    controller_snapshot = dict(self.gello_state.controller_state)

            if controller_snapshot:
                self._session_logger.log_controller_state(controller_snapshot)

            base_list = []
            if self.active_systems.get("body", True):
                base_list = self._bridge.peek_topic("body_target", timeout_s=_to) or []
                logging.debug("[teleop] body_target poll: %d messages", len(base_list))

            if base_list:
                try:
                    msg = json.loads(base_list[0].decode("utf-8"))
                    pose = msg.get("pose", [])
                    # logging.info("[teleop] body_target received: pose=%s", pose)
                    if len(pose) >= 3:
                        x = float(pose[0])
                        y = float(pose[1])
                        # theta = float(pose[2])
                        # Pure yaw rotation: q = [cos(θ/2), 0, 0, sin(θ/2)] in MuJoCo [w,x,y,z]
                        
                        # body_target carries [x, y, theta] (BaseProprioMessage.pose);
                        # rebuild the MuJoCo [w,x,y,z] yaw quaternion from theta. A 7+
                        # element pose is treated as [x,y,z,qx,qy,qz,qw].
                        if len(pose) >= 7:
                            q = pose[3:7]
                            target_quat = np.array([q[-1], q[0], q[1], q[2]])  # [x,y,z,w] -> [w,x,y,z]
                        else:
                            theta = float(pose[2])
                            half = theta * 0.5
                            target_quat = np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)
                        # half = theta * 0.5
                        # quat = np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)
                        ts = float(msg.get("timestamp", time.time()))
                        self._base_targets.set_targets(x, y, target_quat, timestamp=ts)
                except Exception as exc:
                    logging.warning("Failed to deserialize body_target message: %s", exc)

            # Neck pose tracking from Vision Pro via ModPack
            if self._head_enabled:
                neck_list = self._bridge.peek_topic("neck_target_pose", timeout_s=_to) or []
                if neck_list:
                    try:
                        neck_msg = deserialize_neck_message(neck_list[-1])
                        if neck_msg.data_valid and len(neck_msg.neck_pose) == 7:
                            p = neck_msg.neck_pose  # [x, y, z, qx, qy, qz, qw]
                            euler = Rotation.from_quat([p[3], p[4], p[5], p[6]]).as_euler("ZYX")
                            with self._state_lock:
                                self.gello_state.head_qpos = np.array(
                                    [float(euler[0]), float(euler[1])], dtype=np.float64
                                )
                    except Exception as exc:
                        logging.warning("Failed to decode neck_target_pose: %s", exc)

            # Publish current base odometry so vision_pro_process can compensate for base rotation
            snap = None
            if self.wbc is not None:
                try:
                    snap = self.wbc.get_latest_robot_state()
                except Exception as exc:
                    logging.debug("Failed to get robot state: %s", exc)

            if snap is not None and snap.is_valid:
                try:
                    q_snap = np.array(snap.joint_position, dtype=np.float64).reshape(-1)
                    # WBC snapshot ordering: wheels(4) + torso(6) + right_arm(7) + left_arm(7) + head(2)
                    RIGHT_ARM_SLICE = slice(10, 17)
                    LEFT_ARM_SLICE = slice(17, 24)
                    HEAD_SLICE = slice(24, 26)
                    ts_now = time.time()
                    # Read commanded targets per arm (may be None before first IK solve)
                    _jt = self.wbc.joint_targets if self.wbc is not None else None
                    _r_target = _l_target = None
                    if _jt is not None:
                        with _jt.lock:
                            _r_target = _jt.right_qpos
                            _l_target = _jt.left_qpos

                    # Publish each modality independently so partial states are still logged
                    _left_gripper_m, _right_gripper_m = self.wbc.get_latest_gripper_widths()
                    with self._state_lock:
                        _right_gripper_target_m = self._gripper_to_meters(self.gello_state.right_gripper)
                        _left_gripper_target_m = self._gripper_to_meters(self.gello_state.left_gripper)
                    if q_snap.shape[0] >= 17:
                        right_state = {"positions": q_snap[RIGHT_ARM_SLICE].tolist(), "timestamp": ts_now, "data_valid": True, "gripper": _right_gripper_m, "gripper_target": _right_gripper_target_m}
                        if _r_target is not None:
                            right_state["target_positions"] = np.asarray(_r_target, dtype=np.float64).tolist()
                        self._bridge.publish_state("right_arm", right_state)
                    if q_snap.shape[0] >= 24:
                        left_state = {"positions": q_snap[LEFT_ARM_SLICE].tolist(), "timestamp": ts_now, "data_valid": True, "gripper": _left_gripper_m, "gripper_target": _left_gripper_target_m}
                        if _l_target is not None:
                            left_state["target_positions"] = np.asarray(_l_target, dtype=np.float64).tolist()
                        self._bridge.publish_state("left_arm", left_state)
                    if q_snap.shape[0] >= 26:
                        head_state = {"positions": q_snap[HEAD_SLICE].tolist(), "timestamp": ts_now, "data_valid": True}
                        self._bridge.publish_state("head", head_state)
                    for ft_role, ft_valid, wrench in [
                        ("right_ft", getattr(snap, "right_ft_valid", False), getattr(snap, "right_ee_wrench", None)),
                        ("left_ft", getattr(snap, "left_ft_valid", False), getattr(snap, "left_ee_wrench", None)),
                    ]:
                        if ft_valid and wrench is not None:
                            ft_state = {"data": np.asarray(wrench, dtype=np.float64).tolist(), "timestamp": ts_now, "data_valid": True}
                            self._bridge.publish_state(ft_role, ft_state)
                except Exception as exc:
                    logging.debug("Failed to publish arm joint states: %s", exc)

            if self._head_enabled:
                try:
                    with self._state_lock:
                        hq = self.gello_state.head_qpos
                    if hq is not None:
                        head_cmd = {
                            "joint_positions": hq.tolist(),
                            "timestamp": time.time(),
                            "frame_id": "rby1_joint_cmd",
                        }
                        self._bridge.publish_state("head_cmd", head_cmd)
                except Exception as exc:
                    logging.debug("Failed to publish head cmd: %s", exc)

            if self._base_pub_enabled and snap is not None and snap.is_valid:
                try:
                    odom_fn = getattr(self.wbc, "snapshot_odom_SE2", None)
                    if callable(odom_fn):
                        T = np.asarray(odom_fn(snap), dtype=float)
                    else:
                        T = np.asarray(snap.odom_SE2, dtype=float)
                    bx, by = float(T[0, 2]), float(T[1, 2])
                    byaw = math.atan2(float(T[1, 0]), float(T[0, 0]))
                    self._base_pub_seq += 1
                    base_state = {
                        "pose": [bx, by, byaw],
                        "timestamp": time.time(),
                        "sequence_id": self._base_pub_seq,
                    }
                    self._bridge.publish_state("body", base_state)
                except Exception as exc:
                    logging.debug("Failed to publish base pose: %s", exc)

            # Torque feedback: project F/T wrench back to arm joint torques via J^T * F
            dyn_ok = self._dyn_robot is not None
            fb_ok = self._bridge is not None
            snap_ok = snap is not None and snap.is_valid
            torque_gate_open = bool(dyn_ok and fb_ok and snap_ok)

            if self._torque_feedback_debug and self._torque_feedback_requested and not torque_gate_open:
                now = time.monotonic()
                if (now - self._torque_gate_info_last_t) >= 5.0:
                    self._torque_gate_info_last_t = now
                    logging.info(
                        "[torque_feedback] gate CLOSED dyn=%s fb=%s snap_valid=%s",
                        dyn_ok,
                        fb_ok,
                        snap_ok,
                    )

            if self._torque_feedback_debug:
                now = time.monotonic()
                if (
                    self._torque_gate_open_last is None
                    or torque_gate_open != self._torque_gate_open_last
                    or (now - self._torque_diag_last_t) >= self._torque_feedback_debug_period_s
                ):
                    self._torque_diag_last_t = now
                    self._torque_gate_open_last = torque_gate_open
                    snap_ts = getattr(snap, "timestamp", None) if snap is not None else None
                    left_ft_valid = getattr(snap, "left_ft_valid", None) if snap is not None else None
                    right_ft_valid = getattr(snap, "right_ft_valid", None) if snap is not None else None
                    logging.info(
                        "[torque_feedback] gate=%s dyn=%s fb=%s snap_valid=%s active_arms=%s "
                        "left_ft_valid=%s right_ft_valid=%s sent=%s put_fail=%s compute_fail=%s other_fail=%d snap_ts=%s",
                        torque_gate_open,
                        dyn_ok,
                        fb_ok,
                        snap_ok,
                        self.active_arms,
                        left_ft_valid,
                        right_ft_valid,
                        self._torque_sent_count,
                        self._torque_put_fail_count,
                        self._torque_compute_fail_count,
                        self._torque_other_fail_count,
                        snap_ts,
                    )

            if torque_gate_open:
                try:
                    q_snap = np.array(snap.joint_position, dtype=np.float64).reshape(-1)
                    if self._snap_to_dyn_qidx is None:
                        raise RuntimeError("Torque feedback dynamics initialized without snapshot-to-dynamics mapping.")
                    if q_snap.shape[0] <= int(np.max(self._snap_to_dyn_qidx)):
                        raise ValueError(
                            f"snapshot.joint_position length {q_snap.shape[0]} does not match expected joint mapping."
                        )
                    q = q_snap[self._snap_to_dyn_qidx].reshape(-1, 1)
                except Exception as exc:
                    if self._torque_feedback_debug:
                        self._torque_other_fail_count += 1
                        logging.warning("[torque_feedback] failed to read snap.joint_position: %s", exc)
                    q = None

                if q is not None:
                    for arm, dyn_state, ee_idx, arm_cols, ft_valid, wrench in [
                        (
                            "left",
                            self._dyn_state_left,
                            (self._base_link_idx_left, self._left_ee_idx),
                            self._left_arm_cols,
                            snap.left_ft_valid,
                            snap.left_ee_wrench,
                        ),
                        (
                            "right",
                            self._dyn_state_right,
                            (self._base_link_idx_right, self._right_ee_idx),
                            self._right_arm_cols,
                            snap.right_ft_valid,
                            snap.right_ee_wrench,
                        ),
                    ]:
                        if arm not in self.active_arms:
                            continue
                        try:
                            dyn_state.set_q(q)
                            self._dyn_robot.compute_forward_kinematics(dyn_state)
                            # J shape: (6, total_dof); slice to arm DOF only
                            base_idx, target_idx = ee_idx
                            J = self._dyn_robot.compute_body_jacobian(
                                dyn_state, base_idx, target_idx
                            )
                            J_arm = J[:, arm_cols]  # (6, 7)
                            # Snapshot wrench is [Fx, Fy, Fz, Tx, Ty, Tz] from realtime_driver.cc.
                            # Body Jacobian uses spatial vector convention [ω; v], so the dual wrench should be
                            # [Tx, Ty, Tz, Fx, Fy, Fz] to compute joint torques via tau = J^T * wrench.
                            if ft_valid:
                                wrench_raw = np.asarray(wrench, dtype=np.float64).reshape(-1)
                                if wrench_raw.size != 6:
                                    raise ValueError(f"Unexpected wrench size: {wrench_raw.size}")
                                force = wrench_raw[:3]
                                torque = wrench_raw[3:]
                                wrench_vec = np.concatenate([torque, force], axis=0)
                            else:
                                force = np.zeros(3, dtype=np.float64)
                                torque = np.zeros(3, dtype=np.float64)
                                wrench_vec = np.zeros(6, dtype=np.float64)
                            tau = J_arm.T @ wrench_vec  # (7,)
                            # Apply sign correction to map from robot frame to Gello frame
                            tau_gello = (
                                np.array(self.joint_signs.get(arm, [1.0] * 7))
                                * tau
                                * self._torque_feedback_gain
                            )
                        except Exception as exc:
                            if self._torque_feedback_debug:
                                self._torque_compute_fail_count[arm] += 1
                                logging.warning("[torque_feedback] compute failed arm=%s: %s", arm, exc)
                            continue

                        try:
                            torque_msg = self._factory.create_torque_message(
                                arm=arm,
                                joint_torques=tau_gello.tolist(),  # 6 actuated arm DOF
                                gripper_torque=0.0,
                            )
                            topic = Topics.torque_topic(arm)
                            self._bridge.publish_feedback_raw(
                                topic,
                                serialize_torque_message(torque_msg),
                            )
                            if self._torque_feedback_debug:
                                self._torque_sent_count[arm] += 1
                                now = time.monotonic()
                                if (now - self._torque_diag_last_t) >= self._torque_feedback_debug_period_s:
                                    self._torque_diag_last_t = now
                                    logging.info(
                                        "[torque_feedback] sent arm=%s topic=%s ft_valid=%s |force|=%.3f |torque|=%.3f tau6=%s",
                                        arm,
                                        topic,
                                        bool(ft_valid),
                                        float(np.linalg.norm(force)),
                                        float(np.linalg.norm(torque)),
                                        np.array2string(
                                            np.asarray(tau_gello),
                                            precision=6,
                                            suppress_small=True,
                                        ),
                                    )
                        except Exception as exc:
                            if self._torque_feedback_debug:
                                self._torque_put_fail_count[arm] += 1
                                logging.warning("[torque_feedback] put_data failed arm=%s: %s", arm, exc)

            time.sleep(0.005)  # ~200 Hz poll, gello publishes at 100 Hz

    def _apply_mapping(
        self, arm: str, raw: np.ndarray, prev: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply per-arm joint offsets and signs.

        Wraps each joint to the nearest equivalent angle relative to `prev`
        (the last accepted qpos), eliminating discontinuities at the 0/2π boundary.
        On the first call (prev is None) wraps to [-pi, pi].
        """
        offsets = np.array(self.joint_offsets.get(arm, [0.0] * 7), dtype=np.float64)
        signs = np.array(self.joint_signs.get(arm, [1.0] * 7), dtype=np.float64)
        result = signs * raw + offsets
        reference = prev if prev is not None else np.zeros(7)
        return reference + (result - reference + np.pi) % (2 * np.pi) - np.pi

    def _maybe_log_gripper_input(self, arm: str, gripper_position: float) -> None:
        now = time.monotonic()
        last_t = float(self._last_gripper_log_t.get(arm, 0.0))
        last_pos = self._last_gripper_log_pos.get(arm)
        should_log = (now - last_t) >= 1.0
        if last_pos is None:
            should_log = True
        else:
            try:
                should_log = should_log or abs(float(gripper_position) - float(last_pos)) >= 0.05
            except Exception:
                should_log = True

        if should_log:
            self._last_gripper_log_t[arm] = now
            self._last_gripper_log_pos[arm] = float(gripper_position)
            logging.debug("[teleop] gripper input arm=%s pos=%.3f", arm, float(gripper_position))

    def wait_for_first_pose(self) -> tuple:
        """Block until the subscriber loop has received a valid pose for each active arm."""
        while True:
            with self._state_lock:
                left = self.gello_state.left_qpos
                right = self.gello_state.right_qpos
            left_ready = left is not None or "left" not in self.active_arms
            right_ready = right is not None or "right" not in self.active_arms
            if left_ready and right_ready:
                return left, right
            time.sleep(0.05)

    def _gripper_to_meters(self, gripper: float) -> float:
        """Convert gello gripper [0=closed, 1=open] to width in meters."""
        return min(1.0, max(0.0, gripper)) * self._GRIPPER_OPEN_M

    def _get_arm_joint_limits(self, arm: str):
        """Return (lower, upper) arrays of shape (7,) from the MuJoCo model, or None."""
        if self.wbc is None:
            return None, None
        ik = getattr(self.wbc, "ik_solver", None)
        if ik is None:
            return None, None
        joint_names = (
            ik.left_arm_joint_names if arm == "left" else ik.right_arm_joint_names
        )
        lower, upper = [], []
        for name in joint_names:
            jid = mujoco.mj_name2id(ik.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0 or not ik.model.jnt_limited[jid]:
                lower.append(-np.pi)
                upper.append(np.pi)
            else:
                lower.append(float(ik.model.jnt_range[jid, 0]))
                upper.append(float(ik.model.jnt_range[jid, 1]))
        return np.array(lower), np.array(upper)

    def compute_target(self) -> Optional[JointTargets]:
        with self._state_lock:
            left_qpos = self.gello_state.left_qpos
            right_qpos = self.gello_state.right_qpos
            left_gripper = self.gello_state.left_gripper
            right_gripper = self.gello_state.right_gripper
            timestamp = self.gello_state.timestamp

        snapshot = None
        current_qpos = None
        if self.wbc is not None:
            snapshot = self.wbc.get_latest_robot_state()
            if snapshot is not None:
                current_qpos = self.wbc.snapshot_to_qpos(snapshot)

        if left_qpos is None:
            if "left" in self.active_arms or current_qpos is None:
                return None
            left_indices = np.array(self.wbc.ik_solver.left_arm_qpos_indices, dtype=int)
            left_qpos = current_qpos[left_indices]

        if right_qpos is None:
            if "right" in self.active_arms or current_qpos is None:
                return None
            right_indices = np.array(self.wbc.ik_solver.right_arm_qpos_indices, dtype=int)
            right_qpos = current_qpos[right_indices]

        for arm, qpos in (("left", left_qpos), ("right", right_qpos)):
            lo, hi = self._get_arm_joint_limits(arm)
            if lo is not None:
                clipped = np.clip(qpos, lo, hi)
                if not np.array_equal(clipped, qpos):
                    logging.debug(
                        "[teleop] %s arm qpos clamped to joint limits: %s -> %s",
                        arm, np.array2string(qpos, precision=3), np.array2string(clipped, precision=3),
                    )
                if arm == "left":
                    left_qpos = clipped
                else:
                    right_qpos = clipped

        left_width = self._gripper_to_meters(left_gripper)
        right_width = self._gripper_to_meters(right_gripper)

        target = JointTargets(
            left_qpos=left_qpos.copy(),
            right_qpos=right_qpos.copy(),
            left_width=left_width,
            right_width=right_width,
            timestamp=timestamp,
        )
        self._trajectory_recorder.log_target(target, timestamp=time.time())

        return target

    def compute_head_target(self) -> Optional[np.ndarray]:
        """Return latest [pan, tilt] head joint targets, or None if head tracking is disabled."""
        if not self._head_enabled:
            return None
        with self._state_lock:
            hq = self.gello_state.head_qpos
            return None if hq is None else hq.copy()

    def compute_base_target(self) -> Optional[BaseTargets]:
        x, _, _ = self._base_targets.get_target()
        if x is None:
            return None
        return self._base_targets

    def on_target_rejected(self) -> None:
        logging.info("Gello target rejected: move gello arms back near last accepted pose.")
