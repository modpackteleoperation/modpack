"""Whole-body controller thread decoupled from any GUI interaction."""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import mujoco
import numpy as np
import yaml

from loop_rate_limiters import RateLimiter

from rby1.whole_body_ik import RBY1WholeBodyIK
from rby1.base_targets import BaseTargets
from rby1.ee_targets import EETargets, _normalize_quaternion
from rby1.joint_targets import JointTargets

from . import (
    AdmittanceController,
    AdmittanceControllerConfig,
    Config as ControllerConfig,
    RealtimeDriver,
    RobotSnapshot,
)
from ft.calibrator import FTCalibrator

from gripper.gripper_client import GripperClient

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


class RobotStateBuffer:
    """Stores the latest robot snapshot retrieved from the controller."""

    def __init__(self):
        self._lock = threading.Lock()
        self.latest: Optional[RobotSnapshot] = None
        self._gripper_widths: Tuple[float, float] = (0.0, 0.0)

    def store(self, snapshot: RobotSnapshot) -> None:
        with self._lock:
            self.latest = snapshot

    def load(self) -> Optional[RobotSnapshot]:
        with self._lock:
            return self.latest

    def store_gripper_widths(
        self,
        left_width: Optional[float],
        right_width: Optional[float],
    ) -> None:
        with self._lock:
            current_left, current_right = self._gripper_widths
            if left_width is not None:
                current_left = float(left_width)
            if right_width is not None:
                current_right = float(right_width)
            self._gripper_widths = (current_left, current_right)

    def load_gripper_widths(self) -> Tuple[float, float]:
        with self._lock:
            return self._gripper_widths


class RBY1WBC:
    """Worker that streams IK solutions to the realtime controller."""

    def __init__(self, config_path: str = PROJECT_ROOT + "/config/wbc.yaml"):
        # Load Controller Config
        try:
            config_path = Path(config_path)
            with config_path.open("r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            raise Exception(f"Exception while loading config file: {e}")
        if not isinstance(self.config, dict):
            raise ValueError(f"WBC config at {config_path} must be a mapping.")

        def require(name: str):
            if name not in self.config:
                raise KeyError(f"Missing required WBC config key: {name}")
            return self.config[name]

        self.address = require("address")
        self.model_path = PROJECT_ROOT + str(require("model_path"))
        self.init_position = require("init_position")
        self.state_frequency_hz = require("state_frequency_hz")
        self.trajectory_frequency_hz = require("trajectory_frequency_hz")
        self.ik_frequency_hz = require("ik_frequency_hz")
        self.use_interpolation = require("use_interpolation")
        self.command_timeout_sec = require("command_timeout_sec")
        self.base_error_gain = np.asarray(require("base_error_gain"), dtype=float)
        self.base_velocity_gain = np.asarray(require("base_velocity_gain"), dtype=float)
        self.low_pass_freq_hz = require("low_pass_freq_hz")
        self.init_position_max_step_delta = float(
            self.config.get("init_position_max_step_delta", 0.01)
        )

        base_reset_cfg = self.config.get("base_reset", {})
        self.base_reset_enabled = bool(base_reset_cfg.get("enabled", True))
        self.base_reset_pos_tolerance = float(
            base_reset_cfg.get("pos_tolerance_m", 0.01)
        )
        self.base_reset_yaw_tolerance_deg = float(
            base_reset_cfg.get("yaw_tolerance_deg", 0.1)
        )
        self.base_reset_max_linear_speed = float(
            base_reset_cfg.get("max_linear_speed", 0.2)
        )
        self.base_reset_max_yaw_speed = float(base_reset_cfg.get("max_yaw_speed", 0.8))
        self.base_reset_min_linear_speed = float(
            base_reset_cfg.get("min_linear_speed", 0.01)
        )
        self.base_reset_min_yaw_speed = float(base_reset_cfg.get("min_yaw_speed", 0.01))

        base_odometry_cfg = self.config.get("base_odometry", {})
        self.base_odometry_zero_at_start = bool(
            base_odometry_cfg.get("zero_at_start", True)
        )
        self._base_odom_lock = threading.Lock()
        self._base_odom_origin_inv: Optional[np.ndarray] = None

        incremental_cfg = self.config.get("incremental_safety", {})
        self.incremental_safety_enabled = bool(incremental_cfg.get("enabled", False))
        self.max_incremental_translation = float(
            incremental_cfg.get("max_translation_m", 0.05)
        )
        self.max_incremental_rotation_deg = float(
            incremental_cfg.get("max_rotation_deg", 20.0)
        )
        self.max_joint_step_delta = float(
            incremental_cfg.get("max_joint_step_rad", 0.2)
        )

        admittance_cfg = require("admittance")
        self.admittance_enabled = bool(admittance_cfg.get("enabled", False))
        self._admittance_initialized = False
        self._admittance_controller_left: Optional[AdmittanceController] = None
        self._admittance_controller_right: Optional[AdmittanceController] = None
        self._admittance_config_left: Optional[AdmittanceControllerConfig] = None
        self._admittance_config_right: Optional[AdmittanceControllerConfig] = None
        self._admittance_Tr: Optional[np.ndarray] = None
        self._admittance_n_af: int = 0
        self._admittance_wrench_left = np.zeros(6, dtype=float)
        self._admittance_wrench_right = np.zeros(6, dtype=float)

        if self.admittance_enabled:
            self._configure_admittance(admittance_cfg)
            self._ft_calibrator = FTCalibrator()
        else:
            self._ft_calibrator = None

        # Initialize Controller loops
        self.ik_rate = RateLimiter(frequency=self.ik_frequency_hz, warn=False)
        self.state_poll_rate = RateLimiter(
            frequency=self.state_frequency_hz, warn=False
        )

        self.ee_targets = EETargets()
        self.joint_targets = JointTargets()
        self.base_targets = BaseTargets()
        self.robot_state = RobotStateBuffer()

        self._stop = threading.Event()
        self._threads_started = False
        self._model_lock = threading.Lock()

        self.controller = self._init_controller(self.address, self.command_timeout_sec)

        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self._left_ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector_l"
        )
        self._right_ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector_r"
        )
        self._head_ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "head"
        )
        if self._left_ee_site_id < 0 or self._right_ee_site_id < 0:
            raise RuntimeError(
                "End-effector body IDs could not be resolved in the MuJoCo model."
            )

        self.ik_solver = RBY1WholeBodyIK()
        self._build_joint_mapping()
        self._extract_base_origin()

        # Initialize gripper client
        gripper_cfg = self.config.get("gripper", {})
        try:
            self.gripper = self._setup_gripper(gripper_cfg)
            if self.gripper is not None:
                print(f"Successfully initialized gripper")
            else:
                print("Gripper disabled in config")
        except Exception as exc:
            self.gripper = None
            print(f"Failed to initialize gripper: {exc}")

        self._state_thread: Optional[threading.Thread] = None
        self._ik_thread: Optional[threading.Thread] = None

    def start(self, skip_init_position: bool = False) -> None:
        if self._threads_started:
            return
        self._stop.clear()
        self._state_thread = threading.Thread(
            target=self._state_poll_loop, name="state-poll", daemon=True
        )
        self._ik_thread = threading.Thread(
            target=self._ik_loop, name="wbc", daemon=True
        )
        self._state_thread.start()
        self._ik_thread.start()
        self._threads_started = True
        self.wait_for_first_state()
        if self.base_reset_enabled:
            self._reset_base_pose()
        if not skip_init_position:
            self._set_init_position()

    def stop(self, join_timeout: float = 2.0) -> None:
        self._stop.set()
        if self._ik_thread is not None:
            self._ik_thread.join(timeout=join_timeout)
        if self._state_thread is not None:
            self._state_thread.join(timeout=join_timeout)
        self.controller.stop()
        self._threads_started = False

    def _setup_gripper(self, cfg: Mapping[str, Any]) -> Optional[Any]:
        mode = str(cfg.get("mode", "")).lower()
        if mode == "remote":
            auto_initialize = bool(cfg.get("auto_initialize", True))
            auto_homing = bool(cfg.get("auto_homing", True))
            auto_start = bool(cfg.get("auto_start", True))
            verbose_init = bool(cfg.get("verbose_init", False))

            host = cfg.get("host")
            if not host:
                raise ValueError("Remote gripper mode requires 'host' in config.")
            port = int(cfg.get("port", 5678))
            timeout = float(cfg.get("timeout", 2.0))
            gripper = GripperClient(host=host, port=port, timeout=timeout)

            if auto_initialize:
                if not gripper.initialize(verbose=verbose_init):
                    raise RuntimeError("Gripper initialize() returned False")
            if auto_homing:
                gripper.homing()
            if auto_start:
                gripper.start()
            return gripper
        elif mode in ("none", "disabled"):
            return None
        else:
            raise ValueError(f"Unknown gripper mode '{mode}'")

    def update_targets(
        self,
        left_pos: np.ndarray,
        left_quat: np.ndarray,
        right_pos: np.ndarray,
        right_quat: np.ndarray,
        left_width: Optional[float] = None,
        right_width: Optional[float] = None,
        head_pos: Optional[np.ndarray] = None,
        head_quat: Optional[np.ndarray] = None,
        duration: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> bool:
        if not self._passes_incremental_safety(
            left_pos,
            left_quat,
            right_pos,
            right_quat,
            head_pos,
            head_quat,
        ):
            print("[wbc] target command rejected by incremental safety.")
            return False

        if duration is None:
            duration = 1.0 / float(self.trajectory_frequency_hz)

        self.ee_targets.set_targets(
            left_pos,
            left_quat,
            right_pos,
            right_quat,
            left_width=left_width,
            right_width=right_width,
            head_pos=head_pos,
            head_quat=head_quat,
            duration=duration,
            timestamp=timestamp,
        )
        return True

    def update_joint_targets(
        self,
        left_qpos: Optional[np.ndarray],
        right_qpos: Optional[np.ndarray],
        left_width: Optional[float] = None,
        right_width: Optional[float] = None,
        head_qpos: Optional[np.ndarray] = None,
        duration: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> bool:
        # If arm qpos not provided, hold current positions (e.g. head-only update)
        if left_qpos is None or right_qpos is None:
            snapshot = self.get_latest_robot_state()
            current_qpos = (
                self.snapshot_to_qpos(snapshot) if snapshot is not None else None
            )
            if current_qpos is not None:
                left_indices = getattr(self.ik_solver, "left_arm_qpos_indices", [])
                right_indices = getattr(self.ik_solver, "right_arm_qpos_indices", [])
                if left_qpos is None and left_indices:
                    left_qpos = current_qpos[np.array(left_indices, dtype=int)]
                if right_qpos is None and right_indices:
                    right_qpos = current_qpos[np.array(right_indices, dtype=int)]
            if left_qpos is None or right_qpos is None:
                return False

        if not self._passes_joint_safety(left_qpos, right_qpos):
            print("[wbc] joint target command rejected by incremental safety.")
            return False

        if duration is None:
            duration = 1.0 / float(self.trajectory_frequency_hz)

        self.joint_targets.set_targets(
            left_qpos,
            right_qpos,
            left_width=left_width,
            right_width=right_width,
            head_qpos=head_qpos,
            duration=duration,
            timestamp=timestamp,
        )
        return True

    def update_base_targets(self, base_target: BaseTargets) -> None:
        x, y, quat = base_target.get_target()
        if x is not None and y is not None and quat is not None:
            self.base_targets.set_targets(x, y, quat, timestamp=base_target.timestamp)

    def _passes_joint_safety(
        self,
        left_qpos: np.ndarray,
        right_qpos: np.ndarray,
    ) -> bool:
        if not self.incremental_safety_enabled:
            return True

        snapshot = self.robot_state.load()
        if snapshot is None or not snapshot.is_valid:
            return True

        current_qpos = self.snapshot_to_qpos(snapshot)
        if current_qpos is None:
            return True

        left_indices = getattr(self.ik_solver, "left_arm_qpos_indices", [])
        right_indices = getattr(self.ik_solver, "right_arm_qpos_indices", [])

        if left_indices and left_qpos is not None:
            current_left = current_qpos[np.array(left_indices, dtype=int)]
            left_delta = float(np.max(np.abs(np.asarray(left_qpos) - current_left)))
            if left_delta > self.max_joint_step_delta:
                print(
                    f"[wbc] joint safety reject (left): max_delta={left_delta:.3f} rad"
                    f" (limit {self.max_joint_step_delta} rad)"
                )
                return False

        if right_indices and right_qpos is not None:
            current_right = current_qpos[np.array(right_indices, dtype=int)]
            right_delta = float(np.max(np.abs(np.asarray(right_qpos) - current_right)))
            if right_delta > self.max_joint_step_delta:
                print(
                    f"[wbc] joint safety reject (right): max_delta={right_delta:.3f} rad"
                    f" (limit {self.max_joint_step_delta} rad)"
                )
                return False

        return True

    def _passes_incremental_safety(
        self,
        left_pos: np.ndarray,
        left_quat: np.ndarray,
        right_pos: np.ndarray,
        right_quat: np.ndarray,
        head_pos: Optional[np.ndarray] = None,
        head_quat: Optional[np.ndarray] = None,
    ) -> bool:
        if not self.incremental_safety_enabled:
            return True

        snapshot = self.robot_state.load()
        if snapshot is None or not snapshot.is_valid:
            return True

        current_qpos = self.snapshot_to_qpos(snapshot)
        if current_qpos is None:
            return True

        with self._model_lock:
            self.data.qpos[:] = current_qpos
            mujoco.mj_forward(self.model, self.data)

            left_rot = self.data.site_xmat[self._left_ee_site_id]
            right_rot = self.data.site_xmat[self._right_ee_site_id]
            left_quat_current = np.zeros(4)
            right_quat_current = np.zeros(4)
            mujoco.mju_mat2Quat(left_quat_current, left_rot)
            mujoco.mju_mat2Quat(right_quat_current, right_rot)
            left_pos_current = self.data.site_xpos[self._left_ee_site_id].copy()
            right_pos_current = self.data.site_xpos[self._right_ee_site_id].copy()

        left_delta = float(
            np.linalg.norm(np.asarray(left_pos, dtype=float) - left_pos_current)
        )
        left_rot_delta = self._quat_angle_deg(left_quat_current, left_quat)
        if (
            left_delta > self.max_incremental_translation
            or left_rot_delta > self.max_incremental_rotation_deg
        ):
            print(
                f"[wbc] incremental safety reject (left): dpos={left_delta:.3f}m"
                f" drot={left_rot_delta:.1f}deg (limits {self.max_incremental_translation}m"
                f" {self.max_incremental_rotation_deg}deg)"
            )
            return False

        right_delta = float(
            np.linalg.norm(np.asarray(right_pos, dtype=float) - right_pos_current)
        )
        right_rot_delta = self._quat_angle_deg(right_quat_current, right_quat)
        if (
            right_delta > self.max_incremental_translation
            or right_rot_delta > self.max_incremental_rotation_deg
        ):
            print(
                f"[wbc] incremental safety reject (right): dpos={right_delta:.3f}m"
                f" drot={right_rot_delta:.1f}deg (limits {self.max_incremental_translation}m"
                f" {self.max_incremental_rotation_deg}deg)"
            )
            return False

        return True

    @staticmethod
    def _quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
        q1n = _normalize_quaternion(np.asarray(q1, dtype=float))
        q2n = _normalize_quaternion(np.asarray(q2, dtype=float))
        dot = float(np.clip(np.dot(q1n, q2n), -1.0, 1.0))
        angle_rad = 2.0 * math.acos(abs(dot))
        return math.degrees(angle_rad)

    def get_latest_robot_state(self) -> Optional[RobotSnapshot]:
        return self.robot_state.load()

    def get_latest_gripper_widths(self) -> Tuple[float, float]:
        return self.robot_state.load_gripper_widths()

    def get_end_effector_pose(
        self, snapshot: Optional[RobotSnapshot] = None
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Returns (left_pose, right_pose) in the world frame for a snapshot."""
        snap = snapshot if snapshot is not None else self.robot_state.load()
        qpos = self.snapshot_to_qpos(snap)
        if qpos is None:
            return None
        return self._compute_end_effector_world_pose(qpos)

    def wait_for_first_state(self, timeout_sec: float = 5.0) -> Optional[RobotSnapshot]:
        deadline = time.monotonic() + timeout_sec
        snapshot = None
        while time.monotonic() < deadline:
            snapshot = self.get_latest_robot_state()
            if snapshot is not None and snapshot.is_valid:
                return snapshot
            time.sleep(0.005)
        raise Exception("Timeout waiting for first valid robot state snapshot")

    def reset_base_odometry_origin(self, timeout_sec: float = 5.0) -> bool:
        """Use the current raw base odometry as WBC's local zero pose."""

        if not self.base_odometry_zero_at_start:
            return True

        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            snapshot = self.controller.get_latest_robot_state()
            if snapshot is not None and snapshot.is_valid:
                raw_T = self._as_se2_matrix(snapshot.odom_SE2)
                self._set_base_odometry_origin(raw_T)
                x, y, yaw = self._se2_pose(raw_T)
                print(
                    "[wbc] base odometry local origin set at "
                    f"x={x:.4f}, y={y:.4f}, yaw={yaw:.4f} rad"
                )
                return True
            time.sleep(0.005)
        return False

    def snapshot_odom_SE2(self, snapshot: RobotSnapshot) -> np.ndarray:
        """Return snapshot odometry in WBC's local base frame."""

        raw_T = self._as_se2_matrix(snapshot.odom_SE2)
        if not self.base_odometry_zero_at_start:
            return raw_T

        with self._base_odom_lock:
            if self._base_odom_origin_inv is None:
                self._base_odom_origin_inv = np.linalg.inv(raw_T)
            origin_inv = self._base_odom_origin_inv.copy()
        local_T = origin_inv @ raw_T
        local_T[2, :] = [0.0, 0.0, 1.0]
        return local_T

    def snapshot_to_qpos(self, snapshot: RobotSnapshot) -> Optional[np.ndarray]:
        if snapshot is None or not snapshot.is_valid:
            return None
        with self._model_lock:
            return self._snapshot_to_qpos(snapshot)

    def _init_controller(
        self, address: str, command_timeout_sec: float
    ) -> RealtimeDriver:
        config = ControllerConfig()
        config.robot_address = address
        config.command_timeout_us = int(
            round(max(command_timeout_sec, 0.0) * 1_000_000.0)
        )
        config.low_pass_freq_hz = float(self.low_pass_freq_hz)
        controller = RealtimeDriver(config)
        controller.start()
        if not controller.wait_until_ready(timeout_sec=15.0):
            controller.stop()
            raise RuntimeError("Controller did not report ready within 15 seconds")
        print("Realtime controller ready.")
        return controller

    def set_init_position(
        self,
        left_arm_override: Optional[np.ndarray] = None,
        right_arm_override: Optional[np.ndarray] = None,
        head_override: Optional[np.ndarray] = None,
        gripper_override: Optional[np.ndarray] = None,
    ) -> None:
        self._set_init_position(
            left_arm_override=left_arm_override,
            right_arm_override=right_arm_override,
            head_override=head_override,
            gripper_override=gripper_override,
        )

    def _set_init_position(
        self,
        left_arm_override: Optional[np.ndarray] = None,
        right_arm_override: Optional[np.ndarray] = None,
        head_override: Optional[np.ndarray] = None,
        gripper_override: Optional[np.ndarray] = None,
    ) -> None:
        def _to_array(values: Optional[list[float]]) -> Optional[np.ndarray]:
            if values is None:
                return None
            return np.asarray(values, dtype=float)

        torso_targets = _to_array(self.init_position.get("torso"))
        right_targets = _to_array(self.init_position.get("right_arm"))
        left_targets = _to_array(self.init_position.get("left_arm"))
        head_targets = _to_array(self.init_position.get("head"))
        gripper_targets = _to_array(self.init_position.get("grippers"))

        if left_arm_override is not None:
            left_targets = np.asarray(left_arm_override, dtype=float)
        if right_arm_override is not None:
            right_targets = np.asarray(right_arm_override, dtype=float)
        if head_override is not None:
            head_targets = np.asarray(head_override, dtype=float)
        if gripper_override is not None:
            gripper_targets = np.asarray(gripper_override, dtype=float)

        if all(
            target is None
            for target in (
                torso_targets,
                right_targets,
                left_targets,
                head_targets,
                gripper_targets,
            )
        ):
            print(
                f"Init config at {self.init_config_path} did not contain any targets; skipping initial position command."
            )
            return

        print("Setting initial positions...")
        sol_qpos = self.data.qpos.copy()

        torso_indices = getattr(self.ik_solver, "torso_qpos_indices", [])
        right_indices = getattr(self.ik_solver, "right_arm_qpos_indices", [])
        left_indices = getattr(self.ik_solver, "left_arm_qpos_indices", [])
        head_indices = getattr(self.ik_solver, "head_qpos_indices", [])

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

        target_body = self._compute_body_commands(sol_qpos)
        snapshot = self.robot_state.load()
        with self._model_lock:
            start_qpos = self._snapshot_to_qpos(snapshot)
        if self.admittance_enabled:
            self._initialize_admittance(start_qpos)
        start_body = self._compute_body_commands(start_qpos)

        delta = target_body - start_body
        max_delta = float(np.max(np.abs(delta)))
        if max_delta < 1e-6:
            self.controller.set_body_position_targets(target_body.tolist())
        else:
            max_step = max(float(self.init_position_max_step_delta), 1e-6)
            steps = max(10, int(np.ceil(max_delta / max_step)))
            for step in range(1, steps + 1):
                alpha = step / steps
                cmd = start_body + alpha * delta
                self.controller.set_body_position_targets(cmd.tolist())
                self.ik_rate.sleep()
            self.controller.set_body_position_targets(target_body.tolist())

        if self.gripper and gripper_targets is not None:
            self.gripper.set_target(gripper_targets.tolist())
        self.robot_state.store_gripper_widths(
            float(gripper_targets[0]), float(gripper_targets[1])
        )
        print("Initial positions set.")

    def _reset_base_pose(self) -> None:
        """Drive the base back to the odometry origin using twist commands."""

        if self._base_free_adr is None:
            return

        pos_tolerance = self.base_reset_pos_tolerance
        yaw_tolerance = math.radians(self.base_reset_yaw_tolerance_deg)
        max_linear_speed = self.base_reset_max_linear_speed
        max_yaw_speed = self.base_reset_max_yaw_speed
        min_linear_speed = self.base_reset_min_linear_speed
        min_yaw_speed = self.base_reset_min_yaw_speed
        rate = RateLimiter(frequency=self.trajectory_frequency_hz, warn=False)

        target_xy = self._base_origin_pos[:2]
        target_yaw = getattr(self, "_base_origin_yaw", 0.0)
        zero_twist = np.zeros(3, dtype=float)

        while not self._stop.is_set():
            snapshot = self.robot_state.load()
            cur_qpos = self.snapshot_to_qpos(snapshot)
            if cur_qpos is None:
                rate.sleep()
                continue

            if snapshot is not None:
                T = self.snapshot_odom_SE2(snapshot)
                x, y, yaw = self._se2_pose(T)
                print(f"[wbc] odom x={x:.4f}, y={y:.4f}, yaw={yaw:.4f} rad")

            measured_xy = cur_qpos[self._base_free_adr : self._base_free_adr + 2]
            measured_yaw = self._yaw_from_quat(
                cur_qpos[self._base_free_adr + 3 : self._base_free_adr + 7]
            )

            error_xy = target_xy - measured_xy
            yaw_error = self._angle_difference(target_yaw, measured_yaw)

            if (
                np.linalg.norm(error_xy) <= pos_tolerance
                and abs(yaw_error) <= yaw_tolerance
            ):
                break

            velocity_world = np.array(
                [
                    float(error_xy[0]),
                    float(error_xy[1]),
                    float(yaw_error),
                ],
                dtype=float,
            )
            # velocity_world *= self.base_velocity_gain
            lin_norm = float(np.linalg.norm(velocity_world[:2]))
            if 0.0 < lin_norm < min_linear_speed:
                velocity_world[:2] *= min_linear_speed / lin_norm
            yaw_mag = abs(float(velocity_world[2]))
            if 0.0 < yaw_mag < min_yaw_speed:
                velocity_world[2] = math.copysign(min_yaw_speed, velocity_world[2])
            velocity_world[:2] = np.clip(
                velocity_world[:2], -max_linear_speed, max_linear_speed
            )
            velocity_world[2] = float(
                np.clip(velocity_world[2], -max_yaw_speed, max_yaw_speed)
            )

            cy = math.cos(measured_yaw)
            sy = math.sin(measured_yaw)
            vx_body = cy * velocity_world[0] + sy * velocity_world[1]
            vy_body = -sy * velocity_world[0] + cy * velocity_world[1]
            twist_body = np.array([vx_body, vy_body, velocity_world[2]], dtype=float)
            self.controller.set_base_twist_command(twist_body)
            rate.sleep()

        self.controller.set_base_twist_command(zero_twist.tolist())

    def _state_poll_loop(self) -> None:
        while not self._stop.is_set():
            snapshot = self.controller.get_latest_robot_state()
            if snapshot is not None and snapshot.is_valid:
                self.robot_state.store(snapshot)
            self.state_poll_rate.sleep()

    def _ik_loop(self) -> None:
        while not self._stop.is_set():
            snapshot = self.robot_state.load()
            current_qpos: Optional[np.ndarray] = self.snapshot_to_qpos(snapshot)
            now = time.monotonic()

            # Joint-space path: if joint targets are set, bypass IK entirely.
            jt_left, _, jt_left_width, jt_right, _, jt_right_width, jt_head, _ = (
                self.joint_targets.get_for_ik(use_interpolation=self.use_interpolation)
            )

            # Base twist: always send if a base target is set, independent of arm mode.
            if current_qpos is not None:
                bt_x, bt_y, bt_quat = self.base_targets.get_target()
                # print(f"[wbc] base_target: x={bt_x}, y={bt_y}, quat={bt_quat}")
                if bt_x is not None and bt_quat is not None:
                    try:
                        desired_base_qpos = current_qpos.copy()
                        desired_base_qpos[0] = bt_x
                        desired_base_qpos[1] = bt_y
                        desired_base_qpos[3:7] = bt_quat
                        twist = self._compute_base_twist_command(
                            desired_base_qpos, np.zeros_like(current_qpos), current_qpos
                        )
                        # print(f"[wbc] base twist: {twist}")
                        self.controller.set_base_twist_command(twist.tolist())
                    except Exception as exc:
                        print(f"[wbc] base twist command error: {exc}")

            if jt_left is not None and jt_right is not None:
                if current_qpos is None:
                    self.ik_rate.sleep()
                    continue
                sol_qpos = current_qpos.copy()
                left_indices = getattr(self.ik_solver, "left_arm_qpos_indices", [])
                right_indices = getattr(self.ik_solver, "right_arm_qpos_indices", [])
                for i, idx in enumerate(left_indices):
                    if i < len(jt_left):
                        sol_qpos[int(idx)] = float(jt_left[i])
                for i, idx in enumerate(right_indices):
                    if i < len(jt_right):
                        sol_qpos[int(idx)] = float(jt_right[i])
                head_indices = getattr(self.ik_solver, "head_qpos_indices", [])
                if jt_head is not None and len(head_indices) >= 2:
                    for i, idx in enumerate(head_indices):
                        if i < len(jt_head):
                            sol_qpos[int(idx)] = float(jt_head[i])
                try:
                    if (
                        self.gripper
                        and jt_left_width is not None
                        and jt_right_width is not None
                    ):
                        self.gripper.set_target([jt_right_width, jt_left_width])
                    self.robot_state.store_gripper_widths(jt_left_width, jt_right_width)
                    body_targets = self._compute_body_commands(sol_qpos)
                    self.controller.set_body_position_targets(body_targets.tolist())
                except Exception as exc:
                    print(f"[wbc] joint command error: {exc}")
                self.ik_rate.sleep()
                continue

            # EE-space path: solve IK from Cartesian targets.
            (
                left_pos,
                left_quat,
                left_width,
                right_pos,
                right_quat,
                right_width,
                head_pos,
                head_quat,
            ) = self.ee_targets.get_for_ik(use_interpolation=self.use_interpolation)

            if current_qpos is None or left_pos is None or right_pos is None:
                self.ik_rate.sleep()
                continue

            if self.admittance_enabled:
                if not self._admittance_initialized:
                    self._initialize_admittance(current_qpos)

                left_pose_vec, right_pose_vec = self._compute_end_effector_world_pose(
                    current_qpos
                )
                if snapshot is None:
                    print("[wbc] snapshot is None")
                else:
                    if not snapshot.left_ft_valid:
                        print("[wbc] left_ft_valid is False")
                        left_wrench = np.zeros(6, dtype=float)
                    else:
                        left_wrench = snapshot.left_ee_wrench
                    if not snapshot.right_ft_valid:
                        print("[wbc] right_ft_valid is False")
                        right_wrench = np.zeros(6, dtype=float)
                    else:
                        right_wrench = snapshot.right_ee_wrench
                left_wrench, right_wrench = self._ft_calibrator.calibrate(
                    left_wrench,
                    right_wrench,
                    left_pose_vec,
                    right_pose_vec,
                )
                self._admittance_controller_left.set_robot_status(
                    left_pose_vec, left_wrench
                )
                if left_quat is not None:
                    left_pos, left_quat = self._apply_admittance(
                        self._admittance_controller_left,
                        left_pos,
                        left_quat,
                        self._admittance_wrench_left,
                    )
                self._admittance_controller_right.set_robot_status(
                    right_pose_vec, right_wrench
                )
                if right_quat is not None:
                    right_pos, right_quat = self._apply_admittance(
                        self._admittance_controller_right,
                        right_pos,
                        right_quat,
                        self._admittance_wrench_right,
                    )
            ik_start = time.perf_counter()
            sol_qpos, sol_vel, success, _info = self.ik_solver.solve(
                left_target_pos=left_pos,
                left_target_quat=left_quat,
                right_target_pos=right_pos,
                right_target_quat=right_quat,
                head_target_pos=head_pos,
                head_target_quat=head_quat,
                current_qpos=current_qpos,
                dt=self.ik_rate.dt,
            )
            ik_elapsed_ms = (time.perf_counter() - ik_start) * 1000.0
            # print(f"[wbc] IK solve took {ik_elapsed_ms:.3f} ms")
            if not success:
                print(f"[wbc] IK failed: {_info}")

            try:
                if self.gripper and left_width is not None and right_width is not None:
                    self.gripper.set_target([right_width, left_width])
                self.robot_state.store_gripper_widths(left_width, right_width)
                body_targets = self._compute_body_commands(sol_qpos)
                twist = self._compute_base_twist_command(
                    sol_qpos, sol_vel, current_qpos
                )
                self.controller.set_body_position_targets(body_targets.tolist())
                self.controller.set_base_twist_command(twist)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[wbc] command error: {exc}")

            self.ik_rate.sleep()

    def _compute_body_commands(self, sol_qpos: np.ndarray) -> np.ndarray:
        positions_sdk = np.zeros(len(self._sdk_joint_names), dtype=float)
        for i, adr in enumerate(self._sdk_to_mj_qadr):
            if adr is not None:
                positions_sdk[i] = sol_qpos[adr]

        name_to_idx = {name: i for i, name in enumerate(self._sdk_joint_names)}
        torso_idxs = [name_to_idx[f"torso_{i}"] for i in range(6)]
        left_idxs = [name_to_idx[f"left_arm_{i}"] for i in range(7)]
        right_idxs = [name_to_idx[f"right_arm_{i}"] for i in range(7)]
        head_idxs = [name_to_idx[f"head_{i}"] for i in range(2)]
        ordered = torso_idxs + left_idxs + right_idxs + head_idxs
        body_targets = positions_sdk[ordered]
        return body_targets

    def _configure_admittance(self, config: Optional[Mapping[str, Any]]) -> None:
        controller_params = config.get("controller", {})
        self._admittance_config_left = self._build_admittance_config(controller_params)
        self._admittance_config_right = self._build_admittance_config(controller_params)

        axes_config = config.get("force_controlled_axes", {})
        self._admittance_Tr, self._admittance_n_af = self._parse_force_controlled_axes(
            axes_config
        )

        desired_wrench_cfg = config.get("desired_wrench", {})
        self._admittance_wrench_left = desired_wrench_cfg["left"]
        self._admittance_wrench_right = desired_wrench_cfg["right"]

    def _build_admittance_config(
        self, params: Mapping[str, Any]
    ) -> AdmittanceControllerConfig:
        cfg = AdmittanceControllerConfig()
        cfg.dt = float(params.get("dt", cfg.dt))
        cfg.log_to_file = bool(params.get("log_to_file", cfg.log_to_file))
        cfg.log_file_path = str(params.get("log_file_path", cfg.log_file_path))
        cfg.alert_overrun = bool(params.get("alert_overrun", cfg.alert_overrun))

        compliance = params.get("compliance6d", {})
        cfg.compliance6d.stiffness = np.diag(compliance["stiffness"])
        cfg.compliance6d.damping = np.diag(compliance["damping"])
        cfg.compliance6d.inertia = np.diag(compliance["inertia"])
        cfg.compliance6d.stiction = compliance["stiction"]

        cfg.max_spring_force_magnitude = params["max_spring_force_magnitude"]
        cfg.max_spring_torque_magnitude = params["max_spring_torque_magnitude"]

        gains = params.get("direct_force_control_gains", {})
        cfg.direct_force_control_gains.P_trans = gains["P_trans"]
        cfg.direct_force_control_gains.I_trans = gains["I_trans"]
        cfg.direct_force_control_gains.D_trans = gains["D_trans"]
        cfg.direct_force_control_gains.P_rot = gains["P_rot"]
        cfg.direct_force_control_gains.I_rot = gains["I_rot"]
        cfg.direct_force_control_gains.D_rot = gains["D_rot"]

        cfg.direct_force_control_I_limit = params["direct_force_control_I_limit"]
        return cfg

    def _parse_force_controlled_axes(
        self, axes_config: Mapping[str, Any]
    ) -> Tuple[np.ndarray, int]:
        if not axes_config:
            return np.eye(6, dtype=float), 6

        if "matrix" in axes_config:
            matrix = np.asarray(axes_config["matrix"], dtype=float).reshape(6, 6)
            n_af = int(axes_config.get("n_af", 6))
            return matrix, n_af

        mode = str(axes_config.get("mode", "all_force")).lower()
        if mode == "all_force":
            return np.eye(6, dtype=float), 6
        if mode == "translation_force":
            return np.eye(6, dtype=float), 3
        if mode == "rotation_force":
            matrix = np.array(
                [
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                ],
                dtype=float,
            )
            return matrix, 3
        if mode in ("rigid", "velocity"):
            return np.eye(6, dtype=float), 0

        raise ValueError(f"Unsupported force_controlled_axes mode: {mode}")

    def _initialize_admittance(self, current_qpos: np.ndarray) -> None:
        self._admittance_controller_left = AdmittanceController()
        self._admittance_controller_right = AdmittanceController()
        left_pose_vec, right_pose_vec = self._compute_end_effector_world_pose(
            current_qpos
        )
        self._admittance_controller_left.init(
            self._admittance_config_left, left_pose_vec
        )
        self._admittance_controller_right.init(
            self._admittance_config_right, right_pose_vec
        )
        zero_wrench = np.zeros(6, dtype=float)
        self._admittance_controller_left.set_robot_status(left_pose_vec, zero_wrench)
        self._admittance_controller_right.set_robot_status(right_pose_vec, zero_wrench)
        self._admittance_controller_left.set_force_controlled_axis(
            self._admittance_Tr, self._admittance_n_af
        )
        self._admittance_controller_right.set_force_controlled_axis(
            self._admittance_Tr, self._admittance_n_af
        )
        self._admittance_controller_left.set_robot_reference(
            left_pose_vec, self._admittance_wrench_left
        )
        self._admittance_controller_right.set_robot_reference(
            right_pose_vec, self._admittance_wrench_right
        )
        self._admittance_initialized = True

    def _compute_end_effector_world_pose(
        self, qpos: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        with self._model_lock:
            self.data.qpos[:] = qpos
            mujoco.mj_forward(self.model, self.data)
            left_rot = self.data.site_xmat[self._left_ee_site_id]
            right_rot = self.data.site_xmat[self._right_ee_site_id]
            left_quat = np.zeros(4)
            mujoco.mju_mat2Quat(left_quat, left_rot)
            right_quat = np.zeros(4)
            mujoco.mju_mat2Quat(right_quat, right_rot)
            left_pose = np.concatenate(
                (
                    self.data.site_xpos[self._left_ee_site_id],
                    left_quat,
                )
            )
            right_pose = np.concatenate(
                (
                    self.data.site_xpos[self._right_ee_site_id],
                    right_quat,
                )
            )
        return left_pose.astype(float), right_pose.astype(float)

    def _apply_admittance(
        self,
        controller: AdmittanceController,
        target_pos: np.ndarray,
        target_quat: Optional[np.ndarray],
        desired_wrench: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if target_quat is None:
            return target_pos, target_quat

        controller.set_robot_reference(
            np.concatenate([target_pos, target_quat]), desired_wrench
        )
        status, pose_cmd = controller.step()
        if not status:
            return target_pos, target_quat
        pose_cmd = np.asarray(pose_cmd, dtype=float).reshape(7)
        new_pos = pose_cmd[:3]
        new_quat = _normalize_quaternion(pose_cmd[3:])
        return new_pos, new_quat

    def _compute_base_twist_command(
        self, sol_qpos: np.ndarray, sol_qvel: np.ndarray, cur_qpos: np.ndarray
    ) -> np.ndarray:
        measured_x = float(cur_qpos[0])
        measured_y = float(cur_qpos[1])
        measured_yaw = self._yaw_from_quat(cur_qpos[3:7])

        desired_x = float(sol_qpos[0])
        desired_y = float(sol_qpos[1])
        desired_yaw = self._yaw_from_quat(sol_qpos[3:7])

        error = np.array(
            [
                desired_x - measured_x,
                desired_y - measured_y,
                self._angle_difference(desired_yaw, measured_yaw),
            ],
            dtype=float,
        )

        feedback = self.base_error_gain * error

        velocity_desired_world = np.array(
            [float(sol_qvel[0]), float(sol_qvel[1]), float(sol_qvel[5])],
            dtype=float,
        )
        velocity_command_world = (
            feedback + self.base_velocity_gain * velocity_desired_world
        )

        cy = math.cos(measured_yaw)
        sy = math.sin(measured_yaw)
        vx_world = velocity_command_world[0]
        vy_world = velocity_command_world[1]
        vx_body = cy * vx_world + sy * vy_world
        vy_body = -sy * vx_world + cy * vy_world

        return np.array([vx_body, vy_body, velocity_command_world[2]], dtype=float)

    def _snapshot_to_qpos(self, snapshot: RobotSnapshot) -> np.ndarray:
        qpos = np.zeros(self.model.nq, dtype=float)

        base_adr = getattr(self, "_base_free_adr", None)
        base_origin = getattr(self, "_base_origin_pos", np.zeros(3, dtype=float))
        if base_adr is not None:
            T = self.snapshot_odom_SE2(snapshot)
            x, y, yaw = self._se2_pose(T)
            qpos[base_adr + 0] = base_origin[0] + x
            qpos[base_adr + 1] = base_origin[1] + y
            qpos[base_adr + 2] = base_origin[2]
            qpos[base_adr + 3 : base_adr + 7] = self._quat_from_yaw(yaw)

        joint_positions = snapshot.joint_position
        mapping = getattr(self, "_sdk_to_mj_qadr", [])
        count = min(len(joint_positions), len(mapping))
        for idx in range(count):
            adr = mapping[idx]
            if adr is not None:
                qpos[adr] = joint_positions[idx]

        # Gripper joints handling
        left_width, right_width = self.robot_state.load_gripper_widths()
        mujoco_mapping = getattr(self, "_mj_joint_qadr", {})
        l1_index, l2_index = (
            mujoco_mapping["gripper_finger_l1"],
            mujoco_mapping["gripper_finger_l2"],
        )
        r1_index, r2_index = (
            mujoco_mapping["gripper_finger_r1"],
            mujoco_mapping["gripper_finger_r2"],
        )
        qpos[l2_index], qpos[r2_index] = left_width / 2, right_width / 2
        qpos[l1_index], qpos[r1_index] = -left_width / 2, -right_width / 2
        return qpos

    def _build_joint_mapping(self) -> None:
        self._sdk_joint_names = [
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

        self._mj_joint_qadr = {}
        self._base_free_adr = None
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            jtype = int(self.model.jnt_type[i])
            adr = int(self.model.jnt_qposadr[i])
            if jtype == mujoco.mjtJoint.mjJNT_FREE:
                if name == "world_j":
                    self._base_free_adr = adr
                continue
            elif jtype in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
                self._mj_joint_qadr[name] = adr

        if self._base_free_adr is None:
            raise RuntimeError("Base free joint address not found (world_j)")

        self._sdk_to_mj_qadr = []
        missing = []
        for name in self._sdk_joint_names:
            adr = self._mj_joint_qadr.get(name)
            if adr is None:
                missing.append(name)
            self._sdk_to_mj_qadr.append(adr)
        if missing:
            print(f"[mapping] Missing joints in MuJoCo model: {missing}")

    def _extract_base_origin(self) -> None:
        self._base_origin_pos = self.data.qpos[
            self._base_free_adr : self._base_free_adr + 3
        ].copy()
        base_quat = self.data.qpos[self._base_free_adr + 3 : self._base_free_adr + 7]
        self._base_origin_yaw = self._yaw_from_quat(base_quat)

    def _set_base_odometry_origin(self, raw_T: np.ndarray) -> None:
        with self._base_odom_lock:
            self._base_odom_origin_inv = np.linalg.inv(raw_T)

    @staticmethod
    def _as_se2_matrix(T: np.ndarray) -> np.ndarray:
        mat = np.asarray(T, dtype=float).reshape(3, 3).copy()
        mat[2, :] = [0.0, 0.0, 1.0]
        return mat

    @staticmethod
    def _se2_pose(T: np.ndarray) -> Tuple[float, float, float]:
        return (
            float(T[0, 2]),
            float(T[1, 2]),
            math.atan2(float(T[1, 0]), float(T[0, 0])),
        )

    @staticmethod
    def _quat_from_yaw(yaw: float) -> np.ndarray:
        half = 0.5 * yaw
        return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=float)

    @staticmethod
    def _yaw_from_quat(q: np.ndarray) -> float:
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    @staticmethod
    def _angle_difference(target: float, source: float) -> float:
        diff = float(target) - float(source)
        return (diff + math.pi) % (2 * math.pi) - math.pi


__all__ = ["RBY1WBC", "EETargets"]
