"""
ARX5 Cartesian adapter implementing CartesianRobotInterface.

This is the ONLY file that imports arx5_interface in Cartesian mode.
All neck/head-specific hardware logic lives here, including:
  - Hardware initialization (controller, solver, URDF)
  - Vision Pro READY handshake (part of wait_until_ready)
  - EEF command construction and hold/emergency-stop logic
  - EEF-pose step guard that rejects sudden command jumps

The sys.path manipulation for arx5-sdk is done once at the top of this module
so it never leaks into generic process code.
"""

import os
import sys
import time
import traceback
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

# arx5-sdk path setup — only this file needs it
sys.path.append(os.path.abspath("../arx5-sdk/python/"))
sys.path.append(os.path.abspath("../arx5-sdk/lib/"))
import arx5_interface as arx5
from arx5_interface import Arx5CartesianController, EEFState

from modpack.modules.vision_pro.transformation_helper import neck_start_pose
from modpack.orchestration.message_formats import Topics, deserialize_vp_status_message, serialize_message

class ARX5Cartesian:
    """
    Adapter wrapping the ARX5 arm in Cartesian (EEF-pose) control mode.
    Used for the neck/head manipulator.
    Implements CartesianRobotInterface.
    """

    # Default Cartesian safety limits (can be overridden via config)
    DEFAULT_MIN_X = 0.4
    DEFAULT_MAX_X = 0.5
    DEFAULT_MIN_PITCH = 0.18
    DEFAULT_MAX_PITCH = 0.6

    def __init__(
        self,
        model: str = "L5",
        urdf_path: str = "./arm_models/arx5_iphone15pro.urdf",
        interface: str = "can11",
        joint_dof: int = 6,
        preview_time: float = 0.2,
        vp_ready_timeout: float = 60.0,
        bridge=None,
        msg_factory=None,
        min_cmd_x: float = DEFAULT_MIN_X,
        max_cmd_x: float = DEFAULT_MAX_X,
        min_pitch_rad: float = DEFAULT_MIN_PITCH,
        max_abs_pitch_rad: float = DEFAULT_MAX_PITCH,
        enable_step_guard: bool = True,
        max_joint_step_rad: float = 1.0,  # accepted for config back-compat; unused
        max_cmd_pos_step_m: float = 0.15,
        max_cmd_rot_step_rad: float = 0.6,
    ):
        self._model = model
        self._urdf_path = urdf_path
        self._interface = interface
        self._joint_dof = joint_dof
        self._preview_time = preview_time
        self._vp_ready_timeout = vp_ready_timeout

        self._bridge = bridge
        self._msg_factory = msg_factory

        # Safety limits
        self._min_cmd_x = min_cmd_x
        self._max_cmd_x = max_cmd_x
        self._min_pitch_rad = min_pitch_rad
        self._max_abs_pitch_rad = max_abs_pitch_rad

        # EEF-pose step guard: refuse a command whose commanded EEF pose jumps
        # more than the caps below (translation in m, rotation in rad) from the
        # last accepted command, and hold instead.
        self._enable_step_guard = enable_step_guard
        self._max_cmd_pos_step_m = max_cmd_pos_step_m
        self._max_cmd_rot_step_rad = max_cmd_rot_step_rad
        # Last accepted commanded pose, the step reference. None until the first
        # command (which is never guarded — it uses slow 5 s smoothing).
        self._last_cmd_pose: Optional[np.ndarray] = None
        self._last_flip_log_t = 0.0
        self._last_step_log_t = 0.0

        # Internal state
        self._controller: Optional[Arx5CartesianController] = None
        self._solver = None
        self._activated = False
        self._initialized = False
        self._recovery_mode = False
        self._first_command_sent = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def wait_until_ready(self, timeout: float = 30.0, stop_fn=None) -> bool:
        """
        Initialize hardware, home the arm, move to init pose, then wait for
        Vision Pro READY signal. Returns True on success.

        stop_fn: optional callable returning True when shutdown is requested,
                 used to abort the VP-ready wait early.
        """
        if not self._init_hardware():
            return False

        # Move to home then to init pose
        self._controller.reset_to_home()
        reset_pose = np.array([0.15, -0.0405, 0.3, -np.pi / 2, 0, 0])
        self._move_to_pose(reset_pose, duration=2.0)

        print("Neck at init position — waiting for Vision Pro READY...")
        self._publish_neck_status(at_init=True)

        # Heartbeat loop until VP reports ready
        # timeout=0 means wait indefinitely (only stop_fn can abort)
        HB_PERIOD = 0.5
        deadline = None if timeout == 0 else time.monotonic() + (timeout if timeout > 0 else self._vp_ready_timeout)
        last_hb = time.monotonic()
        while deadline is None or time.monotonic() < deadline:
            if stop_fn is not None and stop_fn():
                return False
            if self._check_vp_ready():
                break
            if time.monotonic() - last_hb >= HB_PERIOD:
                self._publish_neck_status(at_init=True)
                last_hb = time.monotonic()
            time.sleep(0.05)

        if stop_fn is not None and stop_fn():
            return False

        # Move to operational start pose
        self._move_to_pose(neck_start_pose.copy(), duration=3.0)
        self._publish_neck_status(at_start=True)
        print("Neck at start position — ready.")
        return True

    def activate(self) -> None:
        self._activated = True
        self._recovery_mode = False

    def deactivate(self) -> None:
        self.hold_position()
        self._activated = False

    def emergency_stop(self) -> None:
        self._enter_safe_mode("emergency_stop")
        self._activated = False

    def close(self) -> None:
        if self._initialized:
            try:
                self._enter_safe_mode("shutdown")
            except Exception as e:
                print(f"[ARX5Cartesian] shutdown safety mode error: {e}")

    # ------------------------------------------------------------------
    # Command / state
    # ------------------------------------------------------------------

    def execute_action(self, action: dict) -> bool:
        """
        action['eef_pose']: np.ndarray shape (6,) — [x, y, z, roll, pitch, yaw]
        """
        if not self._activated or not self._initialized or self._controller is None:
            return False
        try:
            pose6d = np.asarray(action['eef_pose'], dtype=np.float64)
            pose6d = self._apply_limits(pose6d)

            # Step guard: skip the first command (it uses slow 5 s smoothing),
            # then refuse any command whose EEF pose jumps too far from the last
            # accepted command in one step — hold instead of lurching.
            if self._enable_step_guard and self._first_command_sent:
                exceeded, dpos, drot = self._exceeds_pose_step(pose6d)
                self._log_step(dpos, drot, exceeded)
                if exceeded:
                    self.hold_position()
                    return True

            cmd = EEFState(pose6d, 0)
            current = self._controller.get_eef_state()
            if not self._first_command_sent:
                cmd.timestamp = current.timestamp + 5.0
                self._first_command_sent = True
            self._controller.set_eef_cmd(cmd)
            self._last_cmd_pose = pose6d.copy()
            return True
        except Exception as e:
            print(f"[ARX5Cartesian] execute_action error: {e}")
            self._enter_safe_mode("execute_action_error")
            return False

    def get_state(self) -> dict:
        if not self._initialized or self._controller is None:
            return {
                'positions':  np.zeros(6),
                'eef_pose':   np.zeros(6),
                'timestamp':  time.monotonic(),
                'data_valid': False,
            }
        try:
            s = self._controller.get_eef_state()
            pose6d = np.array(s.pose_6d(), dtype=float)
            return {
                'positions':  pose6d,
                'eef_pose':   pose6d,
                'timestamp':  time.monotonic(),
                'data_valid': True,
            }
        except Exception as e:
            print(f"[ARX5Cartesian] get_state error: {e}")
            return {
                'positions':  np.zeros(6),
                'eef_pose':   np.zeros(6),
                'timestamp':  time.monotonic(),
                'data_valid': False,
            }

    # ------------------------------------------------------------------
    # CartesianRobotInterface extras
    # ------------------------------------------------------------------

    def hold_position(self, duration: float = 3.0) -> None:
        if not self._initialized or self._controller is None:
            return
        try:
            s = self._controller.get_eef_state()
            cmd = EEFState(s.pose_6d(), 0)
            cmd.timestamp = s.timestamp + duration
            self._controller.set_eef_cmd(cmd)
        except Exception as e:
            print(f"[ARX5Cartesian] hold_position error: {e}")

    # ------------------------------------------------------------------
    # EEF-pose step guard
    # ------------------------------------------------------------------

    def _exceeds_pose_step(self, pose6d: np.ndarray) -> tuple:
        """Report whether the commanded EEF pose jumps too far from the last
        accepted command. Translation and rotation are checked against separate
        caps (different units).

        Returns (exceeded, pos_step_m, rot_step_rad).
        """
        if self._last_cmd_pose is None:
            return False, 0.0, 0.0
        prev = self._last_cmd_pose
        dpos = float(np.linalg.norm(pose6d[:3] - prev[:3]))
        rel = R.from_euler("xyz", pose6d[3:]) * R.from_euler("xyz", prev[3:]).inv()
        drot = float(rel.magnitude())
        exceeded = dpos > self._max_cmd_pos_step_m or drot > self._max_cmd_rot_step_rad
        return exceeded, dpos, drot

    def _log_step(self, dpos: float, drot: float, exceeded: bool) -> None:
        """Console observability for the step guard (rate-limited)."""
        now = time.time()
        if exceeded:
            if now - self._last_flip_log_t > 0.5:
                self._last_flip_log_t = now
                print(
                    f"[ARX5Cartesian] step-guard: REJECTED command "
                    f"(pos step {dpos:.3f} m > {self._max_cmd_pos_step_m:.2f} or "
                    f"rot step {drot:.3f} rad > {self._max_cmd_rot_step_rad:.2f}) — holding",
                    flush=True,
                )
        elif now - self._last_step_log_t > 2.0:
            self._last_step_log_t = now
            print(
                f"[ARX5Cartesian] step-guard: pos step {dpos:.3f} m, "
                f"rot step {drot:.3f} rad",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_hardware(self) -> bool:
        try:
            print("Initializing ARX5 Cartesian hardware...")
            robot_config = arx5.RobotConfigFactory.get_instance().get_config(self._model)
            self._solver = arx5.Arx5Solver(
                self._urdf_path,
                self._joint_dof,
                robot_config.joint_pos_min,
                robot_config.joint_pos_max,
            )
            robot_config.gravity_vector = np.array([0, 0, -9.81])
            robot_config.urdf_path = self._urdf_path
            ctrl_config = arx5.ControllerConfigFactory.get_instance().get_config(
                "cartesian_controller", robot_config.joint_dof
            )
            ctrl_config.background_send_recv = True
            ctrl_config.default_preview_time = self._preview_time

            self._controller = Arx5CartesianController(robot_config, ctrl_config, self._interface)
            gain = self._controller.get_gain()
            self._controller.set_gain(gain)
            self._initialized = True
            print("ARX5 Cartesian controller initialized.")
            return True
        except Exception as e:
            print(f"[ARX5Cartesian] Hardware init failed: {e}")
            traceback.print_exc()
            return False

    def _move_to_pose(self, pose6d: np.ndarray, duration: float) -> None:
        current = self._controller.get_eef_state()
        cmd = EEFState(pose6d, 0)
        cmd.timestamp = current.timestamp + duration
        self._controller.set_eef_cmd(cmd)

    def _apply_limits(self, pose6d: np.ndarray) -> np.ndarray:
        out = pose6d.copy()
        print(f"Original x:{out[0]}, original roll :{out[4]}")
        out[0] = float(np.clip(out[0], self._min_cmd_x, self._max_cmd_x))
        out[4] = float(np.clip(out[4], self._min_pitch_rad, self._max_abs_pitch_rad))
        print(f"Clipped x:{out[0]}, clipped roll :{out[4]}")
        return out

    def _enter_safe_mode(self, reason: str) -> None:
        if self._recovery_mode:
            self.hold_position()
            return
        self._recovery_mode = True
        print(f"[ARX5Cartesian] SAFE MODE: {reason}")
        if self._initialized and self._controller is not None:
            try:
                s = self._controller.get_eef_state()
                cmd = EEFState(s.pose_6d(), 0)
                cmd.timestamp = s.timestamp + 3.0
                for _ in range(5):
                    self._controller.set_eef_cmd(cmd)
                    time.sleep(0.002)
                print("[ARX5Cartesian] Hold sent.")
            except Exception as e:
                print(f"[ARX5Cartesian] Failed to send hold: {e}")
        self._activated = False

    def _check_vp_ready(self) -> bool:
        """Poll activation channel for a VP_STATUS READY message."""
        if self._bridge is None:
            return False
        data_list = self._bridge.peek_activation_topic(Topics.VP_STATUS, timeout_s=5)
        if not data_list:
            return False
        payload = data_list[0]
        msg = (
            deserialize_vp_status_message(payload)
            if isinstance(payload, (bytes, bytearray))
            else payload
        )
        return bool(getattr(msg, "vision_pro_ready", False))

    def _publish_neck_status(self, at_init: bool = False, at_start: bool = False) -> None:
        if self._bridge is None or self._msg_factory is None:
            return
        msg = self._msg_factory.create_neck_status_message(at_init=at_init, at_start=at_start)
        self._bridge.publish_activation_raw(Topics.NECK_STATUS, serialize_message(msg))

