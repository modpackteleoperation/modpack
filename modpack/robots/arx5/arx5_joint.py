"""
ARX5 Joint-space adapter implementing JointRobotInterface.

This is the ONLY file that imports arx5_interface in joint-position mode.
All arm-specific hardware logic lives here, ported directly from arx5_process.py:
  - Per-arm hardware initialization (controller, solver, URDF)
  - Joint velocity safety validation
  - Per-step command delta limiting
  - Hold / emergency-stop logic with motion verification
  - Preview window before first command execution
"""

import os
import sys
import time
import traceback
from typing import Optional

import numpy as np

# arx5-sdk path setup — only this file needs it
sys.path.append(os.path.abspath("../arx5-sdk/python/"))
sys.path.append(os.path.abspath("../arx5-sdk/lib/"))
import arx5_interface as arx5
from arx5_interface import Arx5JointController, Arx5Solver



class ARX5Joint:
    """
    Adapter wrapping the ARX5 arm in joint-position control mode.
    Supports a single arm (right or left); instantiate once per arm.
    Implements JointRobotInterface.
    """

    def __init__(
        self,
        arm: str,                                        # "right" or "left"
        model: str = "L5",
        urdf_path: str = "./arm_models/arx5_webcam.urdf",
        interface: str = "can8",                         # CAN interface for this arm
        joint_dof: int = 6,
        preview_time: float = 0.1,
        preview_seconds: float = 5.0,
        joint_velocity_limit: float = 10.0,
        enable_cmd_step_limit: bool = True,
        max_cmd_step_rad: float = 1.0,
        gripper_max_width: float = 0.08,                 # m; normalized 1.0 -> fully open
        bridge=None,
        **_kwargs,  # absorb extra config keys (e.g. backpack_logger_port, control_freq)
    ):
        self._arm = arm
        self._model = model
        self._urdf_path = urdf_path
        self._interface = interface
        self._joint_dof = joint_dof
        self._preview_time = preview_time
        self._preview_seconds = preview_seconds
        self._joint_velocity_limit = joint_velocity_limit
        self._enable_cmd_step_limit = enable_cmd_step_limit
        self._max_cmd_step_rad = max_cmd_step_rad
        self._gripper_max_width = gripper_max_width
        self._bridge = bridge

        # Hardware
        self._controller: Optional[Arx5JointController] = None
        self._solver: Optional[Arx5Solver] = None
        self._controller_config = None

        # State
        self._activated = False
        self._initialized = False
        self._recovery_mode = False
        self._first_command_sent = False
        self._first_command_time: Optional[float] = None
        self._preview_ready_sent = False
        self._jts_prev: Optional[np.ndarray] = None

        # Safety tracking
        self._previous_joint_positions: Optional[np.ndarray] = None
        self._previous_timestamp: Optional[float] = None
        self._last_cmd_step_limit_log_time = 0.0
        self._last_safe_command_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def wait_until_ready(self, timeout: float = 30.0) -> bool:
        """Initialize hardware and reset to home. Returns True on success."""
        return self._init_hardware()

    def activate(self) -> None:
        self._activated = True
        self._recovery_mode = False

    def deactivate(self) -> None:
        self.hold_position()
        self._activated = False

    def emergency_stop(self) -> None:
        self._enter_emergency_safe_mode("emergency_stop")
        self._activated = False

    def close(self) -> None:
        if self._initialized:
            try:
                self._enter_emergency_safe_mode("shutdown")
            except Exception as e:
                print(f"[ARX5Joint] shutdown safety mode error: {e}")

    # ------------------------------------------------------------------
    # Command / state
    # ------------------------------------------------------------------

    def execute_action(self, action: dict) -> bool:
        """
        action['joint_pos']: np.ndarray shape (num_joints,)
        action['gripper']:   float
        """
        if not self._activated or not self._initialized or self._controller is None:
            return False

        joint_pos = np.asarray(action['joint_pos'])
        gripper = float(action.get('gripper', 0.0))

        # Velocity safety check
        if not self._validate_velocity(joint_pos, gripper):
            print(f"[ARX5Joint-{self._arm}] velocity violation — halting")
            self.close()
            os.kill(os.getpid(), os.SIGTSTP if hasattr(os, 'SIGTSTP') else 15)
            return False

        current_time = time.time()
        if self._first_command_time is not None:
            time_since_first = current_time - self._first_command_time
            if time_since_first < self._preview_seconds:
                print(f"[ARX5Joint-{self._arm}] Skipping command - in preview "
                      f"({time_since_first:.1f}s / {self._preview_seconds:.1f}s)")
                return True
            else:
                if not self._preview_ready_sent and self._bridge is not None:
                    self._bridge.publish_activation({
                        "command": "preview_ready",
                        "target": "arx",
                        "timestamp": time.time(),
                    })
                    self._preview_ready_sent = True
                self._first_command_time = None
                print(f"[ARX5Joint-{self._arm}] preview complete - resuming normal operation")

        try:
            cmd = arx5.JointState(self._joint_dof)
            current_state = self._controller.get_joint_state()

            is_first = not self._first_command_sent
            if is_first:
                cmd.timestamp = current_state.timestamp + 5.0
                print(f"[ARX5Joint-{self._arm}] First command - smooth start")
                print(f"  Joint positions: {[f'{np.rad2deg(x):.1f}°' for x in joint_pos]}")
                self._first_command_sent = True
                self._first_command_time = current_time

            cmd_target = self._apply_cmd_step_limit(joint_pos)
            cmd.pos()[0:self._joint_dof] = cmd_target
            # gripper arrives normalized [0,1]; the ARX5 controller expects a
            # width in metres. Scale by max width so the gripper tracks
            # continuously (1.0 -> fully open) instead of snapping binary, mirroring
            # the egoverse_ros2 rollout controller.
            gripper_width = gripper * self._gripper_max_width
            cmd.gripper_pos = gripper_width

            # Record the follower's commanded target so get_state() can expose it
            # (logged as target_joint_pos / EE target by the central stream logger).
            self._last_target_joint_pos = np.array(cmd_target, dtype=float)
            self._last_target_gripper = gripper_width

            self._controller.set_joint_cmd(cmd)
            self._jts_prev = np.array(cmd.pos()[0:self._joint_dof], dtype=float)
            return True

        except Exception as e:
            print(f"[ARX5Joint-{self._arm}] execute_action error: {e}")
            self._enter_emergency_safe_mode(f"execute_action_error_{self._arm}")
            return False

    def get_state(self) -> dict:
        if not self._initialized or self._controller is None:
            return {
                'positions':  np.zeros(self._joint_dof),
                'eef_pose':   np.zeros(6),
                'gripper':    0.0,
                'timestamp':  time.monotonic(),
                'data_valid': False,
            }
        try:
            js = self._controller.get_joint_state()
            joint_pos = np.array(js.pos()[0:self._joint_dof], dtype=float)
            eef_pose = self.forward_kinematics(joint_pos)
            return {
                'positions':        joint_pos,
                'eef_pose':         eef_pose,
                'gripper':          float(js.gripper_pos),
                'target_positions': getattr(self, '_last_target_joint_pos', None),
                'gripper_target':   getattr(self, '_last_target_gripper', None),
                'timestamp':        time.monotonic(),
                'data_valid':       True,
            }
        except Exception as e:
            print(f"[ARX5Joint] get_state error: {e}")
            return {
                'positions':  np.zeros(self._joint_dof),
                'eef_pose':   np.zeros(6),
                'gripper':    0.0,
                'timestamp':  time.monotonic(),
                'data_valid': False,
            }

    # ------------------------------------------------------------------
    # JointRobotInterface extras
    # ------------------------------------------------------------------

    def hold_position(self) -> None:
        if not self._initialized or self._controller is None:
            return
        try:
            current_state = self._controller.get_joint_state()
            hold_cmd = arx5.JointState(self._joint_dof)
            hold_cmd.timestamp = current_state.timestamp + 3
            hold_cmd.pos()[0:self._joint_dof] = current_state.pos()[0:self._joint_dof].copy()
            hold_cmd.gripper_pos = current_state.gripper_pos
            self._controller.set_joint_cmd(hold_cmd)
            print(f"[ARX5Joint-{self._arm}] hold command sent")
        except Exception as e:
            print(f"[ARX5Joint-{self._arm}] hold_position error: {e}")

    def forward_kinematics(self, joint_pos: np.ndarray) -> np.ndarray:
        if self._solver is None:
            return np.zeros(6)
        try:
            return np.array(self._solver.forward_kinematics(joint_pos), dtype=float)
        except Exception as e:
            print(f"[ARX5Joint] forward_kinematics error: {e}")
            return np.zeros(6)

    # ------------------------------------------------------------------
    # Safety helpers (ported from arx5_process.py)
    # ------------------------------------------------------------------

    def _validate_velocity(self, joint_pos: np.ndarray, gripper: float) -> bool:
        """Velocity safety check — mirrors validate_gello_message_safety in arx5_process.py."""
        try:
            now = time.time()
            if self._previous_joint_positions is not None and self._previous_timestamp is not None:
                dt = now - self._previous_timestamp
                # At 300 Hz, only check velocity over multiple samples to avoid jitter
                if 0.015 < dt < 1.0:
                    for i, (pos, prev_pos) in enumerate(zip(joint_pos, self._previous_joint_positions)):
                        velocity = abs(pos - prev_pos) / dt
                        if velocity > self._joint_velocity_limit:
                            return False
            self._previous_joint_positions = np.array(joint_pos, dtype=float)
            self._previous_timestamp = now
            return True
        except Exception as e:
            print(f"[ARX5Joint] velocity check error: {e}")
            return False

    def _apply_cmd_step_limit(self, cmd_target: np.ndarray) -> np.ndarray:
        """
        Clamp per-step joint command delta (wrap-aware) to prevent single-frame jumps.
        Ported directly from arx5_process.py::_apply_cmd_step_limit.
        """
        cmd_target = np.asarray(cmd_target, dtype=float)
        if (not self._enable_cmd_step_limit) or (self._jts_prev is None):
            return cmd_target

        prev_target = np.asarray(self._jts_prev, dtype=float)
        # Shortest angular difference on [-pi, pi]
        step_delta = np.arctan2(
            np.sin(cmd_target - prev_target),
            np.cos(cmd_target - prev_target),
        )
        clipped_delta = np.clip(step_delta, -self._max_cmd_step_rad, self._max_cmd_step_rad)

        if np.any(np.abs(step_delta) > self._max_cmd_step_rad):
            now = time.time()
            if now - self._last_cmd_step_limit_log_time > 0.5:
                self._last_cmd_step_limit_log_time = now
                print(
                    f"[CMD-STEP-LIMIT {self._arm}] "
                    f"max|delta|={float(np.max(np.abs(step_delta))):.3f}rad "
                    f"clipped_to={self._max_cmd_step_rad:.3f}rad"
                )

        return prev_target + clipped_delta

    def _enter_emergency_safe_mode(self, reason: str) -> None:
        """Enhanced safe mode with active verification — ported from arx5_process.py."""
        if not self._recovery_mode:
            self._recovery_mode = True
            print(f"[ARX5Joint-{self._arm}] CRITICAL EMERGENCY SAFE MODE: {reason}")

            if self._initialized and self._controller is not None:
                try:
                    current_state = self._controller.get_joint_state()

                    stop_cmd = arx5.JointState(self._joint_dof)
                    stop_cmd.timestamp = current_state.timestamp + 3
                    stop_cmd.pos()[0:self._joint_dof] = current_state.pos()[0:self._joint_dof].copy()
                    stop_cmd.gripper_pos = current_state.gripper_pos

                    # Send multiple times to ensure reception
                    for _ in range(5):
                        self._controller.set_joint_cmd(stop_cmd)
                        time.sleep(0.002)

                    print(f"[ARX5Joint-{self._arm}] HOLD POSITION: "
                          f"{[f'{np.rad2deg(x):.1f}°' for x in current_state.pos()[:self._joint_dof]]}")

                    # Verify robot stopped moving
                    time.sleep(0.1)
                    new_state = self._controller.get_joint_state()
                    joint_velocities = abs(new_state.pos() - current_state.pos())
                    max_velocity = max(joint_velocities)

                    if max_velocity > 0.1:
                        print(f"[ARX5Joint-{self._arm}] WARNING: still moving! Max vel: {max_velocity:.3f}")
                        for _ in range(10):
                            self._controller.set_joint_cmd(stop_cmd)
                            time.sleep(0.001)
                    else:
                        print(f"[ARX5Joint-{self._arm}] successfully stopped")

                except Exception as e:
                    print(f"[ARX5Joint-{self._arm}] CRITICAL: Failed to stop: {e}")

            self._activated = False

        else:
            # Continue sending hold commands periodically
            if self._last_safe_command_time is None or (
                time.time() - self._last_safe_command_time > 0.1
            ):
                self.hold_position()
            self._last_safe_command_time = time.time()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_hardware(self) -> bool:
        try:
            print(f"[ARX5Joint-{self._arm}] Initializing hardware...")

            robot_config = arx5.RobotConfigFactory.get_instance().get_config(self._model)

            # Gravity vector is arm-side-specific (matches arx5_process.py)
            if self._arm == "right":
                robot_config.gravity_vector = np.array([0, -9.81 * np.sqrt(2) / 2, 9.81 * np.sqrt(2) / 2])
            else:
                robot_config.gravity_vector = np.array([0, 9.81 * np.sqrt(2) / 2, 9.81 * np.sqrt(2) / 2])

            ctrl_config = arx5.ControllerConfigFactory.get_instance().get_config(
                "joint_controller", robot_config.joint_dof
            )
            ctrl_config.background_send_recv = True
            ctrl_config.default_preview_time = self._preview_time
            robot_config.urdf_path = self._urdf_path
            self._controller_config = ctrl_config

            self._solver = Arx5Solver(
                self._urdf_path,
                robot_config.joint_dof,
                robot_config.joint_pos_min,
                robot_config.joint_pos_max,
            )

            self._controller = Arx5JointController(robot_config, ctrl_config, self._interface)

            gain = self._controller.get_gain()
            self._controller.set_gain(gain)

            print(f"[ARX5Joint-{self._arm}] Resetting to home position...")
            self._controller.reset_to_home()

            self._initialized = True
            print(f"[ARX5Joint-{self._arm}] Hardware initialized.")
            return True

        except Exception as e:
            print(f"[ARX5Joint-{self._arm}] Hardware init failed: {e}")
            traceback.print_exc()
            return False

