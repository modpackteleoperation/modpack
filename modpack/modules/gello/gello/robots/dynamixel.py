#!/usr/bin/env python3
import numpy as np
import traceback

from typing import Optional
from modpack.modules.gello.gello.robots.robot import Robot
from modpack.modules.gello.gello.dynamixel.driver import DynamixelDriver, FakeDynamixelDriver
from modpack.modules.gello.utils.dynamics_utils import InverseDynamicsCalculator
from modpack.modules.gello.utils.safety_utils import SafetyMonitor
from modpack.modules.gello.utils.filtering_smoothing_utils import LowPassFilter
from modpack.utils import DebugPrinter

class DynamixelRobot(Robot):
    def __init__(
        self,
        real: bool = False,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 57600,
        gripper_config: Optional[dict] = None,
        motor_config: Optional[dict] = None,
        urdf_path: Optional[str] = None,
        base_link: str = "base_link",
        tip_link: str = "end_effector_link",
        safety_config: Optional[dict] = None,
        hardware_config:  Optional[dict] = None,
        zero_position_offsets: Optional[np.ndarray] = None,
        active_motor_ids: Optional[list[int]] = None,
        robot_joint_signs: Optional[list] = None,
        control_freq: float = 100.0,
        debug: bool = False,
    ):
        
        self.prev_sent_time = 0
        self.direction = 1
        self.debug = debug
        self.debug_printer = DebugPrinter(enabled=debug)
        # Safety check for real
        if real and motor_config is None:
            raise ValueError("motor_config must be provided for real hardware!")
        
        self.motor_config = motor_config
        self.hardware_config = hardware_config
        self.loop_counter = 0
        self._last_grav_pos = None
        self._last_grav_torques = None
        self._robot_joint_signs = np.array(robot_joint_signs) if robot_joint_signs is not None else None
        # Build actuated motor mask from hardware_config
        actuated_motors = hardware_config.get('actuated_motors') if hardware_config else None
        if actuated_motors is not None and motor_config is not None:
            joint_ids_sorted = sorted(motor_config.keys())
            self._actuated_mask = np.array([mid in set(actuated_motors) for mid in joint_ids_sorted], dtype=float)
        else:
            self._actuated_mask = None

        joint_ids = []
        joint_signs = np.array([])
        self._full_zero_position_offsets = np.array([])
        self._arm_joint_count = 0

        # Extract ids and offsets from motor_config
        if motor_config is not None:
            full_motor_config = {mid: dict(config) for mid, config in motor_config.items()}
            full_joint_ids = sorted(list(full_motor_config.keys()))

            if active_motor_ids is not None:
                active_motor_id_set = set(active_motor_ids)
                unknown_motor_ids = active_motor_id_set - set(full_joint_ids)
                if unknown_motor_ids:
                    raise ValueError(f"Unknown active motor ids requested: {sorted(unknown_motor_ids)}")
                motor_config = {
                    mid: dict(full_motor_config[mid])
                    for mid in full_joint_ids
                    if mid in active_motor_id_set
                }
            else:
                motor_config = full_motor_config

            joint_ids = sorted(list(motor_config.keys()))
            joint_signs = np.array([motor_config[mid].get("sign", 1) for mid in joint_ids])
            self._arm_joint_count = (
                max(config["joint_idx"] for config in full_motor_config.values()) + 1
            )
        
        # Set arrays
        self._joint_ids = joint_ids
        self._joint_signs = joint_signs
        
        # Handle gripper configuration
        self.gripper_motor_id = None
        self.gripper_open_close = None
        self.gripper_invert = False
        
        if gripper_config is not None and 'gripper_motor_id' in gripper_config:
            self.gripper_motor_id = gripper_config['gripper_motor_id']  # Should be 7
            self.gripper_open_close = (
                gripper_config['open_angle_deg'] * np.pi / 180,
                gripper_config['closed_angle_deg'] * np.pi / 180,
            )
            self.gripper_invert = bool(gripper_config['gripper_invert'])

        # Zero position calibration
        if zero_position_offsets is not None:
            self._full_zero_position_offsets = np.array(zero_position_offsets[:self._arm_joint_count])
            self._zero_position_offsets = np.array([
                self._full_zero_position_offsets[motor_config[mid]['joint_idx']]
                for mid in joint_ids
            ])
            self._zero_position_calibrated = True
        else:
            self._full_zero_position_offsets = np.zeros(self._arm_joint_count)
            self._zero_position_offsets = np.zeros(len(joint_ids))
            self._zero_position_calibrated = False
        
        # Validation checks
        assert len(self._joint_ids) == len(self._joint_signs)
        assert len(self._joint_ids) == len(self._zero_position_offsets)
        assert np.all(np.abs(self._joint_signs) == 1)

        self.safety_monitor = SafetyMonitor(safety_config) if safety_config else None

        # Initialize driver
        if real:
            driver_motor_config = {mid: dict(config) for mid, config in motor_config.items()}
            if self.gripper_motor_id is not None and gripper_config is not None:
                required = ('model', 'control', 'sign', 'current_limit')
                missing = [k for k in required if k not in gripper_config]
                if missing:
                    raise ValueError(f"gripper_config missing required keys: {missing}")
                driver_motor_config[self.gripper_motor_id] = {
                    'joint_idx': len(joint_ids),
                    'model': gripper_config['model'],
                    'control': gripper_config['control'],
                    'offset': 0.0,
                    'sign': gripper_config['sign'],
                    'current_limit': gripper_config['current_limit'],
                }
            
            self._driver = DynamixelDriver(motor_config=driver_motor_config, port=port, baudrate=baudrate)
            self._driver.enable_torque(False)
        else:
            fake_ids = list(motor_config.keys())
            if self.gripper_motor_id is not None:
                fake_ids.append(self.gripper_motor_id)
            self._driver = FakeDynamixelDriver(fake_ids)
        
        self._torque_on = False

        if urdf_path:
            try:
                self.dynamics_calc = InverseDynamicsCalculator(
                    urdf_path=urdf_path,
                    base_link=base_link,
                    tip_link=tip_link,
                    debug=debug,
                )
            except Exception:
                traceback.print_exc()
        
        self.torque_filter = LowPassFilter(n_joints=self.num_dofs_urdf(), dt=0.01)  # 100Hz control loop
        self.obs_torque_filter = LowPassFilter(n_joints=self.num_dofs_urdf(), dt=0.01)

        # Per-joint torque budgets from motor config.
        # gravity_current_limit: mA budget for gravity compensation.
        # contact_current_limit: mA budget for haptic contact feedback (independent budget).
        _tc_map = {'XM540': 2.4, 'XM430': 1.57, 'XL430': 1.15}
        n_urdf = self.num_dofs_urdf()
        self._gravity_torque_limits = np.full(n_urdf, np.inf)
        self._contact_torque_limits = np.full(n_urdf, np.inf)
        if motor_config is not None:
            for cfg in motor_config.values():
                if cfg.get('control') == 'torque':
                    jidx = cfg['joint_idx']
                    if 0 <= jidx < n_urdf:
                        tc = _tc_map.get(cfg['model'], 1.79)
                        if 'gravity_current_limit' in cfg:
                            self._gravity_torque_limits[jidx] = cfg['gravity_current_limit'] * 1e-3 * tc
                        if 'contact_current_limit' in cfg:
                            self._contact_torque_limits[jidx] = cfg['contact_current_limit'] * 1e-3 * tc

        # Haptic feedback config (optional). Robots without haptics (e.g. arx)
        # simply omit the 'haptic_feedback' key; feedback_scale=0 disables the
        # contact-feedback term entirely, leaving only gravity compensation.
        if hardware_config is None:
            raise ValueError("hardware_config must be provided")
        _haptic_cfg = hardware_config.get('haptic_feedback')
        if _haptic_cfg is None:
            self._feedback_scale = 0.0
            self._max_torque_rate = float('inf')
        else:
            self._feedback_scale = float(_haptic_cfg['feedback_scale'])
            self._max_torque_rate = float(_haptic_cfg['max_torque_rate'])
        # Control-loop period for the haptic torque rate limiter. Derived from the
        # GELLO control frequency (gello.frequency) — the single source of truth —
        # rather than a duplicate haptic_feedback.control_freq key.
        self._control_dt = 1.0 / float(control_freq)
        self._last_robot_torque: Optional[np.ndarray] = None  # for rate limiter

    def num_dofs(self) -> int:
        """Get number of degrees of freedom"""
        return len(self._joint_ids)
    
    def num_dofs_urdf(self) -> int:
        """Get the number of degrees of freedom of urdf"""
        if hasattr(self, 'dynamics_calc'):
            return self.dynamics_calc.n_joints
        return self.num_dofs()

    def get_joint_state(self) -> np.ndarray:
        """
        Get ARM state only (6 joints) - for physics/dynamics
        """
        # Read raw positions from the driver
        raw_positions = self._driver.get_joints()  # expected shape: [n_dofs]

        # Only take the first num_dofs() joints (exclude gripper if present)
        raw_positions = np.array(raw_positions[:self.num_dofs()])
        return raw_positions
    
    def get_full_state(self) -> np.ndarray:
        """
        Get FULL state including gripper (7 joints) for publishing to ARX5
        """
        raw_positions = self._driver.get_joints()
        return np.array(raw_positions)

    def get_publish_state(self) -> np.ndarray:
        """
        Get arm state expanded back to the full configured joint layout.
        Missing joints default to their zero offsets so downstream transforms stay stable.
        """
        raw_positions = np.array(self._driver.get_joints())
        full_arm_state = self._full_zero_position_offsets.copy()

        for driver_idx, motor_id in enumerate(self._joint_ids):
            joint_idx = self.motor_config[motor_id]['joint_idx']
            full_arm_state[joint_idx] = raw_positions[driver_idx]

        if self.gripper_motor_id is not None and len(raw_positions) > len(self._joint_ids):
            return np.concatenate([full_arm_state, [raw_positions[len(self._joint_ids)]]])

        return full_arm_state

    def _get_current_velocities(self) -> np.ndarray:
        """Get current velocities wrapper for driver"""
        return self._driver.get_velocities()[:self.num_dofs()]  # Raw velocities from driver

    def command_torque_enhanced(
        self,
        target_positions: np.ndarray,
        kp_gains: np.ndarray,
        kd_gains: np.ndarray,
        robot_torque: Optional[np.ndarray] = None,
    ) -> None:
        """
        Command gravity-compensated torque with optional haptic contact feedback.

        Gravity compensation and haptic contact feedback use independent current budgets:
          gravity_current_limit  -> clamps gravity comp torques per joint
          contact_current_limit  -> clamps incoming robot feedback torques per joint

        Args:
            target_positions: RAW coordinates for correct physics.
            robot_torque:     Per-joint torques from robot (J^T * F_ee), Gello frame, shape (n_urdf,).
                              None or zeros means no haptic feedback this cycle.
        """
        # 1. Get state
        current_velocities_raw = self._get_current_velocities()
        actuated_positions_urdf = (target_positions - self._zero_position_offsets)[:self.num_dofs_urdf()]
        if self._robot_joint_signs is not None:
            n = min(len(self._robot_joint_signs), len(actuated_positions_urdf))
            actuated_positions_urdf[:n] *= self._robot_joint_signs[:n]

        actuated_velocities_raw = current_velocities_raw[:self.num_dofs_urdf()]
        actuated_kp_gains = kp_gains[:self.num_dofs_urdf()]
        actuated_kd_gains = kd_gains[:self.num_dofs_urdf()]

        # Get actual torques (currently unused; placeholder for future feedback path)
        actual_torques_raw = None

        # Apply robot_joint_signs to velocities so KDL receives them in URDF frame
        actuated_velocities_urdf = actuated_velocities_raw.copy()
        if self._robot_joint_signs is not None:
            n = min(len(self._robot_joint_signs), len(actuated_velocities_urdf))
            actuated_velocities_urdf[:n] *= self._robot_joint_signs[:n]

        # 2. Compute contact feedback with rate limiter + independent contact budget
        n_urdf = self.num_dofs_urdf()
        robot_torque_array = np.zeros(n_urdf)
        if robot_torque is not None and len(robot_torque) > 0:
            robot_torque_array[:min(len(robot_torque), n_urdf)] = robot_torque[:n_urdf]
        if self._last_robot_torque is not None:
            max_delta = self._max_torque_rate * self._control_dt
            delta = robot_torque_array - self._last_robot_torque
            robot_torque_array = self._last_robot_torque + np.clip(delta, -max_delta, max_delta)
        self._last_robot_torque = robot_torque_array.copy()
        robot_torque_array *= self._feedback_scale
        contact_feedback = np.clip(robot_torque_array, -self._contact_torque_limits, self._contact_torque_limits)
        
        # 3. Gravity compensation (with position-change caching)
        pos_change = 0 if self._last_grav_pos is None else np.linalg.norm(actuated_positions_urdf - self._last_grav_pos)

        if pos_change < 0.01 and self._last_grav_torques is not None:
            # Reuse cached gravity (stored in motor frame, ready to send)
            gravity_torques_motor = self._last_grav_torques
        else:
            if hasattr(self, 'dynamics_calc'):
                base_torques = self.dynamics_calc.compute_feedforward_torque(
                    current_positions=actuated_positions_urdf,  # URDF absolute
                    current_velocities=actuated_velocities_urdf,
                    kp_gains=actuated_kp_gains,
                    kd_gains=actuated_kd_gains,
                    actual_torques=actual_torques_raw,
                )
            gravity_torques = self.torque_filter.update(base_torques)
            # Clamp to independent gravity budget
            gravity_torques = np.clip(gravity_torques, -self._gravity_torque_limits, self._gravity_torque_limits)
            self._last_grav_pos = actuated_positions_urdf.copy()
            if self._actuated_mask is not None:
                n = min(len(self._actuated_mask), len(gravity_torques))
                gravity_torques[:n] *= self._actuated_mask[:n]
            # Re-apply robot_joint_signs to convert URDF-frame torques back to motor frame
            if self._robot_joint_signs is not None:
                n = min(len(self._robot_joint_signs), len(gravity_torques))
                gravity_torques[:n] *= self._robot_joint_signs[:n]
            # Cache motor-frame torques so cache hits are ready-to-send without re-flipping
            self._last_grav_torques = gravity_torques.copy()
            gravity_torques_motor = gravity_torques

        # 4. Blend: gravity comp with contact feedback
        combined_torques = gravity_torques_motor + contact_feedback

        self.debug_printer.log(lambda: len(combined_torques))
        self._last_grav_pos = actuated_positions_urdf.copy()
        self._driver.set_torques(combined_torques)
    
    def enable_torque(self, mode: bool):
        """Set torque mode wrapper"""
        self._driver.enable_torque(mode)
        self._torque_on = mode
