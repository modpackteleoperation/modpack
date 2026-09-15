#!/usr/bin/env python3
import time
import sys
import os
from pathlib import Path
import numpy as np
import yaml
from typing import Optional
import traceback
from robotmq.utils import serialize
from robotmq import RMQClient

# Repo root on path so `import modpack` works when this file is run as a script
_repo_root = Path(__file__).resolve().parents[3]
sys.path.append(str(_repo_root))

from modpack.modules.gello.gello.robots.dynamixel import DynamixelRobot
from modpack.orchestration.message_formats import (
    MessageFactory,
    Topics,
    TorqueMessage,
    deserialize_torque_message,
)
from modpack.orchestration.process_shutdown import (
    ShutdownFlag,
    register_signals,
)
from modpack.orchestration.robot_config import load_network_config
from modpack.utils import DebugPrinter
from modpack.modules.gello.utils.urdf_utils import resolve_urdf_path

class GelloPublisher:
    def __init__(
        self,
        gello_config: dict,
        arm: str,
        host: str,
        port: str,
        feedback_port: Optional[str] = None,
        shutdown_flag: Optional[ShutdownFlag] = None,
        debug: bool = False,
    ):
        self.robot = None
        self.debug = debug
        self.debug_printer = DebugPrinter(enabled=debug)
        self.message_factory = MessageFactory()
        self.host = host
        self.port = port

        self.data_client = RMQClient(
            client_name=f"{arm}_gello_client",
            server_endpoint=f"tcp://{self.host}:{self.port}"
        )

        self.feedback_client: Optional[RMQClient] = None
        if feedback_port:
            try:
                self.feedback_client = RMQClient(
                    client_name=f"{arm}_torque_feedback_client",
                    server_endpoint=f"tcp://{self.host}:{feedback_port}",
                )
            except Exception as exc:
                print(f"[{arm}] Failed to connect to torque feedback server at {host}:{feedback_port}: {exc}")

        self.running = False
        self.arm = arm
        # Last valid torque message from RBY1; held across loop iterations so a missed
        # poll doesn't reset haptic feedback to zero immediately.
        self._last_robot_joint_torque: Optional[np.ndarray] = None
      
        # How close reads need to be to just the offsets (i.e. we have failed to read which defaults to zeros - offsets)
        # to trigger safety mechanism
        self.safety_tolerance = 1e-6
        self.shutdown_flag = shutdown_flag or ShutdownFlag()

        register_signals(self.shutdown_flag)

        self.config = {"gello_device": gello_config}

    def create_gello_robot(self):
        # Extract top level configs
        gello_device_config = self.config['gello_device']
        hardware_config = gello_device_config['hardware'][self.arm]
        
        port = hardware_config['port']
        port_config = gello_device_config['port_configs'][port]

        # Per-arm calibration: aligns the GELLO leader joint frame with the follower
        # robot's joint zero / direction. Sourced from gello_config_<robot>.yaml
        # port_configs[<port>]. Identity values are valid (e.g. RBY1 currently uses zeros + ones).
        self.joint_offsets = port_config['joint_offsets']
        self.debug_printer.log(lambda: f"JOINT OFFSETS ARE {self.joint_offsets}")

        self.joint_signs = port_config['joint_signs']
        robot_joint_signs = port_config.get('robot_joint_signs', None)

        enhanced_control_config = hardware_config['enhanced_control']
        safety_config = gello_device_config['safety']
        self.frequency = gello_device_config['frequency']

        # We go into safe mode if stale messages for more than 200 ms
        self.corruption_count = 0
        self.corruption_thresh = np.floor(0.2 / (1 / self.frequency))

        baudrate = hardware_config['baudrate']

        # Extract sub-configs
        arm_motor_config = hardware_config['motors']
        gripper_config = port_config.get('gripper')
        zero_position_offsets = hardware_config['zero_position_offsets']
        active_motor_ids = (
            hardware_config.get('active_motor_ids')
            or hardware_config.get('enabled_motor_ids')
            or hardware_config.get('test_motor_ids')
        )
        
        # Gravity comp unpacking
        urdf_path = resolve_urdf_path(enhanced_control_config.get("urdf_path") or None)
        base_link = enhanced_control_config.get('base_link', 'base_link')
        tip_link = enhanced_control_config.get('tip_link', 'end_effector_link')
        gains = enhanced_control_config['gains']
        self.kp_gains = np.array(gains['kp'])
        self.kd_gains = np.array(gains['kd'])

        if active_motor_ids is not None:
            print(f"Restricting {self.arm} arm to motors: {sorted(active_motor_ids)}")

        # Create robot with all required configs including zero position calibration
        robot = DynamixelRobot(
            real=True,
            motor_config=arm_motor_config,
            gripper_config=gripper_config,
            hardware_config=hardware_config,
            urdf_path=urdf_path,
            base_link=base_link,
            tip_link=tip_link,
            safety_config=safety_config,
            port=port,
            baudrate=baudrate,
            zero_position_offsets=zero_position_offsets,
            active_motor_ids=active_motor_ids,
            robot_joint_signs=robot_joint_signs,
            control_freq=self.frequency,
            debug=self.debug,
        )

        if hasattr(robot, "dynamics_calc"):
            self.dynamics_calc = robot.dynamics_calc

        return robot

    def initialize_hardware(self) -> bool:
        """Initialize GELLO hardware connection"""
        try:
            print("Initializing GELLO hardware...")
            
            self.robot = self.create_gello_robot()
            self.robot.enable_torque(True)
            raw = self.robot.get_joint_state()
            print(f"[{self.arm}] raw joint positions at startup: {np.round(raw, 4).tolist()}")
            print(f"[{self.arm}] zero_position_offsets:          {np.round(self.robot._zero_position_offsets, 4).tolist()}")
            return True
            
        except Exception as e:
            print(f"Failed to initialize GELLO hardware: {e}")
            traceback.print_exc()
            return False
    
    def pop_torque_message(self) -> Optional[TorqueMessage]:
        """Poll the latest torque feedback message from the robot. Returns None if unavailable."""
        if self.feedback_client is None:
            return None
        try:
            topic = Topics.torque_topic(self.arm)
            data_list, _ = self.feedback_client.pop_data(topic, n=-1)
            if not data_list:
                return None
            return deserialize_torque_message(data_list[0])
        except Exception:
            self.debug_printer.log(lambda: f"pop_torque_message error: {exc}")
            return None

    def _send_gello_enhanced_control(
        self,
        current_positions_raw: np.ndarray,
        joint_torque: np.ndarray,
    ):
        """Send enhanced control, blending in RBY1 haptic feedback torques."""
        if hasattr(self.robot, 'command_torque_enhanced'):
            self.robot.command_torque_enhanced(
                target_positions=current_positions_raw,
                kp_gains=self.kp_gains,
                kd_gains=self.kd_gains,
                robot_torque=joint_torque,
            )
    
    def data_health_check(self, joint_positions: np.ndarray) -> None:
        """Checks for readings that are exactly equal to joint offsets"""
        for i, (pos, offset) in enumerate(zip(joint_positions, self.robot._zero_position_offsets)):
            if abs(pos - offset) < self.safety_tolerance:
                self.debug_printer.log(lambda: f"\nENCODER CORRUPTION DETECTED:")
                self.debug_printer.log(lambda: f"  Joint {i}: value={pos:.6f} matches offset={offset:.6f}")
                self.debug_printer.log(lambda: f"  This indicates hardware/communication failure")
                self.corruption_count += 1
                break
        
        if self.corruption_count >= self.corruption_thresh:
            return False
        
        # Reset corruption count if valid message
        self.corruption_count = 0
        return True

    def _port_calibration_vectors(self, n_state: int):
        """
        Pad or trim port_config joint_offsets / joint_signs to match
        :meth:`DynamixelRobot.get_publish_state` length (7 arm joints, optional +1 gripper).
        """
        jo = np.asarray(self.joint_offsets, dtype=np.float64).reshape(-1)
        js = np.asarray(self.joint_signs, dtype=np.float64).reshape(-1)
        if jo.size < n_state:
            pad = n_state - int(jo.size)
            jo = np.concatenate([jo, np.zeros(pad, dtype=np.float64)])
            js = np.concatenate([js, np.ones(pad, dtype=np.float64)])
        elif jo.size > n_state:
            jo = jo[:n_state]
            js = js[:n_state]
        return jo, js

    def run(self):
        """
        UPDATED: Main control loop with zero position calibration support and profiling
        
        Architecture:
        1. Read RAW coordinates (with zero calibration) -> Send to enhanced control (correct physics)
        2. Read TRANSFORMED coordinates (unchanged) -> Publish to ARX5 (coordinate alignment)
        """
        print("Starting GELLO Publisher Process with Zero Position Calibration...")
        
        # Initialize hardware
        if not self.initialize_hardware():
            print("Failed to initialize hardware, exiting...")
            return False
        
        # Set running flag
        self.running = True

        loop_period = 1.0 / self.frequency
        
        last_publish_time = time.monotonic()
        
        while self.running and not self.shutdown_flag.is_requested():
            current_time = time.monotonic()
            
            # Rate limiting
            if current_time - last_publish_time < loop_period:
                time.sleep(loop_period - (current_time - last_publish_time))
                continue
            
            # === PHYSICS/CONTROL LOOP (RAW COORDINATES) ===
            # Read GELLO state in RAW coordinates for correct physics
            gello_state_full = self.robot.get_publish_state()

            # Poll latest haptic torque feedback from RBY1; hold last valid on miss
            torque_msg = self.pop_torque_message()
            if torque_msg is not None:
                self._last_robot_joint_torque = np.array(torque_msg.joint_torques[:self.robot.num_dofs_urdf()], dtype=float)

            robot_joint_torque = (
                self._last_robot_joint_torque.copy()
                if self._last_robot_joint_torque is not None
                else np.zeros(self.robot.num_dofs_urdf())
            )

            if gello_state_full is not None:
                arm_state = gello_state_full[:-1] if self.robot.gripper_motor_id is not None else gello_state_full
                n_arm = len(robot_joint_torque)
                signed_joint_torque = robot_joint_torque * np.array(self.joint_signs[:n_arm])
                self._send_gello_enhanced_control(
                    arm_state,
                    joint_torque=signed_joint_torque,
                )
            else:
                self.debug_printer.log("Warning: Failed to read RAW state for physics")
                continue

            # Health check for bad communication
            if not self.data_health_check(gello_state_full):
                break

            # Safety check (joint limits, step delta)
            if self.robot.safety_monitor is not None:
                arm_state = gello_state_full[:-1] if self.robot.gripper_motor_id is not None else gello_state_full
                if not self.robot.safety_monitor.is_safe(arm_state):
                    continue

            # Apply per-port joint calibration (offsets + signs) from gello_config_<robot>.yaml.
            # RBY1's port_config currently uses identity values, so this reduces to the
            # angle-wrap step for RBY1; ARX picks up its real offsets/signs here.
            # State can be 7 (arm) or 8 (arm + gripper); port_config may only list arm DOF.
            gello_state_full = np.asarray(gello_state_full, dtype=np.float64)
            n_full = int(gello_state_full.shape[0])
            jo, js = self._port_calibration_vectors(n_full)
            gello_state_transformed = (gello_state_full - jo) * js
            n_arm = n_full - 1 if self.robot.gripper_motor_id is not None else n_full
            gello_state_transformed[:n_arm] = np.arctan2(
                np.sin(gello_state_transformed[:n_arm]),
                np.cos(gello_state_transformed[:n_arm]),
            )

            self.debug_printer.log(lambda: f"DEBUG TRANSFORM:")
            self.debug_printer.log(lambda: f"  Raw full: {gello_state_full}")
            self.debug_printer.log(lambda: f"  Offsets:  {self.joint_offsets}")
            self.debug_printer.log(lambda: f"  Signs:    {self.joint_signs}")
            self.debug_printer.log(lambda: f"  Result:   {gello_state_transformed}")

            # Normalize gripper to [0, 1]
            if self.robot.gripper_motor_id is not None and self.robot.gripper_open_close is not None:
                open_angle, closed_angle = self.robot.gripper_open_close
                gripper_raw = gello_state_transformed[-1]
                self.debug_printer.log(lambda: gripper_raw)
                gripper_normalized = (gripper_raw - closed_angle) / (open_angle - closed_angle)
                gripper_normalized = min(max(0, gripper_normalized), 1)
                if self.robot.gripper_invert:
                    gripper_normalized = 1.0 - gripper_normalized
                gello_state_transformed[-1] = gripper_normalized

            self.debug_printer.log(lambda: f"gello_state_full shape: {gello_state_full.shape}")

            active_motor_ids = (
                self.config['gello_device']['hardware'][self.arm].get('active_motor_ids')
                or self.config['gello_device']['hardware'][self.arm].get('enabled_motor_ids')
                or self.config['gello_device']['hardware'][self.arm].get('test_motor_ids')
            )
            is_test_mode = active_motor_ids is not None

            if is_test_mode:
                joint_positions = [0.0] * 7
                active_state = gello_state_transformed
                for driver_idx, motor_id in enumerate(self.robot._joint_ids):
                    joint_idx = self.robot.motor_config[motor_id]['joint_idx']
                    if joint_idx < 7 and driver_idx < len(active_state):
                        joint_positions[joint_idx] = float(active_state[driver_idx])
                gripper_position = float(gello_state_transformed[-1]) if self.robot.gripper_motor_id is not None else 0.0
            else:
                joint_positions = gello_state_transformed.tolist()
                if len(joint_positions) < 7:
                    joint_positions.extend([0.0] * (7 - len(joint_positions)))
                else:
                    joint_positions = joint_positions[:7]
                gripper_position = joint_positions[6] if self.robot.gripper_motor_id is not None else 0.0

            msg = self.message_factory.create_gello_message(
                arm=self.arm,
                joint_positions=joint_positions,
                gripper_position=gripper_position
            )
            
            
            # Serialize and publish via put_data
            serialized_msg = serialize(msg.to_dict())
            self.debug_printer.log(lambda: f"{self.arm} msg={msg}")
            self.data_client.put_data(self.arm, serialized_msg)

            last_publish_time += loop_period
        
        self.cleanup()
        print("GELLO Publisher Process terminated")
        return True

    def cleanup(self):
        """Cleanup resources"""
        print("Cleaning up GELLO Publisher...")

        self.running = False

        # Close robot connection
        if self.robot:
            try:
                self.robot.enable_torque(False)
                print("GELLO torque mode disabled")
            except Exception as e:
                print(f"Error disabling torque mode: {e}")



def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: python gello_process.py <config_path> <arm>")
        print("  config_path: path to robot config.yaml (must have a 'gello:' block)")
        print("  <arm>: 'left' or 'right'")
        sys.exit(1)

    config_path = sys.argv[1]
    arm = sys.argv[2]

    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    gello_config = raw.get("gello", {})

    # Debug entrypoint: connect to the local data server. Port comes from the
    # shared network config; host stays localhost for manual single-PC debugging.
    publisher = GelloPublisher(gello_config, arm=arm, host="127.0.0.1",
                               port=load_network_config().data_port)

    print("GELLO Publisher - Zero Position Calibration Architecture")
    print("Physics: RAW (zero calibrated) | Publishing: TRANSFORMED (unchanged)")

    try:
        publisher.run()
    except KeyboardInterrupt:
        print("\n\n=== SHUTDOWN REQUESTED ===")
        print("Cleaning up and exiting...")
    except Exception as e:
        print(f"\n\n=== UNEXPECTED ERROR ===")
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        try:
            publisher.cleanup()
        except Exception as e:
            print(f"Error during cleanup: {e}")

    print("\nGELLO Publisher terminated")
    sys.exit(0)


if __name__ == "__main__":
    main()
