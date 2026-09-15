#!/usr/bin/env python3
"""
Base Process with iPhone teleoperation
"""

import argparse
import multiprocessing as mp
import time
import signal
import sys
import threading
import numpy as np

import robotmq
from robotmq.utils import deserialize
from scipy.spatial.transform import Rotation as R

from modpack.orchestration.message_formats import Topics, serialize_base_proprio_message, deserialize_base_proprio_message, MessageFactory
from modpack.modules.base.constants import POLICY_CONTROL_PERIOD, PUBLISH_PERIOD, CTRL_FREQ
from modpack.robots.arx5.base.mock_vehicle import MockVehicle
from modpack.robots.arx5.base.base_process_client import connect_base
from modpack.modules.base.policies import TeleopPolicy
from modpack.modules.base.webxr_messages import UnifiedWebXRMessage
from modpack.orchestration.process_shutdown import ShutdownFlag, make_signal_handler
from modpack.utils import DebugPrinter

from robologger.loggers.robot_ctrl_logger import RobotCtrlLogger


def _wxyz_from_xyzw(quat_xyzw):
    """Reorder a scipy [x, y, z, w] quaternion to [w, x, y, z] (float32)."""
    return np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32
    )


class MockBase:
    """Lightweight base with no CAN/phoenix6 hardware required (i.e. for RB-Y1m)."""

    def __init__(self):
        self.vehicle = MockVehicle()

    def reset(self):
        self.vehicle.start_control()

    def activate(self):
        pass

    def deactivate(self):
        self.vehicle.set_target_position(self.vehicle.x.copy())

    def emergency_stop(self):
        self.vehicle.set_target_position(self.vehicle.x.copy())

    def execute_action(self, action):
        if 'base_pose' in action:
            self.vehicle.set_target_position(action['base_pose'])
        return True

    def get_state(self):
        return {'base_pose': self.vehicle.x.copy(), 'activated': True}

    def close(self):
        self.vehicle.stop_control()


class BaseProcess:
    """Base process with iPhone WebXR teleoperation."""

    def __init__(self, activation_endpoint, config=None, debug=False, target_robot=None):
        self.config = config or {}
        self.host = self.config['host']
        self.logging_config = self.config.get('logging')
        self.debug = debug
        self.debug_printer = DebugPrinter(enabled=debug)
        self.data_port = self.config['port']
        self.running = False
        self.activated = False
        self.episode_active = False
        self.teleop_enabled = False
        self.episode_recording = False
        self.activation_endpoint = activation_endpoint

        # Base hardware (CAN/phoenix6) and the iPhone WebXR teleop policy both live
        # on the Robot PC (owns_hardware), so the base command path stays local
        # (policy.step -> execute_action in one loop). The gello PC base module runs
        # coordination only, reads measured state from RMQ, and never touches Base().
        self.owns_hardware = bool(self.config.get("owns_base_hardware", False))

        self.base = None
        # Dedicated process running the 250 Hz Vehicle control loop (Robot PC).
        self.base_server_process = None

        self.policy = None
        self.teleop_thread = None

        self.activation_client = None
        self._last_activation_signature = None
        self.msg_factory = MessageFactory()

        self.current_base_pose = np.zeros(3)

        # Latest measured STATE / commanded TARGET, snapshotted by the hot command
        # loop and republished over RMQ by a dedicated thread (Robot PC only) so the
        # blocking cross-PC put_data round-trips stay off the command hot path.
        self._publish_lock = threading.Lock()
        self._latest_state_pose7 = None   # measured [x, y, z, qx, qy, qz, qw]
        self._latest_target_pose = None   # commanded [x, y, theta]
        self.publisher_thread = None

        self.shutdown_flag = ShutdownFlag()
        base_handler = make_signal_handler(self.shutdown_flag)

        def _shutdown_handler(signum, frame):
            base_handler(signum, frame)
            self.running = False
            self._sigtstp_handler(signum, frame)

        signal.signal(signal.SIGTSTP, _shutdown_handler)
        signal.signal(signal.SIGTERM, _shutdown_handler)

        if self.logging_config is not None:
            has_unified_streams = bool(self.logging_config.get("logging_streams"))
            base_port = self.logging_config.get("base_port")
            if has_unified_streams or base_port is None:
                self.logger = None
            else:
                self.logger = RobotCtrlLogger(
                    name="body",
                    endpoint=f"tcp://localhost:{base_port}",
                    attr={
                        "robot_name": "body",
                        "ctrl_freq": CTRL_FREQ,
                    },
                    log_eef_pose=True,
                    log_joint_pos=False,
                    target_type="eef_pose",
                    joint_units=None,
                )
        else:
            self.logger = None

        print("Base Process initialized")

    def _sigtstp_handler(self, signum, frame):
        print("\n" + "=" * 60)
        print("CTRL+Z PRESSED - BASE EMERGENCY STOP")
        print("=" * 60)
        if self.base:
            self.base.emergency_stop()
        self.activated = False
        self.episode_active = False
        print("Base stopped. Reactivate to resume")
        print("=" * 60)

    def start_base(self):
        if not self.owns_hardware:
            print("Remote base hardware: driven on the Robot PC; no local Base() here")
            self.base = None
            return True

        if self.config and self.config.get("use_mock_base", False):
            print("use_mock_base=True: using MockBase (no CAN hardware required)")
            self.base = MockBase()
            self.base.reset()
            print("MockBase ready")
            return True

        try:
            # Deferred on purpose: base_server pulls in phoenix6 (CAN) via base_controller,
            # which only exists on the Robot PC. Importing it at module level would break the
            # owns_hardware=False and use_mock_base paths above, and would drag phoenix6 into
            # the coordinator (which imports this module via base/spec.py -> base/runner.py).
            from modpack.robots.arx5.base.base_server import serve_base_manager

            # Run the 250 Hz Vehicle control loop in a DEDICATED process so the
            # Flask/SocketIO WebXR server and teleop policy in this process can
            # never starve it of the GIL (co-locating them caused chronic 250 Hz
            # step-time violations and the base lag/coast regression). We talk to
            # it over a local RPC proxy with the same surface as the old Base.
            print("Starting base controller process (dedicated 250 Hz loop)...")
            self.base_server_process = mp.Process(target=serve_base_manager, daemon=True)
            self.base_server_process.start()

            self.base = connect_base()
            self.base.reset()
            print("Base controller ready (RPC, isolated process)")
            return True

        except Exception as e:
            print(f"ERROR: Failed to initialize base: {e}")
            return False

    def start_teleop_policy(self):
        try:
            print("Starting iPhone WebXR teleoperation...")
            self.policy = TeleopPolicy()
            timeout = 5.0
            start_time = time.time()
            while self.policy.url is None:
                if time.time() - start_time > timeout:
                    print("ERROR: Timeout waiting for web server")
                    return False
                time.sleep(0.01)
            print(f"iPhone WebXR interface ready at: {self.policy.page_url}")
            print("Open this URL on your iPhone to control the base")
            return True
        except Exception as e:
            print(f"ERROR: Failed to start teleoperation: {e}")
            return False

    def setup_clients(self):
        try:
            print("Connecting clients to servers...")
            self.activation_client = robotmq.RMQClient(
                client_name="base_activation_client",
                server_endpoint=self.activation_endpoint,
            )
            self.data_client = robotmq.RMQClient(
                client_name="base_data_client",
                server_endpoint=f'tcp://{self.host}:{self.data_port}',
            )
            print("Connected to activation server")
            return True
        except Exception as e:
            print(f"ERROR: Failed to connect to activation server: {e}")
            return False

    def process_activation_message(self, activation_msg):
        command = activation_msg.get("command")
        target = activation_msg.get("target", "all")

        if target not in ["base", "all"]:
            return

        if command == "activate":
            print("Base ACTIVATED")
            self.activated = True
            if self.base:
                self.base.activate()

        elif command == "deactivate":
            print("Base DEACTIVATED (paused)")
            self.activated = False
            self.teleop_enabled = False
            self.episode_recording = False
            if self.base:
                self.base.deactivate()

        elif command == "emergency_stop":
            print("Base EMERGENCY STOP")
            self.activated = False
            self.episode_active = False
            if self.base:
                self.base.emergency_stop()

        elif command == "start_episode":
            print("Episode STARTED - iPhone control now ACTIVE")
            self.episode_active = False
            self.episode_recording = True
            self.teleop_enabled = True
            time.sleep(0.05)

            if not self.activated:
                self.activated = True
                if self.base:
                    self.base.reset()
                    self.base.activate()

            if self.policy and hasattr(self.policy, 'process_activation_message'):
                msg = UnifiedWebXRMessage(
                    timestamp=int(1000 * time.time()),
                    episode_active=True,
                    episode_num=activation_msg.get("episode_num", 0),
                    rezero=True,
                )
                msg_dict = msg.to_dict()
                msg_dict['command'] = 'start_episode'
                self.policy.process_activation_message(msg_dict)
                if hasattr(self.policy, 'server'):
                    self.policy.server.broadcast_to_iphone(msg_dict)
                time.sleep(0.1)
                self.episode_active = True
                if hasattr(self.policy, 'teleop_controller') and self.policy.teleop_controller:
                    self.policy.teleop_controller.targets_initialized = False
            else:
                print("Cannot start episode - TeleopPolicy not ready")

        elif command == "end_episode":
            print("Episode ENDED")
            self.episode_active = False
            self.episode_recording = False
            self.teleop_enabled = True
            time.sleep(0.05)

        elif command == "pause_episode":
            print("Episode PAUSED")
            was_active = self.episode_active
            self.episode_active = False
            self.episode_recording = False
            self.teleop_enabled = False
            if was_active:
                time.sleep(0.05)
            if self.policy and hasattr(self.policy, 'process_activation_message'):
                self.policy.process_activation_message({'command': 'pause_episode', 'timestamp': int(1000 * time.time())})

        elif command == "resume_episode":
            print("Episode RESUMED with REZERO")
            self.episode_active = False
            self.episode_recording = True
            self.teleop_enabled = True
            time.sleep(0.05)
            if self.policy and hasattr(self.policy, 'process_activation_message'):
                msg = UnifiedWebXRMessage(
                    timestamp=int(1000 * time.time()),
                    episode_active=True,
                    rezero=True,
                )
                msg_dict = msg.to_dict()
                msg_dict['command'] = 'resume_episode'
                self.policy.process_activation_message(msg_dict)
                if hasattr(self.policy, 'server'):
                    self.policy.server.broadcast_to_iphone(msg_dict)
                time.sleep(0.1)
                self.episode_active = True
                print("Rezero complete - control loop resumed")

        elif command == "reset":
            print("Base RESET")
            self.activated = False
            self.episode_active = False
            self.teleop_enabled = False
            self.episode_recording = False
            if self.base:
                try:
                    self.base.reset()
                except Exception as e:
                    print(f"[BaseProcess] base.reset() failed, deactivating: {e}")
                    self.base.deactivate()

        elif command in ("shutdown", "manager_shutdown"):
            print("Base shutdown requested")
            self.running = False

    def update_base_pose(self):
        try:
            if self.owns_hardware:
                if self.base:
                    pose = self.base.get_state()['base_pose']
                    if pose is not None:
                        self.current_base_pose = np.array(pose)
                return
            # Remote hardware: read the measured pose the Robot PC publishes on
            # Topics.BASE (7-element [x, y, z, qx, qy, qz, qw]).
            data_list, _ = self.data_client.peek_data(Topics.BASE, n=-1)
            if not data_list:
                return
            pose = deserialize_base_proprio_message(data_list[0]).pose
            if pose is None or len(pose) < 3:
                return
            theta = R.from_quat(pose[3:7]).as_euler('xyz')[2] if len(pose) >= 7 else pose[2]
            self.current_base_pose = np.array([pose[0], pose[1], theta])
        except Exception as e:
            print(f"[BaseProcess] get_state error: {e}")

    def teleop_control_loop(self):
        print("Teleoperation control loop started")
        while self.running:
            recording = self.logger is not None and self.logger.update_recording_state()
            loop_start = time.time()
            self.update_base_pose()
            action_msg = None
            action = None

            state_pos_xyz = np.array([self.current_base_pose[0], self.current_base_pose[1], 0], dtype=np.float32)
            state_quat_xyzw = R.from_euler('z', self.current_base_pose[-1]).as_quat()
            state_quat_wxyz = _wxyz_from_xyzw(state_quat_xyzw)

            # Only the hardware owner (Robot PC) publishes measured STATE on
            # Topics.BASE ('body'). The gello side reads it (update_base_pose).
            # Snapshot it here for the publisher thread instead of doing the
            # blocking cross-PC put_data inline — that round-trip must not sit
            # between policy.step and execute_action on the command hot path.
            if self.activated and self.owns_hardware:
                with self._publish_lock:
                    self._latest_state_pose7 = [*state_pos_xyz.tolist(), *state_quat_xyzw.tolist()]

            # iPhone WebXR teleop runs co-located with the hardware (Robot PC),
            # where self.policy is set; policy.step -> execute_action stays local.
            if self.activated and self.teleop_enabled and self.policy is not None:
                obs = {'base_pose': self.current_base_pose.copy()}
                action = self.policy.step(obs)
                if action is not None:
                    target_pos_xyz = np.array([action['base_pose'][0], action['base_pose'][1], 0], dtype=np.float32)
                    target_quat_xyzw = R.from_euler('z', action['base_pose'][-1]).as_quat()
                    target_quat_wxyz = _wxyz_from_xyzw(target_quat_xyzw)

                if recording and self.episode_recording:
                    self.logger.log_state(
                        state_timestamp=time.monotonic(),
                        state_pos_xyz=state_pos_xyz,
                        state_quat_wxyz=state_quat_wxyz,
                    )
                    if action is not None:
                        self.logger.log_target(
                            target_timestamp=time.monotonic(),
                            target_pos_xyz=target_pos_xyz,
                            target_quat_wxyz=target_quat_wxyz,
                        )

            # Robot PC (no WebXR policy): drive from the gello WebXR target on
            # Topics.BASE_TARGET ('body_target'). Both sides also honour the
            # external 'body_cmd' rollout target as a fallback.
            if self.activated and action is None and self.owns_hardware:
                try:
                    tgt_raw, _ = self.data_client.peek_data(Topics.BASE_TARGET, n=-1)
                    if tgt_raw:
                        tgt = deserialize_base_proprio_message(tgt_raw[0]).pose
                        if tgt is not None and len(tgt) >= 3:
                            theta = R.from_quat(tgt[3:7]).as_euler('xyz')[2] if len(tgt) >= 7 else tgt[2]
                            action = {'base_pose': [tgt[0], tgt[1], theta]}
                except Exception as e:
                    print(f"[BaseProcess] body_target consume error: {e}")

            if self.activated and action is None:
                try:
                    cmd_raw, _ = self.data_client.peek_data(f"{Topics.BASE}_cmd", n=-1)
                    if cmd_raw:
                        cmd = deserialize(cmd_raw[0])
                        if isinstance(cmd, dict) and 'base_pose' in cmd:
                            action = {'base_pose': list(cmd['base_pose'])}
                except Exception as e:
                    print(f"[BaseProcess] external base cmd error: {e}")

            action_to_run = action if action is not None else action_msg

            if self.activated and self.teleop_enabled and action_to_run and isinstance(action_to_run, dict):
                try:
                    if self.owns_hardware:
                        # Robot PC: drive the physical base directly from the locally
                        # computed teleop target (no cross-PC hop, no RMQ in the hot
                        # path). Snapshot the target for the publisher thread, which
                        # republishes it on BASE_TARGET ('body_target') off the hot
                        # path so the base_state logger, rby1 teleop, and vision_pro
                        # can still read it.
                        self.base.execute_action(action_to_run)
                        if self.config.get("publish_base_target", True) and 'base_pose' in action_to_run:
                            with self._publish_lock:
                                self._latest_target_pose = list(action_to_run['base_pose'])
                    else:
                        # Gello side: publish the commanded base pose as the action/
                        # target on BASE_TARGET ('body_target'). The Robot PC consumes
                        # this to drive Base(); the base_state logger (target_topic)
                        # and rby1 teleop also read it.
                        if self.config.get("publish_base_target", True) and 'base_pose' in action_to_run:
                            target_msg = self.msg_factory.create_base_message(pose=list(action_to_run['base_pose']))
                            self.data_client.put_data(Topics.BASE_TARGET, serialize_base_proprio_message(target_msg))
                except Exception as e:
                    print(f"Error executing action: {e}")

            elapsed = time.time() - loop_start
            time.sleep(max(0, POLICY_CONTROL_PERIOD - elapsed))

        print("Teleoperation control loop stopped")

    def publisher_loop(self):
        """Republish measured STATE and commanded TARGET over RMQ for remote
        consumers (gello logger, rby1 teleop, vision_pro), OFF the command hot
        path. The blocking cross-PC put_data round-trips live here so they never
        delay policy.step -> execute_action on the Robot PC. Robot PC only."""
        print("Base publisher loop started")
        while self.running:
            loop_start = time.time()
            if self.activated:
                with self._publish_lock:
                    state_pose7 = self._latest_state_pose7
                    target_pose = self._latest_target_pose
                try:
                    if state_pose7 is not None:
                        state_msg = self.msg_factory.create_base_message(pose=state_pose7)
                        self.data_client.put_data(Topics.BASE, serialize_base_proprio_message(state_msg))
                    if target_pose is not None and self.config.get("publish_base_target", True):
                        target_msg = self.msg_factory.create_base_message(pose=target_pose)
                        self.data_client.put_data(Topics.BASE_TARGET, serialize_base_proprio_message(target_msg))
                except Exception as e:
                    print(f"[BaseProcess] publisher put_data error: {e}")
            elapsed = time.time() - loop_start
            time.sleep(max(0, PUBLISH_PERIOD - elapsed))
        print("Base publisher loop stopped")

    def activation_loop(self):
        print("Base activation monitoring started")
        while self.running:
            try:
                data, _ = self.activation_client.peek_data(Topics.ACTIVATION, n=-1)
                if data:
                    activation_msg = deserialize(data[0])
                    signature = (
                        activation_msg.get("timestamp"),
                        activation_msg.get("command"),
                        activation_msg.get("target"),
                    )
                    if signature != self._last_activation_signature:
                        self.process_activation_message(activation_msg)
                        self._last_activation_signature = signature
                time.sleep(0.01)
            except Exception as e:
                print(f"Error in activation loop: {e}")
                time.sleep(0.1)
        print("Base activation monitoring stopped")

    def shutdown(self):
        print("Shutting down base process...")
        self.running = False
        if self.base:
            try:
                self.base.close()
            except Exception as e:
                print(f"Error closing base: {e}")
        # Tear down the dedicated Vehicle control-loop process.
        if self.base_server_process is not None:
            try:
                self.base_server_process.terminate()
                self.base_server_process.join(timeout=5)
            except Exception as e:
                print(f"Error terminating base server process: {e}")
        print("Base process shutdown complete")

    def run(self):
        print("Starting Base Process...")
        print("=" * 60)

        if not self.start_base():
            print("Failed to initialize base")
            return False

        # The iPhone WebXR teleop server + policy run co-located with the base
        # hardware on the Robot PC, so policy.step -> execute_action happens in one
        # loop with no cross-PC hop (and the policy reads fresh local pose feedback).
        # The gello PC base module runs coordination only and never runs the policy.
        if self.owns_hardware:
            if not self.start_teleop_policy():
                print("Failed to start teleoperation")
        else:
            self.policy = None

        if not self.setup_clients():
            print("Cannot connect to activation server")

        self.running = True

        print("\n" + "=" * 60)
        print("BASE PROCESS READY")

        if self.policy:
            print(f"iPhone control: {self.policy.url}")
        print("Waiting for activation and episode start")
        print("=" * 60 + "\n")

        self.teleop_thread = threading.Thread(target=self.teleop_control_loop, daemon=True)
        self.teleop_thread.start()

        # Robot PC republishes STATE/TARGET over RMQ on a dedicated thread so the
        # blocking cross-PC put_data round-trips stay off the command hot path.
        if self.owns_hardware:
            self.publisher_thread = threading.Thread(target=self.publisher_loop, daemon=True)
            self.publisher_thread.start()

        try:
            if self.activation_client:
                self.activation_loop()
            else:
                while self.running:
                    time.sleep(0.01)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received")
        except Exception as e:
            print(f"Error in base process: {e}")
        finally:
            self.shutdown()

        return True


def main() -> None:
    """Standalone entry point for the Robot PC (launched via robot_pc.scripts).

    The CAN hardware (CANivore) is attached to the Robot PC, so the base
    process must run there — not on the gello PC, where the coordinator
    spawns its local modules.
    """
    parser = argparse.ArgumentParser(description="Base process with teleop")
    parser.add_argument("--data-host",       required=True,            help="RMQ data server host (gello PC IP)")
    parser.add_argument("--data-port",       default=5555, type=int,   help="RMQ data server port")
    parser.add_argument("--activation-host", required=True,            help="Activation server host (gello PC IP)")
    parser.add_argument("--activation-port", default=5556, type=int,   help="Activation server port")
    parser.add_argument("--mock",  action="store_true", help="Use MockBase (no CAN hardware required)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    config = {
        "host": args.data_host,
        "port": args.data_port,
        "use_mock_base": args.mock,
        # Standalone entry point runs on the Robot PC, which owns the CAN hardware.
        "owns_base_hardware": True,
    }
    process = BaseProcess(
        activation_endpoint=f"tcp://{args.activation_host}:{args.activation_port}",
        config=config,
        debug=args.debug,
    )
    if not process.run():
        sys.exit(1)


if __name__ == "__main__":
    main()
