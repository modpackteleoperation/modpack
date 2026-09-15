#!/usr/bin/env python3
"""
Backpack-side episode logging.

Runs on the Modpack PC. Subscribes to RMQ state topics published by the
robot PC via RobotBridge and writes all proprio and camera data locally.

Classes:
  BackpackStreamLogger — unified logger driven by logging_streams config list
  BackpackCameraLogger — polls camera topics from the camera server, drives VideoLogger
"""

import time
import threading
import struct
from typing import Dict, List, Optional

import cv2
import numpy as np
import robotmq
from robotmq.utils import deserialize
from scipy.spatial.transform import Rotation as R

from robologger.loggers.robot_ctrl_logger import RobotCtrlLogger
from robologger.loggers.mobile_base_logger import MobileBaseLogger

from modpack.orchestration.message_formats import (
    Topics,
    deserialize_neck_message,
    deserialize_base_proprio_message,
    deserialize_torque_message,
)
from modpack.orchestration.message_formats import (
    KEY_GRIPPER_POSITION,
    KEY_JOINT_POSITIONS,
    KEY_TIMESTAMP,
    normalize_follower_command,
)
from robologger.loggers.video_logger import VideoLogger


def _build_ee_gripper_logger(arm: str, port: int, control_freq: float = 100.0) -> Optional[RobotCtrlLogger]:
    """End-effector (gripper) logger; `name` must match RobotName enum."""
    name = "right_end_effector" if arm == "right" else "left_end_effector"
    return RobotCtrlLogger(
        name=name,
        endpoint=f"tcp://localhost:{port}",
        attr={"robot_name": name, "ctrl_freq": control_freq, "num_joints": 1},
        log_eef_pose=False,
        log_joint_pos=True,
        target_type="joint_pos",
        joint_units="meters",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _eef_to_pos_quat(eef_pose):
    """Convert 6-D (x,y,z,roll,pitch,yaw) -> (pos_xyz, quat_wxyz)."""
    pos = np.array(eef_pose[:3], dtype=np.float32)
    quat_xyzw = R.from_euler('xyz', eef_pose[3:]).as_quat().astype(np.float32)
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
    return pos, quat_wxyz



# ─────────────────────────────────────────────────────────────────────────────
# BackpackStreamLogger — unified logger driven by logging_streams config
# ─────────────────────────────────────────────────────────────────────────────

def _build_stream_logger(entry: dict):
    """
    Factory: returns (entry, logger, optional_gripper_logger) or None.

    Dispatches on entry['logger_type']:
      joint_command   — follower_command, logs joint_positions (+optional gripper scalar)
      joint_state     — robot state dict; logs positions + optional eef_pose; logs GELLO cmd as target
      base_state      — base odometry dict; logs pose via MobileBaseLogger
      cartesian_state — cartesian state dict; logs eef_pose via RobotCtrlLogger
      torque
    """
    logger_type = entry.get("logger_type", "joint_command")
    name = entry.get("logger_name")
    port = entry.get("port")
    if not name or not port:
        return None

    if logger_type == "joint_command":
        joint_dof = int(entry.get("joint_dof", 7))
        n_joints = joint_dof + (1 if entry.get("log_gripper_scalar") else 0)
        lg = RobotCtrlLogger(
            name=name,
            endpoint=f"tcp://localhost:{port}",
            attr={
                "robot_name": name,
                "ctrl_freq": float(entry.get("control_freq", 100.0)),
                "num_joints": n_joints,
                "gripper_in_last_joint_dim": bool(entry.get("log_gripper_scalar")),
            },
            log_eef_pose=False,
            log_joint_pos=True,
            target_type="joint_pos",
            joint_units="radians",
        )
        print(f"[BackpackStreamLogger] {name} ← topic={entry.get('topic')} type=joint_command dof={n_joints}")
        return (entry, lg, None)

    elif logger_type == "joint_state":
        joint_dof = int(entry.get("joint_dof", 6))
        log_eef = entry.get("log_eef", True)
        lg = RobotCtrlLogger(
            name=name,
            endpoint=f"tcp://localhost:{port}",
            attr={
                "robot_name": name,
                "ctrl_freq": float(entry.get("control_freq", 100.0)),
                "num_joints": joint_dof,
            },
            log_eef_pose=bool(log_eef),
            log_joint_pos=True,
            target_type="joint_pos",
            joint_units="radians",
        )
        ee_lg = None
        gp_val = entry.get("gripper_port")
        gn = entry.get("gripper_logger_name")
        if gp_val and gn:
            try:
                arm_hint = "right" if "right" in name else "left"
                ee_lg = _build_ee_gripper_logger(arm_hint, int(gp_val), float(entry.get("control_freq", 100.0)))
                print(f"[BackpackStreamLogger] {gn} ← companion EE gripper port={gp_val}")
            except Exception as e:
                print(f"[BackpackStreamLogger] EE logger build failed ({name}): {e}")
        print(f"[BackpackStreamLogger] {name} ← topic={entry.get('topic')} type=joint_state dof={joint_dof}")
        return (entry, lg, ee_lg)

    elif logger_type == "base_state":
        if not entry.get("target_topic"):
            raise ValueError(
                f"base_state logger '{name}' is missing 'target_topic' in config — "
                "MobileBaseLogger requires a target stream"
            )
        lg = MobileBaseLogger(
            name=name,
            endpoint=f"tcp://localhost:{port}",
            attr={"robot_name": name, "ctrl_freq": float(entry.get("control_freq", 30.0))},
            log_state_pose=True,
            log_state_velocity=False,
            log_target_pose=True,
            log_target_velocity=False,
        )
        print(
            f"[BackpackStreamLogger] {name} ← topic={entry.get('topic')} "
            f"target_topic={entry.get('target_topic')} type=base_state"
        )
        return (entry, lg, None)

    elif logger_type == "cartesian_state":
        lg = RobotCtrlLogger(
            name=name,
            endpoint=f"tcp://localhost:{port}",
            attr={"robot_name": name, "ctrl_freq": float(entry.get("control_freq", 30.0))},
            log_eef_pose=True,
            log_joint_pos=False,
            target_type="eef_pose",
            joint_units=None,
        )
        print(f"[BackpackStreamLogger] {name} ← topic={entry.get('topic')} type=cartesian_state")
        return (entry, lg, None)

    print(f"[BackpackStreamLogger] Unknown logger_type '{logger_type}' for entry {name!r}")
    return None


def _build_grouped_stream_logger(entry: dict) -> list:
    """
    Factory for grouped entries (entry has a 'topics' list).
    Creates one shared RobotCtrlLogger (or MobileBaseLogger); returns one stream
    tuple per sub-topic so each topic gets its own poll thread while sharing the
    same underlying logger instance.
    """
    name = entry.get("logger_name")
    port = entry.get("port")
    if not name or not port:
        return []

    joint_dof = int(entry.get("joint_dof", 7))
    freq = float(entry.get("control_freq", 100.0))
    log_eef = bool(entry.get("log_eef", False))

    lg = RobotCtrlLogger(
        name=name,
        endpoint=f"tcp://localhost:{port}",
        attr={"robot_name": name, "ctrl_freq": freq, "num_joints": joint_dof},
        log_eef_pose=log_eef,
        log_joint_pos=True,
        target_type="joint_pos",
        joint_units="radians",
    )

    ee_lg = None
    gp = entry.get("gripper_port")
    gn = entry.get("gripper_logger_name")
    if gp and gn:
        arm_hint = "right" if "right" in name else "left"
        try:
            ee_lg = _build_ee_gripper_logger(arm_hint, int(gp), freq)
            print(f"[BackpackStreamLogger] {gn} ← companion EE gripper port={gp}")
        except Exception as e:
            print(f"[BackpackStreamLogger] EE logger build failed ({name}): {e}")

    # One lock per shared logger guards update_recording_state() from concurrent
    # threads in the same group (each topic gets its own thread but shares lg/ee_lg).
    lg_lock = threading.Lock()
    ee_lock = threading.Lock() if ee_lg is not None else None

    log_period = 1.0 / freq
    streams = []
    for t in entry.get("topics", []):
        synthetic = {**entry, **t}
        synthetic["topic"] = t.get("topic")
        synthetic["logger_type"] = t.get("role", "joint_command")
        synthetic.pop("topics", None)
        synthetic["__lg_lock"] = lg_lock
        if ee_lock is not None:
            synthetic["__ee_lock"] = ee_lock
        streams.append((synthetic, lg, ee_lg, log_period))
        print(f"[BackpackStreamLogger] {name} ← topic={t.get('topic')} role={t.get('role')} port={port}")

    return streams


class BackpackStreamLogger:
    """
    Unified backpack proprio logger driven by a logging_streams config list.

    Subscribes to state topics on the data server (port 5555) and the feedback
    server (port 5581). BackpackCameraLogger handles logger_type: camera entries.

    Each entry in logging_streams must have: topic, logger_name, port,
    logger_type, control_freq, enabled. Optional fields depend on logger_type
    (see _build_stream_logger). Entry schema mirrors modpack_config.yaml
    robotmq.logging.logging_streams.
    """

    def __init__(
        self,
        host: str,
        port: int,
        logging_streams: List[dict],
        debug: bool = False,
        feedback_port: Optional[int] = None,
    ):
        self.debug = debug
        self.running = False

        self.data_client = robotmq.RMQClient(
            client_name="backpack_stream_logger",
            server_endpoint=f"tcp://{host}:{port}",
        )

        # Torque-role topics are published to the separate feedback server
        # (see coordinator.feedback_server / RobotBridge.publish_feedback_raw).
        self.feedback_client = None
        if feedback_port:
            try:
                self.feedback_client = robotmq.RMQClient(
                    client_name="backpack_stream_logger_feedback",
                    server_endpoint=f"tcp://{host}:{feedback_port}",
                )
            except Exception as e:
                print(f"[BackpackStreamLogger] feedback client connect failed ({host}:{feedback_port}): {e}")

        # Each stream is stored as (entry, lg, extra, log_period) where log_period = 1/control_freq.
        # control_freq is required in every logging_streams entry.
        self._streams: List[tuple] = []
        for entry in logging_streams:
            if entry.get("enabled", True) is False:
                continue
            # Skip entries owned by the robot process (single-PC direct-logger path)
            if entry.get("logger_type") == "camera":
                continue  # handled by BackpackCameraLogger
            if "control_freq" not in entry:
                raise ValueError(
                    f"[BackpackStreamLogger] entry {entry.get('logger_name')!r} is missing "
                    "required field 'control_freq'"
                )
            freq = float(entry["control_freq"])
            if not (1.0 <= freq <= 1000.0):
                raise ValueError(
                    f"[BackpackStreamLogger] entry {entry.get('logger_name')!r} control_freq "
                    f"{freq} is outside valid range [1, 1000] Hz"
                )
            try:
                if "topics" in entry:
                    for r in _build_grouped_stream_logger(entry):
                        self._streams.append(r)
                else:
                    result = _build_stream_logger(entry)
                    if result is not None:
                        entry_out, lg, extra = result
                        self._streams.append((entry_out, lg, extra, 1.0 / freq))
            except Exception as e:
                print(f"[BackpackStreamLogger] skip entry {entry!r}: {e}")

    def run(self) -> None:
        if not self._streams:
            print("[BackpackStreamLogger] No streams configured; thread exiting")
            return
        self.running = True
        # Each stream runs its own thread at its own control_freq
        threads = [
            threading.Thread(target=self._run_stream, args=(s,), daemon=True)
            for s in self._streams
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def _run_stream(self, stream: tuple) -> None:
        """Per-stream loop following the standard robologger pattern: sleep(max(0, period-elapsed))."""
        entry, lg, extra, log_period = stream
        name = entry.get("logger_name", "?")
        hz = round(1.0 / log_period)
        print(f"[BackpackStreamLogger] {name} loop started ({hz} Hz)")
        while self.running:
            t0 = time.monotonic()
            self._poll_stream(stream)
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, log_period - elapsed))
        print(f"[BackpackStreamLogger] {name} loop stopped")

    def _poll_stream(self, stream: tuple) -> None:
        entry, lg, extra, _log_period = stream
        topic = entry.get("topic")
        logger_type = entry.get("logger_type", "joint_command")

        try:
            lg_lock = entry.get("__lg_lock")
            if lg_lock is not None:
                with lg_lock:
                    recording = lg.update_recording_state()
            else:
                recording = lg.update_recording_state()
        except Exception as e:
            if self.debug:
                print(f"[BackpackStreamLogger] update_recording_state ({topic}): {e}")
            return

        # Always drain the EE logger's command queue so stop/start commands are
        # processed even when the arm logger is not recording.
        if extra is not None:
            try:
                ee_lock = entry.get("__ee_lock")
                if ee_lock is not None:
                    with ee_lock:
                        extra.update_recording_state()
                else:
                    extra.update_recording_state()
            except Exception as e:
                if self.debug:
                    print(f"[BackpackStreamLogger] EE update_recording_state ({topic}): {e}")

        if not recording:
            return

        if logger_type == "torque":
            self._log_torque(entry, lg, topic)
            return

        if logger_type == "ft":
            self._log_ft_wrench(entry, lg, topic)
            return

        if logger_type == "base_state":
            # Base proprio is decoded with the typed deserializer (msgpack from the
            # teleop bridge, JSON from base_process) rather than the generic
            # msgpack-only path below, mirroring _log_base_target.
            self._log_base_state(topic, lg)
            target_topic = entry.get("target_topic")
            if target_topic:
                self._log_base_target(target_topic, lg)
            return

        try:
            data_list, _ = self.data_client.peek_data(topic, n=-1)
            if not data_list:
                return
            raw = data_list[0]
            state = deserialize(raw) if isinstance(raw, (bytes, bytearray, memoryview)) else raw
            if not isinstance(state, dict):
                return
        except Exception as e:
            if self.debug:
                print(f"[BackpackStreamLogger] peek_data ({topic}): {e}")
            return

        ts = state.get("timestamp", time.monotonic())

        try:
            if logger_type == "joint_command":
                self._log_joint_command(entry, lg, extra, state, ts)
            elif logger_type == "joint_state":
                self._log_joint_state(entry, lg, extra, state, ts)
            elif logger_type == "cartesian_state":
                self._log_cartesian_state(entry, lg, state, ts)
        except Exception as e:
            if self.debug:
                print(f"[BackpackStreamLogger] log ({topic}, {logger_type}): {e}")

    def _log_joint_command(self, entry: dict, lg, ee_lg, state: dict, ts: float) -> None:
        """Log follower_command: joint_positions + optional gripper scalar.

        When ee_lg is provided (grouped mode), the gripper is routed to the companion
        EE logger via log_target() and is NOT appended to the main arm array.
        When ee_lg is None (flat/legacy mode), the gripper is appended to the main
        arm joints array for backward compatibility.
        """
        msg = normalize_follower_command(
            state, frame_id_default=str(entry.get("frame_id_default") or "")
        )
        if msg is None:
            return
        jpos = msg.get(KEY_JOINT_POSITIONS, [])
        joint_dof = int(entry.get("joint_dof", 7))
        use_g = bool(entry.get("log_gripper_scalar"))
        if len(jpos) < joint_dof:
            return
        vals = [float(x) for x in jpos[:joint_dof]]
        ts = float(msg.get(KEY_TIMESTAMP, ts))
        if use_g:
            g = msg.get(KEY_GRIPPER_POSITION)
            if g is None:
                return
            if ee_lg is not None:
                try:
                    ee_lock = entry.get("__ee_lock")
                    if ee_lock is not None:
                        with ee_lock:
                            ee_recording = ee_lg.update_recording_state()
                    else:
                        ee_recording = ee_lg.update_recording_state()
                    if ee_recording:
                        ee_lg.log_target(
                            target_timestamp=ts,
                            target_joint_pos=np.array([float(g)], dtype=np.float64),
                        )
                except Exception as e:
                    print(f"[StreamLogger] gripper log error: {e}")
            else:
                vals.append(float(g))
        lg.log_target(
            target_timestamp=ts,
            target_joint_pos=np.array(vals, dtype=np.float64),
        )

    def _log_joint_state(self, entry: dict, lg, ee_lg, state: dict, ts: float) -> None:
        """Log robot joint state (positions + optional eef_pose). Target = GELLO leader cmd."""
        joints = state.get("positions")
        if joints is None:
            return
        joint_arr = np.array(joints, dtype=np.float32)
        eef = state.get("eef_pose")
        log_eef = entry.get("log_eef", True)
        if log_eef and eef is not None:
            pos, quat = _eef_to_pos_quat(eef)
            lg.log_state(
                state_timestamp=ts,
                state_pos_xyz=pos,
                state_quat_wxyz=quat,
                state_joint_pos=joint_arr,
            )
        else:
            lg.log_state(
                state_timestamp=ts,
                state_joint_pos=joint_arr,
            )

        # Companion EE gripper
        if ee_lg is not None:
            try:
                ee_lock = entry.get("__ee_lock")
                if ee_lock is not None:
                    with ee_lock:
                        ee_recording = ee_lg.update_recording_state()
                else:
                    ee_recording = ee_lg.update_recording_state()
                if ee_recording:
                    if state.get("gripper") is not None:
                        ee_lg.log_state(
                            state_timestamp=ts,
                            state_joint_pos=np.array([float(state["gripper"])], dtype=np.float32),
                        )
                    if state.get("gripper_target") is not None:
                        ee_lg.log_target(
                            target_timestamp=ts,
                            target_joint_pos=np.array([float(state["gripper_target"])], dtype=np.float32),
                        )
            except Exception as e:
                if self.debug:
                    print(f"[BackpackStreamLogger] EE log ({entry.get('logger_name')}): {e}")

        # Log target in the same message. The gripper target is handled once in
        # the companion-EE block above; logging it here too would double-log it.
        target_joints = state.get("target_positions")
        if target_joints is not None:
            lg.log_target(
                target_timestamp=ts,
                target_joint_pos=np.array(target_joints, dtype=np.float32),
            )

        # Log GELLO leader as target
        gello_topic = entry.get("gello_target_topic")
        if gello_topic:
            try:
                gdata, _ = self.data_client.peek_data(gello_topic, n=-1)
                if gdata:
                    raw_g = gdata[0]
                    gcmd = deserialize(raw_g) if isinstance(raw_g, (bytes, bytearray, memoryview)) else raw_g
                    if isinstance(gcmd, dict):
                        gt = gcmd.get("timestamp", time.monotonic())
                        jp = gcmd.get("joint_positions")
                        if jp is not None:
                            lg.log_target(
                                target_timestamp=gt,
                                target_joint_pos=np.array(
                                    jp[:entry.get("joint_dof", 6)], dtype=np.float32
                                ),
                            )
                        gp = gcmd.get("gripper_position")
                        if ee_lg is not None and gp is not None:
                            ee_lg.log_target(
                                target_timestamp=gt,
                                target_joint_pos=np.array([float(gp)], dtype=np.float32),
                            )
            except Exception as e:
                print(f"[BackpackStreamLogger] log_target ({gello_topic}): {e}")

    def _log_base_state(self, topic: str, lg) -> None:
        """Log base odometry by peeking the state topic.

        Uses the typed base-proprio deserializer (mirrors _log_base_target) so it
        accepts both wire formats published to the base state topic: msgpack from
        the teleop bridge ([x, y, yaw]) and JSON from base_process
        ([x, y, z, qx, qy, qz, qw]). Both normalise to [x, y, yaw].
        """
        try:
            data_list, _ = self.data_client.peek_data(topic, n=-1)
            if not data_list:
                return
            smsg = deserialize_base_proprio_message(data_list[0])
        except Exception as e:
            if self.debug:
                print(f"[BackpackStreamLogger] peek_data ({topic}): {e}")
            return

        pose = getattr(smsg, "pose", None)
        if not pose or len(pose) < 3:
            return
        pose_arr = np.asarray(pose, dtype=np.float64)
        if pose_arr.shape[0] == 3:
            pose_3d = pose_arr
        elif pose_arr.shape[0] >= 7:
            # [x, y, z, qx, qy, qz, qw] → [x, y, yaw]
            yaw = R.from_quat(pose_arr[3:7]).as_euler("xyz")[2]
            pose_3d = np.array([pose_arr[0], pose_arr[1], yaw], dtype=np.float64)
        else:
            return
        ts = float(getattr(smsg, "timestamp", time.monotonic()))
        try:
            lg.log_state(
                state_timestamp=ts,
                state_pose=pose_3d,
            )
        except Exception as e:
            if self.debug:
                print(f"[BackpackStreamLogger] log_state ({topic}): {e}")

    def _log_base_target(self, target_topic: str, lg) -> None:
        """Log base target pose by peeking the companion target topic.

        Accepts both [x, y, yaw] and [x, y, z, qx, qy, qz, qw] wire formats and
        normalises to [x, y, yaw] before storage (mirrors _log_base_state).
        """
        try:
            data_list, _ = self.data_client.peek_data(target_topic, n=-1)
            if not data_list:
                return
            raw = data_list[0]
            tmsg = deserialize_base_proprio_message(raw)
        except Exception as e:
            if self.debug:
                print(f"[BackpackStreamLogger] peek_data ({target_topic}): {e}")
            return

        pose = getattr(tmsg, "pose", None)
        if not pose or len(pose) < 3:
            return
        pose_arr = np.asarray(pose, dtype=np.float64)
        if pose_arr.shape[0] == 3:
            pose_3d = pose_arr
        elif pose_arr.shape[0] >= 7:
            yaw = R.from_quat(pose_arr[3:7]).as_euler("xyz")[2]
            pose_3d = np.array([pose_arr[0], pose_arr[1], yaw], dtype=np.float64)
        else:
            return
        target_ts = float(getattr(tmsg, "timestamp", time.monotonic()))
        try:
            lg.log_target(
                target_timestamp=target_ts,
                target_pose=pose_3d,
            )
        except Exception as e:
            if self.debug:
                print(f"[BackpackStreamLogger] log_target ({target_topic}): {e}")

    def _log_torque(self, entry: dict, lg, topic: str) -> None:
        """Log per-joint torques from the feedback server onto the shared arm logger.

        RobotCtrlLogger.log_state requires state_joint_pos when log_joint_pos=True,
        so we pair the torque with the latest joint_state positions from the sibling
        topic (the grouped entry's logger_name == joint_state topic name).
        """
        if self.feedback_client is None:
            return
        try:
            data_list, _ = self.feedback_client.peek_data(topic, n=-1)
            if not data_list:
                return
            raw = data_list[0]
            tmsg = deserialize_torque_message(bytes(raw)) if isinstance(
                raw, (bytes, bytearray, memoryview)
            ) else raw
        except Exception as e:
            if self.debug:
                print(f"[BackpackStreamLogger] peek_data ({topic}): {e}")
            return

        joint_torques = getattr(tmsg, "joint_torques", None)
        if joint_torques is None:
            return
        ts = float(getattr(tmsg, "timestamp", time.monotonic()))
        joint_dof = int(entry.get("joint_dof", 7))
        torque_arr = np.asarray(joint_torques, dtype=np.float64)
        if torque_arr.shape[0] != joint_dof:
            if self.debug:
                print(
                    f"[BackpackStreamLogger] torque dof mismatch ({topic}): "
                    f"got {torque_arr.shape[0]}, expected {joint_dof}"
                )
            return

        # Pair with latest joint_state positions from the sibling topic.
        joint_state_topic = entry.get("logger_name")
        try:
            jdata, _ = self.data_client.peek_data(joint_state_topic, n=-1)
            if not jdata:
                return
            jraw = jdata[0]
            jstate = deserialize(jraw) if isinstance(jraw, (bytes, bytearray, memoryview)) else jraw
            if not isinstance(jstate, dict):
                return
            joints = jstate.get("positions")
            if joints is None:
                return
            joint_arr = np.asarray(joints, dtype=np.float64)
            if joint_arr.shape[0] != joint_dof:
                return
        except Exception as e:
            if self.debug:
                print(f"[BackpackStreamLogger] peek joint_state ({joint_state_topic}): {e}")
            return

        try:
            lg_lock = entry.get("__lg_lock")
            if lg_lock is not None:
                with lg_lock:
                    lg.log_state(
                        state_timestamp=ts,
                        state_joint_pos=joint_arr,
                        state_joint_torque=torque_arr,
                    )
            else:
                lg.log_state(
                    state_timestamp=ts,
                    state_joint_pos=joint_arr,
                    state_joint_torque=torque_arr,
                )
        except Exception as e:
            if self.debug:
                print(f"[BackpackStreamLogger] log_torque ({topic}): {e}")

    def _log_ft_wrench(self, entry: dict, lg, topic: str) -> None:
        """Log raw 6-DOF F/T wrench onto the shared arm logger, paired with latest joint_state.

        Mirrors _log_torque, but reads from the main data server (the bridge publishes
        snap.{left,right}_ee_wrench there as {"data": [...6 floats...]}) and writes the
        state_ft_wrench kwarg instead of state_joint_torque.
        """
        try:
            data_list, _ = self.data_client.peek_data(topic, n=-1)
            if not data_list:
                return
            raw = data_list[0]
            state = deserialize(raw) if isinstance(raw, (bytes, bytearray, memoryview)) else raw
            if not isinstance(state, dict) or not state.get("data_valid", True):
                return
            wrench = state.get("data")
            if wrench is None:
                return
            wrench_arr = np.asarray(wrench, dtype=np.float64)
            if wrench_arr.shape[0] != 6:
                if self.debug:
                    print(
                        f"[BackpackStreamLogger] ft size mismatch ({topic}): "
                        f"got {wrench_arr.shape[0]}, expected 6"
                    )
                return
            ts = float(state.get("timestamp", time.monotonic()))
        except Exception as e:
            if self.debug:
                print(f"[BackpackStreamLogger] peek_data ft ({topic}): {e}")
            return

        # Pair with latest joint_state positions from the sibling topic.
        joint_state_topic = entry.get("logger_name")
        joint_dof = int(entry.get("joint_dof", 7))
        try:
            jdata, _ = self.data_client.peek_data(joint_state_topic, n=-1)
            if not jdata:
                return
            jraw = jdata[0]
            jstate = deserialize(jraw) if isinstance(jraw, (bytes, bytearray, memoryview)) else jraw
            if not isinstance(jstate, dict):
                return
            joints = jstate.get("positions")
            if joints is None:
                return
            joint_arr = np.asarray(joints, dtype=np.float64)
            if joint_arr.shape[0] != joint_dof:
                return
        except Exception as e:
            if self.debug:
                print(f"[BackpackStreamLogger] peek joint_state for ft ({joint_state_topic}): {e}")
            return

        try:
            lg_lock = entry.get("__lg_lock")
            if lg_lock is not None:
                with lg_lock:
                    lg.log_state(
                        state_timestamp=ts,
                        state_joint_pos=joint_arr,
                        state_ft_wrench=wrench_arr,
                    )
            else:
                lg.log_state(
                    state_timestamp=ts,
                    state_joint_pos=joint_arr,
                    state_ft_wrench=wrench_arr,
                )
        except Exception as e:
            if self.debug:
                print(f"[BackpackStreamLogger] log_ft_wrench ({topic}): {e}")

    def _log_cartesian_state(self, entry: dict, lg, state: dict, ts: float) -> None:
        """Log cartesian (EEF) state; log neck command as target."""
        eef = state.get("eef_pose") or state.get("positions")
        if eef is not None and len(eef) >= 6:
            pos, quat = _eef_to_pos_quat(eef)
            lg.log_state(
                state_timestamp=ts,
                state_pos_xyz=pos,
                state_quat_wxyz=quat,
            )
        try:
            ndata, _ = self.data_client.peek_data(Topics.NECK_CMD, n=-1)
            if ndata:
                nraw = ndata[0]
                nmsg = deserialize_neck_message(nraw) if isinstance(
                    nraw, (bytes, bytearray, memoryview)
                ) else nraw
                if hasattr(nmsg, "neck_pose"):
                    p7 = list(nmsg.neck_pose)
                    if len(p7) == 7:
                        npos = np.array(p7[:3], dtype=np.float32)
                        wxyz = np.array(p7[3:7], dtype=np.float32)
                        lg.log_target(
                            target_timestamp=getattr(nmsg, "timestamp", time.monotonic()),
                            target_pos_xyz=npos,
                            target_quat_wxyz=wxyz,
                        )
        except Exception as e:
            if self.debug:
                print(f"[BackpackStreamLogger] NECK_CMD log_target: {e}")


    def stop(self) -> None:
        self.running = False

# ─────────────────────────────────────────────────────────────────────────────
# BackpackCameraLogger
# ─────────────────────────────────────────────────────────────────────────────

class BackpackCameraLogger:
    """
    Runs on the GELLO PC. Subscribes to wrist-camera topics on the robot PC's camera
    server and writes frames locally via VideoLogger.

    Camera topics are not in the 'robots:' list so they're driven by logging_streams
    entries in the robot's config.yaml.
    """

    _CAM_WIDTH = 1280
    _CAM_HEIGHT = 720
    _CAM_FPS = 30

    def __init__(
        self,
        camera_endpoint: str,
        logging_streams: List[dict],
        debug: bool = False,
    ):
        self.debug = debug
        self.running = False

        self.cam_client = robotmq.RMQClient(
            client_name="backpack_camera_logger",
            server_endpoint=camera_endpoint,
        )

        # cam_name → (VideoLogger, cam_name, w, h, encoding, camera_type)
        self.loggers: Dict[str, tuple] = {}
        # One lock per VideoLogger: guards update_recording_state + log_frame so a
        # rapid stop/start cannot interleave end-episode teardown with new-episode setup.
        self._cam_locks: Dict[str, threading.Lock] = {}

        # Build one VideoLogger per camera entry in logging_streams
        for entry in logging_streams:
            if entry.get("logger_type") != "camera":
                continue
            if entry.get("enabled", True) is False:
                continue
            cam_name = str(entry["camera_key"])
            port = entry.get("port")
            if not port:
                continue
            w = int(entry.get("width", self._CAM_WIDTH))
            h = int(entry.get("height", self._CAM_HEIGHT))
            fps = float(entry.get("fps", self._CAM_FPS))

            # rgb (jpeg, default) for the wrist cams; depth/raw for the iphone streams.
            camera_type = str(entry.get("camera_type", "rgb")).lower()
            if camera_type not in ("rgb", "depth"):
                print(f"[BackpackCameraLogger] {cam_name}: invalid camera_type {camera_type!r}, using rgb")
                camera_type = "rgb"
            encoding = str(entry.get("encoding", "jpeg")).lower()
            if encoding not in ("jpeg", "raw"):
                print(f"[BackpackCameraLogger] {cam_name}: invalid encoding {encoding!r}, using jpeg")
                encoding = "jpeg"
            # depth_range only applies to depth; omit to let the logger default (0, 4) apply.
            depth_kwargs = (
                {"depth_range": tuple(entry["depth_range"])}
                if camera_type == "depth" and entry.get("depth_range")
                else {}
            )

            try:
                lg = VideoLogger(
                    name=cam_name,
                    endpoint=f"tcp://localhost:{port}",
                    attr={"camera_configs": {cam_name: {"type": camera_type, "width": w, "height": h, "fps": fps}}},
                    codec="libx264",
                    defer_encoding=True,
                    **depth_kwargs,
                )
                self.loggers[cam_name] = (lg, cam_name, w, h, encoding, camera_type)
                self._cam_locks[cam_name] = threading.Lock()
                print(
                    f"[BackpackCameraLogger] Logger ready: {cam_name} port={port} "
                    f"({w}x{h}@{fps}fps type={camera_type} enc={encoding})"
                )
            except Exception as e:
                print(f"[BackpackCameraLogger] Failed to build logger for {cam_name}: {e}")

    def run(self) -> None:
        if not self.loggers:
            print("[BackpackCameraLogger] No cameras configured; thread exiting")
            return
        self.running = True
        threads = [
            threading.Thread(target=self._run_camera, args=(cam_name,), daemon=True)
            for cam_name in self.loggers
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def _run_camera(self, cam_name: str) -> None:
        lg = self.loggers[cam_name][0]
        fps = lg.attr.get("camera_configs", {}).get(cam_name, {}).get("fps", self._CAM_FPS)
        loop_period = 1.0 / fps
        print(f"[BackpackCameraLogger] {cam_name} loop started ({fps:.0f} Hz)")
        while self.running:
            t0 = time.monotonic()
            self._poll_camera(cam_name)
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, loop_period - elapsed))
        print(f"[BackpackCameraLogger] {cam_name} loop stopped")

    def _poll_camera(self, cam_name: str) -> None:
        logger_entry = self.loggers.get(cam_name)
        if logger_entry is None:
            return
        lg, _, cam_w, cam_h, encoding, camera_type = logger_entry
        lock = self._cam_locks[cam_name]

        with lock:
            try:
                recording = lg.update_recording_state()
            except Exception as e:
                print(f"[BackpackCameraLogger] disabling {cam_name}: {e}")
                self.loggers.pop(cam_name, None)
                return

            # need to sleep when not recording to avoid busy-looping on the camera server
            if not recording:
                return

            try:
                data_list, _ = self.cam_client.peek_data(cam_name, n=-1)
                if not data_list:
                    return
                raw = bytes(data_list[0])
            except Exception as e:
                if self.debug:
                    print(f"[BackpackCameraLogger] peek_data error ({cam_name}): {e}")
                return

            try:
                # Strip TSB1 timestamp header if present
                if raw[:4] == b"TSB1" and len(raw) >= 12:
                    ts_ns = struct.unpack("!Q", raw[4:12])[0]
                    timestamp = ts_ns / 1e9
                    payload = raw[12:]
                else:
                    timestamp = time.monotonic()
                    payload = raw

                if encoding == "raw":
                    # iphone streams: raw frame bytes, no JPEG. Reshape from the
                    # declared dims (np.frombuffer is read-only → .copy()).
                    if camera_type == "depth":
                        expected = cam_h * cam_w * np.dtype(np.float32).itemsize
                        if len(payload) != expected:
                            if self.debug:
                                print(f"[BackpackCameraLogger] {cam_name}: depth byte mismatch "
                                      f"len={len(payload)} expected={expected}; skipping")
                            return
                        frame = np.frombuffer(payload, dtype=np.float32).reshape(cam_h, cam_w).copy()
                    else:  # raw rgb — payload is already RGB, no decode / no cvtColor
                        expected = cam_h * cam_w * 3
                        if len(payload) != expected:
                            if self.debug:
                                print(f"[BackpackCameraLogger] {cam_name}: rgb byte mismatch "
                                      f"len={len(payload)} expected={expected}; skipping")
                            return
                        frame = np.frombuffer(payload, dtype=np.uint8).reshape(cam_h, cam_w, 3).copy()
                else:  # jpeg (default — wrist cams): decode and convert BGR→RGB
                    arr = np.frombuffer(payload, dtype=np.uint8)
                    frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame_bgr is None:
                        return
                    if frame_bgr.shape[:2] != (cam_h, cam_w):
                        frame_bgr = cv2.resize(frame_bgr, (cam_w, cam_h))
                    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                lg.log_frame(camera_name=cam_name, timestamp=timestamp, frame=frame)

            except Exception as e:
                if self.debug:
                    print(f"[BackpackCameraLogger] log_frame error ({cam_name}): {e}")

    def stop(self):
        self.running = False