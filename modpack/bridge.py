"""
Robot-side bridge to ModPack: pull commands, push state, mirror activation/episode FSM.

Controllers import this module and call methods from their own control loops. ModPack
does not run a generic robot process for you — it exposes this thin RMQ + activation
layer on the robot / controller machine.

Minimal config.yaml::

    name: my_robot
    topology: managed
    roles:
      right_arm: {type: joint_arm, dof: 7}
      left_arm:  {type: joint_arm, dof: 7}
      body:      {type: mobile_base}
    rmq:
      host: 192.168.0.225
      port: 5555
      activation_port: 5556

Topic conventions (derived automatically from role name):
  Command  (GELLO → get_command):    {role}_cmd
  State    (publish_state → logger): {role}
  Activation:                        robot_activation
"""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Set, Union

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as _R

import modpack.schemas as S
from modpack.orchestration.message_formats import Topics

try:
    import robotmq
    from robotmq.utils import deserialize, serialize
except ImportError:  # pragma: no cover
    robotmq = None
    deserialize = None
    serialize = None

# (command_type, value) e.g. ("activate", None)
ActivationCallback = Callable[[str, dict], None]

PEEK_TIMEOUT_S = 0.1


def _as_float_array(x: Any, shape_hint: int) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.size != shape_hint and a.ndim == 0:
        a = np.zeros(shape_hint, dtype=np.float64)
    return a.reshape(-1)[:shape_hint] if a.size >= shape_hint else a


def _dict_to_command(role_type: str, raw: dict, dof: Optional[int] = None) -> Any:
    """Map a deserialized RMQ command dict to schema command objects.

    dof, when given, is the role's declared joint count. GELLO publishes
    joint_positions as [arm joints..., gripper], so trimming to dof drops the
    trailing gripper slot and yields exactly the arm joints the consumer
    expects (the gripper is carried separately below).
    """
    if role_type == S.JOINT_ARM:
        if "joint_pos" in raw:
            j = np.asarray(raw["joint_pos"], dtype=np.float64)
        else:
            jp = raw.get("joint_positions")
            j = np.asarray(jp, dtype=np.float64) if jp is not None else np.zeros(1)
        if dof is not None:
            j = j[:dof]
        g = float(raw.get("gripper", raw.get("gripper_position", 0.0)))
        return S.JointArmCommand(joint_pos=j, gripper=g)
    if role_type == S.CARTESIAN_ARM:
        raw_pose = raw.get("eef_pose") or raw.get("neck_pose")
        p = np.asarray(raw_pose, dtype=np.float64)
        if p.size < 6:
            p = np.zeros(6, dtype=np.float64)
        elif p.size == 7:
            # neck_pose arrives as [x, y, z, qx, qy, qz, qw] — convert to [x, y, z, r, p, y]
            rpy = _R.from_quat(p[3:7]).as_euler("xyz")
            p = np.concatenate([p[:3], rpy])
        return S.CartesianArmCommand(eef_pose=p.reshape(-1)[:6])
    if role_type == S.MOBILE_BASE:
        if "base_pose" in raw:
            p = _as_float_array(raw["base_pose"], 3)
        elif "target_pose" in raw:
            p = _as_float_array(raw["target_pose"], 3)
        else:
            p = np.zeros(3, dtype=np.float64)
        return S.MobileBaseCommand(base_pose=p)
    raise ValueError(f"Unknown role type {role_type!r} for command mapping")


def _state_to_bytes(state: Any) -> bytes:
    if is_dataclass(state):
        payload = asdict(state)
    elif isinstance(state, dict):
        payload = state
    else:
        raise TypeError("state must be a dict or a schemas dataclass")
    if serialize is None:
        raise RuntimeError("robotmq is not installed; cannot serialize state")
    return serialize(payload)




def _default_activation_targets(roles: List[dict]) -> Set[str]:
    """{"all"} plus any per-role activation_target (must match ActivationMessage.target)."""
    s: Set[str] = {"all"}
    for r in roles:
        t = r.get("activation_target")
        if t is not None:
            s.add(str(t))
    return s


class RobotBridge:
    """
    RMQ + activation sidecar. Connect once, then poll `get_command` / `publish_state`
    from your controller loop. Activation is updated on a background thread.
    """

    def __init__(
        self,
        roles: List[dict],
        host: str,
        data_port: int,
        activation_port: int,
        feedback_port: int = 5581,
        camera_port: int = 5570,
        camera_host: Optional[str] = None,
        activation_targets: Optional[Set[str]] = None,
        activation_callback: Optional[ActivationCallback] = None,
    ):
        if robotmq is None:
            raise RuntimeError("robotmq is required for RobotBridge; install it in the robot env")
        self._roles_cfg = roles
        self._by_id = {r["id"]: r for r in roles}
        self._host = host
        self._data_port = data_port
        self._activation_port = activation_port
        self._feedback_port = feedback_port
        self._camera_port = camera_port
        # The camera broker is co-located with the robot bridge (robot PC in
        # managed topology, gello PC in unmanaged), so it lives on a different
        # host than the data/activation/feedback brokers in managed topology.
        # Default to localhost; only the camera client uses this.
        self._camera_host = camera_host or "localhost"
        self._activation_targets: Set[str] = activation_targets or _default_activation_targets(roles)
        self._activation_callback = activation_callback

        self._data_client: Any = None
        self._activation_client: Any = None
        self._feedback_client: Any = None
        self._camera_client: Any = None
        self._connected = False
        self._last_activation_sig: Optional[tuple] = None

        self._lock = threading.Lock()
        self._activated = False
        self._teleop_enabled = False
        self._episode_recording = False
        self._run_activation = True
        self._activation_thread: Optional[threading.Thread] = None

    @classmethod
    def from_config(
        cls,
        path: Union[str, Path],
        activation_callback: Optional[ActivationCallback] = None,
    ) -> RobotBridge:
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        return cls.from_dict(doc, activation_callback=activation_callback)

    @classmethod
    def from_dict(
        cls,
        doc: dict,
        activation_callback: Optional[ActivationCallback] = None,
        activation_target: Optional[str] = None,
    ) -> "RobotBridge":
        rmq = doc.get("rmq") or doc.get("robotmq")
        if not isinstance(rmq, dict):
            raise ValueError("config must include an 'rmq:' block with host, port, activation_port")
        host = str(rmq["host"])
        data_port = int(rmq["port"])
        activation_port = int(rmq["activation_port"])
        feedback_port = int(rmq.get("feedback_port", 5581))
        camera_port = int(rmq.get("cam_port", 5570))
        camera_host = rmq.get("cam_host")  # None -> localhost default (see __init__)

        if "roles" not in doc or doc["roles"] is None:
            raise ValueError("config must include a non-empty 'roles:' mapping (see modpack config docs)")
        raw_roles = doc["roles"]
        if not raw_roles:
            raise ValueError("no roles in config (empty 'roles')")

        # Accept both new dict form {role_name: {type, ...}} and old list form [{id, type, ...}].
        if isinstance(raw_roles, dict):
            roles = []
            for name, cfg in raw_roles.items():
                r = dict(cfg)
                r["id"] = name
                r.setdefault("command_topic", f"{name}_cmd")
                r.setdefault("state_topic", name)
                roles.append(r)
        else:
            roles = list(raw_roles)

        for r in roles:
            if "id" not in r or "type" not in r:
                raise ValueError("each role needs 'id' and 'type'")
            if r["type"] not in S.ROLE_TYPES:
                raise ValueError(f"Unknown role type {r['type']!r}; expected one of {S.ROLE_TYPES}")
        # A runner that owns a single role passes activation_target to scope this
        # bridge to ONLY that target (e.g. 'arx' for an arm). Without it the
        # bridge would accept activations for ANY role in the config (the union
        # below), so a neck activate would engage the arms. 'all'-broadcast
        # safety commands still apply via _activation_applies regardless.
        if activation_target is not None:
            at = {str(activation_target)}
        else:
            act_raw = doc.get("activation", {}) or {}
            if isinstance(act_raw, dict) and "targets" in act_raw:
                at = {str(x) for x in act_raw["targets"]}
            elif "activation_targets" in doc:
                at = {str(x) for x in doc["activation_targets"]}
            else:
                at = _default_activation_targets(roles)

        return cls(
            roles=roles,
            host=host,
            data_port=data_port,
            activation_port=activation_port,
            feedback_port=feedback_port,
            camera_port=camera_port,
            camera_host=camera_host,
            activation_targets=at,
            activation_callback=activation_callback,
        )

    def connect(self) -> None:
        if self._connected:
            return
        self._data_client = robotmq.RMQClient(
            client_name="modpack_bridge_data",
            server_endpoint=f"tcp://{self._host}:{self._data_port}",
        )
        self._activation_client = robotmq.RMQClient(
            client_name="modpack_bridge_activation",
            server_endpoint=f"tcp://{self._host}:{self._activation_port}",
        )
        self._feedback_client = robotmq.RMQClient(
            client_name="modpack_bridge_feedback",
            server_endpoint=f"tcp://{self._host}:{self._feedback_port}",
        )
        self._camera_client = robotmq.RMQClient(
            client_name="modpack_bridge_camera",
            server_endpoint=f"tcp://{self._camera_host}:{self._camera_port}",
        )
        self._connected = True
        self._run_activation = True
        self._activation_thread = threading.Thread(target=self._activation_loop, daemon=True)
        self._activation_thread.start()

    def disconnect(self) -> None:
        self._run_activation = False
        if self._activation_thread is not None:
            self._activation_thread.join(timeout=2.0)
            self._activation_thread = None
        self._data_client = None
        self._activation_client = None
        self._feedback_client = None
        self._camera_client = None
        self._connected = False

    @property
    def is_active(self) -> bool:
        """True when the coordinator allows teleop (activated and not paused by episode FSM)."""
        with self._lock:
            return self._activated and self._teleop_enabled

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._episode_recording

    def get_command(self, role: str) -> Any:
        """Latest command for role id or None (sensor / no data / unknown role)."""
        r = self._by_id.get(role)
        if not r or r["type"] == S.SENSOR_STREAM:
            return None
        if not self._connected or not self._data_client:
            return None
        topic = r.get("command_topic")
        if not topic:
            return None
        try:
            data_list, _ = self._data_client.peek_data(topic, n=-1)
            if not data_list:
                return None
            raw0 = data_list[0]
            payload = deserialize(raw0) if isinstance(raw0, (bytes, bytearray)) else raw0
            if not isinstance(payload, dict):
                return None
            return _dict_to_command(r["type"], payload, r.get("dof"))
        except Exception as e:
            print(f"[Bridge] get_command deserialize error on {topic}: {e}")
            return None

    def publish_state(self, role: str, state: Any) -> None:
        """Publish state dict or schema dataclass to the role's state topic.

        For video_stream roles the state must be a dict with frame: bytes
        and topic: str; the raw bytes are forwarded without serialization.
        """
        r = self._by_id.get(role)
        if not r or not self._data_client:
            return
        if r.get("type") == S.VIDEO_STREAM:
            payload = state if isinstance(state, dict) else (asdict(state) if is_dataclass(state) else {})
            frame = payload.get("frame")
            topic = payload.get("topic") or r.get("topic")
            if not isinstance(frame, (bytes, bytearray)) or not topic:
                return
            self._data_client.put_data(topic, bytes(frame))
            return
        st = r.get("state_topic")
        if not st:
            return
        self._data_client.put_data(st, _state_to_bytes(state))

    def peek_topic(self, topic: str, *, timeout_s: float = PEEK_TIMEOUT_S) -> Optional[list]:
        """Peek latest data from any topic on the data server.

        For non-command inbound flows that don't map to a declared role
        (e.g. neck_target_pose, body_target). Returns the raw data
        list or None.
        """
        if not self._connected or not self._data_client:
            return None
        data_list, _ = self._data_client.peek_data(topic, n=-1, timeout_s=timeout_s)
        return data_list if data_list else None

    def peek_activation_topic(self, topic: str, *, timeout_s: float = PEEK_TIMEOUT_S) -> Optional[list]:
        """Peek latest data from any topic on the activation server."""
        if not self._connected or not self._activation_client:
            return None
        data_list, _ = self._activation_client.peek_data(topic, n=-1, timeout_s=timeout_s)
        return data_list if data_list else None

    def publish_activation_raw(self, topic: str, payload: bytes) -> None:
        """Publish raw bytes to a specific topic on the activation server."""
        if not self._activation_client:
            return
        self._activation_client.put_data(topic, payload)

    def publish_feedback_raw(self, topic: str, payload: bytes) -> None:
        """Publish raw bytes to the torque feedback server (robot→gello direction)."""
        if not self._feedback_client:
            return
        self._feedback_client.put_data(topic, payload)

    def publish_camera_raw(self, topic: str, payload: bytes) -> None:
        """Publish raw bytes to the camera server."""
        if not self._camera_client:
            return
        self._camera_client.put_data(topic, payload)

    @property
    def camera_endpoint(self) -> str:
        return f"tcp://{self._camera_host}:{self._camera_port}"

    def publish_activation_command(self, command: str, target: str = "all") -> None:
        """Post a typed activation command to the activation bus."""
        if not self._activation_client or serialize is None:
            return
        msg = {
            "command": command,
            "target": target,
            "timestamp": time.time(),
            "source": "robot_bridge",
        }
        self._activation_client.put_data(Topics.ACTIVATION, serialize(msg))

    def publish_activation(self, message: dict) -> None:
        """Send a control message on the activation bus (e.g. preview_ready for ARX)."""
        if not self._activation_client or serialize is None:
            return
        self._activation_client.put_data(Topics.ACTIVATION, serialize(message))

    # Commands that a target="all" broadcast is allowed to deliver to every
    # bridge regardless of role. These are lifecycle/safety signals that must
    # reach all processes. Activation/episode commands are NOT in this set:
    # target="all" must NOT activate a bridge whose role wasn't addressed, so
    # e.g. start_episode (published target=all) or a neck activate cannot engage
    # the arms — only an activation addressed to this bridge's own target does.
    _ALL_BROADCAST_COMMANDS = frozenset(
        {"emergency_stop", "shutdown", "manager_shutdown"}
    )

    def _activation_applies(self, target: str, command: str) -> bool:
        t = (target or "all").strip()
        if t == "all":
            return command in self._ALL_BROADCAST_COMMANDS
        if t in self._activation_targets:
            return True
        return False

    def _activation_loop(self) -> None:
        if deserialize is None or self._activation_client is None:
            print("[Bridge] activation loop: no deserialize or client — exiting immediately")
            return
        print("[Bridge] activation loop started")
        while self._run_activation and self._connected:
            try:
                data, _ = self._activation_client.peek_data(Topics.ACTIVATION, n=-1)
                if data:
                    msg = deserialize(data[0])
                    if not isinstance(msg, dict):
                        time.sleep(0.01)
                        continue
                    sig = (msg.get("timestamp"), msg.get("command"), msg.get("target"))
                    target = str(msg.get("target", "all"))
                    command = str(msg.get("command", ""))
                    if not self._activation_applies(target, command):
                        time.sleep(0.01)
                        continue
                    if sig != self._last_activation_sig:
                        self._last_activation_sig = sig
                        self._process_activation_message(msg)
                time.sleep(0.01)
            except Exception as e:
                print(f"[Bridge] activation loop exception: {e}")
                time.sleep(0.1)
        print(f"[Bridge] activation loop exited: _run_activation={self._run_activation} _connected={self._connected}")

    def _process_activation_message(self, msg: dict) -> None:
        command = msg.get("command")
        if self._activation_callback is not None:
            try:
                self._activation_callback(str(command or ""), msg)
            except Exception as e:
                print(f"[Bridge] activation callback error: {e}")

        if command == "activate":
            with self._lock:
                self._activated = True
                self._teleop_enabled = True
        elif command == "deactivate":
            with self._lock:
                self._activated = False
                self._teleop_enabled = False
                self._episode_recording = False
        elif command == "emergency_stop":
            with self._lock:
                self._activated = False
                self._teleop_enabled = False
                self._episode_recording = False
        elif command == "start_episode":
            with self._lock:
                self._episode_recording = True
                self._teleop_enabled = True
                if not self._activated:
                    self._activated = True
        elif command == "end_episode":
            with self._lock:
                self._episode_recording = False
        elif command == "pause_episode":
            with self._lock:
                self._episode_recording = False
                self._teleop_enabled = False
        elif command == "resume_episode":
            with self._lock:
                self._episode_recording = True
                self._teleop_enabled = True
        elif command == "reset":
            with self._lock:
                self._activated = False
                self._teleop_enabled = False
                self._episode_recording = False
        elif command in ("shutdown", "manager_shutdown"):
            self._run_activation = False
        # preview_ready and unknown commands: ignore
