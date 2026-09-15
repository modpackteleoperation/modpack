"""
Robot config schema and loader.

Each robot lives at modpack/robots/<robot>/config.yaml.  The config is
the single source of truth for everything the orchestration layer needs to know
about that robot:

  - topology (managed vs unmanaged)
  - roles (name → type + dof)
  - modules to activate on the Modpack PC
  - per-module config overrides

Usage
-----
    from modpack.orchestration.robot_config import load_config

    cfg = load_config("rby1")                  # reads robots/rby1/config.yaml
    cfg = load_config("/abs/path/config.yaml") # explicit path

The robot config is loaded once at startup by SimpleMessageQueueManager and used
to derive active_systems and logging_streams so no robot-specific code
ever needs to live in orchestration or core modules.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml

from modpack.modules import MODPACK_CONFIG_PATH

# Canonical location of built-in robot packages relative to this file.
_ROBOTS_DIR = Path(__file__).parent.parent / "robots"


# =============================================================================
# Robot config schema
# =============================================================================

@dataclass
class LoggingStreamEntry:
    """One entry in a robot config's logging_streams list."""
    logger_name: str
    port: int
    logger_type: str = "joint_state"   # joint_state | joint_command | base_state | cartesian_state
    control_freq: float = 100.0
    joint_dof: int = 7
    joint_units: str = "radians"
    log_eef: bool = False
    enabled: bool = True
    topic: Optional[str] = None
    frame_id_default: Optional[str] = None
    # Gripper companion logger
    gripper_port: Optional[int] = None
    gripper_logger_name: Optional[str] = None
    log_gripper_scalar: bool = False
    # Two-PC: RMQ topic for GELLO leader command (e.g. right / left) → log_target on the arm
    gello_target_topic: Optional[str] = None
    # Companion target topic for base_state entries (publishes commanded base pose
    # alongside the state topic; mirrors gripper_port for arm entries).
    target_topic: Optional[str] = None
    # Grouped sub-topic list (e.g. head entry with joint_state + joint_command sub-topics)
    topics: Optional[List[dict]] = None

    @classmethod
    def from_dict(cls, d: dict) -> "LoggingStreamEntry":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    def to_dict(self) -> dict:
        """Return a plain dict compatible with BackpackStreamLogger entry format."""
        d: dict = {
            "logger_name": self.logger_name,
            "port": self.port,
            "logger_type": self.logger_type,
            "control_freq": self.control_freq,
            "joint_dof": self.joint_dof,
            "joint_units": self.joint_units,
            "log_eef": self.log_eef,
            "enabled": self.enabled,
        }
        if self.topic is not None:
            d["topic"] = self.topic
        if self.frame_id_default is not None:
            d["frame_id_default"] = self.frame_id_default
        if self.gripper_port is not None:
            d["gripper_port"] = self.gripper_port
        if self.gripper_logger_name is not None:
            d["gripper_logger_name"] = self.gripper_logger_name
        if self.log_gripper_scalar:
            d["log_gripper_scalar"] = True
        if self.gello_target_topic is not None:
            d["gello_target_topic"] = self.gello_target_topic
        if self.target_topic is not None:
            d["target_topic"] = self.target_topic
        if self.topics is not None:
            d["topics"] = self.topics
        return d


@dataclass
class CameraStreamEntry:
    """One camera entry in a robot config's logging_streams list."""
    camera_key: str
    port: int
    width: int = 960
    height: int = 720
    fps: float = 30.0
    enabled: bool = True
    logger_type: str = "camera"

    @classmethod
    def from_dict(cls, d: dict) -> "CameraStreamEntry":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    def to_dict(self) -> dict:
        return {
            "logger_type": "camera",
            "camera_key": self.camera_key,
            "port": self.port,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "enabled": self.enabled,
        }


@dataclass
class RobotConfig:
    """Declarative description of one robot package.

    Attributes
    ----------
    name
        Unique robot identifier, e.g. "rby1".  Must match the folder name
        under modpack/robots/<name>/.
    topology
        "managed"   — modpack owns a robot PC: the coordinator SSH-launches the
                      robot-side runners (robot_pc.scripts) and RMQ runs over
                      the network. has_robot_pc is True.
        "unmanaged" — modpack publishes commands/state into RMQ but does not
                      launch or manage any robot PC; whatever consumes them
                      (e.g. a separate inference machine) runs independently.
                      has_robot_pc is False.
    subsystems
        List of subsystem names (keys in REGISTRY) to activate for this robot.
        Merged with whatever active_systems the operator explicitly sets.
    logging_streams
        Stream definitions for episode logging.  Each entry declares source
        so BackpackStreamLogger knows whether to subscribe or skip.
    has_robot_pc
        Derived from topology; True for "managed".
    module_overrides
        Per-subsystem config overrides keyed by subsystem name.  Each value is
        an arbitrary dict forwarded to the subsystem's setup_fn / start_fn
        via manager.module_overrides[name].
    gello
        Raw gello: block from config.yaml with motor specs, port calibration,
        URDF paths, control gains for this robot's GELLO leader arms.
    robot_pc
        Robot PC launch config: scripts list of commands to SSH-launch,
        plus optional conda_env override.
    """
    name: str
    topology: Literal["managed", "unmanaged"]
    roles: Dict[str, Any] = field(default_factory=dict)
    subsystems: List[str] = field(default_factory=list)
    logging_streams: List[Union[LoggingStreamEntry, CameraStreamEntry]] = field(default_factory=list)
    module_overrides: Dict[str, Any] = field(default_factory=dict)
    debug: Dict[str, bool] = field(default_factory=dict)
    gello: Dict[str, Any] = field(default_factory=dict)
    robot_pc: Dict[str, Any] = field(default_factory=dict)
    activation_buttons: Dict[int, str] = field(default_factory=dict)

    @property
    def has_robot_pc(self) -> bool:
        return self.topology == "managed"

    @property
    def data_topics(self) -> List[tuple]:
        """State and command topics derived from declared roles.

        Returns (topic, ttl_s) pairs for registration on the data server.
        Adding a new role to config.yaml automatically registers its topics
        without touching coordinator.py or message_formats.py.
        """
        out = []
        for role_id, role_cfg in (self.roles or {}).items():
            cfg = role_cfg if isinstance(role_cfg, dict) else {}
            ttl = float(cfg.get("topic_ttl_s", 2.0))
            out.append((role_id, ttl))
            out.append((f"{role_id}_cmd", 10.0))
            if cfg.get("type") == "mobile_base":
                out.append((f"{role_id}_target", ttl))
            # Register each role's declared command topic when it differs from the
            # generic '<role>_cmd' (e.g. the neck's 'neck_target_pose', or a joint
            # arm's gello topic 'right'/'left'). This is keyed off the role config,
            # not which arms are active, so the robot-side consumer can always poll
            # its command topic even when that arm's publisher isn't running —
            # peek on a registered-but-empty topic is a no-op, peek on an
            # unregistered topic spams "Topic not found" on the server.
            command_topic = cfg.get("command_topic")
            if command_topic and command_topic != f"{role_id}_cmd":
                out.append((command_topic, 10.0))
        return out

    def validate(self) -> None:
        """Raise ValueError for any structural inconsistency."""
        valid_topologies = {"managed", "unmanaged"}
        if self.topology not in valid_topologies:
            raise ValueError(
                f"robot config '{self.name}': topology must be one of "
                f"{sorted(valid_topologies)}, got {self.topology!r}"
            )
        for s in self.logging_streams:
            if isinstance(s, CameraStreamEntry):
                continue
            if not s.topic and not s.topics:
                raise ValueError(
                    f"robot config '{self.name}': logging stream {s.logger_name!r} has no 'topic' or 'topics' field"
                )

    def logging_streams_as_dicts(self) -> List[dict]:
        """Return logging_streams as plain dicts for BackpackStreamLogger."""
        return [s.to_dict() for s in self.logging_streams]

    def active_systems_dict(self) -> Dict[str, bool]:
        """Return a dict suitable for merging into coordinator's active_systems."""
        return {name: True for name in self.subsystems}


# =============================================================================
# Loader
# =============================================================================

def load_config(robot_name_or_path: str) -> RobotConfig:
    """Load and validate a robot config from disk.

    Args:
        robot_name_or_path:
            Either a robot name (e.g. "rby1") resolved to
            modpack/robots/<name>/config.yaml or an explicit
            absolute or relative path to a config.yaml file.

    Returns:
        A validated :class:`RobotConfig` instance.

    Raises:
        FileNotFoundError: If the robot config file does not exist.
        ValueError: If the robot config fails schema validation.
    """
    path = _resolve_path(robot_name_or_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Robot config not found: {path}\n"
            f"Expected layout: modpack/robots/<robot>/config.yaml"
        )

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    return _parse(data, path)


def _resolve_path(robot_name_or_path: str) -> Path:
    p = Path(robot_name_or_path)
    if p.is_absolute() or os.sep in robot_name_or_path or robot_name_or_path.endswith(".yaml"):
        return p
    return _ROBOTS_DIR / robot_name_or_path / "config.yaml"


def _parse(data: dict, source: Path) -> RobotConfig:
    name = data.get("name") or source.parent.name
    topology = data.get("topology", "unmanaged")

    raw_streams = data.get("logging_streams", [])
    streams: List[Union[LoggingStreamEntry, CameraStreamEntry]] = []
    for entry in raw_streams:
        try:
            if entry.get("logger_type") == "camera":
                streams.append(CameraStreamEntry.from_dict(entry))
            else:
                streams.append(LoggingStreamEntry.from_dict(entry))
        except TypeError as exc:
            raise ValueError(
                f"Bad logging_stream entry in {source}: {entry!r}\n{exc}"
            ) from exc

    raw_buttons = data.get("activation_buttons") or {}
    activation_buttons = {int(k): str(v) for k, v in raw_buttons.items()}

    cfg = RobotConfig(
        name=name,
        topology=topology,
        roles=dict(data.get("roles") or {}),
        subsystems=list(data.get("modules") or data.get("subsystems") or []),
        logging_streams=streams,
        module_overrides=dict(data.get("module_overrides") or {}),
        debug={k: bool(v) for k, v in (data.get("debug") or {}).items()},
        gello=dict(data.get("gello") or {}),
        robot_pc=dict(data.get("robot_pc") or {}),
        activation_buttons=activation_buttons,
    )
    cfg.validate()
    return cfg


# =============================================================================
# Network config schema and loader
# =============================================================================
#
# RMQ network endpoints (host, the GELLO-PC/robot-PC IPs, broker ports) are
# owned by modpack and live solely in modpack/modules/modpack_config.yaml under
# the `robotmq:` block.  Both the coordinator and the standalone teleop
# launchers load that one block through load_network_config() instead of each
# keeping its own copy.  No IP/port literals live here: a missing key raises.

@dataclass
class NetworkConfig:
    """RMQ network endpoints shared across the modpack framework.

    bind_host is what RMQ servers bind to (``0.0.0.0``); it is intentionally
    kept distinct from gello_pc_ip, the IP that clients connect to.
    """
    bind_host: str
    gello_pc_ip: str
    robot_pc_ip: str
    data_port: int
    activation_port: int
    cam_port: int
    feedback_port: int
    logging: Dict[str, Any] = field(default_factory=dict)

    def as_rmq_dict(self) -> dict:
        """Return the ``rmq:`` block shape ``RobotBridge.from_dict`` expects.

        ``host`` here is the connect target (gello_pc_ip), matching what the
        old teleop_gello.yaml ``rmq:`` block carried.
        """
        return {
            "host": self.gello_pc_ip,
            "port": self.data_port,
            "activation_port": self.activation_port,
            "feedback_port": self.feedback_port,
            "cam_port": self.cam_port,
        }


def load_network_config(path: Union[str, Path] = MODPACK_CONFIG_PATH) -> NetworkConfig:
    """Load RMQ network config from modpack_config.yaml's ``robotmq:`` block.

    Every value comes from the file (the single source of truth); a missing key
    raises rather than silently falling back to a baked-in default.
    """
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    block = data["robotmq"]
    return NetworkConfig(
        bind_host=block["host"],
        gello_pc_ip=block["gello_pc_ip"],
        robot_pc_ip=block["robot_pc_ip"],
        data_port=block["port"],
        activation_port=block["activation_port"],
        cam_port=block["cam_port"],
        feedback_port=block["feedback_port"],
        logging=dict(block.get("logging") or {}),
    )
