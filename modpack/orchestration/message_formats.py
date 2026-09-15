#!/usr/bin/env python3
"""
Message formats for communication via robotmq
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field, fields
import json
import struct
from robotmq import serialize, deserialize

@dataclass
class GelloMessage:
    """
    Core message from GELLO to ARX5
    Designed for minimal latency and maximum reliability
    """
    # Message metadata (required fields first)
    timestamp: float                    # Unix timestamp when message created
    sequence_id: int                   # Monotonic sequence number
    arm: str                           # Which arm message corresponds to
    joint_positions: List[float]       # 6 arm joint positions in radians
    gripper_position: float            # Gripper position [0.0, 1.0]
    
    # Optional fields with defaults
    source: str = "gello_process"      # Source identifier
    data_valid: bool = True           # Overall data validity
    corruption_flags: List[str] = None # Any corruption detected
    joint_velocities: Optional[List[float]] = None  # Optional velocity data
    
    def __post_init__(self):
        """Validate message on creation"""
        if self.corruption_flags is None:
            self.corruption_flags = []

        # Length is robot-specific; each adapter validates for its own DOF
        if len(self.joint_positions) < 1:
            self.data_valid = False
            self.corruption_flags.append(f"INVALID_JOINT_COUNT: {len(self.joint_positions)}")
        
        # Validate finite values
        if not all(np.isfinite(self.joint_positions)):
            self.data_valid = False
            self.corruption_flags.append("NON_FINITE_JOINTS")
            
        if not np.isfinite(self.gripper_position):
            self.data_valid = False
            self.corruption_flags.append("NON_FINITE_GRIPPER")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GelloMessage':
        """Create from dictionary"""
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'GelloMessage':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))

@dataclass
class BaseProprioMessage:
    """
    Base state broadcast from the mobile base process.
    """
    timestamp: float
    sequence_id: int
    pose: List[float]                     # [x, y, theta]
    source: str = "base_process"
    data_valid: bool = True
    corruption_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseProprioMessage":
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "BaseProprioMessage":
        return cls.from_dict(json.loads(json_str))


@dataclass
class SystemReadyMessage:
    timestamp: float
    node: str                 # "gello_pc" or "robot_pc"
    sequence_id: int
    ready: bool
    detail: Optional[str] = None
    source: str = "system_ready_publisher"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    @classmethod
    def from_dict(cls, data):
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})
    def to_json(self): return json.dumps(self.to_dict())
    @classmethod
    def from_json(cls, json_str): return cls.from_dict(json.loads(json_str))


@dataclass
class EpisodeStateMessage:
    """
    Broadcast when an episode changes state so both PCs stay synchronized.
    """
    timestamp: float
    sequence_id: int
    source_node: str              # "gello_pc" | "robot_pc"
    event: str                    # "start" | "pause" | "resume" | "stop"
    episode_index: int
    logging_state: Optional[str] = None
    detail: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeStateMessage":
        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "EpisodeStateMessage":
        return cls.from_dict(json.loads(json_str))

@dataclass
class NeckMessage:
    """
    Core message from Vision Pro to neck process
    """
    # Message metadata (required fields first)
    timestamp: float                    # Unix timestamp when message created
    sequence_id: int                   # Monotonic sequence number
    neck_pose: List[float]       # Length 7 vector with x,y,z, quat

    # Optional fields with defaults
    source: str = "vision_pro_process"      # Source identifier
    data_valid: bool = True           # Overall data validity
    corruption_flags: List[str] = None # Any corruption detected
    
    def __post_init__(self):
        """Validate message on creation"""
        if self.corruption_flags is None:
            self.corruption_flags = []
            
        # Validate pose length (3 position + 4 quaternion)
        if len(self.neck_pose) != 7:
            self.data_valid = False
            self.corruption_flags.append(f"INVALID_POSE_LEN: {len(self.neck_pose)}")
            return

        # Validate finite values in pose
        if not all(np.isfinite(self.neck_pose)):
            self.data_valid = False
            self.corruption_flags.append("NON_FINITE_POSE")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NeckMessage':
        """Create from dictionary"""
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'NeckMessage':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))

@dataclass
class JointStateMessage:
    timestamp: float              # Unix time from controller or sensor
    sequence_id: int              # Monotonic counter per arm
    arm: str                      # 'right' | 'left'
    joint_positions: List[float]  # len == DOF, radians
    joint_velocities: Optional[List[float]] = None  # len == DOF, rad/s
    gripper_position: float = 0.0 # normalized 0..1 or radians
    gripper_velocity: Optional[float] = None
    source: str = "arx5_process"
    data_valid: bool = True
    corruption_flags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JointStateMessage':
        """Create from dictionary"""
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'JointStateMessage':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))


@dataclass 
class ActivationMessage:
    """
    Activation/control message for ARX5 process
    """
    timestamp: float
    command: str  # 'activate', 'deactivate', 'emergency_stop', 'reset'
    target: str = "all"
    source: str = "activation_manager"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActivationMessage':
        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)
    
@dataclass 
class NeckActivationMessage:
    """
    Activation/control message for neck process
    """
    timestamp: float
    command: str  # 'activate', 'deactivate', 'emergency_stop', 'reset'
    target: str = "neck"
    source: str = "activation_manager"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NeckActivationMessage':
        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)
    

@dataclass
class VisionProStatusMessage:
    """
    Status feedback from vision pro to guide neck activation
    """
    timestamp: float
    sequence_id: int
    source: str = "vision_pro_process"
    
    # Vision Pro state
    vision_pro_ready: bool = False
    vision_pro_activated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VisionProStatusMessage":
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    @classmethod
    def from_json(cls, json_str: str) -> "VisionProStatusMessage":
        return cls.from_dict(json.loads(json_str))
    
@dataclass
class NeckiPhoneStatusMessage:
    """
    Status feedback from neck iphone to guide neck activation
    """
    timestamp: float
    sequence_id: int
    # iPhone state
    iphone_publishing: bool
    
    source: str = "read_iphone_process"
    

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NeckiPhoneStatusMessage":
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    @classmethod
    def from_json(cls, json_str: str) -> "NeckiPhoneStatusMessage":
        return cls.from_dict(json.loads(json_str))

@dataclass
class NeckStatusMessage:
    """
    Status feedback from neck
    """
    timestamp: float
    sequence_id: int
    source: str = "neck_process"
    
    # Neck state
    at_init: bool = False
    at_start: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NeckStatusMessage":
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    @classmethod
    def from_json(cls, json_str: str) -> "NeckStatusMessage":
        return cls.from_dict(json.loads(json_str))


@dataclass
class TorqueMessage:
    timestamp: float
    sequence_id: int
    arm: str                          # 'right' | 'left'
    joint_torques: List[float]        # per-joint torques in Nm
    gripper_torque: Optional[float] = None
    source: str = "rby1_teleop"
    data_valid: bool = True
    corruption_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TorqueMessage':
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'TorqueMessage':
        return cls.from_dict(json.loads(json_str))


class MessageValidator:
    """
    Utility class for message validation and corruption detection
    """
    
    @staticmethod
    def validate_gello_message(msg: GelloMessage) -> bool:
        """Comprehensive validation of GELLO message"""
        # Check basic structure
        if not isinstance(msg.joint_positions, list) or len(msg.joint_positions) < 1:
            return False
            
        # Check value ranges (generous limits)
        for pos in msg.joint_positions:
            if not np.isfinite(pos) or abs(pos) > 10.0:  # 10 rad = ~573 degrees
                return False
        
        # Check gripper range
        if not np.isfinite(msg.gripper_position) or msg.gripper_position < -1.0 or msg.gripper_position > 2.0:
            return False
            
        # Check timestamp reasonableness (within last hour)
        current_time = time.time()
        if abs(msg.timestamp - current_time) > 3600:
            return False
            
        return True
    

class MessageFactory:
    """
    Factory for creating properly formatted messages
    """
    
    def __init__(self):
        self.sequence_counter = 0
    
    def create_gello_message(self,
                            arm: str, 
                           joint_positions: List[float],
                           gripper_position: float,
                           joint_velocities: Optional[List[float]] = None) -> GelloMessage:
        """Create a GELLO message with auto-incrementing sequence"""
        
        self.sequence_counter += 1
        
        return GelloMessage(
            timestamp=time.time(),
            sequence_id=self.sequence_counter,
            arm=arm,
            joint_positions=joint_positions.copy() if hasattr(joint_positions, 'copy') else list(joint_positions),
            gripper_position=float(gripper_position),
            joint_velocities=list(joint_velocities) if joint_velocities else None
        )
    
    def create_torque_message(
        self,
        arm: str,
        joint_torques: List[float],
        gripper_torque: Optional[float] = None,
    ) -> 'TorqueMessage':
        self.sequence_counter += 1
        return TorqueMessage(
            timestamp=time.time(),
            sequence_id=self.sequence_counter,
            arm=arm,
            joint_torques=joint_torques.copy() if hasattr(joint_torques, "copy") else list(joint_torques),
            gripper_torque=float(gripper_torque) if gripper_torque is not None else None,
        )

    
    def create_neck_message(self,
                           neck_pose: List[float],
                           ) -> NeckMessage:
        """Create a neck message with auto-incrementing sequence"""
        
        self.sequence_counter += 1
        now_ns = time.time_ns()
        
        return NeckMessage(
            timestamp=now_ns / 1e9,
            sequence_id=self.sequence_counter,
            neck_pose=neck_pose.copy() if hasattr(neck_pose, 'copy') else list(neck_pose),
        )
    
    def create_base_message(self,
                            pose: List[float],
                            ) -> BaseProprioMessage:
        """Create a base message with auto-incrementing sequence"""
        
        self.sequence_counter += 1
        now_ns = time.time_ns()
        
        return BaseProprioMessage(
            timestamp=now_ns / 1e9,
            sequence_id=self.sequence_counter,
            pose=pose,
        )


    
    def create_activation_message(self, command: str) -> ActivationMessage:
        """Create activation message"""
        return ActivationMessage(
            timestamp=time.time(),
            command=command
        )
    
    def create_ready_message(self, node: str, ready: bool) -> SystemReadyMessage:
        """Create system ready message"""
        self.sequence_counter += 1
        return SystemReadyMessage(
            timestamp=time.time(),
            node=node,
            sequence_id=self.sequence_counter,
            ready=ready
        )
    
    
    def create_vision_pro_status_message(self, **kwargs) -> VisionProStatusMessage:
        """Create vision pro status message with current timestamp"""
        return VisionProStatusMessage(
            timestamp=time.time(),
            sequence_id=self.sequence_counter,
            **kwargs
        )
    
    def create_neck_iphone_status_message(self, **kwargs) -> NeckiPhoneStatusMessage:
        """Create neck iphone status message with current timestamp"""
        return NeckiPhoneStatusMessage(
            timestamp=time.time(),
            sequence_id=self.sequence_counter,
            **kwargs
        )
    
    def create_neck_status_message(self, **kwargs) -> NeckStatusMessage:
        """Create neck status message with current timestamp"""
        return NeckStatusMessage(
            timestamp=time.time(),
            sequence_id=self.sequence_counter,
            **kwargs
        )

    def create_episode_state_message(self, **kwargs) -> EpisodeStateMessage:
        """Create an episode state message with sequence tracking."""
        self.sequence_counter += 1
        return EpisodeStateMessage(
            timestamp=time.time(),
            sequence_id=self.sequence_counter,
            **kwargs
        )

    


# Message topic names for robotmq
class Topics:
    """Message queue topic definitions"""
    # System readiness
    GELLO_PC_STATUS = 'gello_pc_status'
    ROBOT_PC_STATUS = 'robot_pc_status'
    EPISODE_STATE = 'episode_state'

    # Gello 
    GELLO_LEFT = "left"
    GELLO_RIGHT = "right"

    # ARX5
    ACTIVATION = "robot_activation"
    RIGHT_RGB = 'right_wrist_camera_0'
    LEFT_RGB = 'left_wrist_camera_0'
    RIGHT = 'right_arm'
    RIGHT_GRIPPER = 'right_end_effector'
    LEFT = 'left_arm'
    LEFT_GRIPPER = 'left_end_effector'


    # Base
    BASE = 'body'
    BASE_ACTION = 'body_action'
    BASE_TARGET = 'body_target'  # commanded base target pose (base action)
    
    # Neck
    NECK_ACTIVATION = "neck_activation" 
    NECK_CMD = "neck_target_pose"
    NECK = 'head' # Can use JointStateMessage, just make sure to change source
    IPHONE = "head_camera_0"
    IPHONE_RGB = "iphone_rgb"
    IPHONE_DEPTH = "iphone_depth"
    IPHONE_INTRINSICS = "iphone_intrinsics"
    IPHONE_EXTRINSICS = "iphone_extrinsics"
    NECK_IPHONE_STATUS = 'neck_iphone_status'
    NECK_STATUS = 'neck_status'

    # Torque feedback (rby1)
    RIGHT_ARM_TORQUE = 'right_arm_torque'
    LEFT_ARM_TORQUE = 'left_arm_torque'

    # Raw FT wrench (rby1) — 6-DOF [Fx, Fy, Fz, Tx, Ty, Tz]
    RIGHT_FT = 'right_ft'
    LEFT_FT = 'left_ft'

    # Vision Pro
    VP_STATUS = "vision_pro_status"
    VP_PUBLISH_ACTIVATION = 'vision_pro_publish_activation'

    @staticmethod
    def torque_topic(arm: str) -> str:
        """Get torque feedback topic for a given arm."""
        if arm in ('right', 'right_arm'):
            return Topics.RIGHT_ARM_TORQUE
        return Topics.LEFT_ARM_TORQUE

    
    @staticmethod
    def gello_topic(arm: str) -> str:
        """Get topic name for specific arm"""
        return f"{arm}"
    
# Serialization utilities for robotmq compatibility
def serialize_message(msg) -> bytes:
    """Serialize message to bytes for robotmq transmission"""
    if hasattr(msg, 'to_json'):
        return msg.to_json().encode('utf-8')
    else:
        return json.dumps(asdict(msg)).encode('utf-8')
    
def serialize_system_ready_message(msg: SystemReadyMessage) -> bytes:
    return msg.to_json().encode("utf-8")

def deserialize_system_ready_message(data: bytes) -> SystemReadyMessage:
    return SystemReadyMessage.from_json(data.decode("utf-8"))

def serialize_episode_state_message(msg: EpisodeStateMessage) -> bytes:
    return msg.to_json().encode("utf-8")

def deserialize_episode_state_message(data: bytes) -> EpisodeStateMessage:
    return EpisodeStateMessage.from_json(data.decode("utf-8"))

def deserialize_vp_status_message(data: bytes) -> VisionProStatusMessage:
    """Deserialize status message from bytes"""
    json_str = data.decode('utf-8')
    return VisionProStatusMessage.from_dict(json.loads(json_str))

def deserialize_neck_iphone_status_message(data: bytes) -> NeckiPhoneStatusMessage:
    """Deserialize status message from neck iphone from bytes"""
    json_str = data.decode('utf-8')
    return NeckiPhoneStatusMessage.from_dict(json.loads(json_str))

def deserialize_neck_status_message(data: bytes) -> NeckStatusMessage:
    """Deserialize status message from neck iphone from bytes"""
    json_str = data.decode('utf-8')
    return NeckStatusMessage.from_dict(json.loads(json_str))
def serialize_neck_message(msg: NeckMessage) -> bytes:
    """Serialize NeckMessage to bytes using msgpack"""
    return serialize(asdict(msg))


def deserialize_neck_message(data: bytes) -> NeckMessage:
    """Deserialize NeckMessage from msgpack bytes"""
    msg_dict = deserialize(data)
    return NeckMessage.from_dict(msg_dict)


def serialize_timestamped_bytes(raw_bytes: bytes, timestamp_ns: Optional[int] = None) -> bytes:
    ts_ns = time.time_ns() if timestamp_ns is None else int(timestamp_ns)
    return b"TSB1" + struct.pack("!Q", ts_ns) + raw_bytes


def deserialize_timestamped_bytes(data: bytes) -> Dict[str, Any]:
    raw = bytes(data)
    if raw.startswith(b"TSB1") and len(raw) >= 12:
        ts_ns = struct.unpack("!Q", raw[4:12])[0]
        return {
            "timestamp": ts_ns / 1e9,
            "timestamp_ns": ts_ns,
            "data": raw[12:],
        }
    try:
        payload = json.loads(raw.decode("utf-8"))
        if isinstance(payload.get("data"), list):
            payload["data"] = bytes(payload["data"])
        return payload
    except (UnicodeDecodeError, json.JSONDecodeError):
        now_ns = time.time_ns()
        return {
            "timestamp": now_ns / 1e9,
            "timestamp_ns": now_ns,
            "data": raw,
        }


def serialize_torque_message(msg: 'TorqueMessage') -> bytes:
    return msg.to_json().encode('utf-8')

def deserialize_torque_message(data: bytes) -> 'TorqueMessage':
    return TorqueMessage.from_json(data.decode('utf-8'))

def serialize_base_proprio_message(msg: BaseProprioMessage) -> bytes:
    return msg.to_json().encode('utf-8')

def deserialize_base_proprio_message(data: bytes) -> BaseProprioMessage:
    try:
        return BaseProprioMessage.from_dict(deserialize(data))
    except Exception:
        return BaseProprioMessage.from_json(data.decode('utf-8'))


# Testing utilities

# =============================================================================
# Follower command
# =============================================================================

KEY_JOINT_POSITIONS = "joint_positions"
KEY_TIMESTAMP = "timestamp"
KEY_GRIPPER_POSITION = "gripper_position"


def normalize_follower_command(data, frame_id_default: str = ""):
    """Parse a deserialized payload into a dict with required fields materialized.

    Returns None if the message is not a valid follower command.
    """
    if not isinstance(data, dict):
        return None
    jpos = data.get(KEY_JOINT_POSITIONS)
    if not isinstance(jpos, (list, tuple)) or len(jpos) < 1:
        return None
    ts = data.get(KEY_TIMESTAMP)
    if ts is None:
        return None
    out = {
        KEY_JOINT_POSITIONS: [float(x) for x in jpos],
        KEY_TIMESTAMP: float(ts),
    }
    if KEY_GRIPPER_POSITION in data and data[KEY_GRIPPER_POSITION] is not None:
        out[KEY_GRIPPER_POSITION] = float(data[KEY_GRIPPER_POSITION])
    fid = data.get("frame_id") or frame_id_default
    if fid:
        out["frame_id"] = str(fid)
    if data.get("source") is not None:
        out["source"] = str(data["source"])
    return out


if __name__ == "__main__":
    # Basic functionality test
    print("Testing message formats...")
    
    # Test GELLO message
    factory = MessageFactory()
    msg = factory.create_gello_message(
        joint_positions=[0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
        gripper_position=0.7
    )
    
    print(f"Created GELLO message: {msg}")
    print(f"Valid: {MessageValidator.validate_gello_message(msg)}")
    
    # Test serialization
    serialized = serialize_message(msg)
    deserialized = GelloMessage.from_json(serialized.decode('utf-8'))
    print(f"Serialization round-trip successful: {msg.sequence_id == deserialized.sequence_id}")
    
    # Test activation message
    activation = factory.create_activation_message("activate")
    print(f"Created activation message: {activation}")
    
    print("Message format tests completed!")
