"""
Role type data contracts for the ModPack bridge.

For each role type:
  - command_schema: what bridge.get_command(role) returns
  - state_schema:   the fields bridge.publish_state(role, state) expects

Asymmetric by design:
  - Command dataclasses are used: get_command() constructs and returns them, and
    controllers read them by attribute (cmd.joint_pos, cmd.gripper).
  - State dataclasses are reference shapes only. publish_state() accepts a
    plain dict (the common case) or any dataclass, and does not validate either,
    it just serializes whatever it is handed. Controllers publish hand-built dicts
    and may include extra fields beyond what the State dataclass lists. Treat these
    as the documented baseline, not an enforced contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# =============================================================================
# Role types
# =============================================================================

JOINT_ARM = "joint_arm"
CARTESIAN_ARM = "cartesian_arm"
MOBILE_BASE = "mobile_base"
SENSOR_STREAM = "sensor_stream"
VIDEO_STREAM = "video_stream"

ROLE_TYPES = {JOINT_ARM, CARTESIAN_ARM, MOBILE_BASE, SENSOR_STREAM, VIDEO_STREAM}


# =============================================================================
# Command schemas (what get_command() returns per role type)
# =============================================================================

@dataclass
class JointArmCommand:
    """Command for a joint-space arm."""
    joint_pos: np.ndarray   # shape (dof,) — target joint angles in radians
    gripper: float          # [0, 1] — target gripper opening


@dataclass
class CartesianArmCommand:
    """Command for a Cartesian-space arm (e.g. neck)."""
    eef_pose: np.ndarray    # shape (6,) — [x, y, z, roll, pitch, yaw]


@dataclass
class MobileBaseCommand:
    """Command for a mobile base."""
    base_pose: np.ndarray   # shape (3,) — [x, y, theta]


# Sensor streams receive no commands — get_command() returns None for them.


# =============================================================================
# State schemas (advisory) — baseline fields publish_state() expects per role type.
# Not constructed or validated at runtime; controllers publish plain dicts and may
# add extra fields (e.g. target_positions, gripper_target). See module docstring.
# =============================================================================

@dataclass
class JointArmState:
    """State for a joint-space arm."""
    positions: np.ndarray   # shape (dof,) — current joint angles in radians
    gripper: float          # current gripper opening
    timestamp: float        # time.monotonic()
    data_valid: bool
    eef_pose: Optional[np.ndarray] = None  # shape (6,) if FK available


@dataclass
class CartesianArmState:
    """State for a Cartesian-space arm."""
    positions: np.ndarray   # shape (6,) — current EEF pose [x,y,z,r,p,y]
    timestamp: float
    data_valid: bool


@dataclass
class MobileBaseState:
    """State for a mobile base."""
    positions: np.ndarray   # shape (3,) — [x, y, theta]
    timestamp: float
    data_valid: bool


@dataclass
class SensorStreamState:
    """State for a sensor stream (read-only role)."""
    data: np.ndarray        # shape matches manifest declaration
    timestamp: float
    data_valid: bool


@dataclass
class VideoStreamState:
    """State for a video stream (read-only role). Payload is pre-compressed bytes."""
    frame: bytes            # compressed image bytes (e.g. JPEG)
    topic: str              # routing key — matches role config 'topic'
    timestamp: float
    data_valid: bool
