"""Joint-space policy rollout bridge with absolute base commands.

Architecture mirrors rby1_wbc_policy.py (asynchronous inference + scheduled
action buffer) but dispatches joint-position targets instead of Cartesian IK.

Action format (per step):
    left_qpos:           float32[7]
    right_qpos:          float32[7]
    head_qpos:           float32[2]
    left_gripper_width:  float32
    right_gripper_width: float32
    base_xy_yaw:         float32[3]  -- absolute (x, y, yaw) in world frame

Observation format matches the rby1 logger schema (see config.yaml):
    left_arm_state_joint_pos    [2, 7]
    right_arm_state_joint_pos   [2, 7]
    head_state_joint_pos        [2, 2]
    left_eef_state_joint_pos    [2, 1]
    right_eef_state_joint_pos   [2, 1]
    body_state_pos_xyz          [2, 3]
    body_state_quat_wxyz        [2, 4]
    timestamp                   [2]

Optional --include-torque adds projected joint torques (J_arm^T * F_ee).
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence

import dill
import numpy as np
import zmq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PROJECT_ROOT_STR = str(PROJECT_ROOT)

from camera.camera_stream import AravisCameraStreamer
from control.rby1_policy import RBY1PolicyRobot
from rby1.base_targets import BaseTargets

GRIPPER_WIDTH_LIMITS = (0.0, 0.085)

# Flat layout for --initial-actions .npy files. Mirrors the action dict the
# policy server emits; base_xy_yaw is included for round-trip with policy
# dumps but skipped when commanding the ready pose.
INITIAL_ACTION_DIM = 21
INITIAL_ACTION_SLICES = {
    "left_qpos": slice(0, 7),
    "right_qpos": slice(7, 14),
    "head_qpos": slice(14, 16),
    "left_gripper_width": slice(16, 17),
    "right_gripper_width": slice(17, 18),
    "base_xy_yaw": slice(18, 21),
}

DEFAULT_CAMERA_LATENCIES = {
    "camera_left_main_rgb": 0.06,
    "camera_right_main_rgb": 0.06,
    "camera_head_main_rgb": 0.1,
    "camera_head_main_right_rgb": 0.1,
    "camera_head_ultrawide_rgb": 0.1,
}

# Cameras that use the sticky black-crop preprocessing (wrist cams) on the
# policy-server side. Mirrors _WRIST_CAMERA_SLASH_KEYS in policy_server_rby1.py
# but keyed on the bare obs name we use locally.
_WRIST_CAMERA_KEYS = {
    "camera_left_main_rgb",
    "camera_right_main_rgb",
}


def _find_black_crop_box(frame: np.ndarray) -> tuple[int, int, int, int]:
    """Sticky square crop box that excludes black borders (training-matched)."""
    mask = frame.max(axis=2) > 0
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        fh, fw = frame.shape[:2]
        s = min(fh, fw)
        return 0, s, 0, s
    r0, r1 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1]) + 1
    c0, c1 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1]) + 1
    ch, cw = r1 - r0, c1 - c0
    s = min(ch, cw)
    yr = (ch - s) // 2
    xr = (cw - s) // 2
    return r0 + yr, r0 + yr + s, c0 + xr, c0 + xr + s


def _resize_and_pad(frame: np.ndarray, out_wh: tuple[int, int]) -> np.ndarray:
    """Aspect-preserving resize + zero-pad to out_wh (training-matched)."""
    import cv2

    w, h = out_wh
    fh, fw = frame.shape[:2]
    scale = min(w / fw, h / fh)
    new_w, new_h = round(fw * scale), round(fh * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((h, w, frame.shape[2]), dtype=frame.dtype)
    y0, x0 = (h - new_h) // 2, (w - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _preprocess_for_model(
    frame_bgr: np.ndarray,
    out_wh: tuple[int, int],
    is_wrist: bool,
    crop_box: Optional[tuple[int, int, int, int]],
) -> tuple[np.ndarray, Optional[tuple[int, int, int, int]]]:
    """Mirror policy_server_rby1._preprocess_frames for one frame, but return
    uint8 HWC BGR ready for cv2.VideoWriter (skipping the float/CHW step the
    server does for the model). Result is what the model sees, color-correct
    for playback."""
    import cv2

    # The server does BGR→RGB, resize/pad or crop+resize, then float32/255.
    # For visualization we want the same spatial transforms but keep BGR/uint8.
    frame = frame_bgr
    if frame.shape[:2] != (out_wh[1], out_wh[0]):
        if is_wrist:
            if crop_box is None:
                # The server computes the crop on the RGB-converted frame, but
                # the black-pixel test is channel-symmetric so BGR works too.
                crop_box = _find_black_crop_box(frame)
            r0, r1, c0, c1 = crop_box
            frame = cv2.resize(
                frame[r0:r1, c0:c1], out_wh, interpolation=cv2.INTER_LINEAR
            )
        else:
            frame = _resize_and_pad(frame, out_wh)
    return frame, crop_box


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class JointScheduledAction:
    """Timestamped joint-space action for the scheduled action buffer."""

    timestamp: float
    duration: float
    payload: Dict[
        str, np.ndarray
    ]  # left_qpos, right_qpos, head_qpos, widths, base_xy_yaw


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _yaw_quat_wxyz(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=float)


def _parse_latency_overrides(entries: Optional[Sequence[str]]) -> Dict[str, float]:
    if not entries:
        return {}
    out: Dict[str, float] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                f"Invalid camera latency override '{entry}', expected KEY=SECONDS"
            )
        key, val = entry.split("=", 1)
        try:
            out[key.strip()] = float(val)
        except ValueError:
            raise ValueError(f"Invalid latency value in override '{entry}'") from None
    return out


def _camera_latency_for(
    key: str, global_override: Optional[float], overrides: Dict[str, float]
) -> float:
    if global_override is not None:
        return float(global_override)
    if key in overrides:
        return float(overrides[key])
    return float(DEFAULT_CAMERA_LATENCIES.get(key, 0.0))


def _merge_camera_timestamps(camera_ts: Dict[str, np.ndarray]) -> np.ndarray:
    """Slowest camera wins — use latest (max) timestamp as anchor."""
    if not camera_ts:
        raise ValueError("camera_ts must contain at least one stream")
    arrays = [np.asarray(ts, dtype=float).reshape(-1) for ts in camera_ts.values()]
    ref = arrays[0]
    for arr in arrays[1:]:
        if arr.shape != ref.shape:
            raise ValueError("Camera timestamp arrays must share the same shape")
        ref = np.maximum(ref, arr)
    return ref


# ---------------------------------------------------------------------------
# Initial (ready) position from a saved action file
# ---------------------------------------------------------------------------


# Map modality_layout names (from box_placing_new_strategy.npz and similar)
# to the action-dict fields the bridge consumes. Names not in this map are
# ignored; the file may include them (e.g. body/base) but the ready-pose move
# only commands arms + head + grippers.
#
# Note: the *_end_effector slots are labeled "target_joint_pos" in the layout
# but the stored values are gripper widths in meters (range ~0–0.1), matching
# the legacy .npy convention — we treat them as widths.
_MODALITY_TO_TARGET_KEY = {
    "left_arm:target_joint_pos": "left_qpos",
    "right_arm:target_joint_pos": "right_qpos",
    "head:target_joint_pos": "head_qpos",
    "left_end_effector:target_joint_pos": "left_gripper_width",
    "right_end_effector:target_joint_pos": "right_gripper_width",
}
_REQUIRED_TARGET_KEYS = ("left_qpos", "right_qpos", "head_qpos")
_GRIPPER_TARGET_KEYS = ("left_gripper_width", "right_gripper_width")
_TORQUE_WIRE_KEYS = ("left_arm_state_joint_torque", "right_arm_state_joint_torque")


def _slices_from_modality_layout(
    layout: Sequence[str],
) -> Dict[str, slice]:
    """Convert a modality_layout array (e.g. ['body:target_pose:3', ...]) to
    per-target-key slices. Raises if total dim ≠ INITIAL_ACTION_DIM or if any
    required modality is missing."""

    slices: Dict[str, slice] = {}
    offset = 0
    for entry in layout:
        parts = str(entry).split(":")
        if len(parts) != 3:
            raise ValueError(f"Malformed modality_layout entry: {entry!r}")
        modality_key = f"{parts[0]}:{parts[1]}"
        size = int(parts[2])
        target_key = _MODALITY_TO_TARGET_KEY.get(modality_key)
        if target_key is not None:
            slices[target_key] = slice(offset, offset + size)
        offset += size
    if offset != INITIAL_ACTION_DIM:
        raise ValueError(
            f"modality_layout totals {offset} dims, expected {INITIAL_ACTION_DIM}"
        )
    missing = [k for k in _REQUIRED_TARGET_KEYS if k not in slices]
    if missing:
        raise ValueError(
            f"modality_layout missing required modalities: {missing}. "
            f"Saw {[str(e) for e in layout]}"
        )
    return slices


def _load_initial_target(path: Path) -> Dict[str, np.ndarray]:
    """Load initial actions from a .npy or .npz file and average to a target.

    Supported formats:
    - .npy: legacy flat layout described by INITIAL_ACTION_SLICES
      (left_qpos[7], right_qpos[7], head_qpos[2], lgrip[1], rgrip[1],
      base_xy_yaw[3]). Gripper fields are interpreted as widths in meters
      and clipped to GRIPPER_WIDTH_LIMITS.
    - .npz: must contain first_actions of shape [N, 21] and a
      modality_layout string array describing the per-modality split.
      Layouts that don't expose gripper *widths* (e.g. box_placing_new_strategy
      stores target_joint_pos instead) cause grippers to be omitted from
      the target so the WBC holds their current width.

    Mean is taken across axis 0. The base slot, if present, is always ignored
    — the ready-position step does not command the base.
    """
    suffix = path.suffix.lower()
    if suffix == ".npz":
        data = np.load(str(path))
        if "first_actions" not in data.files:
            raise ValueError(
                f"{path}: .npz missing 'first_actions'. Keys: {data.files}"
            )
        arr = np.asarray(data["first_actions"], dtype=float)
        if arr.ndim != 2 or arr.shape[1] != INITIAL_ACTION_DIM:
            raise ValueError(
                f"{path}: first_actions shape {arr.shape}, expected [N, {INITIAL_ACTION_DIM}]"
            )
        if "modality_layout" not in data.files:
            raise ValueError(
                f"{path}: .npz lacks 'modality_layout' — cannot infer slice order safely"
            )
        slices = _slices_from_modality_layout(data["modality_layout"])
        mean = arr.mean(axis=0)
        target: Dict[str, np.ndarray] = {
            "left_qpos": mean[slices["left_qpos"]].copy(),
            "right_qpos": mean[slices["right_qpos"]].copy(),
            "head_qpos": mean[slices["head_qpos"]].copy(),
        }
        for grip_key in _GRIPPER_TARGET_KEYS:
            if grip_key in slices:
                target[grip_key] = float(
                    np.clip(mean[slices[grip_key]][0], *GRIPPER_WIDTH_LIMITS)
                )
        return target

    # Legacy .npy path
    arr = np.load(str(path), allow_pickle=False)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != INITIAL_ACTION_DIM:
        raise ValueError(
            f"Expected initial actions of shape [N, {INITIAL_ACTION_DIM}], got {arr.shape}"
        )
    mean = arr.mean(axis=0)
    return {
        "left_qpos": mean[INITIAL_ACTION_SLICES["left_qpos"]].copy(),
        "right_qpos": mean[INITIAL_ACTION_SLICES["right_qpos"]].copy(),
        "head_qpos": mean[INITIAL_ACTION_SLICES["head_qpos"]].copy(),
        "left_gripper_width": float(
            np.clip(
                mean[INITIAL_ACTION_SLICES["left_gripper_width"]][0],
                *GRIPPER_WIDTH_LIMITS,
            )
        ),
        "right_gripper_width": float(
            np.clip(
                mean[INITIAL_ACTION_SLICES["right_gripper_width"]][0],
                *GRIPPER_WIDTH_LIMITS,
            )
        ),
    }


def _move_to_ready_position(
    robot: RBY1PolicyRobot,
    target: Dict[str, np.ndarray],
    duration: float,
) -> None:
    """Ramp arms/head/grippers to the target pose over `duration` seconds.

    Uses the WBC's own joint interpolation (joint_targets lerps from current
    to target over the given duration), so this is a single dispatch.
    """
    action: Dict[str, np.ndarray] = {
        "left_qpos": target["left_qpos"],
        "right_qpos": target["right_qpos"],
        "head_qpos": target["head_qpos"],
    }
    if "left_gripper_width" in target:
        action["left_gripper_width"] = target["left_gripper_width"]
    if "right_gripper_width" in target:
        action["right_gripper_width"] = target["right_gripper_width"]
    robot.apply_joint_action(
        action,
        duration=float(max(duration, 1e-3)),
        timestamp=time.monotonic(),
    )


# ---------------------------------------------------------------------------
# ZMQ policy client
# ---------------------------------------------------------------------------


class PolicyClient:
    def __init__(self, ip: str, port: int, timeout: float = 2.0) -> None:
        self._ctx = zmq.Context.instance()
        self._ip = ip
        self._port = port
        self._timeout = timeout
        self._socket = None
        self._connect()
        self.observation_keys: Dict[str, object] = {}

    def _connect(self) -> None:
        """Create and connect a fresh REQ socket with proper initialization."""
        if self._socket is not None:
            try:
                self._socket.close(0)
            except Exception:
                pass
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.connect(f"tcp://{self._ip}:{self._port}")
        self._socket.setsockopt(zmq.RCVTIMEO, int(max(self._timeout, 0.1) * 1000))
        self._socket.setsockopt(zmq.SNDTIMEO, int(max(self._timeout, 0.1) * 1000))
        # Set linger to prevent hanging on close
        self._socket.setsockopt(zmq.LINGER, 0)
        # Give server time to be ready
        time.sleep(0.1)

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close(0)
            except Exception:
                pass

    def request_observation_keys(self) -> Dict[str, object]:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self._socket.send(dill.dumps("get_obs_keys"))
                reply_bytes = self._socket.recv()
                reply = dill.loads(reply_bytes)
                if isinstance(reply, dict):
                    self.observation_keys = reply
                    return reply
            except zmq.Again:
                # Timeout or would-block: socket state may be corrupted, reconnect
                time.sleep(0.1)
                self._connect()
                continue
            except Exception as e:
                # Any other exception: socket state is corrupted, reconnect
                print(f"[policy] Socket error: {e}, reconnecting...")
                time.sleep(0.1)
                self._connect()
                if attempt < max_retries - 1:
                    continue
                raise
            time.sleep(0.5)
        raise RuntimeError("Failed to get observation keys after retries")

    def infer(self, obs: Dict[str, np.ndarray]) -> Optional[Dict]:
        max_retries = 2
        for attempt in range(max_retries):
            try:
                self._socket.send(dill.dumps(obs))
                result_bytes = self._socket.recv()
                result = dill.loads(result_bytes)
                return result
            except zmq.Again:
                # Timeout: socket is in bad state, reconnect
                self._connect()
                if attempt < max_retries - 1:
                    continue
                return None
            except Exception as e:
                # Any other error: socket is corrupted, reconnect
                print(f"[policy] Infer error: {e}, reconnecting...")
                self._connect()
                if attempt < max_retries - 1:
                    continue
                return None
        return None


# ---------------------------------------------------------------------------
# Action scheduling
# ---------------------------------------------------------------------------


def build_scheduled_actions(
    actions: Dict[str, np.ndarray],
    timestamps: np.ndarray,
    fallback_dt: float,
    now: float,
) -> List[JointScheduledAction]:
    """Slice stacked action arrays into a list of timestamped single-step payloads."""

    if not actions or "left_qpos" not in actions or "right_qpos" not in actions:
        return []

    left = np.asarray(actions["left_qpos"], dtype=float)
    right = np.asarray(actions["right_qpos"], dtype=float)
    length = min(left.shape[0], right.shape[0])
    if length == 0:
        return []

    timestamps = np.asarray(timestamps, dtype=float).reshape(-1)

    def _ts(idx: int) -> float:
        if timestamps.size == 0:
            return now + fallback_dt * (idx + 1)
        i = min(idx, timestamps.size - 1)
        ts = float(timestamps[i])
        return ts if np.isfinite(ts) else now + fallback_dt * (idx + 1)

    result: List[JointScheduledAction] = []
    for idx in range(length):
        payload: Dict[str, np.ndarray] = {
            "left_qpos": left[idx].reshape(-1).copy(),
            "right_qpos": right[idx].reshape(-1).copy(),
        }
        for key in ("head_qpos",):
            arr = actions.get(key)
            if arr is not None:
                a = np.asarray(arr, dtype=float)
                if idx < a.shape[0]:
                    payload[key] = a[idx].reshape(-1).copy()
        for key in ("left_gripper_width", "right_gripper_width"):
            arr = actions.get(key)
            if arr is not None:
                a = np.asarray(arr, dtype=float).reshape(-1)
                if idx < a.size:
                    payload[key] = np.clip(float(a[idx]), *GRIPPER_WIDTH_LIMITS)
        arr = actions.get("base_xy_yaw")
        if arr is not None:
            a = np.asarray(arr, dtype=float)
            if idx < a.shape[0]:
                payload["base_xy_yaw"] = a[idx].reshape(-1).copy()

        ts = _ts(idx)
        dur = (
            float(max(fallback_dt, _ts(idx + 1) - ts))
            if idx + 1 < length
            else float(fallback_dt)
        )
        result.append(JointScheduledAction(timestamp=ts, duration=dur, payload=payload))

    result.sort(key=lambda a: a.timestamp)
    return result


# ---------------------------------------------------------------------------
# Absolute base target dispatcher
# ---------------------------------------------------------------------------


class AbsoluteBaseDispatcher:
    """Forward absolute world-frame (x, y, yaw) targets to update_base_targets."""

    def __init__(self, robot: RBY1PolicyRobot) -> None:
        self._robot = robot

    def apply(self, x: float, y: float, yaw: float) -> bool:
        target = BaseTargets()
        target.set_targets(float(x), float(y), _yaw_quat_wxyz(float(yaw)))
        update_fn = getattr(
            self._robot._backend, "update_base_targets", None
        )  # noqa: SLF001
        if update_fn is None:
            return False
        update_fn(target)
        return True


# ---------------------------------------------------------------------------
# Optional torque projection (J_arm^T * F_ee).
#
# TODO: This block is a copy of teleop/teleop_gello.py:192-301 (init) and
# teleop/teleop_gello.py:611-682 (per-step). Gello-specific bits stripped.
# ---------------------------------------------------------------------------

_TORQUE_SNAP_JOINT_NAMES = [
    "wheel_fr",
    "wheel_fl",
    "wheel_rr",
    "wheel_rl",
    "torso_0",
    "torso_1",
    "torso_2",
    "torso_3",
    "torso_4",
    "torso_5",
    "right_arm_0",
    "right_arm_1",
    "right_arm_2",
    "right_arm_3",
    "right_arm_4",
    "right_arm_5",
    "right_arm_6",
    "left_arm_0",
    "left_arm_1",
    "left_arm_2",
    "left_arm_3",
    "left_arm_4",
    "left_arm_5",
    "left_arm_6",
    "head_0",
    "head_1",
]


# Emit raw Jacobian-projected joint torques (N·m, robot frame). Matches the
# unscaled torque produced by
# policy/diffusion_policy/dataset/mobile_gello_teleop_rby1_dataset.py::_load_torque,
# which now inverts the joint_signs / torque_feedback_gain that teleop_gello.py
# applies before publishing to the feedback channel.


class _TorqueProjector:
    def __init__(self, urdf_path: str) -> None:
        try:
            import rby1_sdk.dynamics as rby_dyn  # type: ignore
        except ImportError:
            sdk_py = str((PROJECT_ROOT / "rby1-sdk" / "python").resolve())
            if os.path.isdir(sdk_py) and sdk_py not in sys.path:
                sys.path.insert(0, sdk_py)
            import rby1_sdk.dynamics as rby_dyn  # type: ignore  # noqa: F401

        p = Path(os.path.expanduser(os.path.expandvars(str(urdf_path).strip())))
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        p = p.resolve()
        if not p.exists():
            raise FileNotFoundError(str(p))

        logging.info("[torque] loading URDF: %s", str(p))
        robot_config = rby_dyn.load_robot_from_urdf(str(p), "base")
        self._dyn_robot = rby_dyn.Robot(robot_config)

        if hasattr(self._dyn_robot, "get_joint_names"):
            all_joint_names = list(self._dyn_robot.get_joint_names())
        else:
            attr = getattr(self._dyn_robot, "joint_names", None)
            all_joint_names = list(attr() if callable(attr) else attr)

        snap_name_to_idx = {name: i for i, name in enumerate(_TORQUE_SNAP_JOINT_NAMES)}
        try:
            self._snap_to_dyn_qidx = np.array(
                [snap_name_to_idx[name] for name in all_joint_names], dtype=int
            )
        except KeyError as exc:
            raise KeyError(
                f"Dynamics joint name not found in WBC snapshot order: {exc}. "
                f"dyn_joint_names={all_joint_names}"
            ) from exc

        self._right_arm_cols = [
            all_joint_names.index(f"right_arm_{i}") for i in range(7)
        ]
        self._left_arm_cols = [all_joint_names.index(f"left_arm_{i}") for i in range(7)]
        self._dyn_state_left = self._dyn_robot.make_state(
            ["base", "ee_left"], all_joint_names
        )
        self._dyn_state_right = self._dyn_robot.make_state(
            ["base", "ee_right"], all_joint_names
        )
        left_links = list(self._dyn_state_left.get_link_names())
        right_links = list(self._dyn_state_right.get_link_names())
        self._base_link_idx_left = left_links.index("base")
        self._left_ee_idx = left_links.index("ee_left")
        self._base_link_idx_right = right_links.index("base")
        self._right_ee_idx = right_links.index("ee_right")
        logging.info("[torque] dynamics initialized.")

    def compute(self, snapshot) -> Dict[str, np.ndarray]:
        zeros = np.zeros(7, dtype=float)
        result = {"left": zeros.copy(), "right": zeros.copy()}
        if snapshot is None or not getattr(snapshot, "is_valid", True):
            return result
        try:
            q_snap = np.array(snapshot.joint_position, dtype=np.float64).reshape(-1)
        except Exception:
            return result
        if q_snap.shape[0] <= int(np.max(self._snap_to_dyn_qidx)):
            return result
        q = q_snap[self._snap_to_dyn_qidx].reshape(-1, 1)

        for arm, dyn_state, base_idx, ee_idx, arm_cols, ft_valid_attr, wrench_attr in [
            (
                "left",
                self._dyn_state_left,
                self._base_link_idx_left,
                self._left_ee_idx,
                self._left_arm_cols,
                "left_ft_valid",
                "left_ee_wrench",
            ),
            (
                "right",
                self._dyn_state_right,
                self._base_link_idx_right,
                self._right_ee_idx,
                self._right_arm_cols,
                "right_ft_valid",
                "right_ee_wrench",
            ),
        ]:
            try:
                if not bool(getattr(snapshot, ft_valid_attr, False)):
                    continue
                wrench = getattr(snapshot, wrench_attr, None)
                if wrench is None:
                    continue
                wrench_raw = np.asarray(wrench, dtype=np.float64).reshape(-1)
                if wrench_raw.size != 6:
                    continue
                dyn_state.set_q(q)
                self._dyn_robot.compute_forward_kinematics(dyn_state)
                J = self._dyn_robot.compute_body_jacobian(dyn_state, base_idx, ee_idx)
                J_arm = J[:, arm_cols]
                # Snapshot wrench: [Fx, Fy, Fz, Tx, Ty, Tz]
                # Body Jacobian dual convention: [omega; v] -> reorder to [Tx, Ty, Tz, Fx, Fy, Fz]
                wrench_vec = np.concatenate([wrench_raw[3:], wrench_raw[:3]])
                result[arm] = (J_arm.T @ wrench_vec).reshape(-1)
            except Exception as exc:
                logging.debug("[torque] %s projection failed: %s", arm, exc)
        return result


# ---------------------------------------------------------------------------
# Joint-space command thread
# ---------------------------------------------------------------------------


class JointCommandThread:
    """Mirrors RBY1PolicyRobot._command_worker but dispatches joint-space actions.

    Runs at controller rate, samples the scheduled action buffer, and calls
    apply_joint_action + AbsoluteBaseDispatcher each tick.
    """

    def __init__(
        self,
        robot: RBY1PolicyRobot,
        base_dispatcher: AbsoluteBaseDispatcher,
        control_dt: float,
        disable_base: bool = False,
    ) -> None:
        self._robot = robot
        self._base_dispatcher = base_dispatcher
        self._control_dt = float(control_dt)
        self._disable_base = disable_base
        self._action_buffer: Deque[JointScheduledAction] = deque()
        self._last_action: Optional[JointScheduledAction] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._worker, name="joint-command", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def queue_actions(self, actions: List[JointScheduledAction]) -> None:
        if not actions:
            return
        now = time.monotonic()
        with self._lock:
            # Drop future entries from the buffer that are superseded by the new chunk.
            while (
                self._action_buffer
                and self._action_buffer[-1].timestamp >= actions[0].timestamp
            ):
                self._action_buffer.pop()
            # Trim stale entries, but keep the most-recent past entry so
            # _sample_payload always has a `prev` to dispatch when every
            # remaining buffered timestamp is still in the future (the typical
            # case with arm_execution_latency > 0 — every freshly-queued chunk
            # is entirely in the future).
            while (
                len(self._action_buffer) >= 2
                and self._action_buffer[1].timestamp < now
            ):
                self._action_buffer.popleft()
            self._action_buffer.extend(actions)

    def _sample_payload(self, now: float) -> Optional[JointScheduledAction]:
        """Return the action whose timestamp bracket contains now."""
        with self._lock:
            prev: Optional[JointScheduledAction] = None
            for action in self._action_buffer:
                if action.timestamp > now:
                    break
                prev = action
            if prev is None and self._last_action is not None:
                prev = self._last_action
            return prev

    def _worker(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            action = self._sample_payload(now)
            if action is not None:
                payload = action.payload
                base_xy_yaw = payload.get("base_xy_yaw")
                try:
                    self._robot.apply_joint_action(
                        payload, duration=self._control_dt, timestamp=now
                    )
                except Exception as exc:
                    logging.debug("[joint-cmd] apply_joint_action failed: %s", exc)
                if base_xy_yaw is not None and not self._disable_base:
                    try:
                        self._base_dispatcher.apply(
                            float(base_xy_yaw[0]),
                            float(base_xy_yaw[1]),
                            float(base_xy_yaw[2]),
                        )
                    except Exception as exc:
                        logging.debug("[joint-cmd] base dispatcher failed: %s", exc)
                with self._lock:
                    self._last_action = action
            wait = max(0.0, self._control_dt - (time.monotonic() - now))
            if self._stop.wait(timeout=wait):
                break


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RBY1 joint-space + relative-base policy bridge"
    )
    parser.add_argument("--policy-ip", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8766)
    parser.add_argument(
        "--control-dt",
        type=float,
        default=0.1,
        help="Controller tick period in seconds (10 Hz default).",
    )
    parser.add_argument(
        "--policy-interval",
        type=float,
        default=0.1,
        help="Target interval between inference calls in seconds (10 Hz default).",
    )
    parser.add_argument("--camera-horizon", type=int, default=2)
    parser.add_argument("--camera-stride", type=int, default=6)
    parser.add_argument("--obs-frequency", type=float, default=60.0)
    parser.add_argument("--policy-timeout", type=float, default=2.0)
    parser.add_argument(
        "--arm-execution-latency",
        type=float,
        default=0.2753,
        help="Measured arm execution latency (s) for stale-action filtering.",
    )
    parser.add_argument(
        "--camera-latency",
        type=float,
        default=None,
        help="Global camera latency (s); defaults to per-camera values.",
    )
    parser.add_argument(
        "--camera-latency-override",
        action="append",
        help="Per-camera latency override KEY=SECONDS; repeatable.",
    )
    parser.add_argument(
        "--proprioception-latency",
        type=float,
        default=0.005,
        help="Proprioception latency (s) for observation alignment.",
    )
    parser.add_argument(
        "--state-only", action="store_true", help="Skip camera streaming."
    )
    parser.add_argument(
        "--sim-only", action="store_true", help="Run without the realtime controller."
    )
    parser.add_argument("--sim-model", default=None)
    parser.add_argument("--sim-viewer", action="store_true")
    parser.add_argument("--gripper-width-offset", type=float, default=0.005)
    parser.add_argument(
        "--disable-base", action="store_true", help="Ignore base_xy_yaw in actions."
    )
    parser.add_argument("--urdf-path", default="rby1-sdk/models/rby1m/urdf/model.urdf")
    parser.add_argument(
        "--initial-actions",
        default=None,
        help=(
            "Path to initial_actions.npy of shape [N, "
            f"{INITIAL_ACTION_DIM}] (flat: left_qpos[7], right_qpos[7], "
            "head_qpos[2], left_gripper_width, right_gripper_width, base_xy_yaw[3]). "
            "Mean across N becomes the ready-position target; prompts the operator "
            "before moving and again before starting inference."
        ),
    )
    parser.add_argument(
        "--ready-duration",
        type=float,
        default=5.0,
        help="Seconds to ramp from current pose to the ready position.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Collect observations and run inference normally, but discard the "
            "action chunks instead of queuing them to the robot. Pairs with "
            "--dry-run-log-dir to persist obs and replies for offline analysis."
        ),
    )
    parser.add_argument(
        "--dry-run-log-dir",
        default=None,
        help=(
            "Directory to dump one .npz per inference cycle (obs + reply) when "
            "--dry-run is set. Defaults to log_captures/dry_run_<timestamp>."
        ),
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help=(
            "Write one .mp4 per camera_*_rgb obs into --video-dir, encoded "
            "live as the rollout runs. Works in both dry-run and live modes."
        ),
    )
    parser.add_argument(
        "--video-dir",
        default=None,
        help=(
            "Directory for --record-video output. Defaults to "
            "log_captures/rollout_<timestamp>."
        ),
    )
    parser.add_argument(
        "--processed-resolution",
        default="224x224",
        help=(
            "WxH the policy server resizes/pads camera frames to before "
            "inference. When --record-video is set, an additional "
            "<cam>_processed.mp4 per camera is written showing exactly what "
            "the model sees (same resize/pad/crop pipeline). Match this to "
            "the training task yaml's image_width/image_height."
        ),
    )
    args = parser.parse_args()

    proc_w, proc_h = (int(x) for x in args.processed_resolution.lower().split("x"))
    processed_out_wh = (proc_w, proc_h)

    control_dt = max(args.control_dt, 1e-2)
    inference_period = max(args.policy_interval, 1e-2)
    camera_latency_overrides = _parse_latency_overrides(args.camera_latency_override)

    robot = RBY1PolicyRobot(
        config_path=PROJECT_ROOT_STR + "/config/wbc.yaml",
        use_sim=args.sim_only,
        sim_model_path=args.sim_model,
        sim_viewer=args.sim_viewer,
    )
    robot.start()

    camera_streamer: Optional[AravisCameraStreamer] = None
    policy_client: Optional[PolicyClient] = None
    torque_projector: Optional[_TorqueProjector] = None
    torque_horizon: Optional[int] = None
    stop_event = threading.Event()
    video_dir: Optional[Path] = None
    video_writers: Dict[str, object] = {}
    processed_video_writers: Dict[str, object] = {}
    processed_crop_boxes: Dict[str, Optional[tuple]] = {}

    base_dispatcher = AbsoluteBaseDispatcher(robot)
    cmd_thread = JointCommandThread(
        robot=robot,
        base_dispatcher=base_dispatcher,
        control_dt=control_dt,
        disable_base=args.disable_base,
    )

    try:
        robot.wait_until_ready()
        required_samples = max(
            int(
                args.camera_horizon
                * args.camera_stride
                / (robot.dt * args.obs_frequency)
            ),
            20,
        )
        if not robot.wait_for_observations(required_samples, timeout=5.0):
            raise TimeoutError("Timed out waiting for initial robot observations")

        if not args.state_only and not args.sim_only:
            camera_streamer = AravisCameraStreamer()
            camera_streamer.start()
            try:
                camera_streamer.wait_until_ready(
                    min_frames=max(args.camera_horizon * args.camera_stride, 1),
                    timeout=2.0,
                )
            except TimeoutError as exc:
                print(f"[camera] {exc}")
        elif args.sim_only and not args.state_only:
            print("[camera] Skipping camera streamer in simulation mode.")

        policy_client = PolicyClient(
            ip=args.policy_ip,
            port=args.policy_port,
            timeout=args.policy_timeout,
        )
        policy_client.request_observation_keys()
        print(f"[policy] Required observation keys: {policy_client.observation_keys}")

        # Auto-enable torque projection iff the policy advertises torque obs keys.
        # The handshake also tells us the history length the policy expects.
        needs_torque = any(
            k in policy_client.observation_keys for k in _TORQUE_WIRE_KEYS
        )
        if needs_torque:
            for k in _TORQUE_WIRE_KEYS:
                shp = policy_client.observation_keys.get(k)
                if shp is not None:
                    torque_horizon = int(shp[0])
                    break
            try:
                torque_projector = _TorqueProjector(args.urdf_path)
            except Exception as exc:
                requested = [
                    k for k in _TORQUE_WIRE_KEYS if k in policy_client.observation_keys
                ]
                raise RuntimeError(
                    f"Policy requires torque observations {requested} but "
                    f"_TorqueProjector init failed: {exc}"
                ) from exc
            robot.register_torque_projector(torque_projector)
            print(
                f"[torque] Auto-enabled from policy metadata (horizon={torque_horizon})"
            )

        if args.initial_actions is not None and not args.dry_run:
            init_path = Path(
                os.path.expanduser(os.path.expandvars(args.initial_actions))
            )
            if not init_path.is_absolute():
                init_path = PROJECT_ROOT / init_path
            target = _load_initial_target(init_path)
            grip_str = (
                f", l_grip={target['left_gripper_width']:.4f}, "
                f"r_grip={target['right_gripper_width']:.4f}"
                if "left_gripper_width" in target and "right_gripper_width" in target
                else " (grippers hold current width)"
            )
            print(
                f"[init] Loaded ready-position target from {init_path}: "
                f"left={target['left_qpos']}, right={target['right_qpos']}, "
                f"head={target['head_qpos']}"
                f"{grip_str}"
            )
            input("[init] Press Enter to move to READY position...")
            print(f"[init] Moving to ready position over {args.ready_duration:.1f}s...")
            _move_to_ready_position(robot, target, duration=args.ready_duration)
            time.sleep(float(max(args.ready_duration, 0.0)))
            print("[init] At ready position.")
            input("[init] Press Enter to START inference...")
        elif args.initial_actions is not None and args.dry_run:
            print(
                "[dry-run] Skipping ready-position move (robot stays put in dry-run)."
            )

        dry_run_dir: Optional[Path] = None
        if args.dry_run:
            log_root = args.dry_run_log_dir or str(PROJECT_ROOT / "log_captures")
            dry_run_dir = Path(os.path.expanduser(os.path.expandvars(log_root)))
            if not dry_run_dir.is_absolute():
                dry_run_dir = PROJECT_ROOT / dry_run_dir
            if args.dry_run_log_dir is None:
                dry_run_dir = dry_run_dir / time.strftime("dry_run_%Y%m%d_%H%M%S")
            dry_run_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[dry-run] Logging obs + replies to {dry_run_dir} (robot will not move)"
            )
        else:
            cmd_thread.start()

        video_fps = max(1.0, 1.0 / inference_period)
        if args.record_video:
            vid_root = args.video_dir or str(PROJECT_ROOT / "log_captures")
            video_dir = Path(os.path.expanduser(os.path.expandvars(vid_root)))
            if not video_dir.is_absolute():
                video_dir = PROJECT_ROOT / video_dir
            if args.video_dir is None:
                video_dir = video_dir / time.strftime("rollout_%Y%m%d_%H%M%S")
            video_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[video] Recording per-camera mp4s to {video_dir} @ {video_fps:.1f} fps"
            )

        def inference_worker() -> None:
            while not stop_event.is_set():
                loop_start = time.monotonic()

                # ── gather camera obs + timestamps ────────────────────────────
                camera_obs: Optional[Dict[str, np.ndarray]] = None
                camera_timestamps: Dict[str, np.ndarray] = {}
                anchor_timestamps: Optional[np.ndarray] = None

                if camera_streamer is not None:
                    try:
                        camera_result = camera_streamer.get_observation_window(
                            horizon=args.camera_horizon,
                            stride=args.camera_stride,
                            obs_frequency=args.obs_frequency,
                            include_timestamps=True,
                        )
                        camera_obs, camera_timestamps = camera_result
                    except RuntimeError as exc:
                        print(f"[camera] {exc}")

                # ── build synchronized observation window ─────────────────────
                # Mirror rby1_wbc_policy.py exactly: correct camera timestamps
                # by measured latency, use the latest (slowest camera) as the
                # anchor, then interpolate proprioception to that anchor.
                if camera_obs is not None and camera_timestamps:
                    adjusted_ts = {
                        k: np.asarray(ts, dtype=float)
                        - _camera_latency_for(
                            k, args.camera_latency, camera_latency_overrides
                        )
                        for k, ts in camera_timestamps.items()
                    }
                    anchor_timestamps = _merge_camera_timestamps(adjusted_ts)
                    query_timestamps = anchor_timestamps + float(
                        args.proprioception_latency
                    )
                else:
                    # No cameras: anchor at now; sample_joint_observations_at
                    # clamps to the latest buffer entry.
                    anchor_timestamps = np.array([time.monotonic()], dtype=float)
                    query_timestamps = anchor_timestamps

                # Interpolate joint-state obs to query_timestamps from the buffer.
                ik = getattr(robot._backend, "ik_solver", None) or getattr(
                    robot._backend, "_ik", None
                )  # noqa: SLF001
                if ik is None:
                    print("[robot] No IK solver found on backend")
                    if stop_event.wait(timeout=control_dt):
                        break
                    continue
                try:
                    robot_obs = robot.sample_joint_observations_at(query_timestamps, ik)
                except Exception as exc:
                    print(f"[robot] Failed to align joint observations: {exc}")
                    if stop_event.wait(timeout=control_dt):
                        break
                    continue

                # Base SE2 obs: interpolated from the same buffer timestamps.
                # We read the latest value — base pose changes slowly relative
                # to the obs window, and the buffer doesn't store SE2 history.
                base_T = robot.get_current_base_SE2()
                if base_T is not None:
                    x = float(base_T[0, 2])
                    y = float(base_T[1, 2])
                    yaw = math.atan2(float(base_T[1, 0]), float(base_T[0, 0]))
                    n = query_timestamps.size
                    robot_obs["body_state_pos_xyz"] = np.tile(
                        np.array([x, y, 0.0], dtype=float), (n, 1)
                    )
                    robot_obs["body_state_quat_wxyz"] = np.tile(
                        _yaw_quat_wxyz(yaw), (n, 1)
                    )

                # Torque history: sample the ring buffer at the camera cadence
                # ending at the most-recent anchor. Mirrors training, which
                # resamples torque to the camera master clock and keeps the
                # most-recent torque_obs_horizon samples.
                if torque_horizon is not None:
                    anchor = float(query_timestamps[-1])
                    dt_cam = float(robot.dt * args.obs_frequency)
                    torque_query_ts = anchor - dt_cam * np.arange(
                        torque_horizon - 1, -1, -1
                    )
                    try:
                        robot_obs.update(robot.sample_torque_at(torque_query_ts))
                    except Exception as exc:
                        print(f"[torque] Skipping cycle -- sample failed: {exc}")
                        if stop_event.wait(timeout=control_dt):
                            break
                        continue

                obs_dict: Dict[str, np.ndarray] = dict(robot_obs)
                if camera_obs is not None:
                    obs_dict.update(camera_obs)
                obs_dict["timestamp"] = anchor_timestamps

                # ── append per-camera video frames ────────────────────────────
                if args.record_video and video_dir is not None:
                    try:
                        import cv2

                        for k, v in obs_dict.items():
                            if not k.endswith("_rgb"):
                                continue
                            arr = np.asarray(v)
                            if arr.ndim == 4:
                                frame = arr[-1]
                            elif arr.ndim == 3:
                                frame = arr
                            else:
                                continue
                            writer = video_writers.get(k)
                            if writer is None:
                                h, w = frame.shape[:2]
                                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                                writer = cv2.VideoWriter(
                                    str(video_dir / f"{k}.mp4"),
                                    fourcc,
                                    video_fps,
                                    (w, h),
                                )
                                if not writer.isOpened():
                                    print(f"[video] Failed to open writer for {k}")
                                    continue
                                video_writers[k] = writer
                            writer.write(frame)

                            # Mirror the server's resize/pad or black-crop+resize
                            # so the *_processed mp4 shows what the model sees.
                            is_wrist = k in _WRIST_CAMERA_KEYS
                            processed_frame, new_box = _preprocess_for_model(
                                frame,
                                processed_out_wh,
                                is_wrist=is_wrist,
                                crop_box=(
                                    processed_crop_boxes.get(k) if is_wrist else None
                                ),
                            )
                            if is_wrist:
                                processed_crop_boxes[k] = new_box
                            proc_writer = processed_video_writers.get(k)
                            if proc_writer is None:
                                ph, pw = processed_frame.shape[:2]
                                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                                proc_writer = cv2.VideoWriter(
                                    str(video_dir / f"{k}_processed.mp4"),
                                    fourcc,
                                    video_fps,
                                    (pw, ph),
                                )
                                if not proc_writer.isOpened():
                                    print(
                                        f"[video] Failed to open processed writer for {k}"
                                    )
                                    continue
                                processed_video_writers[k] = proc_writer
                            proc_writer.write(processed_frame)
                    except Exception as exc:
                        print(f"[video] Frame write failed: {exc}")

                # ── inference ─────────────────────────────────────────────────
                reply = policy_client.infer(obs_dict)
                infer_end = time.monotonic()

                if reply is None or "actions" not in reply:
                    print("[policy] Inference timeout or malformed reply")
                    if stop_event.wait(timeout=control_dt):
                        break
                    continue

                # Only execute the first N steps of each chunk (16 = full chunk).
                chunk_limit = 16
                actions_raw = {
                    k: (np.asarray(v)[:chunk_limit] if v is not None else None)
                    for k, v in reply["actions"].items()
                }
                action_timestamps = np.asarray(
                    reply.get("timestamps", []), dtype=float
                )[:chunk_limit]

                base = actions_raw.get("base_xy_yaw")
                if base is not None and len(base):
                    cur = robot.get_current_base_SE2()
                    cx = float(cur[0, 2]) if cur is not None else 0.0
                    cy = float(cur[1, 2]) if cur is not None else 0.0
                    cyaw = (
                        math.atan2(float(cur[1, 0]), float(cur[0, 0]))
                        if cur is not None
                        else 0.0
                    )
                    b0, bL = base[0], base[-1]
                    print(
                        f"[base] cur=({cx:+.3f},{cy:+.3f},{cyaw:+.3f}) "
                        f"a0=({b0[0]:+.3f},{b0[1]:+.3f},{b0[2]:+.3f}) "
                        f"aN=({bL[0]:+.3f},{bL[1]:+.3f},{bL[2]:+.3f}) "
                        f"dx={bL[0]-cx:+.3f} dy={bL[1]-cy:+.3f} dyaw={bL[2]-cyaw:+.3f}"
                    )

                # ── stale-action filtering ─────────────────────────
                arm_cutoff = float(infer_end + max(args.arm_execution_latency, 0.0))
                if action_timestamps.size:
                    finite_ts = action_timestamps.reshape(-1)
                    valid_mask = np.isfinite(finite_ts) & (finite_ts > arm_cutoff)
                    if not np.any(valid_mask):
                        if not args.dry_run:
                            print(
                                f"[policy] Dropping chunk; all timestamps precede execution cutoff "
                                f"(>{arm_cutoff:.3f}s)."
                            )
                        # In dry-run, don't drop obs — save for debugging
                        if not args.dry_run:
                            if stop_event.wait(timeout=control_dt):
                                break
                            continue
                    else:
                        first_valid = int(np.argmax(valid_mask))
                        if first_valid > 0 and not args.dry_run:
                            print(f"[policy] Skipping {first_valid} stale actions.")
                            action_timestamps = action_timestamps[first_valid:]
                            actions_raw = {
                                k: (np.asarray(v)[first_valid:] if v is not None else v)
                                for k, v in actions_raw.items()
                            }

                # ── gripper width offset ──────────────────────────────────────
                for key in ("left_gripper_width", "right_gripper_width"):
                    if key in actions_raw and actions_raw[key] is not None:
                        actions_raw[key] = np.clip(
                            np.asarray(actions_raw[key], dtype=float)
                            - args.gripper_width_offset,
                            *GRIPPER_WIDTH_LIMITS,
                        )

                # ── schedule or save (dry-run) ────────────────────────────────
                scheduled = build_scheduled_actions(
                    actions=actions_raw,
                    timestamps=action_timestamps,
                    fallback_dt=control_dt,
                    now=time.monotonic(),
                )

                # In dry-run, always save obs + actions (even if stale)
                if args.dry_run and dry_run_dir is not None:
                    cycle_tag = f"cycle_{int(loop_start * 1e6):020d}"
                    dump_path = dry_run_dir / f"{cycle_tag}.npz"
                    try:
                        flat: Dict[str, np.ndarray] = {}
                        for k, v in obs_dict.items():
                            flat[f"obs__{k}"] = np.asarray(v)
                        for k, v in actions_raw.items():
                            if v is not None:
                                flat[f"act__{k}"] = np.asarray(v)
                        flat["action_timestamps"] = np.asarray(action_timestamps)
                        flat["infer_end_monotonic"] = np.asarray([infer_end])
                        flat["arm_cutoff"] = np.asarray([arm_cutoff])
                        np.savez_compressed(dump_path, **flat)
                        print(f"[dry-run] Saved {dump_path}")
                    except Exception as exc:
                        print(f"[dry-run] Failed to write {dump_path}: {exc}")

                    try:
                        import cv2  # local import: only needed in dry-run

                        for k, v in obs_dict.items():
                            if not k.endswith("_rgb"):
                                continue
                            arr = np.asarray(v)
                            if arr.ndim == 4:
                                frame = arr[-1]
                            elif arr.ndim == 3:
                                frame = arr
                            else:
                                continue
                            png_path = dry_run_dir / f"{cycle_tag}__{k}.png"
                            cv2.imwrite(str(png_path), frame)
                    except Exception as exc:
                        print(f"[dry-run] Failed to write image PNGs: {exc}")
                elif scheduled:
                    if args.dry_run:
                        if dry_run_dir is not None:
                            dump_path = (
                                dry_run_dir / f"cycle_{int(loop_start * 1e6):020d}.npz"
                            )
                            try:
                                flat: Dict[str, np.ndarray] = {}
                                for k, v in obs_dict.items():
                                    flat[f"obs__{k}"] = np.asarray(v)
                                for k, v in actions_raw.items():
                                    if v is not None:
                                        flat[f"act__{k}"] = np.asarray(v)
                                flat["action_timestamps"] = np.asarray(
                                    action_timestamps
                                )
                                flat["infer_end_monotonic"] = np.asarray([infer_end])
                                np.savez_compressed(dump_path, **flat)
                            except Exception as exc:
                                print(f"[dry-run] Failed to write {dump_path}: {exc}")
                    else:
                        cmd_thread.queue_actions(scheduled)

                elapsed = time.monotonic() - loop_start
                wait_time = max(0.0, inference_period - elapsed)
                tag = "[dry-run]" if args.dry_run else "[policy]"
                print(
                    f"{tag} Inference cycle {elapsed:.3f}s, "
                    f"next in {wait_time:.3f}s, "
                    f"chunk={len(scheduled)} steps"
                )
                if stop_event.wait(timeout=wait_time):
                    break

        inf_thread = threading.Thread(
            target=inference_worker, name="policy-inference", daemon=True
        )
        inf_thread.start()

        try:
            while inf_thread.is_alive():
                if stop_event.wait(timeout=0.1):
                    break
        except KeyboardInterrupt:
            print("[main] Interrupted, shutting down...")

    finally:
        stop_event.set()
        # Wait for the inference thread to exit before tearing down resources
        # it touches — VideoWriter in particular is not thread-safe, and a
        # concurrent write()/release() drops the mp4's trailer.
        if "inf_thread" in locals() and inf_thread.is_alive():
            # Generous timeout: the worker may be mid-infer() blocked on ZMQ
            # (bounded by --policy-timeout, default 2s).
            inf_thread.join(timeout=max(args.policy_timeout + 1.0, 3.0))
        cmd_thread.stop()
        if policy_client is not None:
            policy_client.close()
        if camera_streamer is not None:
            camera_streamer.stop()
        for writer in list(video_writers.values()) + list(
            processed_video_writers.values()
        ):
            try:
                writer.release()
            except Exception:
                pass
        if (video_writers or processed_video_writers) and video_dir is not None:
            n_total = len(video_writers) + len(processed_video_writers)
            print(f"[video] Wrote {n_total} mp4(s) to {video_dir}")
        robot.stop()


if __name__ == "__main__":
    main()
