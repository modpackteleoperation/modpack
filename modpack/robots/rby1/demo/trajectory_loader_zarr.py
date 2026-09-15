"""Load a robologger zarr episode directory for joint-space trajectory replay."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import zarr
from scipy.interpolate import make_interp_spline

REPLAY_HZ = 100.0


def _deduplicate_samples(
    data: np.ndarray,
    ts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sort samples by timestamp and keep the last sample for duplicate times."""
    data = np.asarray(data)
    ts = np.asarray(ts, dtype=np.float64).reshape(-1)

    if data.shape[0] != ts.shape[0]:
        raise ValueError(
            f"Sample count mismatch: data has {data.shape[0]} rows but timestamps has {ts.shape[0]} entries."
        )
    if ts.size == 0:
        raise ValueError("Cannot resample an empty trajectory.")

    order = np.argsort(ts, kind="stable")
    sorted_ts = ts[order]
    sorted_data = data[order]

    keep_mask = np.ones(sorted_ts.shape[0], dtype=bool)
    keep_mask[:-1] = sorted_ts[:-1] != sorted_ts[1:]

    dedup_ts = sorted_ts[keep_mask]
    dedup_data = sorted_data[keep_mask]
    if dedup_ts.size < 2:
        raise ValueError("Need at least two unique timestamps to resample trajectory data.")
    return dedup_data, dedup_ts


def _resample(data: np.ndarray, src_ts: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    data, src_ts = _deduplicate_samples(data, src_ts)
    if data.ndim == 1:
        return make_interp_spline(src_ts, data, k=1)(target_ts)
    return np.stack(
        [make_interp_spline(src_ts, data[:, i], k=1)(target_ts) for i in range(data.shape[1])],
        axis=1,
    )


def _open(zarr_path: str, dataset: str) -> np.ndarray:
    return np.array(zarr.open_group(zarr_path, mode="r")[dataset])


def _load_arm(episode_dir: Path, side: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (joints, joint_ts, gripper, gripper_ts) for one arm."""
    arm_path = str(episode_dir / f"{side}_arm.zarr")
    if not os.path.exists(arm_path):
        raise FileNotFoundError(f"Required zarr store not found: {arm_path}")
    joints = _open(arm_path, "target_joint_pos")   # (T, 7)
    joint_ts = _open(arm_path, "target_timestamps") # (T,)

    ee_path = str(episode_dir / f"{side}_end_effector.zarr")
    if not os.path.exists(ee_path):
        raise FileNotFoundError(f"Required zarr store not found: {ee_path}")
    gripper = _open(ee_path, "target_joint_pos")    # (T, 1)
    gripper_ts = _open(ee_path, "target_timestamps")

    return joints, joint_ts, gripper, gripper_ts


def _load_head(episode_dir: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    path = str(episode_dir / "head.zarr")
    if not os.path.exists(path):
        return None, None
    try:
        joints = _open(path, "target_joint_pos")    # (T, 2)
        ts = _open(path, "target_timestamps")
        return joints, ts
    except Exception:
        return None, None


def _load_body(episode_dir: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    path = str(episode_dir / "body.zarr")
    if not os.path.exists(path):
        return None, None
    try:
        pose = _open(path, "target_pose")           # (T, 3) [x, y, yaw]
        ts = _open(path, "target_timestamps")
        return pose, ts
    except Exception:
        return None, None


def load_trajectory_zarr(
    episode_dir,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load a robologger episode directory and return per-timestep target lists at 50 Hz.

    All streams are resampled onto a 50 Hz output grid derived from the
    overlapping time range of left_arm and right_arm target timestamps.

    Args:
        episode_dir: Path to episode directory containing <name>.zarr stores.

    Returns:
        qpos_list:   list of dicts with "left_arm" (7,), "right_arm" (7,),
                     and optionally "head" (2,).
        widths_list: list of dicts with "left_width" and "right_width" (floats, metres).
        base_list:   list of dicts with "x", "y", "yaw" (floats).
                     Empty list if no body.zarr is present.
    """
    episode_dir = Path(episode_dir)

    left_joints,  left_ts,  left_grip,  left_grip_ts  = _load_arm(episode_dir, "left")
    right_joints, right_ts, right_grip, right_grip_ts = _load_arm(episode_dir, "right")
    left_joints, left_ts = _deduplicate_samples(left_joints, left_ts)
    right_joints, right_ts = _deduplicate_samples(right_joints, right_ts)

    # Main timeline: left_arm timestamps clipped to overlap with right_arm
    t_start = max(left_ts[0], right_ts[0])
    t_end   = min(left_ts[-1], right_ts[-1])
    if t_start >= t_end:
        raise ValueError("left_arm and right_arm timestamps do not overlap.")
    main_ts = left_ts[(left_ts >= t_start) & (left_ts <= t_end)]

    # Output grid at replay frequency
    output_ts = np.arange(main_ts[0], main_ts[-1], 1.0 / REPLAY_HZ)

    left_joints  = _resample(left_joints,  left_ts,       output_ts)
    right_joints = _resample(right_joints, right_ts,      output_ts)
    left_grip    = _resample(left_grip,    left_grip_ts,  output_ts)
    right_grip   = _resample(right_grip,   right_grip_ts, output_ts)

    head_joints_raw, head_ts_raw = _load_head(episode_dir)
    head_joints: Optional[np.ndarray] = None
    if head_joints_raw is not None and head_ts_raw is not None:
        ht_start = max(output_ts[0], head_ts_raw[0])
        ht_end   = min(output_ts[-1], head_ts_raw[-1])
        if ht_start < ht_end:
            head_joints = _resample(head_joints_raw, head_ts_raw, output_ts)

    body_pose_raw, body_ts_raw = _load_body(episode_dir)
    body_pose: Optional[np.ndarray] = None
    if body_pose_raw is not None and body_ts_raw is not None:
        bt_start = max(output_ts[0], body_ts_raw[0])
        bt_end   = min(output_ts[-1], body_ts_raw[-1])
        if bt_start < bt_end:
            body_pose = _resample(body_pose_raw, body_ts_raw, output_ts)

    T = len(output_ts)
    qpos_list:   List[Dict] = []
    widths_list: List[Dict] = []
    base_list:   List[Dict] = []

    for i in range(T):
        q: Dict = {
            "left_arm":  left_joints[i].copy(),
            "right_arm": right_joints[i].copy(),
        }
        if head_joints is not None:
            q["head"] = head_joints[i].copy()
        qpos_list.append(q)

        widths_list.append({
            "left_width":  float(left_grip[i, 0]),
            "right_width": float(right_grip[i, 0]),
        })

        if body_pose is not None:
            base_list.append({
                "x":   float(body_pose[i, 0]),
                "y":   float(body_pose[i, 1]),
                "yaw": float(body_pose[i, 2]),
            })

    return qpos_list, widths_list, base_list
