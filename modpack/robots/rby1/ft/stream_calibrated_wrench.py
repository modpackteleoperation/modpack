#!/usr/bin/env python3
"""Continuously print calibrated FT wrenches using ft_sensor.yaml parameters."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from control.rby1_wbc import RBY1WBC


def _parse_tilts(spec: str) -> List[Tuple[float, float]]:
    tilts: List[Tuple[float, float]] = []
    for segment in spec.split(";"):
        segment = segment.strip()
        if not segment:
            continue
        parts = segment.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid tilt segment '{segment}'. Expected 'x,y'.")
        tilts.append((float(parts[0]), float(parts[1])))
    return tilts


@dataclass
class FTCALParams:
    force_offset: np.ndarray
    torque_offset: np.ndarray
    gravity: np.ndarray
    com: np.ndarray
    force_transform: np.ndarray
    torque_transform: np.ndarray


@dataclass
class PhaseInterval:
    name: str
    start: float
    end: float


def _load_ft_config(path: Path) -> Dict[str, FTCALParams]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    params: Dict[str, FTCALParams] = {}
    for arm in ("left", "right"):
        entry = data.get(arm)
        if entry is None:
            continue
        params[arm] = FTCALParams(
            force_offset=np.array(
                [entry["offset"]["fx"], entry["offset"]["fy"], entry["offset"]["fz"]], dtype=float
            ),
            torque_offset=np.array(
                [entry["offset"]["tx"], entry["offset"]["ty"], entry["offset"]["tz"]], dtype=float
            ),
            gravity=np.array([entry["gravity"]["x"], entry["gravity"]["y"], entry["gravity"]["z"]], dtype=float),
            com=np.array([entry["com"]["x"], entry["com"]["y"], entry["com"]["z"]], dtype=float),
            force_transform=np.array(entry["force_transform"], dtype=float),
            torque_transform=np.array(entry.get("torque_transform", np.eye(3)), dtype=float),
        )
    return params


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()


def _wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    return np.array([quat[1], quat[2], quat[3], quat[0]], dtype=float)


def _xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    return np.array([quat[3], quat[0], quat[1], quat[2]], dtype=float)


def _apply_global_tilts(initial_quat: np.ndarray, tilt_x: float, tilt_y: float) -> np.ndarray:
    base = Rotation.from_quat(_wxyz_to_xyzw(initial_quat))
    tilt = Rotation.from_euler("xy", [tilt_x, tilt_y], degrees=True)
    result = tilt * base
    return _normalize(_xyzw_to_wxyz(result.as_quat()))


def _normalize(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=float)
    norm = np.linalg.norm(q)
    if norm < 1e-9:
        raise ValueError("Quaternion norm is zero")
    return q / norm


def _transform_wrench(raw: np.ndarray, params: FTCALParams) -> np.ndarray:
    force = params.force_transform @ raw[:3]
    torque = params.torque_transform @ raw[3:]
    return np.concatenate([force, torque])


def _calibrate_wrench(wrench: np.ndarray, rot_world_from_tool: np.ndarray, params: FTCALParams) -> np.ndarray:
    g_tool = rot_world_from_tool.T @ params.gravity
    force = wrench[:3] + params.force_offset - g_tool
    torque = wrench[3:] + params.torque_offset - np.cross(params.com, g_tool)
    return np.concatenate([force, torque])


def _format_wrench(vec: np.ndarray) -> str:
    force = vec[:3]
    torque = vec[3:]
    return (
        f"F=({force[0]: .3f}, {force[1]: .3f}, {force[2]: .3f}) "
        f"T=({torque[0]: .3f}, {torque[1]: .3f}, {torque[2]: .3f})"
    )


def _plot_samples(
    times: np.ndarray,
    raw: np.ndarray,
    calibrated: np.ndarray,
    title: str,
    phases: List[PhaseInterval],
) -> "plt.Figure":
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(times, raw[:, 0], label="Fx")
    axes[0].plot(times, raw[:, 1], label="Fy")
    axes[0].plot(times, raw[:, 2], label="Fz")
    axes[0].set_ylabel("Raw Force (N)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    axes[1].plot(times, raw[:, 3], label="Tx")
    axes[1].plot(times, raw[:, 4], label="Ty")
    axes[1].plot(times, raw[:, 5], label="Tz")
    axes[1].set_ylabel("Raw Torque (N·m)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    axes[2].plot(times, calibrated[:, 0], label="Fx")
    axes[2].plot(times, calibrated[:, 1], label="Fy")
    axes[2].plot(times, calibrated[:, 2], label="Fz")
    axes[2].set_ylabel("Cal Force (N)")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, linestyle="--", alpha=0.3)

    axes[3].plot(times, calibrated[:, 3], label="Tx")
    axes[3].plot(times, calibrated[:, 4], label="Ty")
    axes[3].plot(times, calibrated[:, 5], label="Tz")
    axes[3].set_ylabel("Cal Torque (N·m)")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, linestyle="--", alpha=0.3)

    colors = {"move": "tab:orange", "stable": "tab:green"}
    for phase in phases:
        phase_type = "move" if "move" in phase.name.lower() else "stable"
        color = colors[phase_type]
        for ax in axes:
            ax.axvspan(phase.start, phase.end, color=color, alpha=0.12)
        axes[0].text(
            0.5 * (phase.start + phase.end),
            axes[0].get_ylim()[1],
            phase.name,
            ha="center",
            va="bottom",
            fontsize=9,
            color=color,
        )
    fig.suptitle(f"Wrench Trace - {title}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _can_show_plots() -> bool:
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    import matplotlib
    from matplotlib.rcsetup import interactive_bk
    return matplotlib.get_backend().lower() in {b.lower() for b in interactive_bk}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream calibrated FT readings across multiple orientations.")
    parser.add_argument("--wbc-config", default=str(Path(PROJECT_ROOT, "config/wbc.yaml")), help="Path to WBC config.")
    parser.add_argument("--ft-config", default=str(Path(PROJECT_ROOT, "config/ft_sensor.yaml")), help="Path to ft_sensor.yaml.")
    parser.add_argument("--arm", choices=("left", "right"), default="right", help="Which arm to move and record.")
    parser.add_argument("--rate", type=float, default=100.0, help="Sampling frequency (Hz).")
    parser.add_argument("--stable-duration", type=float, default=5.0, help="Seconds to remain still per stable phase.")
    parser.add_argument("--move-duration", type=float, default=5.0, help="Seconds allocated to each move.")
    parser.add_argument(
        "--tilts",
        type=str,
        default="15,0;0,15;-15,-15",
        help="Semicolon separated world-frame tilt degrees `x,y` for successive orientations.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable matplotlib plotting (headless-safe).")
    parser.add_argument("--save-plot", type=str, default="", help="Save plot to this path instead of showing a window.")
    args = parser.parse_args()

    ft_params = _load_ft_config(Path(args.ft_config))
    if not ft_params:
        raise RuntimeError("No FT parameters loaded; please populate ft_sensor.yaml.")
    if args.arm not in ft_params:
        raise RuntimeError(f"No FT parameters found for arm '{args.arm}'.")

    tilts = _parse_tilts(args.tilts)

    wbc = RBY1WBC(config_path=args.wbc_config)
    wbc.start()
    try:
        period = 1.0 / max(args.rate, 0.1)
        snapshot = wbc.wait_for_first_state(timeout_sec=10.0)
        poses0 = wbc.get_end_effector_pose(snapshot)
        if poses0 is None:
            raise RuntimeError("Unable to read initial end-effector pose.")
        left_pose0, right_pose0 = poses0
        left_pos = left_pose0[:3].copy()
        right_pos = right_pose0[:3].copy()
        left_quat = left_pose0[3:].copy()
        right_quat = right_pose0[3:].copy()
        left_width, right_width = wbc.get_latest_gripper_widths()

        wbc.update_targets(
            left_pos,
            left_quat,
            right_pos,
            right_quat,
            left_width=left_width,
            right_width=right_width,
            duration=max(0.5, args.stable_duration),
        )

        arm_quat = left_quat if args.arm == "left" else right_quat
        orientations = [_normalize(arm_quat)]
        for tx, ty in tilts:
            orientations.append(_apply_global_tilts(arm_quat, tx, ty))

        t0 = time.monotonic()
        samples_time: List[float] = []
        samples_raw: List[np.ndarray] = []
        samples_cal: List[np.ndarray] = []
        phases: List[PhaseInterval] = []

        def collect_sample(timestamp: float) -> None:
            snapshot_local = wbc.get_latest_robot_state()
            if snapshot_local is None or not snapshot_local.is_valid:
                return
            poses = wbc.get_end_effector_pose(snapshot_local)
            if poses is None:
                return
            left_pose, right_pose = poses
            if args.arm == "left" and snapshot_local.left_ft_valid and "left" in ft_params:
                params = ft_params["left"]
                raw = np.asarray(snapshot_local.left_ee_wrench, dtype=float)
                wrench_tool = _transform_wrench(raw, params)
                rot = _quat_wxyz_to_matrix(left_pose[3:])
                calibrated = _calibrate_wrench(wrench_tool, rot, params)
                samples_raw.append(wrench_tool.copy())
                samples_cal.append(calibrated.copy())
                samples_time.append(timestamp)
            if args.arm == "right" and snapshot_local.right_ft_valid and "right" in ft_params:
                params = ft_params["right"]
                raw = np.asarray(snapshot_local.right_ee_wrench, dtype=float)
                wrench_tool = _transform_wrench(raw, params)
                rot = _quat_wxyz_to_matrix(right_pose[3:])
                calibrated = _calibrate_wrench(wrench_tool, rot, params)
                samples_raw.append(wrench_tool.copy())
                samples_cal.append(calibrated.copy())
                samples_time.append(timestamp)

        def run_phase(name: str, duration: float, command: Optional[Callable[[], None]] = None) -> None:
            start = time.monotonic()
            if command is not None:
                command()
            while True:
                now = time.monotonic()
                if now - start >= duration:
                    break
                collect_sample(now - t0)
                time.sleep(period)
            phases.append(PhaseInterval(name=name, start=start - t0, end=time.monotonic() - t0))

        print("Holding initial orientation...")
        run_phase("Stable 0", args.stable_duration)

        left_target_quat = left_quat.copy()
        right_target_quat = right_quat.copy()

        for idx, target_quat in enumerate(orientations[1:], start=1):
            def make_command(quat: np.ndarray):
                def _cmd():
                    nonlocal left_target_quat, right_target_quat
                    if args.arm == "left":
                        left_target_quat = quat.copy()
                    else:
                        right_target_quat = quat.copy()
                    wbc.update_targets(
                        left_pos,
                        left_target_quat,
                        right_pos,
                        right_target_quat,
                        left_width=left_width,
                        right_width=right_width,
                        duration=args.move_duration,
                    )
                return _cmd

            run_phase(f"Move {idx}", args.move_duration, make_command(target_quat))
            run_phase(f"Stable {idx}", args.stable_duration)

        if samples_cal:
            title = "Left Arm" if args.arm == "left" else "Right Arm"
            fig = _plot_samples(
                np.array(samples_time),
                np.array(samples_raw),
                np.array(samples_cal),
                title,
                phases,
            )
            save_path = args.save_plot.strip()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"Saved plot to {save_path}")

        if not args.no_plot:
            if _can_show_plots():
                import matplotlib.pyplot as plt

                plt.show()
            elif not args.save_plot.strip():
                print("No display detected; skipping plot window. Use --save-plot PATH or --no-plot.")
    except KeyboardInterrupt:
        print("\nStopping stream early...")
    finally:
        wbc.stop()


if __name__ == "__main__":
    main()
