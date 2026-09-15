#!/usr/bin/env python3
"""Estimate arm execution latency by tracking a sinusoidal end-effector path."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Dict, MutableMapping, Sequence, Tuple

import numpy as np
from scipy import signal
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.rby1_policy import RBY1PolicyRobot


def _install_sigint_handler(stop_flag: MutableMapping[str, bool]) -> None:
    import signal

    def _handler(signum: int, frame: object | None) -> None:
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _estimate_latency(
    commands: Sequence[Tuple[float, float]],
    measurements: Sequence[Tuple[float, float]],
) -> float | None:
    """Cross-correlate command and measured offsets to recover delay."""
    cmd_t, cmd_v = map(np.asarray, zip(*commands))
    meas_t, meas_v = map(np.asarray, zip(*measurements))

    start = max(cmd_t[0], meas_t[0])
    end = min(cmd_t[-1], meas_t[-1])
    if end <= start:
        return None

    cmd_dt = np.median(np.diff(cmd_t)) if cmd_t.size > 1 else None
    meas_dt = np.median(np.diff(meas_t)) if meas_t.size > 1 else None
    finite_dts = [dt for dt in (cmd_dt, meas_dt) if dt and dt > 0]
    if not finite_dts:
        return None

    sample_period = max(0.001, min(finite_dts) / 4.0)
    grid = np.arange(start, end, sample_period)

    cmd_series = np.interp(grid, cmd_t, cmd_v)
    meas_series = np.interp(grid, meas_t, meas_v)

    cmd_zero = cmd_series - np.mean(cmd_series)
    meas_zero = meas_series - np.mean(meas_series)

    corr = signal.correlate(meas_zero, cmd_zero, mode="full")
    lags = signal.correlation_lags(meas_zero.size, cmd_zero.size, mode="full")
    best_lag = lags[int(np.argmax(corr))]
    return best_lag * sample_period


def _render_plot(
    commands: Sequence[Tuple[float, float]],
    measurements: Sequence[Tuple[float, float]],
    latency: float | None,
    axis: str,
    arm: str,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmd_t, cmd_v = zip(*commands)
    meas_t, meas_v = zip(*measurements)
    t0 = min(cmd_t[0], meas_t[0])
    cmd_t_rel = np.array(cmd_t) - t0
    meas_t_rel = np.array(meas_t) - t0

    plt.figure(figsize=(10, 4))
    plt.plot(cmd_t_rel, cmd_v, label="Command offset", linewidth=2)
    plt.plot(meas_t_rel, meas_v, label="Measured offset", linewidth=2, alpha=0.8)
    if latency is not None:
        direction = "lag" if latency > 0 else "lead"
        plt.title(f"{arm.title()} arm latency ~ {abs(latency):.3f}s ({direction}) on {axis}-axis")
    else:
        plt.title(f"{arm.title()} arm command vs measurement on {axis}-axis")
    plt.xlabel("Time since start [s]")
    plt.ylabel("Offset [m]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure robot execution latency")
    parser.add_argument("--duration", type=float, default=20.0, help="Total duration in seconds")
    parser.add_argument("--frequency", type=float, default=1.0, help="Sine wave frequency in Hz")
    parser.add_argument("--amplitude", type=float, default=0.03, help="Sine wave amplitude in meters (or radians for head)")
    parser.add_argument("--sample-rate", type=float, default=20.0, help="Command/update rate in Hz")
    parser.add_argument("--axis", choices=["x", "y", "z"], default="x", help="Axis to oscillate (arms)")
    parser.add_argument("--arm", choices=["left", "right", "head"], default="right", help="Effector to command")
    parser.add_argument(
        "--head-axis",
        choices=["roll", "pitch", "yaw"],
        default="yaw",
        help="Rotation axis to oscillate when --arm=head",
    )
    parser.add_argument("--use-sim", action="store_true", help="Run against the MuJoCo simulation backend")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional CSV path for debugging traces")
    parser.add_argument("--output-plot", type=str, default=None, help="Optional PNG path for plotting traces")
    args = parser.parse_args()

    robot = RBY1PolicyRobot(use_sim=args.use_sim)
    robot.start()
    robot.wait_until_ready(timeout=10.0)

    obs = robot.get_observation_window(horizon=1)
    left_tf = obs["gripper_left_tf"][0]
    right_tf = obs["gripper_right_tf"][0]
    head_tf = obs["head_tf"][0]

    active_tf = left_tf if args.arm == "left" else right_tf if args.arm == "right" else head_tf
    base_pose = np.array(active_tf, dtype=float)

    stop_flag: Dict[str, bool] = {"stop": False}
    _install_sigint_handler(stop_flag)

    axis_index = {"x": 0, "y": 1, "z": 2}.get(args.axis, 0)
    command_period = 1.0 / max(args.sample_rate, 1e-3)
    start_time = time.monotonic()
    commands: list[Tuple[float, float]] = []
    measurements: list[Tuple[float, float]] = []

    print(f"Streaming sinusoidal targets on {args.arm} arm along {args.axis}-axis...")
    while not stop_flag["stop"] and (time.monotonic() - start_time) < args.duration:
        now = time.monotonic()
        phase = now - start_time
        offset = args.amplitude * math.sin(2.0 * math.pi * args.frequency * phase)

        if args.arm == "head":
            axis_vec = {"roll": np.array([1.0, 0.0, 0.0]), "pitch": np.array([0.0, 1.0, 0.0]), "yaw": np.array([0.0, 0.0, 1.0])}[
                args.head_axis
            ]
            base_rot = Rotation.from_matrix(base_pose[:3, :3])
            delta_rot = Rotation.from_rotvec(axis_vec * offset)
            target_rot = (delta_rot * base_rot).as_matrix()
            target_tf = np.array(base_pose, dtype=float)
            target_tf[:3, :3] = target_rot
        else:
            target_tf = np.array(base_pose, dtype=float)
            target_tf[axis_index, 3] = base_pose[axis_index, 3] + offset

        payload = {
            "left_tf": target_tf if args.arm == "left" else left_tf,
            "right_tf": target_tf if args.arm == "right" else right_tf,
            "head_tf": target_tf if args.arm == "head" else head_tf,
        }
        robot.apply_action(payload, duration=command_period, timestamp=now)
        commands.append((now, offset))

        obs = robot.get_observation_window(horizon=1)
        if args.arm == "head":
            measured_tf = obs["head_tf"][0]
            rel_rot = Rotation.from_matrix(measured_tf[:3, :3]) * Rotation.from_matrix(base_pose[:3, :3]).inv()
            meas_offset = float(np.dot(rel_rot.as_rotvec(), axis_vec))
        else:
            measured_tf = obs["gripper_left_tf" if args.arm == "left" else "gripper_right_tf"][0]
            meas_offset = float(np.asarray(measured_tf[axis_index, 3] - base_pose[axis_index, 3]).item())
        measurements.append((float(obs["timestamp"][-1]), meas_offset))

        sleep_dt = command_period - (time.monotonic() - now)
        if sleep_dt > 0:
            time.sleep(sleep_dt)

    robot.stop()

    if len(commands) < 2 or len(measurements) < 2:
        print("Insufficient samples collected for latency estimation")
        return

    latency = _estimate_latency(commands, measurements)
    if latency is None:
        print("Unable to estimate latency from collected traces")
    else:
        direction = "lag" if latency > 0 else "lead"
        print(f"Estimated {args.arm} arm execution latency: {abs(latency):.4f}s ({direction})")

    if args.output_csv:
        out_path = Path(args.output_csv)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["command_timestamp", "command_offset", "measurement_timestamp", "measured_offset"])
            for (t_cmd, cmd), (t_meas, meas) in zip(commands, measurements):
                writer.writerow([t_cmd, cmd, t_meas, meas])
        print(f"Wrote {len(commands)} samples to {out_path}")

    if args.output_plot:
        _render_plot(commands, measurements, latency, args.axis, args.arm, Path(args.output_plot))
        print(f"Wrote plot to {args.output_plot}")


if __name__ == "__main__":
    main()

# under 10Hz
# yaw: 0.3006s
# pitch: 0.3254s
