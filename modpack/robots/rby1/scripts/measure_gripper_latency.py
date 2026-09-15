#!/usr/bin/env python3
"""Estimate gripper execution latency via sinusoidal commands."""

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
    """Cross-correlate command and measured gripper width to recover delay."""
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

    # Oversample the finest stream to improve correlation resolution.
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
    plt.plot(cmd_t_rel, cmd_v, label="Command width", linewidth=2)
    plt.plot(meas_t_rel, meas_v, label="Measured width", linewidth=2, alpha=0.8)
    if latency is not None:
        direction = "lag" if latency > 0 else "lead"
        plt.title(f"Gripper latency ~ {abs(latency):.3f}s ({direction})")
    else:
        plt.title("Gripper command vs measurement")
    plt.xlabel("Time since start [s]")
    plt.ylabel("Width [m]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure gripper execution latency")
    parser.add_argument("--duration", type=float, default=20.0, help="Total duration in seconds")
    parser.add_argument("--frequency", type=float, default=0.5, help="Sine wave frequency in Hz")
    parser.add_argument("--amplitude", type=float, default=0.02, help="Sine wave amplitude in meters")
    parser.add_argument("--offset", type=float, default=0.04, help="Sine wave offset in meters")
    parser.add_argument("--sample-rate", type=float, default=10.0, help="Command/update rate in Hz")
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

    stop_flag: Dict[str, bool] = {"stop": False}
    _install_sigint_handler(stop_flag)

    command_period = 1.0 / max(args.sample_rate, 1e-3)
    start_time = time.monotonic()
    commands: list[Tuple[float, float]] = []
    measurements: list[Tuple[float, float]] = []

    print("Streaming sinusoidal gripper commands...")
    while not stop_flag["stop"] and (time.monotonic() - start_time) < args.duration:
        now = time.monotonic()
        phase = now - start_time
        target_width = args.offset + args.amplitude * math.sin(2.0 * math.pi * args.frequency * phase)
        target_width = float(np.clip(target_width, 0.0, 0.085))
        payload = {
            "left_tf": left_tf,
            "right_tf": right_tf,
            "head_tf": head_tf,
            "left_gripper_width": target_width,
            "right_gripper_width": target_width,
        }
        robot.apply_action(payload, duration=command_period, timestamp=now)
        commands.append((now, target_width))

        obs = robot.get_observation_window(horizon=1)
        measurements.append(
            (
                float(obs["timestamp"][-1]),
                float(np.asarray(obs["gripper_left_gripper_width"][-1]).item()),
            )
        )

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
        print(f"Estimated gripper execution latency: {abs(latency):.4f}s ({direction})")

    if args.output_csv:
        out_path = Path(args.output_csv)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["command_timestamp", "command_width", "measurement_timestamp", "measured_width"])
            for (t_cmd, cmd), (t_meas, meas) in zip(commands, measurements):
                writer.writerow([t_cmd, cmd, t_meas, meas])
        print(f"Wrote {len(commands)} samples to {out_path}")

    if args.output_plot:
        _render_plot(commands, measurements, latency, Path(args.output_plot))
        print(f"Wrote plot to {args.output_plot}")


if __name__ == "__main__":
    main()
