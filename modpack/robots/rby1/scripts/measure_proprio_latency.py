#!/usr/bin/env python3
"""Measure proprioception latency using hardware timestamps or ICMP ping."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, MutableMapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control import Config as ControllerConfig, RealtimeDriver


def _install_sigint_handler(stop_flag: MutableMapping[str, bool]) -> None:
    import signal

    def _handler(signum: int, frame: object | None) -> None:
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _summarize(latencies: np.ndarray) -> str:
    if latencies.size == 0:
        return "no proprioception samples"
    return (
        f"n={latencies.size} min/med/p95/max = "
        f"{latencies.min():.4f}s/{np.median(latencies):.4f}s/"
        f"{np.percentile(latencies, 95):.4f}s/{latencies.max():.4f}s"
    )


def _measure_ping_latency(address: str, count: int) -> float | None:
    proc = subprocess.run(
        ["ping", "-c", str(count), address],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        if "min/avg/max" in line:
            parts = line.split("=")[-1].strip().split("/")
            if len(parts) >= 2:
                try:
                    avg_ms = float(parts[1])
                except ValueError:
                    return None
                return 0.5 * (avg_ms / 1000.0)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure proprioception latency")
    parser.add_argument("--address", default="192.168.30.1:50051", help="Robot gRPC address")
    parser.add_argument("--seconds", type=float, default=10.0, help="Measurement duration")
    parser.add_argument("--ping-address", default=None, help="Optional address to ping when timestamps are unavailable")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional CSV path for samples")
    args = parser.parse_args()

    cfg = ControllerConfig()
    cfg.robot_address = args.address
    driver = RealtimeDriver(cfg)
    driver.start()
    if not driver.wait_until_ready(timeout_sec=15.0):
        driver.stop()
        raise SystemExit("Driver did not become ready in time")

    stop_flag: Dict[str, bool] = {"stop": False}
    _install_sigint_handler(stop_flag)

    rows: list[tuple[float, float]] = []
    end_time = time.time() + float(args.seconds)
    print("Collecting proprioception latency samples...")
    while not stop_flag["stop"] and time.time() < end_time:
        snap = driver.get_latest_robot_state()
        if snap is None or not getattr(snap, "is_valid", False):
            time.sleep(0.002)
            continue
        hw_ts = float(getattr(snap, "timestamp_ns", 0)) / 1e9
        recv_ts = time.time()
        if hw_ts <= 0.0:
            time.sleep(0.002)
            continue
        rows.append((recv_ts, hw_ts))
        time.sleep(0.002)

    driver.stop()

    if not rows:
        print("No proprioception samples collected")
        return

    recv_ts = np.array([row[0] for row in rows], dtype=float)
    hw_ts = np.array([row[1] for row in rows], dtype=float)

    # The robot timestamps are on its own clock; align by removing the observed
    # clock offset (smallest observed difference) so we report transport delay.
    raw_offsets = recv_ts - hw_ts
    clock_offset = float(np.min(raw_offsets))
    latencies = raw_offsets - clock_offset
    print(f"Estimated host-robot clock offset: {clock_offset:.3f}s")
    print(_summarize(latencies))

    if args.ping_address and latencies.size == 0:
        approx = _measure_ping_latency(args.ping_address, count=10)
        if approx is None:
            print(f"Failed to measure ICMP latency to {args.ping_address}")
        else:
            print(f"Estimated proprioception latency via ping: {approx:.4f}s")

    if args.output_csv:
        out_path = Path(args.output_csv)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["recv_timestamp", "hardware_timestamp", "transport_latency_seconds"])
            for rt, ht, latency in zip(recv_ts, hw_ts, latencies):
                writer.writerow([rt, ht, latency])
        print(f"Wrote {len(rows)} samples to {out_path}")


if __name__ == "__main__":
    main()
