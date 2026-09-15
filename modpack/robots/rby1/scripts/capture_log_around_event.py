#!/usr/bin/env python3
"""Stream R-PC logs around a control-manager event to debug DefaultControl spawn.

Captures up to four windows you can diff:
  baseline      -- robot idle, no operation
  enable        -- streams logs across enable_control_manager()
  cancel        -- streams logs across cancel_control() then waits for respawn
  disable_race  -- streams logs across the cancel+retry-disable race

Each window is written to a separate timestamped file in the output dir.
Run all four back-to-back with the same robot state (no power cycle between)
for a clean diff.

Usage:
  python capture_log_around_event.py --address 192.168.30.1:50051 --model m
  python capture_log_around_event.py --mode enable
  python capture_log_around_event.py --duration 10 --out /tmp/captures
"""

import argparse
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import rby1_sdk


def fmt_log(log) -> str:
    ts = log.timestamp.isoformat(timespec="microseconds")
    rts = log.robot_system_timestamp.isoformat(timespec="microseconds")
    return f"{ts} | rpc={rts} | {log.level.name:<8} | {log.message}"


def stream_for(robot, seconds: float, out_path: Path, action=None) -> int:
    """Stream logs for `seconds`. Optionally invoke `action()` after 1s pre-baseline.

    Returns the number of log entries captured.
    """
    count = 0
    lock = threading.Lock()
    f = out_path.open("w")
    f.write(f"# capture started: {datetime.now().isoformat()}\n")
    f.write(f"# duration: {seconds}s\n")
    if action:
        f.write(f"# action: {action.__name__}\n")
    f.write("# ---\n")
    f.flush()

    def cb(logs):
        nonlocal count
        with lock:
            for log in logs:
                f.write(fmt_log(log) + "\n")
                count += 1
            f.flush()

    robot.start_log_stream(cb, 50.0)
    try:
        if action:
            time.sleep(1.0)  # pre-action baseline so trigger line is easy to find
            f.write(f"# >>> ACTION FIRING at {datetime.now().isoformat()}\n")
            f.flush()
            try:
                result = action()
                f.write(f"# >>> ACTION RETURNED: {result!r}\n")
                f.flush()
            except Exception as e:
                f.write(f"# >>> ACTION RAISED: {e!r}\n")
                f.flush()
            time.sleep(max(0.0, seconds - 1.0))
        else:
            time.sleep(seconds)
    finally:
        robot.stop_log_stream()
        f.write(f"# capture ended: {datetime.now().isoformat()} (entries={count})\n")
        f.close()
    return count


def cmd_baseline(robot, out_dir: Path, seconds: float) -> Path:
    p = out_dir / f"01_baseline_{datetime.now():%H%M%S}.log"
    print(f"[baseline] streaming {seconds}s of idle logs -> {p}")
    n = stream_for(robot, seconds, p)
    print(f"[baseline] captured {n} entries")
    return p


def cmd_enable(robot, out_dir: Path, seconds: float) -> Path:
    p = out_dir / f"02_enable_{datetime.now():%H%M%S}.log"
    print(f"[enable] streaming {seconds}s around enable_control_manager() -> {p}")

    def action():
        return robot.enable_control_manager()
    action.__name__ = "enable_control_manager"

    n = stream_for(robot, seconds, p, action=action)
    print(f"[enable] captured {n} entries")
    return p


def cmd_cancel(robot, out_dir: Path, seconds: float) -> Path:
    p = out_dir / f"03_cancel_{datetime.now():%H%M%S}.log"
    print(f"[cancel] streaming {seconds}s around cancel_control() -> {p}")

    def action():
        return robot.cancel_control()
    action.__name__ = "cancel_control"

    n = stream_for(robot, seconds, p, action=action)
    print(f"[cancel] captured {n} entries")
    return p


def cmd_disable_race(robot, out_dir: Path, seconds: float) -> Path:
    p = out_dir / f"04_disable_race_{datetime.now():%H%M%S}.log"
    print(f"[disable_race] streaming {seconds}s around forced disable -> {p}")

    def action():
        robot.cancel_control()
        deadline = time.time() + 2.0
        attempts = 0
        while time.time() < deadline:
            attempts += 1
            if robot.disable_control_manager():
                return f"disabled after {attempts} attempts"
            time.sleep(0.020)
        return f"FAILED after {attempts} attempts"
    action.__name__ = "force_disable_via_race"

    n = stream_for(robot, seconds, p, action=action)
    print(f"[disable_race] captured {n} entries")
    return p


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--address", default="192.168.30.1:50051")
    p.add_argument("--model", default="a", choices=["a", "m"], help="robot model")
    p.add_argument("--out", default="./log_captures", help="output directory")
    p.add_argument("--duration", type=float, default=6.0,
                   help="seconds per capture window")
    p.add_argument(
        "--mode", default="all",
        choices=["all", "baseline", "enable", "cancel", "disable_race"],
        help="which capture(s) to run",
    )
    args = p.parse_args()

    out_dir = Path(args.out) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_dir.resolve()}")

    if args.model == "a":
        robot = rby1_sdk.create_robot_a(args.address)
    else:
        robot = rby1_sdk.create_robot_m(args.address)

    if not robot.connect():
        print(f"ERROR: failed to connect to {args.address}", file=sys.stderr)
        sys.exit(1)
    print(f"connected to {args.address}")

    try:
        robot.sync_time()
    except Exception as e:
        print(f"warning: sync_time failed: {e}")

    cms = robot.get_control_manager_state()
    print(f"control manager state at start: state={cms.state}, ctrl={cms.control_state}")

    paths = []
    if args.mode in ("all", "baseline"):
        paths.append(cmd_baseline(robot, out_dir, args.duration))
        time.sleep(1)
    if args.mode in ("all", "enable"):
        paths.append(cmd_enable(robot, out_dir, args.duration))
        time.sleep(1)
    if args.mode in ("all", "cancel"):
        paths.append(cmd_cancel(robot, out_dir, args.duration))
        time.sleep(1)
    if args.mode in ("all", "disable_race"):
        paths.append(cmd_disable_race(robot, out_dir, args.duration))

    print("\n=== captures complete ===")
    for p in paths:
        print(f"  {p}")
    if len(paths) >= 2:
        print("\nDiff hint:")
        print(f"  diff <(grep -v '^#' {paths[0]}) <(grep -v '^#' {paths[1]}) | less")
    print("\nGrep hint:")
    print(f"  grep -E 'DefaultControl|Control started|Control finished|priority' {out_dir}/*.log")


if __name__ == "__main__":
    main()
