#!/usr/bin/env python3
"""Force the control manager into Idle state by racing the DefaultControl respawn.

Background:
  When `enable_control_manager()` is called, the firmware spawns a control named
  `DefaultControl` (gravity compensation on torso + arms) at priority 99. This
  control respawns ~200ms after every `cancel_control()`. As a result, the
  obvious `disable_control_manager()` returns False with "There are still
  active controls" because DefaultControl is still alive.

  The working sequence (as observed in server logs at 08:46:41 in our 2026-05-04
  trace) is:
    1. cancel_control()
    2. spam disable_control_manager() at ~22ms intervals
    3. one of them lands in the post-cancel pre-respawn window and succeeds.

  In the trace, disable succeeded on the 4th attempt, ~68ms after cancel.

Usage:
  python force_disable.py --address 192.168.30.1:50051 --model m
  python force_disable.py --timeout 5.0 --interval 0.020
  python force_disable.py --servo-off-on-failure
"""

import argparse
import sys
import time

import rby1_sdk


def force_disable(robot, timeout: float = 2.0, interval: float = 0.020) -> bool:
    """Cancel any active control and spam disable until it succeeds.

    Returns True if disable succeeded, False on timeout.
    """
    print(f"  cancel_control() ...", flush=True)
    try:
        robot.cancel_control()
    except Exception as e:
        print(f"  cancel_control raised: {e}", flush=True)

    deadline = time.time() + timeout
    attempts = 0
    t0 = time.time()
    while time.time() < deadline:
        attempts += 1
        ok = robot.disable_control_manager()
        elapsed_ms = (time.time() - t0) * 1000
        if ok:
            print(f"  disabled after {attempts} attempts ({elapsed_ms:.1f}ms)", flush=True)
            return True
        time.sleep(interval)
    print(f"  FAILED after {attempts} attempts ({(time.time()-t0)*1000:.1f}ms)", flush=True)
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--address", default="192.168.30.1:50051")
    p.add_argument("--model", default="a", choices=["a", "m"])
    p.add_argument("--timeout", type=float, default=2.0,
                   help="max seconds to keep retrying disable")
    p.add_argument("--interval", type=float, default=0.020,
                   help="seconds between disable retries")
    p.add_argument("--servo-off-on-failure", action="store_true",
                   help="if disable race fails, fall back to servo_off(.*)")
    args = p.parse_args()

    if args.model == "a":
        robot = rby1_sdk.create_robot_a(args.address)
    else:
        robot = rby1_sdk.create_robot_m(args.address)

    if not robot.connect():
        print(f"ERROR: failed to connect to {args.address}", file=sys.stderr)
        sys.exit(1)
    print(f"connected to {args.address}")

    cms = robot.get_control_manager_state()
    print(f"before: state={cms.state}, ctrl={cms.control_state}")

    if cms.state == rby1_sdk.ControlManagerState.State.Idle:
        print("already idle, nothing to do")
        return

    ok = force_disable(robot, timeout=args.timeout, interval=args.interval)

    cms = robot.get_control_manager_state()
    print(f"after:  state={cms.state}, ctrl={cms.control_state}")

    if not ok and args.servo_off_on_failure:
        print("falling back to servo_off('.*')...", flush=True)
        try:
            robot.servo_off(".*")
            print("servo_off succeeded")
        except Exception as e:
            print(f"servo_off raised: {e}", file=sys.stderr)
            sys.exit(2)

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
