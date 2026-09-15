#!/usr/bin/env python3
"""
GELLO Calibration Script

Calibrates joint_offsets and gripper open/close angles for any robot config
that uses the standard gello: config format (RBY1, ARX5, etc.).

Usage:
    python zero_calib.py <config_path> <arm> [--skip-gripper]

Examples:
    python zero_calib.py modpack/robots/rby1/config.yaml right
    python zero_calib.py modpack/robots/arx5/config.yaml left --skip-gripper
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
import yaml

sys.path.append(str(Path(__file__).parent.parent.parent))

from modpack.modules.gello.gello.robots.dynamixel import DynamixelRobot


def _get_gripper_angles(port_config: dict):
    """Return (open_angle_deg, closed_angle_deg) from either config format."""
    if 'gripper' in port_config:
        g = port_config['gripper']
        return g.get('open_angle_deg'), g.get('closed_angle_deg')
    return (
        port_config.get('gripper_open_angle_deg'),
        port_config.get('gripper_closed_angle_deg'),
    )


def _has_nested_gripper(port_config: dict) -> bool:
    return 'gripper' in port_config


def create_robot(config_path: str, arm: str) -> tuple:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    gello_config = config['gello']
    hardware_config = gello_config['hardware'][arm]
    port = hardware_config['port']
    port_config = gello_config['port_configs'][port]

    arm_motor_config = hardware_config['motors']
    baudrate = hardware_config['baudrate']

    # Build gripper_config in the format DynamixelRobot expects
    gripper_config = None
    if _has_nested_gripper(port_config):
        gripper_config = port_config['gripper']

    robot = DynamixelRobot(
        real=True,
        motor_config=arm_motor_config,
        gripper_config=gripper_config,
        hardware_config=hardware_config,
        port=port,
        baudrate=baudrate,
        zero_position_offsets=None,
    )
    return robot, port_config, port


def calibrate_joint_offsets(robot: DynamixelRobot, port_config: dict) -> list:
    joint_signs = np.array(port_config.get('joint_signs', [1] * 7), dtype=float)
    n_joints = len(robot._joint_ids)
    joint_signs = joint_signs[:n_joints]

    target = np.zeros(n_joints)

    print("\n" + "=" * 70)
    print("STEP 1: Joint Offset Calibration")
    print("=" * 70)
    print("Move the arm to its comfortable neutral pose.")
    print("When in this pose, the robot should be at all-zero joint positions.")
    print()
    input("Press Enter when the arm is in position...")

    raw = robot.get_joint_state()[:n_joints]
    print(f"\nRaw readings: {np.round(raw, 4).tolist()}")

    print(f"\n{'Joint':<8} {'Raw':<14} {'Sign':<6} {'Offset':<14}")
    print("-" * 45)

    new_offsets = []
    for i in range(n_joints):
        offset = raw[i] - (target[i] / joint_signs[i])
        new_offsets.append(float(offset))
        print(f"{i:<8} {raw[i]:>8.4f} ({np.rad2deg(raw[i]):>6.1f}°)  "
              f"{int(joint_signs[i]):<6} {offset:>8.4f} ({np.rad2deg(offset):>6.1f}°)")

    # Verification
    print("\nVerification:")
    print(f"{'Joint':<8} {'After offset':<14} {'After sign':<14} {'Target':<10} {'OK?'}")
    print("-" * 55)
    all_ok = True
    for i in range(n_joints):
        after_offset = raw[i] - new_offsets[i]
        after_sign = after_offset * joint_signs[i]
        ok = abs(after_sign - target[i]) < 0.001
        all_ok = all_ok and ok
        print(f"{i:<8} {after_offset:>8.4f}        {after_sign:>8.4f}        "
              f"{target[i]:>6.4f}    {'✓' if ok else '✗'}")

    if not all_ok:
        print("\nWarning: small numerical differences (likely floating point, usually fine)")

    # Preserve existing gripper offset if present
    existing = port_config.get('joint_offsets', [])
    if len(existing) > n_joints:
        new_offsets.append(existing[n_joints])
    elif robot.gripper_motor_id is not None:
        new_offsets.append(0.0)

    return new_offsets


def calibrate_gripper(robot: DynamixelRobot) -> tuple:
    if robot.gripper_motor_id is None:
        print("\nNo gripper motor configured — skipping gripper calibration.")
        return None, None

    print("\n" + "=" * 70)
    print("STEP 2: Gripper Calibration")
    print("=" * 70)

    input("Open the gripper FULLY, then press Enter...")
    raw_open = robot.get_full_state()
    open_deg = int(round(np.rad2deg(raw_open[-1])))
    print(f"  Open angle: {open_deg}°")

    input("Close the gripper FULLY, then press Enter...")
    raw_closed = robot.get_full_state()
    closed_deg = int(round(np.rad2deg(raw_closed[-1])))
    print(f"  Closed angle: {closed_deg}°")

    return open_deg, closed_deg


def print_results(port: str, new_offsets: list, open_deg, closed_deg, port_config: dict):
    print("\n" + "=" * 70)
    print("RESULTS — copy into your config file")
    print("=" * 70)
    print(f"\nUnder port_configs -> {port}:\n")

    rounded = [round(v, 6) for v in new_offsets]
    print(f"  joint_offsets: {rounded}")

    if open_deg is not None and closed_deg is not None:
        if _has_nested_gripper(port_config):
            gripper_invert = port_config['gripper'].get('gripper_invert', False)
            gripper_motor_id = port_config['gripper'].get('gripper_motor_id', 8)
            print(f"  gripper:")
            print(f"    open_angle_deg: {open_deg}")
            print(f"    closed_angle_deg: {closed_deg}")
            print(f"    gripper_invert: {str(gripper_invert).lower()}")
            print(f"    gripper_motor_id: {gripper_motor_id}")
        else:
            print(f"  gripper_open_angle_deg: {open_deg}")
            print(f"  gripper_closed_angle_deg: {closed_deg}")

    print()


def main():
    parser = argparse.ArgumentParser(description="GELLO calibration script")
    parser.add_argument("config_path", help="Path to robot config.yaml")
    parser.add_argument("arm", choices=["left", "right"], help="Which arm to calibrate")
    parser.add_argument("--skip-gripper", action="store_true", help="Skip gripper calibration")
    args = parser.parse_args()

    if not os.path.exists(args.config_path):
        print(f"Config not found: {args.config_path}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("GELLO CALIBRATION")
    print("=" * 70)
    print(f"Config: {args.config_path}")
    print(f"Arm:    {args.arm}")

    print("\nInitializing hardware...")
    robot, port_config, port = create_robot(args.config_path, args.arm)
    robot.enable_torque(True)
    print("✓ Connected")

    try:
        new_offsets = calibrate_joint_offsets(robot, port_config)

        open_deg, closed_deg = None, None
        if not args.skip_gripper:
            open_deg, closed_deg = calibrate_gripper(robot)

        print_results(port, new_offsets, open_deg, closed_deg, port_config)

    finally:
        try:
            robot.enable_torque(False)
            print("✓ Torque disabled")
        except Exception:
            pass

    print("=" * 70)
    print("CALIBRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
