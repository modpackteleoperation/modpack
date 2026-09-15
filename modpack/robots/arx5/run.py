"""
ARX5 bridge runner.

Usage (Robot PC — each mode runs in its own process):
    python -m modpack.robots.arx5.run --arm right
    python -m modpack.robots.arx5.run --arm left
    python -m modpack.robots.arx5.run --neck
    python -m modpack.robots.arx5.run --cameras
    python -m modpack.robots.arx5.run --iphone --camera-endpoint tcp://... --activation-host ... --activation-port ...
"""
from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path

import yaml

from modpack.bridge import RobotBridge
from modpack.orchestration.message_formats import MessageFactory
from modpack.orchestration.robot_config import load_network_config
from modpack.robots.arx5.arx5_joint import ARX5Joint
from modpack.robots.arx5.arx5_neck_cartesian import ARX5Cartesian
from modpack.robots.arx5.read_iphone_camera import ReadiPhone
from modpack.robots.arx5.camera_process import CameraPublisher

_CONFIG = "modpack/robots/arx5/config.yaml"


def _load_config() -> dict:
    with open(_CONFIG, "r") as f:
        doc = yaml.safe_load(f) or {}
    # Pull the RMQ network block from the single source (modpack_config.yaml);
    # as_rmq_dict() sets host = gello_pc_ip, which the robot PC connects to.
    if "rmq" not in doc and "robotmq" not in doc:
        doc["rmq"] = load_network_config().as_rmq_dict()
    return doc


def run_arm(arm: str) -> None:
    stop = False
    robot = None  # assigned below; referenced by callbacks via closure
    def _on_stop():
        nonlocal stop
        stop = True

    def _activation_cb(command: str, _msg: dict) -> None:
        print(f"[arx5/{arm}] activation_cb: {command}")
        # execute_action() gates on the robot's own _activated flag, so the
        # activate/deactivate commands must be forwarded to the robot — gating
        # the loop on bridge.is_active alone is not enough (the arm reads
        # commands but silently refuses to execute them while _activated=False).
        if robot is None:
            return  # activation arrived before hardware init; loop isn't running yet
        if command == "activate":
            robot.activate()
        elif command in ("deactivate", "reset", "pause_episode"):
            robot.deactivate()
        elif command in ("emergency_stop", "shutdown", "manager_shutdown"):
            robot.deactivate()
            _on_stop()

    # Scope this bridge to the 'arx' target (shared by both arms) so a neck/body
    # activation cannot engage the arm — the bridge otherwise accepts any role's
    # target because _load_config() loads all roles.
    bridge = RobotBridge.from_dict(
        _load_config(), activation_callback=_activation_cb, activation_target="arx"
    )
    can_interface = "can8" if arm == "right" else "can5"

    def _safe_shutdown(signum=None, frame=None):
        if robot is not None:
            robot.hold_position()
        bridge.disconnect()
        print(f"[arx5/{arm}] holding position — suspending process")
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGSTOP)

    signal.signal(signal.SIGTERM, _safe_shutdown)
    signal.signal(signal.SIGINT,  lambda *_: _on_stop())

    bridge.connect()
    robot  = ARX5Joint(arm=arm, interface=can_interface, bridge=bridge)
    robot.wait_until_ready()

    role = f"{arm}_arm"
    print(f"[arx5/{arm}] Bridge connected — waiting for activation")

    try:
        while not stop:
            # Gate on the robot's own activation (set only by an arx 'activate' —
            # the single 'v' tap), NOT bridge.is_active, which also flips true on
            # start_episode/resume_episode (target=all, 's' key) and would engage
            # the arm with no 'v'.
            if not robot._activated:
                time.sleep(0.01)
                continue
            cmd = bridge.get_command(role)
            if cmd is not None:
                robot.execute_action({"joint_pos": cmd.joint_pos, "gripper": cmd.gripper})
            state = robot.get_state()
            bridge.publish_state(role, state)
        # shutdown via activation message — hold then suspend
        _safe_shutdown()
    finally:
        robot.close()
        bridge.disconnect()


def run_neck() -> None:
    stop = False
    def _on_stop():
        nonlocal stop
        stop = True

    def _activation_cb(command: str, _msg: dict) -> None:
        print(f"[arx5/neck] activation_cb: {command}")
        if command == "activate":
            robot.activate()
        elif command in ("deactivate",):
            robot.deactivate()
        elif command in ("emergency_stop", "shutdown", "manager_shutdown"):
            robot.deactivate()
            _on_stop()

    cfg = _load_config()
    neck_cfg = cfg.get("neck", {}) or {}
    # Scope this bridge to the 'neck' target so an arx/body activation cannot
    # engage the neck (the bridge otherwise accepts any role's target).
    bridge = RobotBridge.from_dict(
        cfg, activation_callback=_activation_cb, activation_target="neck"
    )
    signal.signal(signal.SIGINT, lambda *_: _on_stop())

    bridge.connect()
    robot = ARX5Cartesian(
        bridge=bridge,
        msg_factory=MessageFactory(),
        enable_step_guard=bool(neck_cfg.get("enable_step_guard", True)),
        max_cmd_pos_step_m=float(neck_cfg.get("max_cmd_pos_step_m", 0.15)),
        max_cmd_rot_step_rad=float(neck_cfg.get("max_cmd_rot_step_rad", 0.6)),
    )

    def _safe_shutdown(signum=None, frame=None):
        robot.hold_position()
        bridge.disconnect()
        print("[arx5/neck] holding position — suspending process")
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGSTOP)

    signal.signal(signal.SIGTERM, _safe_shutdown)

    if not robot.wait_until_ready(timeout=0, stop_fn=lambda: stop):
        _safe_shutdown()
        return  # unreachable

    print("[arx5/neck] Bridge connected — waiting for activation")

    while not stop:
        # Gate on the robot's own activation (set only by a neck 'activate'), NOT
        # bridge.is_active, which also flips true on start_episode/resume_episode
        # (target=all, 's' key) and would engage the neck with no activation.
        if not robot._activated:
            time.sleep(0.01)
            continue
        cmd = bridge.get_command("neck")
        if cmd is not None:
            robot.execute_action({"eef_pose": cmd.eef_pose})
        state = robot.get_state()
        bridge.publish_state("neck", state)

    _safe_shutdown()


def run_cameras(arms: list[str], debug: bool = False) -> None:
    """Start wrist camera capture and publish via bridge (Robot PC)."""
    bridge = RobotBridge.from_dict(_load_config())
    bridge.connect()
    print(f"[arx5/cameras] Bridge connected — starting cameras for arms: {arms}")
    pub = CameraPublisher(
        arms=arms,
        debug=debug,
        bridge=bridge,
    )
    pub.run()


def run_iphone(
    logging_endpoint: str | None = None,
    debug: bool = False,
) -> None:
    """Start the arx5 Record3D iPhone camera capture (Robot PC)."""
    _neck_config = Path(__file__).parent / "neck_config.yaml"

    signal.signal(signal.SIGTERM, lambda *_: None)
    signal.signal(signal.SIGINT,  lambda *_: None)

    bridge = RobotBridge.from_dict(_load_config())
    bridge.connect()

    print("[arx5/iphone] Starting Record3D iPhone capture")
    pub = ReadiPhone(
        bridge=bridge,
        neck_config_path=str(_neck_config),
        logging_endpoint=logging_endpoint,
        debug=debug,
    )
    pub.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="ARX5 bridge runner")
    parser.add_argument("--arm",     choices=["right", "left"], help="Run joint arm adapter")
    parser.add_argument("--neck",    action="store_true",       help="Run neck (Cartesian) adapter")
    parser.add_argument("--cameras", action="store_true",       help="Run wrist camera capture (Robot PC)")
    parser.add_argument("--iphone",  action="store_true",       help="Run Record3D iPhone capture (Robot PC)")
    parser.add_argument("--logging-endpoint", default=None, help="Logger RMQ endpoint (iphone mode)")
    parser.add_argument("--debug",   action="store_true",       help="Enable debug output")
    args = parser.parse_args()

    modes = [bool(args.arm), args.neck, args.cameras, args.iphone]
    if sum(modes) != 1:
        parser.error("Specify exactly one of: --arm right|left, --neck, --cameras, --iphone")

    if args.cameras:
        run_cameras(arms=["right", "left"], debug=args.debug)
    elif args.iphone:
        run_iphone(
            logging_endpoint=args.logging_endpoint,
            debug=args.debug,
        )
    elif args.neck:
        run_neck()
    else:
        run_arm(args.arm)


if __name__ == "__main__":
    main()
