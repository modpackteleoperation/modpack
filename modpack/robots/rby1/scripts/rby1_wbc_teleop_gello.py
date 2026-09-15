"""Gello teleoperation frontend for the RBY1 whole-body controller."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional
import yaml
from camera.camera_stream_publisher import CameraStreamPublisher


# PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
# MODPACK_ROOT = str(Path(__file__).resolve().parent.parent.parent)
MODPACK_ROOT = Path(__file__).resolve().parent.parent.parent.parent
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

PROJECT_ROOT = Path(MODPACK_ROOT) / "robots" / "rby1"


# def _add_modpack_root() -> None:
#     project_root = Path(PROJECT_ROOT)
#     env_root = os.environ.get("MODPACK_ROOT")
#     candidates = [Path(env_root).expanduser()] if env_root else []

#     # candidates.extend(
#     #     [
#     #         project_root.parent / "modpack",
#     #         project_root,
#     #     ]
#     # )
#     sys.path.insert(0, candidate_str)

#     # for candidate in candidates:
#     #     if (candidate / "modpack").is_dir():
#     #         candidate_str = str(candidate)
#     #         if candidate_str not in sys.path:
#     #             sys.path.insert(0, candidate_str)
#     #         return

#     raise ModuleNotFoundError(
#         "Could not locate the 'modpack' package. "
#         "Set MODPACK_ROOT to the modpack repository root."
#     )

sys.path.insert(0, MODPACK_ROOT)
sys.path.insert(0, PROJECT_ROOT)
# _add_modpack_root()
print()

from modpack.bridge import RobotBridge
from modpack.orchestration.robot_config import load_network_config
from control.rby1_wbc import RBY1WBC
from rby1.base_targets import BaseTargets
from rby1.joint_targets import JointTargets
from rby1.rby1_wbc_app import RBY1WBCApp
from teleop.teleop_gello import TeleopGello


class RBY1WBCTeleop(RBY1WBCApp):
    def __init__(self, wbc: RBY1WBC, teleop: TeleopGello, headless: bool = False, active_systems: Optional[dict] = None) -> None:
        self.teleop = teleop
        self._active_systems = active_systems or {"arms": True, "body": True}
        super().__init__(wbc=wbc, headless=headless)
        self.teleop.start()

    def get_joint_target(self) -> Optional[JointTargets]:
        target = None
        if self._active_systems.get("arms", True):
            target = self.teleop.compute_target()

        head_qpos = self.teleop.compute_head_target()
        if head_qpos is not None:
            if target is None:
                target = JointTargets(head_qpos=head_qpos)
            else:
                target.head_qpos = head_qpos

        return target

    def get_base_target(self) -> Optional[BaseTargets]:
        if not self._active_systems.get("body", True):
            return None
        return self.teleop.compute_base_target()

    def on_target_rejected(self, target: Any) -> None:
        handler = getattr(self.teleop, "on_target_rejected", None)
        if callable(handler):
            handler()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RBY1 whole-body teleop frontend for Gello."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Skip launching the MuJoCo viewer.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Persist computed teleop trajectory as a dataset-style pickle under demo/.",
    )
    args = parser.parse_args()

    headless = bool(args.headless)
    save_trajectory = bool(args.save)
    if headless:
        os.environ.setdefault("MUJOCO_GL", "egl")

    config_path = Path(PROJECT_ROOT) / "config" / "teleop_gello.yaml"
    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise Exception(f"Exception while loading config file: {e}")
    if not isinstance(config, dict):
        raise ValueError(f"Gello config at {config_path} must be a mapping.")

    active_systems = config.get("active_systems", {"arms": False, "body": False})
    urdf_path = str(config.get("urdf_path") or "")
    teleop_enabled = any(
        bool(active_systems.get(key, False))
        for key in ("arms", "body", "head", "base_pub")
    )

    # Network endpoints come from modpack_config.yaml (one source of truth);
    # the teleop config keeps only robot-specific keys (roles, calibration, tuning).
    net = load_network_config()
    try:
        # Keep this file's roles (the bridge needs head_cmd); use net's endpoints.
        # Unmanaged rby1: the camera server lives on the gello PC (same host as the
        # data/activation/feedback servers), so point the camera client there too.
        # Without cam_host the bridge defaults the camera client to localhost and
        # cross-PC camera frames are lost.
        bridge = RobotBridge.from_dict(
            {**config, "rmq": {**net.as_rmq_dict(), "cam_host": net.gello_pc_ip}}
        )
        bridge.connect()
    except Exception as _bridge_exc:
        print(f"[rby1_teleop] bridge unavailable ({_bridge_exc}); state will publish via direct RMQ")
        bridge = None

    teleop = TeleopGello(
        wbc=None,  # set after wbc is created below
        host=net.gello_pc_ip,
        port=net.data_port,
        activation_host=net.gello_pc_ip,
        activation_port=net.activation_port,
        active_arms=list(config.get("active_arms", ["left", "right"])),
        joint_offsets=config.get("joint_offsets"),
        joint_signs=config.get("joint_signs"),
        save_trajectory=save_trajectory,
        active_systems=active_systems,
        torque_feedback_enabled=bool(config.get("torque_feedback_enabled", False)),
        torque_feedback_gain=float(config.get("torque_feedback_gain", 1.0)),
        torque_feedback_debug=bool(config.get("torque_feedback_debug", False)),
        torque_feedback_debug_period_s=float(config.get("torque_feedback_debug_period_s", 2.0)),
        urdf_path=urdf_path,
        bridge=bridge,
    )

    wbc = RBY1WBC()
    if not wbc.reset_base_odometry_origin():
        print("[startup] Failed to capture base odometry origin; first WBC state will set it.")

    if teleop_enabled:
        if not teleop.initialize():
            wbc.stop()
            raise RuntimeError("Gello teleoperation could not be initialized!")

    if teleop_enabled and active_systems.get("arms", False):
        print("[startup] Waiting for first Gello pose...")
        wbc.start(skip_init_position=True)
        left, right = teleop.wait_for_first_pose()
        print("[startup] Got first Gello pose — ramping to it as init position.")
        wbc.set_init_position(left_arm_override=left, right_arm_override=right)
    else:
        wbc.start()

    teleop.wbc = wbc

    cam_pub = None
    if active_systems.get("camera", False):
        cam_pub = CameraStreamPublisher(
            bridge=bridge,
            vp_ip=str(config.get("vp_ip", "192.168.0.57")),
            vp_rgb_port=int(config.get("vp_rgb_port", 6007)),
            camera_key=str(config.get("camera_key", "camera_head_main_rgb")),
            camera_keys=config.get("camera_keys"),
            camera_binning=config.get("camera_binning"),
            maximize_camera_resolution=bool(config.get("camera_maximize_resolution", False)),
            fps=float(config.get("camera_fps", 15.0)),
            rgb_quality=int(config.get("vp_rgb_quality", 30)),
        )
        cam_pub.start()

    gui: Optional[RBY1WBCTeleop] = None
    try:
        if teleop_enabled:
            gui = RBY1WBCTeleop(wbc=wbc, teleop=teleop, headless=headless, active_systems=active_systems)
            gui.run()
        else:
            print("[startup] Teleop disabled; running camera-only/controller-only loop.")
            while True:
                time.sleep(0.1)
    finally:
        try:
            if gui is not None:
                gui.close()
            if teleop_enabled:
                teleop.stop()
            if cam_pub is not None:
                cam_pub.stop()
        finally:
            wbc.stop()
    if bridge is not None:
        bridge.disconnect()


if __name__ == "__main__":
    main()
