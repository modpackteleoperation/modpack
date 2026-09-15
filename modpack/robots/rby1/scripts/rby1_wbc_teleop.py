"""Unified teleoperation frontend for the RBY1 whole-body controller."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Ensure project root is on sys.path regardless of current working directory.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from control.rby1_wbc import RBY1WBC
from rby1.ee_targets import EETargets
from rby1.rby1_wbc_app import RBY1WBCApp
from teleop.teleop_iphone import TeleopIphone
from teleop.teleop_vr import TeleopVR


class RBY1WBCTeleop(RBY1WBCApp):
    def __init__(self, wbc: RBY1WBC, teleop: Any, headless: bool = False) -> None:
        self.teleop = teleop
        super().__init__(wbc=wbc, headless=headless)
        self.teleop.start()

    def get_target(self) -> Optional[EETargets]:
        return self.teleop.compute_target()

    def on_target_rejected(self, target: EETargets) -> None:
        handler = getattr(self.teleop, "on_target_rejected", None)
        if callable(handler):
            handler()

def build_teleop(
    mode: str,
    wbc: RBY1WBC,
    config: Dict[str, Any],
    save_trajectory: bool,
) -> Any:
    if mode == "vr":
        local_ip = config.get("local_ip")
        meta_quest_ip = config.get("meta_quest_ip")
        local_port = int(config.get("local_port", 5005))
        meta_quest_port = int(config.get("meta_quest_port", 6000))
        if not local_ip or not meta_quest_ip:
            raise ValueError("Meta Quest mode requires 'local_ip' and 'meta_quest_ip' in config/teleop_vr.yaml.")
        teleop = TeleopVR(
            wbc=wbc,
            local_ip=str(local_ip),
            meta_quest_ip=str(meta_quest_ip),
            local_port=local_port,
            meta_quest_port=meta_quest_port,
            save_trajectory=save_trajectory,
        )
    elif mode == "iphone":
        host = str(config.get("host", "0.0.0.0"))
        port = int(config.get("port", 5555))
        portrait = bool(config.get("portrait", False))
        teleop = TeleopIphone(
            wbc=wbc,
            host=host,
            port=port,
            save_trajectory=save_trajectory,
            use_portrait_mode=portrait,
        )
    else:
        raise ValueError(f"Unsupported teleop mode: {mode}")
    return teleop

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RBY1 whole-body teleop frontend for VR or iPhone clients."
    )
    parser.add_argument(
        "--mode",
        choices=["vr", "iphone"],
        help="Select teleop mode; defaults to 'vr' when not provided.",
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

    mode = args.mode or "vr"
    headless = bool(args.headless)
    save_trajectory = bool(args.save)
    if headless:
        os.environ.setdefault("MUJOCO_GL", "egl")

    # Load config
    config_path = Path(PROJECT_ROOT + f"/config/teleop_{mode}.yaml")
    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise Exception(f"Exception while loading config file: {e}")
    if not isinstance(config, dict):
        raise ValueError(f"WBC config at {config_path} must be a mapping.")

    # Run 
    wbc = RBY1WBC()
    wbc.start()
    teleop = build_teleop(
        mode,
        wbc,
        config,
        save_trajectory=save_trajectory,
    )
    if not teleop.initialize():
        wbc.stop()
        raise RuntimeError("Teleoperation can not be initialized!")

    gui: Optional[RBY1WBCTeleop] = None
    try:
        gui = RBY1WBCTeleop(wbc=wbc, teleop=teleop, headless=headless)
        gui.run()
    finally:
        try:
            if gui is not None:
                gui.close()
            teleop.stop()
        finally:
            wbc.stop()


if __name__ == "__main__":
    main()
