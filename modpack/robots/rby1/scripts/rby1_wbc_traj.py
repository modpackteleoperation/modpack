"""Replay a robologger zarr episode as joint-space targets via the RBY1 WBC."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional
import numpy as np

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from control.rby1_wbc import RBY1WBC
from demo.trajectory_loader_zarr import load_trajectory_zarr
from rby1.base_targets import BaseTargets
from rby1.joint_targets import JointTargets
from rby1.rby1_wbc_app import RBY1WBCApp


class RBY1WBCTrajectory(RBY1WBCApp):
    def __init__(
        self,
        wbc: RBY1WBC,
        headless: bool = False,
        qpos_list: list | None = None,
        widths_list: list | None = None,
        base_list: list | None = None,
    ) -> None:
        self.qpos_list   = qpos_list   if qpos_list   is not None else []
        self.widths_list = widths_list if widths_list is not None else []
        self.base_list   = base_list   if base_list   is not None else []
        self.trajectory_index = 0
        self.num_steps = min(len(self.qpos_list), len(self.widths_list))
        super().__init__(wbc=wbc, headless=headless)

    def on_target_rejected(self, target: JointTargets) -> None:
        rejected_index = max(self.trajectory_index - 1, 0)
        print(f"[trajectory] target {rejected_index} rejected; skipping to next command.")

    def get_joint_target(self) -> Optional[JointTargets]:
        if self.trajectory_index == 0:
            input("Press [Enter] to start streaming the trajectory.")

        if self.trajectory_index >= self.num_steps:
            print("[trajectory] streaming complete.")
            return None

        qpos_entry  = self.qpos_list[self.trajectory_index]
        width_entry = self.widths_list[self.trajectory_index]
        self.trajectory_index += 1

        return JointTargets(
            left_qpos=qpos_entry["left_arm"],
            right_qpos=qpos_entry["right_arm"],
            head_qpos=qpos_entry.get("head"),
            left_width=float(width_entry["left_width"]),
            right_width=float(width_entry["right_width"]),
            timestamp=time.monotonic(),
        )

    def get_base_target(self) -> Optional[BaseTargets]:
        idx = self.trajectory_index - 1  # already incremented by get_joint_target
        if not self.base_list or idx < 0 or idx >= len(self.base_list):
            return None
        b = self.base_list[idx]
        half = b["yaw"] * 0.5
        quat = np.array([math.cos(half), 0.0, 0.0, math.sin(half)])
        t = BaseTargets()
        t.set_targets(b["x"], b["y"], quat)
        return t


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a robologger zarr episode as joint-space targets via the RBY1 WBC"
    )
    parser.add_argument(
        "--zarr",
        required=True,
        help="Path to a robologger episode directory (contains left_arm.zarr, right_arm.zarr, etc.)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Skip launching the MuJoCo viewer (useful for debugging controller only).",
    )
    args = parser.parse_args()

    if args.headless:
        os.environ.setdefault("MUJOCO_GL", "egl")

    qpos_list, widths_list, base_list = load_trajectory_zarr(args.zarr)
    print(f"[trajectory] loaded {len(qpos_list)} steps from {args.zarr}")
    if not qpos_list or not widths_list:
        raise ValueError("Trajectory is empty after loading.")

    first_qpos = qpos_list[0]
    first_widths = widths_list[0]

    wbc = RBY1WBC()
    wbc.start(skip_init_position=True)
    wbc.set_init_position(
        left_arm_override=first_qpos["left_arm"],
        right_arm_override=first_qpos["right_arm"],
        head_override=first_qpos.get("head"),
        gripper_override=np.array(
            [
                float(first_widths["left_width"]),
                float(first_widths["right_width"]),
            ],
            dtype=float,
        ),
    )

    gui = None
    try:
        gui = RBY1WBCTrajectory(
            wbc=wbc,
            headless=args.headless,
            qpos_list=qpos_list,
            widths_list=widths_list,
            base_list=base_list,
        )
        gui.run()
    finally:
        if gui is not None:
            gui.close()
        wbc.stop()


if __name__ == "__main__":
    main()
