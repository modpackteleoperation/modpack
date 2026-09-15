"""GUI frontend that drives targets for the standalone RBY1 WBC thread."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path regardless of current working directory.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from control.rby1_wbc import RBY1WBC
from rby1.ee_targets import EETargets
from rby1.rby1_wbc_app import RBY1WBCApp


class RBY1WBCGui(RBY1WBCApp):
    def __init__(self, wbc: RBY1WBC):
        self.mocap_ids = None
        super().__init__(
            wbc=wbc,
            headless=False,
            model_path=PROJECT_ROOT + "/model/rby1/rby1_mocap.xml",
        )
        if self.visualizer is not None:
            self.mocap_ids = self.visualizer.init_mocap_targets()

    def get_target(self) -> EETargets | None:
        if self.visualizer is None:
            return None
        (
            left_pos,
            left_quat,
            right_pos,
            right_quat,
            head_pos,
            head_quat,
        ) = self.visualizer.get_mocap_targets(self.mocap_ids)
        
        return EETargets(
            left_pos=left_pos,
            left_quat=left_quat,
            right_pos=right_pos,
            right_quat=right_quat,
            left_width=None,
            right_width=None,
            head_pos=head_pos,
            head_quat=head_quat,
            duration=self.trajectory_rate.dt,
            timestamp=time.monotonic(),
        )

    def on_target_rejected(self, target: EETargets) -> None:
        if self.visualizer is None or self.mocap_ids is None:
            return

        snapshot = self.wbc.get_latest_robot_state()
        qpos = self.wbc.snapshot_to_qpos(snapshot)
        if qpos is None:
            return

        with self._visualizer_lock:
            left_pos, left_quat = self.visualizer.site_pose("end_effector_l", qpos)
            right_pos, right_quat = self.visualizer.site_pose("end_effector_r", qpos)
            head_pos, head_quat = self.visualizer.site_pose("head", qpos)

            self.visualizer.data.mocap_pos[self.mocap_ids["left"]] = left_pos
            self.visualizer.data.mocap_pos[self.mocap_ids["right"]] = right_pos
            self.visualizer.data.mocap_pos[self.mocap_ids["head"]] = head_pos
            self.visualizer.data.mocap_quat[self.mocap_ids["left"]] = left_quat
            self.visualizer.data.mocap_quat[self.mocap_ids["right"]] = right_quat
            self.visualizer.data.mocap_quat[self.mocap_ids["head"]] = head_quat

def main() -> None:
    parser = argparse.ArgumentParser(description="RBY1 whole-body IK GUI decoupled from WBC thread")
    args = parser.parse_args()

    wbc = RBY1WBC()
    wbc.start()
    gui = None
    try:
        gui = RBY1WBCGui(wbc=wbc)
        gui.run()
    finally:
        if gui is not None:
            gui.close()
        wbc.stop()

if __name__ == "__main__":
    main()
