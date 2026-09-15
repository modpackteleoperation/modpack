from __future__ import annotations

import threading
import time
from typing import Optional, Union

from loop_rate_limiters import RateLimiter

from .base_targets import BaseTargets
from .ee_targets import EETargets
from .joint_targets import JointTargets
from .state_visualizer import StateVisualizer
from control.rby1_wbc import RBY1WBC
from .whole_body_ik import RBY1WholeBodyIK

class RBY1WBCApp:
    """Reusable visualizer + trajectory loop for streaming WBC targets."""

    def __init__(
        self,
        wbc: RBY1WBC,
        headless: bool = False,
        model_path: Optional[str] = None,
    ) -> None:
        self.wbc = wbc
        self.headless = headless
        self.model_path = model_path or self.wbc.model_path

        self.visualizer: Optional[StateVisualizer] = None
        self.viewer_rate: Optional[RateLimiter] = None
        self._visualizer_lock = threading.Lock()
        if not self.headless:
            snapshot = self.wbc.wait_for_first_state()
            qpos = self.wbc.snapshot_to_qpos(snapshot)
            self.visualizer = StateVisualizer(
                model_path=self.model_path, initial_qpos=qpos, print_errors=False
            )
            self.viewer_rate = RateLimiter(frequency=60.0, warn=False)

        self.trajectory_rate = RateLimiter(
            frequency=self.wbc.trajectory_frequency_hz, warn=False
        )

        self._stop_event = threading.Event()
        self._trajectory_thread: Optional[threading.Thread] = None

    # ----- Hooks for subclasses -------------------------------------------------
    def get_target(self) -> Optional[EETargets]:
        return None

    def get_joint_target(self) -> Optional[JointTargets]:
        return None

    def get_base_target(self) -> Optional[BaseTargets]:
        return None

    def on_target_rejected(self, target: EETargets) -> None:
        """Hook for subclasses to respond when a target is rejected."""
        return

    # ----- Runtime --------------------------------------------------------------
    def visualize_loop(self) -> None:
        snapshot = self.wbc.get_latest_robot_state()
        qpos = self.wbc.snapshot_to_qpos(snapshot)
        targets = self.wbc.ee_targets.get_target()
        with self._visualizer_lock:
            self.visualizer.render(qpos, targets)
        self.viewer_rate.sleep()

    def trajectory_loop(self) -> None:
        while not self._stop_event.is_set():
            target = self.get_target()
            joint_target = self.get_joint_target()
            base_target = self.get_base_target()

            if base_target is not None:
                self.wbc.update_base_targets(base_target)

            if target is None and joint_target is None:
                self.trajectory_rate.sleep()
                continue

            if target is not None:
                duration = target.duration if target.duration and target.duration > 0.0 else self.trajectory_rate.dt
                timestamp = target.timestamp if target.timestamp and target.timestamp > 0.0 else time.monotonic()
                accepted = self.wbc.update_targets(
                    left_pos=target.left_pos,
                    left_quat=target.left_quat,
                    right_pos=target.right_pos,
                    right_quat=target.right_quat,
                    left_width=target.left_width,
                    right_width=target.right_width,
                    head_pos=target.head_pos,
                    head_quat=target.head_quat,
                    duration=duration,
                    timestamp=timestamp,
                )
                if not accepted:
                    self.on_target_rejected(target)
            elif joint_target is not None:
                duration = joint_target.duration if joint_target.duration and joint_target.duration > 0.0 else self.trajectory_rate.dt
                timestamp = joint_target.timestamp if joint_target.timestamp and joint_target.timestamp > 0.0 else time.monotonic()
                accepted = self.wbc.update_joint_targets(
                    left_qpos=joint_target.left_qpos,
                    right_qpos=joint_target.right_qpos,
                    left_width=joint_target.left_width,
                    right_width=joint_target.right_width,
                    head_qpos=getattr(joint_target, "head_qpos", None),
                    duration=duration,
                    timestamp=timestamp,
                )
                if not accepted:
                    self.on_target_rejected(joint_target)

            self.trajectory_rate.sleep()

        self._stop_event.set()

    def run(self) -> None:
        self._trajectory_thread = threading.Thread(
            target=self.trajectory_loop, name="trajectory_streamer", daemon=True
        )
        self._trajectory_thread.start()

        try:
            if self.headless or self.visualizer is None:
                while not self._stop_event.is_set() and self._trajectory_thread.is_alive():
                    time.sleep(0.01)
            else:
                viewer = self.visualizer.viewer
                while viewer.is_running() and not self._stop_event.is_set():
                    self.visualize_loop()
        except KeyboardInterrupt:
            self._stop_event.set()
        finally:
            self._stop_event.set()

    def close(self) -> None:
        self._stop_event.set()
        if self._trajectory_thread is not None:
            self._trajectory_thread.join(timeout=1.0)
            
        if not self.headless and self.visualizer is not None:
            try:
                self.visualizer.viewer.close()
            except Exception:
                pass
