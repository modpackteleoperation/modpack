import numpy as np
from typing import Dict, Optional


class SafetyMonitor:
    """
    Two independent latching safety checks for the GELLO leader arm.

    Joint limits: latches when any joint exits [min, max]; unlocks when all
    joints return inside their limits.

    Step delta: latches when any joint moves more than max_step_delta rad in
    one cycle; unlocks when all joints return within latch_return_tolerance of
    the position at violation time.

    Both checks are independent, so either can be latched while the other is not.
    While latched, commands are suppressed (is_safe() returns False) and the
    console prints per-joint distance from the return condition each cycle.
    """

    def __init__(self, safety_config: dict):
        required = ('enable_joint_limits', 'enable_step_delta', 'max_step_delta', 'latch_return_tolerance')
        missing = [k for k in required if k not in safety_config]
        if missing:
            raise ValueError(f"safety_config missing required keys: {missing}")
        self.enable_joint_limits = safety_config['enable_joint_limits']
        self.enable_step_delta = safety_config['enable_step_delta']
        self.max_step_delta = float(safety_config['max_step_delta'])
        self.latch_return_tolerance = float(safety_config['latch_return_tolerance'])

        # joint_limits: {joint_idx: {min: float, max: float}}
        raw_limits = safety_config.get('joint_limits') or {}
        self.joint_limits: Dict[int, Dict] = {}
        for k, v in raw_limits.items():
            if isinstance(v, dict) and 'min' in v and 'max' in v:
                self.joint_limits[int(k)] = {'min': float(v['min']), 'max': float(v['max'])}
                print(f"[SafetyMonitor] Joint {k} limits: [{v['min']:.3f}, {v['max']:.3f}] rad")

        # Joint limit latch state
        self._jl_latched = False

        # Step delta latch state
        self._sd_latched = False
        self._sd_latch_position: Optional[np.ndarray] = None
        self._sd_prev_positions: Optional[np.ndarray] = None

        print(
            f"[SafetyMonitor] init — joint_limits={len(self.joint_limits)} joints, "
            f"enable_joint_limits={self.enable_joint_limits}, "
            f"enable_step_delta={self.enable_step_delta}, "
            f"max_step_delta={self.max_step_delta} rad"
        )

    def is_safe(self, joint_positions: np.ndarray) -> bool:
        positions = np.asarray(joint_positions, dtype=float)
        safe = True

        if self.enable_joint_limits:
            if not self._check_joint_limits(positions):
                safe = False

        if self.enable_step_delta:
            if not self._check_step_delta(positions):
                safe = False

        return safe

    # =========================================================================
    # Joint limit check
    # =========================================================================

    def _check_joint_limits(self, positions: np.ndarray) -> bool:
        if self._jl_latched:
            # Check if all joints are back inside limits
            all_clear = True
            for i, pos in enumerate(positions):
                if i not in self.joint_limits:
                    continue
                lim = self.joint_limits[i]
                if pos < lim['min'] or pos > lim['max']:
                    dist = max(lim['min'] - pos, pos - lim['max'])
                    print(f"[SAFETY] Joint limit latched — Joint {i}: {dist:.3f} rad outside limit")
                    all_clear = False
            if all_clear:
                self._jl_latched = False
                print("[SAFETY] Joint limit latch cleared — resuming")
                return True
            return False

        # Not latched — check for new violations
        violated = False
        for i, pos in enumerate(positions):
            if i not in self.joint_limits:
                continue
            lim = self.joint_limits[i]
            if pos < lim['min'] or pos > lim['max']:
                print(
                    f"[SAFETY] Joint {i} limit violation: {pos:.3f} "
                    f"not in [{lim['min']:.3f}, {lim['max']:.3f}]"
                )
                violated = True

        if violated:
            self._jl_latched = True
        return not violated

    # =========================================================================
    # Step delta check
    # =========================================================================

    def _check_step_delta(self, positions: np.ndarray) -> bool:
        if self._sd_prev_positions is None:
            self._sd_prev_positions = positions.copy()
            return True

        if self._sd_latched:
            # Check if all joints returned within tolerance of latch position
            dists = np.abs(positions - self._sd_latch_position)
            offending = np.where(dists > self.latch_return_tolerance)[0]
            if len(offending) == 0:
                self._sd_latched = False
                self._sd_latch_position = None
                self._sd_prev_positions = positions.copy()
                print("[SAFETY] Step delta latch cleared — resuming")
                return True
            for i in offending:
                print(
                    f"[SAFETY] Step delta latched — Joint {i}: "
                    f"{dists[i]:.3f} rad from return position"
                )
            return False

        # Not latched — check for new violations
        deltas = np.abs(positions - self._sd_prev_positions)
        offending = np.where(deltas > self.max_step_delta)[0]

        if len(offending) > 0:
            for i in offending:
                print(
                    f"[SAFETY] Joint {i} step delta violation: "
                    f"{deltas[i]:.3f} rad (limit {self.max_step_delta:.3f})"
                )
            self._sd_latched = True
            self._sd_latch_position = self._sd_prev_positions.copy()
            return False

        self._sd_prev_positions = positions.copy()
        return True
