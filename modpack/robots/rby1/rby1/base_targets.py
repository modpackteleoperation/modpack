from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class BaseTargets:
    """Thread-safe shared base pose target for mobility commands."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    timestamp: float = 0.0

    x: Optional[float] = None
    y: Optional[float] = None
    quat: Optional[np.ndarray] = None  # [w, x, y, z] — MuJoCo convention

    def set_targets(
        self,
        x: float,
        y: float,
        quat: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> None:
        with self.lock:
            self.x = float(x)
            self.y = float(y)
            self.quat = np.array(quat, dtype=np.float64)
            self.timestamp = time.monotonic() if timestamp is None else float(timestamp)

    def get_target(self) -> Tuple[Optional[float], Optional[float], Optional[np.ndarray]]:
        with self.lock:
            quat = None if self.quat is None else self.quat.copy()
            return self.x, self.y, quat


__all__ = ["BaseTargets"]
