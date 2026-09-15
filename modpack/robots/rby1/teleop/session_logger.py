from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "log"


def generate_session_name(prefix: str = "teleop") -> str:
    """Return a timestamped session name shared across logs and trajectories."""
    return f"{prefix}_{time.strftime('%Y%m%d-%H%M%S')}"


class SessionLogger:
    """Thread-safe JSONL logger for controller samples and socket timeouts."""

    def __init__(self, session_name: str, log_dir: Path | None = None) -> None:
        self.session_name = session_name
        self._log_dir = log_dir or LOG_DIR
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._controller_path = self._log_dir / f"{self.session_name}.jsonl"
        self._timeout_path = self._log_dir / f"{self.session_name}_timeouts.jsonl"
        self._lock = threading.Lock()
        self._controller_samples: List[Dict] = []
        self._socket_timeouts: List[float] = []
        self._closed = False

    @staticmethod
    def _sanitize_pose_entry(hand: Dict) -> Dict:
        return {
            "position": [float(x) for x in hand.get("position", [])],
            "rotation": [float(x) for x in hand.get("rotation", [])],
        }

    def log_controller_state(self, controller_state: Dict) -> None:
        entry: Dict[str, object] = {"timestamp": float(time.time())}
        hands = controller_state.get("hands", {})
        left = hands.get("left")
        right = hands.get("right")
        head = controller_state.get("head")
        if left:
            entry["left"] = self._sanitize_pose_entry(left)
        if right:
            entry["right"] = self._sanitize_pose_entry(right)
        if head:
            entry["head"] = self._sanitize_pose_entry(head)
        with self._lock:
            self._controller_samples.append(entry)

    def log_socket_timeout(self) -> None:
        with self._lock:
            self._socket_timeouts.append(time.time())

    def close(self) -> None:
        if self._closed:
            return
        with self._lock:
            controller_samples = list(self._controller_samples)
            socket_timeouts = list(self._socket_timeouts)
        timeout_entries = [{"timestamp": ts} for ts in socket_timeouts]
        self._write_jsonl(self._controller_path, controller_samples)
        self._write_jsonl(self._timeout_path, timeout_entries)
        self._closed = True

    def _write_jsonl(self, path: Path, entries: List[Dict]) -> None:
        if not entries:
            return
        with path.open("w", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(entry))
                fh.write("\n")
