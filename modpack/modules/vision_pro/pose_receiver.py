#!/usr/bin/env python3
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Swift packet format:
#   [0]      : UInt8 valid
#   [1..8]   : Float64 timestamp seconds (monotonic)
#   [9..]    : 16 Float32 matrix entries, column-major (numpy order="F")
PKT_SIZE = 1 + 8 + 16 * 4  # 73 bytes


@dataclass
class PosePacket:
    valid: bool
    ts: float
    Tw_device: np.ndarray  # 4x4 float32


class PoseReceiver:
    """
    Background UDP receiver for the native Vision Pro pose stream.
    Keeps only the latest valid packet.
    """

    def __init__(self, bind_ip: str = "0.0.0.0", port: int = 5005, timeout_s: float = 1.0):
        self.bind_ip = bind_ip
        self.port = int(port)
        self.timeout_s = float(timeout_s)

        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        self._lock = threading.Lock()
        self._latest: Optional[PosePacket] = None
        self._last_rx_walltime: float = 0.0
        self._last_addr: Optional[Tuple[str, int]] = None

    def start(self) -> None:
        if self._thread is not None:
            return

        # Clear any stop signal left set by a prior stop(); otherwise a restart on
        # the same instance spawns a thread whose `while not self._stop.is_set()`
        # loop exits immediately and the receiver silently dies (see TeleopVR.start).
        self._stop.clear()

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Allow rebinding a socket left lingering by a prior instance (common on a
        # quick restart) instead of failing the whole launch with "address in use".
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._sock.bind((self.bind_ip, self.port))
        except OSError as exc:
            self._sock.close()
            self._sock = None
            raise RuntimeError(
                f"[PoseReceiver] Could not bind UDP {self.bind_ip}:{self.port} ({exc}). "
                "Another Vision Pro / pose receiver may already be running."
            ) from exc
        self._sock.settimeout(self.timeout_s)

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[PoseReceiver] Listening UDP on {self.bind_ip}:{self.port}")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None

        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
        self._sock = None
        print("[PoseReceiver] Stopped")

    def _run(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                data, addr = self._sock.recvfrom(2048)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[PoseReceiver] recv error: {e}")
                continue

            if len(data) != PKT_SIZE:
                continue

            try:
                valid_u8 = data[0]
                ts = struct.unpack_from("<d", data, 1)[0]
                mat = np.frombuffer(data, dtype=np.float32, count=16, offset=1 + 8)
                Tw = mat.reshape((4, 4), order="F").copy()  # column-major -> 4x4
            except Exception as e:
                print(f"[PoseReceiver] unpack error: {e}")
                continue

            # Basic sanity
            if not np.isfinite(Tw).all():
                continue

            pkt = PosePacket(valid=bool(valid_u8), ts=float(ts), Tw_device=Tw)

            with self._lock:
                self._last_rx_walltime = time.time()
                self._last_addr = (addr[0], addr[1])
                # Keep latest packet even if invalid (so you can observe invalid periods)
                self._latest = pkt

    def get_latest(self) -> Optional[PosePacket]:
        with self._lock:
            return self._latest

    def clear_latest(self) -> None:
        """Drop the cached packet so callers wait for a fresh post-event sample."""
        with self._lock:
            self._latest = None
