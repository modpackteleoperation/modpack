"""Utilities for coordinating graceful shutdown across processes."""

import signal
from typing import Callable

class ShutdownFlag:
    """Lightweight flag shared between components to signal shutdown."""

    def __init__(self) -> None:
        self._requested = False

    def request(self) -> None:
        self._requested = True

    def is_requested(self) -> bool:
        return self._requested


def make_signal_handler(shutdown_flag: ShutdownFlag) -> Callable[[int, object], None]:
    """Return a signal handler that sets the provided shutdown flag."""

    def _handler(signum: int, frame: object) -> None:
        print(f"\nReceived signal {signum}, requesting shutdown...")
        shutdown_flag.request()

    return _handler

def register_signals(
    shutdown_flag: ShutdownFlag, *signals_to_register: int
) -> None:
    """
    Install handlers for the provided signals that set the shutdown flag.

    If no signals are specified, defaults to handling SIGINT and SIGTERM.
    """
    handler = make_signal_handler(shutdown_flag)
    signals = signals_to_register or (signal.SIGINT, signal.SIGTERM)

    for sig in signals:
        signal.signal(sig, handler)
