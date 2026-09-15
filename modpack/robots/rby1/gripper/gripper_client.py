import argparse
import json
import socket
import threading
from typing import Any, Dict, List, Optional, Sequence

from gripper.network_protocol import (
    COMMAND_HOMING,
    COMMAND_INITIALIZE,
    COMMAND_PING,
    COMMAND_SET_TARGET,
    COMMAND_START,
    COMMAND_STATUS,
    COMMAND_STOP,
    ProtocolError,
    decode_message,
    encode_message,
)


class GripperClient:
    """
    Lightweight proxy that mirrors the Gripper API but forwards all commands to
    a TCP server running on the machine that hosts the real gripper hardware.
    """

    def __init__(self, host: str, port: int = 5678, timeout: float = 2.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._connect()

    def initialize(self, verbose: bool = False) -> bool:
        return bool(self._request(COMMAND_INITIALIZE, verbose=verbose))

    def homing(self) -> bool:
        return bool(self._request(COMMAND_HOMING))

    def start(self) -> bool:
        return bool(self._request(COMMAND_START))

    def stop(self) -> bool:
        return bool(self._request(COMMAND_STOP))

    def set_target(self, target_width: Sequence[float]) -> None:
        width = self._parse_width(target_width)
        self._request(COMMAND_SET_TARGET, width=width)

    def status(self) -> Dict[str, Any]:
        result = self._request(COMMAND_STATUS)
        if not isinstance(result, dict):
            raise ProtocolError("Malformed status response")
        return result

    def ping(self) -> str:
        result = self._request(COMMAND_PING)
        if isinstance(result, str):
            return result
        raise ProtocolError("Unexpected ping response")

    def close(self) -> None:
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                self._sock.close()
                self._sock = None
            self._buffer.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def _request(self, command: str, **payload: Any) -> Any:
        message = {"command": command}
        if payload:
            message.update(payload)

        attempt = 0
        while True:
            with self._lock:
                try:
                    self._send(encode_message(message))
                    response = self._receive()
                except (OSError, ProtocolError):
                    self._buffer.clear()
                    self._reconnect_locked()
                    attempt += 1
                    if attempt > 1:
                        raise
                    continue
            if not isinstance(response, dict):
                raise ProtocolError("Server response is not a JSON object")
            if not response.get("ok", False):
                raise RuntimeError(response.get("error", "Unknown error"))
            return response.get("result")

    def _send(self, data: bytes) -> None:
        self._ensure_socket()
        assert self._sock is not None
        self._sock.sendall(data)

    def _receive(self) -> Dict[str, Any]:
        self._ensure_socket()
        assert self._sock is not None
        while True:
            newline_idx = self._buffer.find(b"\n")
            if newline_idx != -1:
                line = bytes(self._buffer[:newline_idx])
                del self._buffer[: newline_idx + 1]
                return decode_message(line)
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ProtocolError("Connection closed by server")
            self._buffer.extend(chunk)

    def _parse_width(self, width: Sequence[float]) -> List[float]:
        if len(width) != 2:
            raise ValueError("Gripper target must be a sequence of two floats")
        try:
            left = float(width[0])
            right = float(width[1])
        except (TypeError, ValueError) as exc:
            raise ValueError("Gripper width values must be numeric") from exc
        return [left, right]

    def _connect(self) -> None:
        with self._lock:
            self._reconnect_locked()

    def _ensure_socket(self) -> None:
        if self._sock is None:
            self._reconnect_locked()

    def _reconnect_locked(self) -> None:
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._sock.close()
        self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._buffer.clear()


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send commands to a remote RB-Y1 gripper server.")
    parser.add_argument("--host", required=True, help="Gripper server host/IP")
    parser.add_argument("--port", type=int, default=5678, help="Gripper server port (default: 5678)")
    parser.add_argument("--timeout", type=float, default=2.0, help="Socket timeout in seconds")
    return parser.parse_args()


def _print_help() -> None:
    print(
        "Available commands:\n"
        "  set-target <left> <right>  Set finger widths in meters\n"
        "  status                     Query the gripper status\n"
        "  ping                       Ping the gripper server\n"
        "  start                      Start the gripper control loop\n"
        "  stop                       Stop the gripper control loop\n"
        "  initialize                 Re-run gripper initialization\n"
        "  homing                     Run the gripper homing routine\n"
        "  help                       Show this message\n"
        "  quit/exit                  Close the client\n"
    )


def _command_loop(client: GripperClient) -> None:
    _print_help()
    while True:
        try:
            raw = input("gripper> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue
        lowered = raw.lower()
        if lowered in {"quit", "exit"}:
            break
        if lowered in {"help", "?"}:
            _print_help()
            continue

        parts = raw.split()
        command = parts[0].lower()
        args = parts[1:]

        try:
            if command == "set-target":
                if len(args) != 2:
                    print("Usage: set-target <left> <right>")
                    continue
                left, right = float(args[0]), float(args[1])
                client.set_target([left, right])
                print(f"Target set to ({left:.3f}, {right:.3f}) m")
            elif command == "status":
                status = client.status()
                print(json.dumps(status, indent=2))
            elif command == "ping":
                print(client.ping())
            elif command == "start":
                client.start()
                print("Gripper loop started")
            elif command == "stop":
                client.stop()
                print("Gripper loop stopped")
            elif command == "initialize":
                rv = client.initialize(verbose=True)
                print("Initialize returned", rv)
            elif command == "homing":
                rv = client.homing()
                print("Homing returned", rv)
            else:
                print(f"Unknown command: {command}. Type 'help' to list commands.")
        except Exception as exc:  # noqa: BLE001
            print(f"Error while executing {command}: {exc}")


def main() -> None:
    args = _parse_cli_args()
    with GripperClient(args.host, port=args.port, timeout=args.timeout) as client:
        _command_loop(client)

if __name__ == "__main__":
    main()
