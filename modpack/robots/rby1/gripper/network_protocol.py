import json
from typing import Any, Dict

COMMAND_INITIALIZE = "initialize"
COMMAND_HOMING = "homing"
COMMAND_START = "start"
COMMAND_STOP = "stop"
COMMAND_SET_TARGET = "set_target"
COMMAND_STATUS = "status"
COMMAND_PING = "ping"


class ProtocolError(Exception):
    """Raised when a malformed message is encountered."""


def encode_message(payload: Dict[str, Any]) -> bytes:
    """
    Serialize a payload dict into the newline-delimited JSON format shared by the
    gripper client and server.
    """
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("ascii") + b"\n"


def decode_message(raw_line: bytes) -> Dict[str, Any]:
    """Parse a newline-delimited JSON message."""
    try:
        return json.loads(raw_line.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Invalid JSON: {exc}") from exc
