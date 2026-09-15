import argparse
import logging
import signal
import socketserver
import threading
from typing import Any, Dict, List, Optional

from gripper.gripper import Gripper
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


class _ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


class GripperServer(_ThreadedTCPServer):
    """
    TCP server that exposes the Gripper API to remote machines. Each request is a
    newline-delimited JSON object with a `command` field.
    """

    def __init__(
        self,
        host: str,
        port: int,
        auto_initialize: bool = True,
        auto_homing: bool = True,
        auto_start: bool = True,
        verbose_init: bool = False,
    ):
        self.gripper = Gripper()
        self.lock = threading.Lock()
        super().__init__((host, port), GripperRequestHandler)

        if auto_initialize:
            logging.info("Initializing gripper hardware...")
            if not self.gripper.initialize(verbose=verbose_init):
                raise RuntimeError("Failed to initialize gripper hardware")
        if auto_homing:
            logging.info("Running gripper homing routine...")
            self.gripper.homing()
        if auto_start:
            logging.info("Starting gripper control loop...")
            self.gripper.start()

    def shutdown(self):
        super().shutdown()
        try:
            self.gripper.stop()
        except Exception as exc:  # noqa: broad-except
            logging.warning("Failed to stop gripper cleanly: %s", exc)


class GripperRequestHandler(socketserver.StreamRequestHandler):
    """Request handler that accepts newline-delimited JSON payloads."""

    def handle(self):
        peer = "%s:%s" % self.client_address
        logging.info("Client connected: %s", peer)
        try:
            while True:
                raw_line = self.rfile.readline()
                if not raw_line:
                    break
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    payload = decode_message(raw_line)
                    response = self._process_command(payload)
                except ProtocolError as exc:
                    logging.warning("Malformed message from %s: %s", peer, exc)
                    response = {"ok": False, "error": str(exc)}
                except Exception as exc:  # noqa: broad-except
                    logging.exception("Error while handling command from %s", peer)
                    response = {"ok": False, "error": str(exc)}
                self.wfile.write(encode_message(response))
        finally:
            logging.info("Client disconnected: %s", peer)

    def _process_command(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        cmd = payload.get("command")
        if not cmd:
            raise ProtocolError("Missing 'command' field")

        gripper: Gripper = self.server.gripper
        if cmd == COMMAND_PING:
            return {"ok": True, "result": "pong"}

        if cmd == COMMAND_STATUS:
            return {"ok": True, "result": self._status_payload(gripper)}

        with self.server.lock:
            if cmd == COMMAND_INITIALIZE:
                verbose = bool(payload.get("verbose", False))
                rv = gripper.initialize(verbose=verbose)
                return {"ok": rv, "result": rv}
            if cmd == COMMAND_HOMING:
                rv = gripper.homing()
                return {"ok": bool(rv), "result": bool(rv)}
            if cmd == COMMAND_START:
                gripper.start()
                return {"ok": True, "result": True}
            if cmd == COMMAND_STOP:
                gripper.stop()
                return {"ok": True, "result": True}
            if cmd == COMMAND_SET_TARGET:
                width = self._parse_width(payload)
                gripper.set_target(width)
                return {"ok": True, "result": {"width": width}}

        raise ProtocolError(f"Unsupported command '{cmd}'")

    def _parse_width(self, payload: Dict[str, Any]) -> List[float]:
        width = payload.get("width")
        if width is None:
            width = payload.get("target")
        if (
            not isinstance(width, (list, tuple))
            or len(width) != 2
        ):
            raise ProtocolError("Command 'set_target' expects 'width'=[w_l,w_r]")
        try:
            left = float(width[0])
            right = float(width[1])
        except (TypeError, ValueError) as exc:
            raise ProtocolError("Width values must be numeric") from exc
        return [left, right]

    def _status_payload(self, gripper: Gripper) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "min_q": gripper.min_q.tolist(),
            "max_q": gripper.max_q.tolist(),
        }
        if gripper.target_q is not None:
            result["target_q"] = gripper.target_q.tolist()
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RB-Y1 gripper TCP server")
    parser.add_argument("--host", default="0.0.0.0", help="Interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5678, help="Port to listen on (default: 5678)")
    parser.add_argument(
        "--skip-init",
        action="store_true",
        help="Skip calling Gripper.initialize at startup",
    )
    parser.add_argument(
        "--skip-homing",
        action="store_true",
        help="Skip homing sequence at startup",
    )
    parser.add_argument(
        "--skip-start",
        action="store_true",
        help="Skip starting the background gripper loop at startup",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--verbose-init",
        action="store_true",
        help="Enable verbose logging from Gripper.initialize",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    server = GripperServer(
        host=args.host,
        port=args.port,
        auto_initialize=not args.skip_init,
        auto_homing=not args.skip_homing,
        auto_start=not args.skip_start,
        verbose_init=args.verbose_init,
    )
    logging.info("Gripper server listening on %s:%s", args.host, args.port)

    def handle_signal(signum, _frame):
        logging.info("Received signal %s, shutting down...", signum)
        server.shutdown()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handle_signal)

    try:
        server.serve_forever()
    finally:
        server.server_close()
        logging.info("Server stopped.")


if __name__ == "__main__":
    main()
