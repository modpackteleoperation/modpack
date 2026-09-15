"""Subprocess target for the GELLO publisher process."""
import os
import sys

from modpack.modules.gello.gello_process import GelloPublisher
from modpack.orchestration.process_shutdown import ShutdownFlag


def run_gello_process(arm: str, log_path, cfg) -> None:
    sys.stdout = open(log_path, "w", buffering=1)
    sys.stderr = sys.stdout
    os.environ["PYTHONUNBUFFERED"] = "1"

    pub = GelloPublisher(
        cfg.gello_config,
        arm,
        cfg.host,
        cfg.port,
        cfg.feedback_port,
        ShutdownFlag(),
        cfg.debug_flags["gello"],
    )
    pub.run()
