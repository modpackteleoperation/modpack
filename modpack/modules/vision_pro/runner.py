"""Subprocess entry point for the Vision Pro process."""
import os
import sys
from modpack.modules.vision_pro.vision_pro_process import VisionPro

def run_vision_pro_process(log_path, cfg) -> None:
    sys.stdout = open(log_path, "w", buffering=1)
    sys.stderr = sys.stdout
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["AMENT_PREFIX_PATH"] = "/opt/ros/humble"

    sub = VisionPro(
        cfg.fixed_gello_ip,
        cfg.port,
        cfg.activation_host,
        cfg.activation_port,
        cfg.debug_flags["vision_pro"],
    )
    sub.run()
