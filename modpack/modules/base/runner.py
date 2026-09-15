"""Subprocess target for the base process."""
import os
import sys

from modpack.modules.base.base_process import BaseProcess


def run_base_process(log_path, cfg) -> None:
    sys.stdout = open(log_path, "w", buffering=1)
    sys.stderr = sys.stdout
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["AMENT_PREFIX_PATH"] = "/opt/ros/humble"

    base_overrides = dict(cfg.module_overrides.get("base", {}) or {})
    rmq = {**dict(cfg.manager_cfg.get("robotmq", {}) or {}), **base_overrides}
    # The data server (body / body_target / body_cmd topics) runs on the gello PC.
    # The manager config host is the bind address (0.0.0.0); the base data client
    # must dial the gello PC explicitly so the Robot PC base can read body_target.
    rmq["host"] = cfg.fixed_gello_ip
    # Base hardware (CAN/phoenix6) lives on the Robot PC only, which also hosts the
    # iPhone WebXR teleop policy so the base command + control are co-located (no
    # cross-PC hop). The gello PC base module runs coordination only and never
    # touches Base().
    rmq["owns_base_hardware"] = (cfg.pc_id == "robot")

    process = BaseProcess(
        activation_endpoint=f"tcp://{cfg.fixed_gello_ip}:{cfg.activation_port}",
        config=rmq,
        debug=cfg.debug_flags.get("base", False),
    )
    process.run()
