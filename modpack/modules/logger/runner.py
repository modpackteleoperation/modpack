"""Subprocess target for the backpack stream + camera logger process."""
import os
import sys
import threading

from modpack.modules.logger.stream_logger import BackpackCameraLogger, BackpackStreamLogger


def run_backpack_logger_process(log_path, cfg) -> None:
    sys.stdout = open(log_path, "w", buffering=1)
    sys.stderr = sys.stdout
    os.environ["PYTHONUNBUFFERED"] = "1"

    # Use the endpoint already computed by the coordinator from the robot config's topology.
    camera_endpoint = cfg.camera_endpoint or f"tcp://{cfg.fixed_gello_ip}:{cfg.manager_cfg['robotmq'].get('cam_port', 5570)}"

    stream_logger = BackpackStreamLogger(
        host=cfg.fixed_gello_ip,
        port=cfg.port,
        logging_streams=cfg.logging_streams,
        feedback_port=cfg.feedback_port,
    )
    cam = BackpackCameraLogger(
        camera_endpoint=camera_endpoint,
        logging_streams=cfg.logging_streams,
    )

    t1 = threading.Thread(target=stream_logger.run, daemon=True)
    t2 = threading.Thread(target=cam.run, daemon=True)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
