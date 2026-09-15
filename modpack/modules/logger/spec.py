"""SubsystemSpec registration for the backpack stream logger module.

Importing this module registers the logger spec with REGISTRY.
The logger is always-on (active_key=None) — it starts on every GELLO PC run.
"""

import multiprocessing as mp
import time

from modpack.orchestration._registry import REGISTRY, SubsystemSpec
from modpack.modules.logger.runner import run_backpack_logger_process


def _logger_setup(manager) -> None:
    manager.backpack_logger_process = None
    manager.backpack_log_path = None


def _logger_start(manager, cfg, timestamp: float) -> bool:
    log_path = manager.log_dir / f"backpack_logger_{timestamp}.log"
    manager.backpack_log_path = log_path
    process = mp.Process(target=run_backpack_logger_process, args=(log_path, cfg))
    process.start()
    manager.backpack_logger_process = process
    time.sleep(0.5)
    alive = process.is_alive()
    if not alive:
        print("WARNING: Backpack logger process died immediately")
    return alive


def _logger_shutdown(manager) -> None:
    proc = getattr(manager, "backpack_logger_process", None)
    if proc and proc.is_alive():
        print("Terminating backpack logger process...")
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()


def _logger_log_entries(manager):
    return [("Backpack Logger", getattr(manager, "backpack_log_path", None))]


REGISTRY.register(SubsystemSpec(
    name="logger",
    active_key=None,   # always-on on the GELLO PC
    pc="gello",
    log_label="Backpack Logger",
    setup_fn=_logger_setup,
    start_fn=_logger_start,
    shutdown_fn=_logger_shutdown,
    log_entries_fn=_logger_log_entries,
))
