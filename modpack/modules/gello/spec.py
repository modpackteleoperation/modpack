"""SubsystemSpec registration for the GELLO publisher module.

Importing this module registers the gello spec with REGISTRY.
"""
import multiprocessing as mp
import time

from modpack.orchestration._registry import REGISTRY, SubsystemSpec
from modpack.orchestration.process_runners import ProcessRunConfig
from modpack.modules.gello.runner import run_gello_process


def _gello_setup(manager) -> None:
    manager.gello_processes = {}
    manager.gello_log_files = {}
    manager.gello_log_paths = {}
    manager.gello_topics = {}
    manager.gello_ready_flags = {arm: False for arm in manager.arms}
    manager._gello_launch_pending = bool(manager.active_systems.get("gello", False))


def _gello_start(manager, cfg: ProcessRunConfig, timestamp: float) -> bool:
    if not manager.active_systems.get("gello"):
        manager._gello_launch_pending = False
        return True

    if getattr(manager, "_gello_launch_pending", False):
        print("\nGELLO is pending activation (quadruple tap to launch).")
        return True

    print("\nStarting GELLO arm processes...")
    for arm in manager.arms:
        log_path = manager.log_dir / f"gello_{arm}_{timestamp}.log"
        manager.gello_log_paths[arm] = log_path
        print(f"  Log: {log_path}")
        process = mp.Process(target=run_gello_process, args=(arm, log_path, cfg))
        process.start()
        manager.gello_processes[arm] = process
        print(f"GELLO {arm} process started (PID: {process.pid})")
        time.sleep(0.5)
        if not process.is_alive():
            print(f"WARNING: GELLO {arm} process died immediately")
            manager.gello_ready_flags[arm] = False
            return False
        manager.gello_ready_flags[arm] = True

    manager._gello_launch_pending = False
    return True


def _gello_shutdown(manager) -> None:
    for arm, process in list(getattr(manager, "gello_processes", {}).items()):
        if process and process.is_alive():
            print(f"Terminating GELLO {arm} process...")
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()


def _gello_ready(manager):
    if not manager.active_systems.get("gello"):
        return None
    if manager.pc_id != "gello":
        return None
    flags = getattr(manager, "gello_ready_flags", {})
    pending = getattr(manager, "_gello_launch_pending", False)
    if pending:
        return (False, "GELLO not yet launched")
    all_ready = all(flags.values()) if flags else False
    if not all_ready:
        return (False, "GELLO not ready")
    return (True, "")


def _gello_log_entries(manager):
    entries = []
    for arm, log_path in getattr(manager, "gello_log_paths", {}).items():
        entries.append((f"GELLO {arm}", log_path))
    return entries


REGISTRY.register(SubsystemSpec(
    name="gello",
    active_key="gello",
    pc="gello",
    log_label="GELLO",
    setup_fn=_gello_setup,
    start_fn=_gello_start,
    shutdown_fn=_gello_shutdown,
    ready_fn=_gello_ready,
    log_entries_fn=_gello_log_entries,
))
