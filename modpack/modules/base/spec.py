"""SubsystemSpec registration for the base process module.

Importing this module registers the base spec with REGISTRY.
"""
import multiprocessing as mp
import time
from modpack.orchestration._registry import REGISTRY, SubsystemSpec
from modpack.orchestration.process_runners import ProcessRunConfig
from modpack.modules.base.runner import run_base_process


def _base_setup(manager) -> None:
    manager.base_process = None
    manager.base_log_path = None
    manager.base_ready_flag = False


def _base_start(manager, cfg: "ProcessRunConfig", timestamp: float) -> bool:
    if not manager.active_systems.get("base"):
        return True

    log_path = manager.log_dir / f"base_{timestamp}.log"
    manager.base_log_path = log_path
    print(f"Starting Base process (logs: {log_path})")
    print("Base process includes RPC server + iPhone WebXR control")

    manager.base_process = mp.Process(target=run_base_process, args=(log_path, cfg))
    manager.base_process.start()
    print(f"Base process started (PID: {manager.base_process.pid})")
    print("Waiting for base to initialize (includes web server startup)...")
    time.sleep(0.5)

    if not manager.base_process.is_alive():
        print("WARNING: Base process died immediately")
        manager.base_ready_flag = False
        return False

    print("Base process running")
    manager.base_ready_flag = True
    return True


def _base_shutdown(manager) -> None:
    proc = getattr(manager, "base_process", None)
    if proc and proc.is_alive():
        print("Terminating Base process...")
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()


def _base_ready(manager):
    if not manager.active_systems.get("base"):
        return None
    activated = getattr(manager, "_activated_subsystems", {}).get("base", False)
    module_alive = (
        getattr(manager, "base_ready_flag", False)
        and getattr(manager, "base_process", None) is not None
        and manager.base_process.is_alive()
    )
    # The base hardware lives on the Robot PC. The gello PC runs the base module
    # (WebXR/teleop) but never owns hardware, so its readiness also requires the
    # Robot PC's base to be ready (reported over the peer-readiness bus).
    if getattr(manager, "pc_id", "gello") == "robot":
        ok = activated and module_alive
    else:
        robot_ready = getattr(manager, "_peer_ready", {}).get("robot_pc", {}).get("ready", False)
        ok = activated and module_alive and robot_ready
    return ok, "base not ready"


def _base_log_entries(manager):
    return [("Base Process", getattr(manager, "base_log_path", None))]


REGISTRY.register(SubsystemSpec(
    name="base",
    active_key="base",
    pc="both",
    log_label="Base",
    setup_fn=_base_setup,
    start_fn=_base_start,
    shutdown_fn=_base_shutdown,
    ready_fn=_base_ready,
    log_entries_fn=_base_log_entries,
))
