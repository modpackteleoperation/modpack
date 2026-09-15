"""SubsystemSpec registration for the Vision Pro module.

Importing this module registers the vision_pro spec with REGISTRY.
"""
import os
import time

from robotmq.utils import serialize

from modpack.orchestration._registry import REGISTRY, SubsystemSpec
from modpack.orchestration.message_formats import Topics, serialize_message
from modpack.modules.vision_pro.runner import run_vision_pro_process


def _vp_setup(manager) -> None:
    manager.vision_pro_activated = False
    manager.vision_pro_ready = False
    manager._vision_pro_ready_notified = False
    manager.vp_bypass_neck = os.getenv("VP_BYPASS_NECK") == "1"
    manager._vp_bypass_last_pub = 0.0
    manager.neck_at_init_flag = False
    manager.neck_ready_for_commands = False
    manager.neck_activated = False
    manager._notified_neck_ready = False
    manager.iphone_ready_flag = False
    manager._notified_iphone_ready = False
    # Bus topics (VP_STATUS, NECK_STATUS, NECK_CMD, ...) are registered
    # unconditionally by the coordinator — see _register_activation_topics


def _vp_ready(manager):
    if not manager.active_systems.get("vision_pro"):
        return None
    if getattr(manager, "vp_bypass_neck", False):
        with manager._status_lock:
            manager.neck_ready_for_commands = True
        if not manager._notified_neck_ready:
            print("DEBUG: Neck READY (bypass)")
            manager._notified_neck_ready = True
        _maybe_publish_vp_bypass_status(manager)
    with manager._status_lock:
        vp_ready = manager.vision_pro_ready
        neck_ready = manager.neck_ready_for_commands
    if vp_ready and neck_ready and not manager._vision_pro_ready_notified:
        print("Vision Pro stream ready: press 'v' when you want to start publishing head data.")
        manager._vision_pro_ready_notified = True
    return None


def _vp_activation(manager, subsystem: str, new_state: bool) -> None:
    with manager._status_lock:
        neck_init_ok = manager.neck_at_init_flag or getattr(manager, "vp_bypass_neck", False)
        neck_start_ok = manager.neck_ready_for_commands or getattr(manager, "vp_bypass_neck", False)
        iphone_ok = manager.iphone_ready_flag or not manager.active_systems.get("iphone", False)
        vp_ready = manager.vision_pro_ready
        vp_activated = manager.vision_pro_activated

    if not vp_ready:
        if not (neck_init_ok and iphone_ok):
            print("\nCannot signal Vision Pro READY yet - waiting for prerequisites")
            print(f"  - neck_at_init_flag: {manager.neck_at_init_flag}")
            print(f"  - iphone_ready_flag: {iphone_ok}")
            print(f"  - vp_bypass_neck: {getattr(manager, 'vp_bypass_neck', False)}")
            return
        try:
            vp_status = manager.message_factory.create_vision_pro_status_message(vision_pro_ready=True)
            manager.publish_to_activation(Topics.VP_STATUS, serialize_message(vp_status))
            with manager._status_lock:
                manager.vision_pro_ready = True
            print("\nVision Pro READY signaled. Triple tap again to start/stop publishing.")
        except Exception as e:
            print(f"Failed to publish Vision Pro READY: {e}")
        return

    if not (neck_start_ok and iphone_ok):
        print("\nVision Pro publishing not ready yet - waiting for neck/iPhone status")
        print(f"  - neck_ready_for_commands (at_start): {neck_start_ok}")
        print(f"  - iphone_ready_flag: {iphone_ok}")
        print(f"  - vp_bypass_neck: {getattr(manager, 'vp_bypass_neck', False)}")
        return

    with manager._status_lock:
        manager.vision_pro_activated = not vp_activated
        new_activated = manager.vision_pro_activated

    command = "activate" if new_activated else "deactivate"
    status = "ACTIVATED" if new_activated else "DEACTIVATED"
    print(f"\nVISION PRO {status}")

    try:
        activation_msg = {
            "command": command,
            "target": "vision_pro",
            "timestamp": time.time(),
            "source": "system_manager",
        }
        manager.publish_to_activation(Topics.VP_PUBLISH_ACTIVATION, serialize(activation_msg))
        print(f"Vision Pro command sent: {command}")
    except Exception as e:
        print(f"CRITICAL ERROR sending Vision Pro command: {e}")

    try:
        vp_status = manager.message_factory.create_vision_pro_status_message(
            vision_pro_ready=True, vision_pro_activated=new_activated
        )
        manager.publish_to_activation(Topics.VP_STATUS, serialize_message(vp_status))
    except Exception as e:
        print(f"Failed to broadcast VP activation state: {e}")

    time.sleep(5)
    try:
        neck_cmd = {
            "timestamp": time.time(),
            "command": "start_episode" if new_activated else "deactivate",
            "target": "neck",
            "source": "system_manager",
        }
        manager.publish_to_activation(Topics.NECK_ACTIVATION, serialize(neck_cmd))
    except Exception as e:
        print(f"Failed to publish neck activation: {e}")

    # Activate/deactivate the neck bridge (listens on robot_activation, not NECK_ACTIVATION)
    try:
        manager.publish_activation_command(command, target="neck")
    except Exception as e:
        print(f"Failed to publish neck bridge activation: {e}")


def _vp_shutdown(manager) -> None:
    with manager._status_lock:
        manager.vision_pro_activated = False
        manager.vision_pro_ready = False


def _maybe_publish_vp_bypass_status(manager) -> None:
    now = time.monotonic()
    if now - getattr(manager, "_vp_bypass_last_pub", 0.0) < 1.0:
        return
    manager._vp_bypass_last_pub = now
    try:
        neck_status = manager.message_factory.create_neck_status_message(at_init=True, at_start=True)
        manager.publish_to_activation(Topics.NECK_STATUS, serialize_message(neck_status))
        iphone_required = manager.active_systems.get("iphone", False)
        if iphone_required and not getattr(manager, "iphone_ready_flag", False):
            return
        vp_status = manager.message_factory.create_vision_pro_status_message(vision_pro_ready=True)
        manager.publish_to_activation(Topics.VP_STATUS, serialize_message(vp_status))
        if not manager.vision_pro_ready:
            print("DEBUG: Vision Pro READY (bypass)")
        manager.vision_pro_ready = True
    except Exception as e:
        print(f"[VisionPro] bypass status publish error: {e}")



REGISTRY.register(SubsystemSpec(
    name="vision_pro",
    active_key="vision_pro",
    pc="gello",
    log_label="Vision Pro",
    runner=run_vision_pro_process,
    setup_fn=_vp_setup,
    ready_fn=_vp_ready,
    activation_fn=_vp_activation,
    shutdown_fn=_vp_shutdown,
))
