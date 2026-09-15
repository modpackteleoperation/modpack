"""Activation monitor and status publisher threads.

ActivationMonitor wraps the two background threads that were previously
inlined inside SimpleMessageQueueManager:

* poll thread which subscribes to peer/neck/VP/iPhone/episode RMQ topics and
  updates state flags on the manager.
* publish thread which broadcasts this PC's readiness status at 2 Hz.

Usage inside the manager::

    self.activation_monitor = ActivationMonitor(self)
    # start both threads:
    self.activation_monitor.start()
    # stop both threads on shutdown:
    self.activation_monitor.stop()
"""
from __future__ import annotations

import threading
import time

from modpack.orchestration.message_formats import (
    Topics,
    deserialize_neck_iphone_status_message,
    deserialize_neck_status_message,
    deserialize_vp_status_message,
    deserialize_system_ready_message,
    deserialize_episode_state_message,
)

class ActivationMonitor:
    """Background polling + status-publishing threads for activation state.

    Holds a reference to manager to read/write activation flags
    (neck_ready_for_commands, vision_pro_ready, etc.).  All state
    mutations use manager._status_lock where appropriate.
    """

    def __init__(self, manager) -> None:
        self._mgr = manager
        self._poll_running = False
        self._pub_running = False
        self._poll_thread: threading.Thread | None = None
        self._pub_thread: threading.Thread | None = None

    # =========================================================================
    # Public interface
    # =========================================================================

    def start(self) -> None:
        """Start the poll thread and the status-publisher thread."""
        mgr = self._mgr

        self._pub_running = True
        self._pub_thread = threading.Thread(
            target=self._publish_loop,
            daemon=True,
            name="status_publisher",
        )
        self._pub_thread.start()

        self._poll_running = True
        mgr._monitor_running = True  # kept for back-compat reads in manager
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="activation_monitor",
        )
        self._poll_thread.start()

    def stop(self) -> None:
        """Signal both threads to stop and join them."""
        self._poll_running = False
        self._pub_running = False
        mgr = self._mgr
        mgr._monitor_running = False

        if self._poll_thread:
            self._poll_thread.join(timeout=2.0)
            if self._poll_thread.is_alive():
                print("Warning: activation monitor thread did not stop cleanly")
            else:
                print("Activation monitor thread stopped")
        if self._pub_thread:
            self._pub_thread.join(timeout=2.0)

    # =========================================================================
    # Poll thread
    # =========================================================================

    def _poll_loop(self) -> None:
        mgr = self._mgr
        while self._poll_running:
            try:
                # Peer readiness
                if mgr.node_id == "gello_pc":
                    peer_key, peer_topic = "robot_pc", Topics.ROBOT_PC_STATUS
                else:
                    peer_key, peer_topic = "gello_pc", Topics.GELLO_PC_STATUS

                data_list, _ = mgr.activation_client.peek_data(peer_topic, timeout_s=0.5, n=-1)
                if data_list:
                    payload = data_list[0]
                    try:
                        status_msg = (
                            deserialize_system_ready_message(payload)
                            if isinstance(payload, (bytes, bytearray))
                            else payload
                        )
                        with mgr._status_lock:
                            if status_msg.ready:
                                mgr._peer_ready[peer_key]["ready"] = True
                                mgr._peer_ready[peer_key]["timestamp"] = time.monotonic()
                            else:
                                mgr._peer_ready[peer_key]["ready"] = False
                                mgr._peer_ready[peer_key]["timestamp"] = 0.0
                    except Exception as e:
                        print(f"[ActivationMonitor] peer status parse error: {e}")

                # Neck status
                data_list, _ = mgr.activation_client.peek_data(Topics.NECK_STATUS, timeout_s=0.5, n=-1)
                if data_list:
                    payload = data_list[0]
                    try:
                        neck_status = (
                            deserialize_neck_status_message(payload)
                            if isinstance(payload, (bytes, bytearray))
                            else payload
                        )
                        with mgr._status_lock:
                            mgr.neck_at_init_flag = bool(getattr(neck_status, "at_init", False))
                            prev_ready = mgr.neck_ready_for_commands
                            mgr.neck_ready_for_commands = bool(getattr(neck_status, "at_start", False))
                            if mgr.neck_ready_for_commands and not prev_ready:
                                mgr._notified_neck_ready = True
                    except Exception as e:
                        print(f"[ActivationMonitor] neck status parse error: {e}")

                # Vision Pro status
                data_list, _ = mgr.activation_client.peek_data(Topics.VP_STATUS, timeout_s=0.5, n=-1)
                if data_list:
                    payload = data_list[0]
                    try:
                        vp_status = (
                            deserialize_vp_status_message(payload)
                            if isinstance(payload, (bytes, bytearray))
                            else payload
                        )
                        with mgr._status_lock:
                            prev_ready = mgr.vision_pro_ready
                            mgr.vision_pro_ready = bool(getattr(vp_status, "vision_pro_ready", False))
                            if mgr.node_id == "robot_pc":
                                prev_activated = mgr.vision_pro_activated
                                remote_activated = bool(getattr(vp_status, "vision_pro_activated", False))
                                if remote_activated != prev_activated:
                                    print(f"DEBUG: Robot PC synced VP activation to {remote_activated}")
                                    mgr.vision_pro_activated = remote_activated
                            if mgr.vision_pro_ready and not prev_ready:
                                print("DEBUG: Vision Pro READY (status received)")
                    except Exception as e:
                        print(f"[ActivationMonitor] VP status parse error: {e}")

                # iPhone status
                data_list, _ = mgr.activation_client.peek_data(
                    Topics.NECK_IPHONE_STATUS, timeout_s=0.5, n=-1
                )
                if data_list:
                    payload = data_list[0]
                    try:
                        iphone_status = (
                            deserialize_neck_iphone_status_message(payload)
                            if isinstance(payload, (bytes, bytearray))
                            else payload
                        )
                        with mgr._status_lock:
                            prev_ready = mgr.iphone_ready_flag
                            mgr.iphone_ready_flag = bool(
                                getattr(iphone_status, "iphone_publishing", False)
                            )
                            if mgr.iphone_ready_flag and not prev_ready:
                                print("DEBUG: iPhone READY (status received)")
                                mgr._notified_iphone_ready = True
                    except Exception as e:
                        print(f"[ActivationMonitor] iPhone status parse error: {e}")

                # Episode state sync
                data_list, _ = mgr.activation_client.peek_data(
                    Topics.EPISODE_STATE, timeout_s=0.5, n=-1
                )
                if data_list:
                    payload = data_list[0]
                    try:
                        episode_status = (
                            deserialize_episode_state_message(payload)
                            if isinstance(payload, (bytes, bytearray))
                            else payload
                        )
                        mgr.episode_manager.apply_remote_state(episode_status)
                    except Exception as e:
                        print(f"[ActivationMonitor] episode state sync error: {e}")

                if mgr.node_id == "robot_pc":
                    mgr._maybe_handle_robot_shutdown_activation()

                time.sleep(0.2)

            except Exception as e:
                print(f"Activation monitor thread error: {e}")
                time.sleep(1.0)

    # =========================================================================
    # Publish thread
    # =========================================================================

    def _publish_loop(self) -> None:
        mgr = self._mgr
        while self._pub_running:
            try:
                mgr._publish_local_readiness()
                time.sleep(0.5)
            except Exception as e:
                print(f"Status publisher error: {e}")
                time.sleep(1.0)
