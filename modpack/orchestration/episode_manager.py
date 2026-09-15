"""Episode lifecycle management.

EpisodeManager owns all episode-related state (count, state, logging state)
and methods.  SimpleMessageQueueManager instantiates one and delegates all
episode operations to it.

The manager keeps thin wrapper methods (_start_episode, _stop_episode,
etc.) because EpisodeController calls those by name on the manager object.
"""
from __future__ import annotations

from typing import Callable, Optional
from robotmq.utils import serialize
from modpack.orchestration.enums_and_events import LoggingState
from modpack.orchestration.message_formats import Topics, serialize_episode_state_message
from modpack.orchestration.states import SystemState


class EpisodeManager:
    """Manages episode lifecycle: start, stop, pause, resume, delete, and
    cross-PC synchronisation.

    Parameters
    ----------
    main_logger
        The MainLogger instance (from robologger).
    logging_config
        The robotmq.logging sub-dict from the manager YAML config.
    logging_streams
        The list of enabled logging-stream dicts.
    message_factory
        MessageFactory used to serialise episode-state messages.
    node_id
        "gello_pc" or "robot_pc".
    pc_id
        "gello" or "robot".
    publish_activation_fn
        Callable (command, target) that publishes to the activation bus.
    collect_readiness_fn
        Callable () -> (bool, Optional[str]) that returns local subsystem
        readiness (used by _episode_start_ready).
    has_robot_pc
        Whether a separate robot PC exists in this topology.
    peer_ready_fn
        Callable () -> dict returning the _peer_ready dict.
    broadcast_io_fn
        Callable () -> io returning the activation server or client for
        broadcasting episode state messages.
    """

    def __init__(
        self,
        main_logger,
        logging_config: dict,
        logging_streams: list,
        message_factory,
        node_id: str,
        pc_id: str,
        publish_activation_fn: Callable,
        collect_readiness_fn: Callable,
        has_robot_pc: bool,
        peer_ready_fn: Callable,
        broadcast_io_fn: Callable,
    ) -> None:
        self._main_logger = main_logger
        self._logging_config = logging_config
        self._logging_streams = logging_streams
        self._message_factory = message_factory
        self.node_id = node_id
        self.pc_id = pc_id
        self._publish_activation = publish_activation_fn
        self._collect_readiness = collect_readiness_fn
        self._has_robot_pc = has_robot_pc
        self._peer_ready_fn = peer_ready_fn
        self._broadcast_io_fn = broadcast_io_fn

        # State owned by this manager
        self.episode_count: int = 0
        self.episode_state: SystemState = SystemState.IDLE
        self.logging_state: LoggingState = LoggingState.IDLE
        self._last_deleted_episode: Optional[int] = None

    # =========================================================================
    # Metadata
    # =========================================================================

    def episode_metadata(self) -> dict:
        """Build metadata dict stored in metadata.zarr for training."""
        lc = self._logging_config
        meta: dict = {
            "log_raw_gello_leader": bool(lc.get("log_raw_gello_leader", True)),
            "logging_streams": [],
        }
        for e in self._logging_streams:
            if e.get("enabled", True) is False:
                continue
            if "topics" in e:
                for t in e["topics"]:
                    meta["logging_streams"].append(
                        {
                            "topic": t.get("topic"),
                            "logger_name": e.get("logger_name"),
                            "port": e.get("port"),
                            "logger_type": t.get("role", "joint_command"),
                            "frame_id_default": t.get("frame_id_default", ""),
                        }
                    )
            else:
                meta["logging_streams"].append(
                    {
                        "topic": e.get("topic"),
                        "logger_name": e.get("logger_name"),
                        "port": e.get("port"),
                        "logger_type": e.get("logger_type", "joint_command"),
                        "frame_id_default": e.get("frame_id_default", ""),
                    }
                )
        return meta

    # =========================================================================
    # Episode control
    # =========================================================================

    def start(self) -> None:
        """Start a new episode if all subsystems are ready."""
        ok, reason = self._episode_start_ready()
        if not ok:
            print(f"\nStart blocked: {reason}")
            return
        self.episode_count += 1
        self.episode_state = SystemState.ACTIVE

        print(f"\nEPISODE {self.episode_count} STARTED")
        print("iPhone control is now ACTIVE")
        print("First iPhone message will auto-rezero: phone direction → robot forward")
        self._publish_activation("start_episode", target="all")

        self.episode_count = self._logging_handler(desired_state=LoggingState.ACTIVE)
        self._broadcast_episode_state(event="start", episode_idx=self.episode_count)

    def stop(self) -> None:
        self.episode_state = SystemState.IDLE
        print(f"\nEPISODE {self.episode_count} ENDED")
        self._publish_activation("end_episode", target="all")
        print("iPhone control is now INACTIVE")
        print("Ready for next episode (press 's' to start)")
        self._logging_handler(desired_state=LoggingState.STOPPED)
        self._broadcast_episode_state(event="stop")

    def pause(self) -> None:
        self.episode_state = SystemState.PAUSED
        print(f"\nEPISODE {self.episode_count} PAUSED")
        print("iPhone control disabled")
        self._publish_activation("pause_episode", target="all")
        self._logging_handler(desired_state=LoggingState.PAUSE)
        self._broadcast_episode_state(event="pause")

    def resume(self) -> None:
        self.episode_state = SystemState.ACTIVE
        print(f"\nEPISODE {self.episode_count} RESUMED")
        print("iPhone will rezero on next frame")
        self._publish_activation("resume_episode", target="all")
        self._logging_handler(desired_state=LoggingState.RESUME)
        self._broadcast_episode_state(event="resume")

    def delete_last(self, broadcast: bool = True) -> None:
        """Delete the most recent recording."""
        deleted_idx = self._logging_handler(desired_state=LoggingState.DELETE)
        if deleted_idx is None:
            print("No recorded episode to delete or logger unavailable.")
            return
        self.episode_count = max(0, deleted_idx - 1)
        print(f"Episode counter reset to {self.episode_count}. Long press delete complete.")
        if broadcast:
            self._broadcast_episode_state(
                event="delete", episode_idx=deleted_idx, detail="last_episode_deleted"
            )

    def request_remote_delete(self) -> None:
        """Ask the robot PC to delete its most recent episode."""
        print("Requesting robot PC to delete last episode...")
        self._broadcast_episode_state(
            event="delete_request",
            episode_idx=self.episode_count,
            detail="request_last_episode_delete",
        )

    # =========================================================================
    # Cross-PC synchronisation
    # =========================================================================

    def apply_remote_state(self, episode_msg) -> None:
        """Apply an episode transition that originated on the peer PC."""
        if not episode_msg or getattr(episode_msg, "source_node", None) == self.node_id:
            return

        event = getattr(episode_msg, "event", "").lower()
        episode_idx = getattr(episode_msg, "episode_index", self.episode_count)

        target_state = None
        desired_logging = None
        if event == "start":
            target_state = SystemState.ACTIVE
            desired_logging = LoggingState.ACTIVE
        elif event == "pause":
            target_state = SystemState.PAUSED
            desired_logging = LoggingState.PAUSE
        elif event == "resume":
            target_state = SystemState.ACTIVE
            desired_logging = LoggingState.RESUME
        elif event == "stop":
            target_state = SystemState.IDLE
            desired_logging = LoggingState.STOPPED
        elif event == "delete":
            self.episode_count = max(0, episode_idx - 1)
            print(f"EPISODE {episode_idx} delete acknowledged (from {episode_msg.source_node})")
            return
        elif event == "delete_request":
            if self.pc_id == "robot":
                self.delete_last(broadcast=True)
            return
        else:
            return

        prev_state = self.episode_state
        prev_episode = self.episode_count
        self.episode_state = target_state
        self.episode_count = episode_idx

        if (prev_state != target_state) or (prev_episode != episode_idx):
            print(f"\nEPISODE {self.episode_count} {event.upper()} (synced from {episode_msg.source_node})")

        if desired_logging:
            self._logging_handler(desired_logging)

    # =========================================================================
    # Logging state machine
    # =========================================================================

    def _logging_handler(self, desired_state: LoggingState) -> Optional[int]:
        """Switch logging state; returns the episode index on success."""
        logger = self._main_logger
        if logger is None:
            self.logging_state = desired_state
            return self.episode_count

        if desired_state is LoggingState.ACTIVE and self.logging_state in (
            LoggingState.IDLE, LoggingState.STOPPED
        ):
            try:
                episode_idx = logger.start_recording(
                    episode_config=self.episode_metadata(),
                )
                print(f"Recording episode {episode_idx}")
                self.logging_state = desired_state
                return episode_idx
            except Exception as e:
                print(f"Failed to start recording: {e}")

        elif desired_state is LoggingState.PAUSE and self.logging_state is LoggingState.ACTIVE:
            if self._set_remote_loggers_state("pause"):
                print("Recording paused")
            else:
                print("Failed to pause remote loggers (continuing)")
            self.logging_state = LoggingState.PAUSE
            return self.episode_count

        elif desired_state is LoggingState.RESUME and self.logging_state is LoggingState.PAUSE:
            if self._set_remote_loggers_state("resume"):
                print("Recording resumed")
            else:
                print("Failed to resume remote loggers (continuing)")
            self.logging_state = LoggingState.ACTIVE
            return self.episode_count

        elif desired_state is LoggingState.STOPPED and self.logging_state in (
            LoggingState.ACTIVE, LoggingState.PAUSE
        ):
            print("Stopping recording...")
            # A paused episode left MainLogger.is_recording=False (set when we sent
            # 'pause' in _set_remote_loggers_state); stop_recording() early-returns
            # when not recording, so re-enable it first to finalize the episode.
            if self.logging_state is LoggingState.PAUSE:
                logger.is_recording = True
            episode_idx = logger.stop_recording()
            if episode_idx is not None:
                print(f"Stopped recording episode {episode_idx}")
                self.logging_state = desired_state
                return episode_idx

        elif desired_state is LoggingState.DELETE:
            delete_fn = getattr(logger, "delete_last_episode", None)
            if delete_fn is None:
                print("Logger does not support deleting episodes")
            else:
                try:
                    print("Deleting last episode...")
                    episode_idx = delete_fn(confirm=False)
                    self._last_deleted_episode = episode_idx
                    if episode_idx is not None:
                        print(f"Deleted episode {episode_idx}")
                        self.logging_state = LoggingState.IDLE
                        return episode_idx
                    print("No recorded episode to delete")
                except Exception as e:
                    print(f"Failed to delete episode: {e}")

        return None

    def _set_remote_loggers_state(self, command: str) -> bool:
        """Send pause/resume commands to remote zarr loggers."""
        logger = self._main_logger
        if logger is None:
            return False
        try:
            alive_loggers = logger.get_alive_loggers()
        except Exception as exc:
            print(f"Failed to query logger status for '{command}': {exc}")
            return False
        if not alive_loggers:
            print(f"No remote loggers alive to receive '{command}' command")
            return False
        payload = {"type": command}
        try:
            for name in alive_loggers:
                client = logger.clients.get(name)
                if not client:
                    continue
                client.put_data(topic="command", data=serialize(payload))
            if command == "pause":
                logger.is_recording = False
            elif command == "resume":
                logger.is_recording = True
            return True
        except Exception as exc:
            print(f"Failed to send '{command}' command to loggers: {exc}")
            return False

    # =========================================================================
    # Broadcast helpers
    # =========================================================================

    def _broadcast_episode_state(
        self,
        event: str,
        episode_idx: Optional[int] = None,
        detail: Optional[str] = None,
    ) -> None:
        io = self._broadcast_io_fn()
        if not io:
            return
        try:
            idx = episode_idx if episode_idx is not None else self.episode_count
            msg = self._message_factory.create_episode_state_message(
                source_node=self.node_id,
                event=event,
                episode_index=int(idx),
                logging_state=getattr(self.logging_state, "name", None),
                detail=detail,
            )
            io.put_data(Topics.EPISODE_STATE, serialize_episode_state_message(msg))
        except Exception as exc:
            print(f"Failed to broadcast episode state ({event}): {exc}")

    # =========================================================================
    # Readiness check
    # =========================================================================

    def _episode_start_ready(self) -> tuple:
        if self._has_robot_pc:
            peer_ready = self._peer_ready_fn()
            peer_key = "robot_pc" if self.node_id == "gello_pc" else "gello_pc"
            if not peer_ready.get(peer_key, {}).get("ready", False):
                return False, f"peer not ready: {peer_key}"

        local_ok, detail = self._collect_readiness()
        if not local_ok:
            return False, detail or "local subsystems not ready"

        return True, None
