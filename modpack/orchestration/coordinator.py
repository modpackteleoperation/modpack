#!/usr/bin/env python3
"""
Message Queue System Manager
=============================
Use 'q' to quit cleanly.
Each node handles activation messages appropriately.
ARX5 and Base have Ctrl+Z for physical safety.
Process outputs are redirected to separate log files to avoid interfering with input.

Adding a new module
-------------------
See docs/ADD_A_MODULE.md for the complete guide.
The short version: write a runner function, register a SubsystemSpec, add a
YAML flag. No changes to this file required.
"""

import time
import signal
import os
import sys
import select
import termios
import tty
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
import robotmq
import yaml
import threading
import argparse
from robotmq.utils import serialize, deserialize

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from modpack.modules import MODPACK_CONFIG_PATH
from modpack.orchestration.message_formats import (
    Topics,
    serialize_system_ready_message,
    ActivationMessage,
    MessageFactory,
)
from robologger.loggers.main_logger import MainLogger
from robologger.classes import Morphology
from modpack.orchestration.states import SystemState
from modpack.orchestration.enums_and_events import (
    LoggingState,
    EpisodeEvent,
)
from modpack.orchestration.remote_pc import RemotePCManager
from modpack.orchestration.process_runners import ProcessRunConfig
import modpack.modules.base.spec        # noqa: F401 — registers base
import modpack.modules.gello.spec       # noqa: F401 — registers gello
import modpack.modules.logger.spec      # noqa: F401 — registers logger
import modpack.modules.vision_pro.spec  # noqa: F401 — registers vision_pro

# Import registry and all built-in subsystem registrations
from modpack.orchestration._registry import REGISTRY
from modpack.orchestration.robot_config import RobotConfig, load_config, load_network_config

from modpack.orchestration.episode_manager import EpisodeManager
from modpack.orchestration.activation_monitor import ActivationMonitor
from modpack.orchestration._registry import SubsystemInstance


class EpisodeController:
    def __init__(self, manager: "SimpleMessageQueueManager") -> None:
        self.manager = manager

    def handle(self, event: "EpisodeEvent") -> None:
        """Dispatch episode-control events based on current state."""
        if event is EpisodeEvent.TRIPLE_TAP:
            if self.manager.pc_id == "robot":
                self.manager._delete_last_episode()
            else:
                self.manager._request_remote_delete()
            return

        if self.manager.episode_state is SystemState.IDLE:
            if event is EpisodeEvent.SINGLE_TAP:
                self.manager._start_episode()

        elif self.manager.episode_state is SystemState.ACTIVE:
            if event is EpisodeEvent.SINGLE_TAP:
                self.manager._stop_episode()
            elif event is EpisodeEvent.DOUBLE_TAP:
                # TODO: couple logging with pausing episode function
                self.manager._pause_episode()

        elif self.manager.episode_state is SystemState.PAUSED:
            if event is EpisodeEvent.DOUBLE_TAP:
                # TODO: couple logging with resuming episode function
                self.manager._resume_episode()


class SimpleMessageQueueManager:
    """
    Coordinator that starts/stops subsystem processes, manages the activation
    bus, and owns the episode lifecycle.

    Subsystems are declared in builtin_subsystems.py via the REGISTRY.
    To add a new hardware module, see docs/ADD_A_MODULE.md.
    """

    def __init__(
        self,
        pc_id: str,
        manager_config: str = str(MODPACK_CONFIG_PATH),
        announce_robot_logs: bool = False,
        robot_log_mount_root: Optional[str] = None,
        arx_skip_gello_wait: bool = False,
        vp_stream_mode: str = "pcd",
        robot: Optional[str] = None,
        modules_override: Optional[list] = None,
    ):
        self.pc_id = pc_id
        self.vp_stream_mode = vp_stream_mode

        with open(manager_config, "r") as f:
            self.manager_cfg = yaml.safe_load(f)
        # Network endpoints come from one place: modpack_config.yaml's robotmq: block.
        self.net = load_network_config(manager_config)
        # active_systems and debug are controlled by the robot config (modules:).
        self.active_systems: dict = {}
        self.debug_flags: dict = self.manager_cfg.get("debug", {})
        self.logging_config = self.net.logging
        self._logging_streams: list = []
        self.port = self.net.data_port
        self.cam_port = self.net.cam_port
        self.activation_port = self.net.activation_port

        robot_name = robot

        # Load manifest if a robot name is provided; merge active_systems + logging_streams
        self.robot_config: Optional[RobotConfig] = None
        if robot_name:
            try:
                self.robot_config = load_config(robot_name)
                print(f"[coordinator] Loaded manifest for robot '{self.robot_config.name}' "
                      f"(topology={self.robot_config.topology})")
                # Manifest defines the robot's required modules; active_systems starts empty.
                self.active_systems = dict(self.robot_config.active_systems_dict())
                # --modules override: enable only the explicitly requested subset.
                if modules_override is not None:
                    self.active_systems = {k: (k in modules_override) for k in self.active_systems}
                    for mod in modules_override:
                        self.active_systems.setdefault(mod, True)
                self._logging_streams = self.robot_config.logging_streams_as_dicts()
                # Expose module_overrides so subsystem hooks can read them
                self.module_overrides: dict = self.robot_config.module_overrides
                # Per-module debug flags from the robot config take precedence
                self.debug_flags.update(self.robot_config.debug)
                # Gello hardware config comes from the robot config
                self.gello_config: dict = dict(self.robot_config.gello or {})
            except FileNotFoundError as exc:
                print(f"[coordinator] WARNING: {exc}")
            except ValueError as exc:
                raise RuntimeError(f"[coordinator] Invalid manifest for robot '{robot_name}': {exc}") from exc
        else:
            self.module_overrides = {}
            self.gello_config = {}

        self.fixed_gello_ip = self.net.gello_pc_ip
        self.fixed_robot_pc_ip = self.net.robot_pc_ip
        self.feedback_port = self.net.feedback_port
        if self.robot_config is not None:
            self.has_robot_pc = self.robot_config.has_robot_pc
            self.target_robot = self.robot_config.name
            self._activation_button_map: dict[int, str] = dict(self.robot_config.activation_buttons or {})
        else:
            self.target_robot = "unknown"
            self.has_robot_pc = False
            self._activation_button_map = {}
        self.host = self.fixed_gello_ip
        self.activation_host = self.fixed_gello_ip

        self._backpack_owns: bool = True

        self.running = False
        self._status_lock = threading.Lock()
        self.arx_skip_gello_wait = arx_skip_gello_wait

        self.log_dir = Path(tempfile.gettempdir()) / "modpack_logs"
        self.log_dir.mkdir(exist_ok=True)

        self.logger_endpoints = {}
        self.keyboard_enabled = False

        # Neck/iPhone/VP status flags. ActivationMonitor reads these every loop
        # regardless of which modules are enabled, so they must exist even when
        # the vision_pro module (which also sets them in _vp_setup) is disabled.
        self.neck_at_init_flag = False
        self.neck_ready_for_commands = False
        self.neck_activated = False
        self._notified_neck_ready = False
        self.iphone_ready_flag = False
        self._notified_iphone_ready = False
        self.vision_pro_ready = False
        self.vision_pro_activated = False

        # ── RMQ servers ──────────────────────────────────────────────────────
        self.data_server = None
        if self.pc_id == "gello":
            try:
                print("Initializing robotmq server...")
                self.data_server = robotmq.RMQServer(
                    server_name="data_publisher",
                    server_endpoint=f"tcp://{self.net.bind_host}:{self.port}",
                )
                for _topic, _ttl in self.robot_config.data_topics:
                    self.data_server.add_topic(_topic, message_remaining_time_s=_ttl)
                for _ls in self._logging_streams:
                    if _ls.get("enabled", True) is False:
                        continue
                    _sub_topics = _ls.get("topics")
                    if _sub_topics:
                        for _sub in _sub_topics:
                            _t = _sub.get("topic")
                            if _t:
                                self.data_server.add_topic(_t, message_remaining_time_s=2.0)
                    else:
                        _t = _ls.get("topic")
                        if not _t:
                            continue
                        self.data_server.add_topic(_t, message_remaining_time_s=2.0)
                        _tt = _ls.get("target_topic")
                        if _tt:
                            self.data_server.add_topic(_tt, message_remaining_time_s=2.0)
            except Exception as e:
                print(f"Failed to initialize robotmq server: {e}")

        # Camera server
        self.camera_server = None
        _camera_server_pc = "robot" if self.has_robot_pc else "gello"
        if self.pc_id == _camera_server_pc:
            try:
                print("Initializing camera robotmq server...")
                self.camera_server = robotmq.RMQServer(
                    server_name="camera_publisher",
                    server_endpoint=f"tcp://{self.net.bind_host}:{self.cam_port}",
                )
                self.camera_endpoint = f"tcp://localhost:{self.cam_port}"
            except Exception as e:
                print(f"Failed to initialize camera robotmq server: {e}")
        if self.has_robot_pc and self.pc_id == "gello" and not self.camera_server:
            self.camera_endpoint = f"tcp://{self.fixed_robot_pc_ip}:{self.cam_port}"
        if self.camera_server is not None:
            self.camera_server.add_topic(Topics.IPHONE_RGB, message_remaining_time_s=10.0)
            for _ls in self._logging_streams:
                if _ls.get("logger_type") != "camera":
                    continue
                if _ls.get("enabled", True) is False:
                    continue
                _cam_key = _ls.get("camera_key")
                if _cam_key:
                    try:
                        self.camera_server.add_topic(_cam_key, message_remaining_time_s=10.0)
                        print(f"[coordinator] Camera topic registered: {_cam_key}")
                    except Exception as e:
                        print(f"[coordinator] Failed to register camera topic {_cam_key!r}: {e}")

        self._activation_topics_registered = False
        self.activation_monitor_client = None
        self.activation_client = None

        if self.pc_id == "gello":
            self.setup_activation_server()

        # Torque feedback server (GELLO PC only)
        self.feedback_server = None
        if self.pc_id == "gello":
            try:
                self.feedback_server = robotmq.RMQServer(
                    server_name="torque_feedback_server",
                    server_endpoint=f"tcp://{self.net.bind_host}:{self.feedback_port}",
                )
                self.feedback_server.add_topic(Topics.RIGHT_ARM_TORQUE, message_remaining_time_s=2.0)
                self.feedback_server.add_topic(Topics.LEFT_ARM_TORQUE, message_remaining_time_s=2.0)
            except Exception as e:
                print(f"Failed to initialize torque feedback server: {e}")
                self.feedback_server = None

        try:
            self.activation_client = robotmq.RMQClient(
                client_name="",
                server_endpoint=f"tcp://{self.fixed_gello_ip}:{self.activation_port}",
            )
        except Exception as e:
            print(f"Failed to initialize activation client: {e}")

        # ── Per-arm setup (needed before subsystem setup_fn calls) ──────────
        self.arms = self.gello_config.get("active_arms", ["right", "left"])

        # Gello command topics ('right'/'left') are registered unconditionally
        # via robot_config.data_topics (keyed off each role's command_topic), not
        # gated by active_arms — the robot-side consumer polls its command topic
        # even when that arm's gello publisher isn't running, and peeking an
        # unregistered topic spams "Topic not found" on the data server.

        # ── Run setup_fn for every active subsystem ─────────────────────────
        for spec in REGISTRY:
            if spec.setup_fn is None:
                continue
            if spec.active_key is not None and not self.active_systems.get(spec.active_key):
                continue
            try:
                spec.setup_fn(self)
            except Exception as e:
                print(f"Warning: setup_fn for subsystem '{spec.name}' raised: {e}")

        # Unified logging_streams endpoint registration
        if self.pc_id == "gello":
            for _ls in self._logging_streams:
                if _ls.get("enabled", True) is False:
                    continue
                if _ls.get("logger_type") == "camera":
                    _cn = _ls.get("camera_key")
                    _cp = _ls.get("port")
                    if _cn and _cp:
                        self.logger_endpoints[_cn] = f"tcp://localhost:{_cp}"
                    continue
                _ln = _ls.get("logger_name")
                _p = _ls.get("port")
                if _ln and _p:
                    self.logger_endpoints[_ln] = f"tcp://localhost:{_p}"
                _gn = _ls.get("gripper_logger_name")
                _gp = _ls.get("gripper_port")
                if _gn and _gp:
                    self.logger_endpoints[_gn] = f"tcp://localhost:{_gp}"

        self._register_activation_topics()

        # ── Activation state ─────────────────────────────────────────────────
        self._activated_subsystems: dict[str, bool] = {}
        self.node_id = "gello_pc" if self.pc_id == "gello" else "robot_pc"

        self._peer_ready = {
            "gello_pc": {"ready": False, "timestamp": 0.0},
            "robot_pc": {"ready": False, "timestamp": 0.0},
        }
        self._peer_ready_timeout = 2.0
        self.gello_ready_flags = {arm: False for arm in self.arms}
        self._gello_launch_pending = bool(self.active_systems.get("gello", False))
        self._last_local_ready_state = None
        self._last_local_ready_detail = None

        self.message_factory = MessageFactory()
        self._activation_monitor_error_logged = False
        self.old_settings = None
        self._release_window = 0.5
        self._double_tap_window = 0.5
        self._long_press_threshold = 0.75

        # ── MainLogger ───────────────────────────────────────────────────────
        runs_cfg = self.manager_cfg.get("runs", {})
        self.main_logger = MainLogger(
            name="main_logger",
            root_dir=runs_cfg.get("root_dir", "data"),
            project_name=runs_cfg.get("project_name", "demo_project"),
            task_name=runs_cfg.get("task_name", "task_1"),
            run_name=runs_cfg.get("run_name", "run_001"),
            logger_endpoints=self.logger_endpoints,
            morphology=Morphology.WHEEL_BASED_BI_MANUAL,
            success_config="hardcode_true",
        )
        print(self.logger_endpoints)

        # ── EpisodeManager ───────────────────────────────────────────────────
        self.episode_manager = EpisodeManager(
            main_logger=self.main_logger,
            logging_config=self.logging_config,
            logging_streams=self._logging_streams,
            message_factory=self.message_factory,
            node_id=self.node_id,
            pc_id=self.pc_id,
            publish_activation_fn=self.publish_activation_command,
            collect_readiness_fn=self._collect_local_readiness,
            has_robot_pc=self.has_robot_pc,
            peer_ready_fn=lambda: self._peer_ready,
            broadcast_io_fn=lambda: (
                getattr(self, "activation_server", None)
                or getattr(self, "activation_client", None)
            ),
        )
        # ── Signal handlers ──────────────────────────────────────────────────
        signal.signal(signal.SIGTERM, self._sigterm_handler)

        self._init_episode_controller()

        # ── SSH remote PC ────────────────────────────────────────────────────
        self._announce_robot_logs = announce_robot_logs
        self.robot_log_mount_root = robot_log_mount_root or os.getenv("ROBOT_LOG_MOUNT_ROOT")
        _robot_pc_block = dict(
            (self.robot_config.robot_pc if self.robot_config is not None else None) or {}
        )
        _script_template_vars = {
            "gello_pc_ip":     self.net.gello_pc_ip,
            "robot_pc_ip":     self.net.robot_pc_ip,
            "port":            self.net.data_port,
            "activation_port": self.net.activation_port,
            "cam_port":        self.net.cam_port,
            "feedback_port":   self.net.feedback_port,
        }
        _raw_scripts = list(_robot_pc_block.get("scripts") or [])
        _scripts = [s.format(**_script_template_vars) for s in _raw_scripts]
        self._remote_pc = RemotePCManager(
            pc_id=self.pc_id,
            user=_robot_pc_block.get("user") or self.manager_cfg.get("robot_pc", {}).get("user", "robot"),
            ip=self.fixed_robot_pc_ip,
            workspace=_robot_pc_block.get("workspace") or self.manager_cfg.get("robot_pc", {}).get("workspace", "~/modpack"),
            manager_cfg=self.manager_cfg,
            scripts=_scripts,
            conda_env=_robot_pc_block.get("conda_env", "modpack"),
            arx_skip_gello_wait=arx_skip_gello_wait,
            vp_stream_mode=self.vp_stream_mode,
            gello_pc_ip=self.fixed_gello_ip,
        )
        self._remote_shutdown_triggered = False
        self._shutdown_started = False

        # ── ActivationMonitor (created; started in start_system) ─────────────
        self.activation_monitor = ActivationMonitor(self)
        self._monitor_thread = None  # kept for back-compat; monitor owns its thread

        print("Simple Message Queue Manager initialized")

    # =========================================================================
    # Episode state properties — delegate to EpisodeManager
    # =========================================================================

    @property
    def episode_count(self) -> int:
        return self.episode_manager.episode_count

    @episode_count.setter
    def episode_count(self, value: int) -> None:
        self.episode_manager.episode_count = value

    @property
    def episode_state(self):
        return self.episode_manager.episode_state

    @property
    def logging_state(self):
        return self.episode_manager.logging_state

    # =========================================================================
    # Episode lifecycle — thin wrappers so EpisodeController keeps working
    # =========================================================================

    def _start_episode(self) -> None:
        self.episode_manager.start()

    def _stop_episode(self) -> None:
        self.episode_manager.stop()

    def _pause_episode(self) -> None:
        self.episode_manager.pause()

    def _resume_episode(self) -> None:
        self.episode_manager.resume()

    def _delete_last_episode(self, broadcast: bool = True) -> None:
        self.episode_manager.delete_last(broadcast=broadcast)

    def _request_remote_delete(self) -> None:
        self.episode_manager.request_remote_delete()

    def _logging_handler(self, desired_state) -> Optional[int]:
        result = self.episode_manager._logging_handler(desired_state)
        return result

    def _set_remote_loggers_state(self, command: str) -> bool:
        return self.episode_manager._set_remote_loggers_state(command)

    def _broadcast_episode_state(self, event, episode_idx=None, detail=None) -> None:
        self.episode_manager._broadcast_episode_state(event, episode_idx, detail)

    def _episode_start_ready(self):
        return self.episode_manager._episode_start_ready()

    # =========================================================================
    # Signal handlers
    # =========================================================================

    def _sigterm_handler(self, signum, frame):
        print(f"\nSIGTERM received - sending graceful shutdown")
        self.publish_activation_command("shutdown", target="all")
        time.sleep(2)
        self.shutdown()
        self._logging_handler(desired_state=LoggingState.STOPPED)
        sys.exit(0)

    def _init_episode_controller(self) -> None:
        self._episode_controller = EpisodeController(self)
        self._episode_press_pending = False
        self._episode_press_start = 0.0
        self._episode_last_event = 0.0
        self._long_press_threshold = 0.75
        self._episode_double_tap = False
        self._episode_triple_tap = False
        self._episode_tap_count = 0
        self._activation_press_pending = False
        self._activation_last_event = 0.0
        self._activation_tap_count = 0

    # =========================================================================
    # Episode / activation button press handling (unchanged logic)
    # =========================================================================

    def _handle_episode_button_press(self, timestamp: float) -> None:
        if not hasattr(self, "_episode_controller"):
            self._init_episode_controller()

        if self._episode_press_pending:
            if timestamp - self._episode_last_event <= self._double_tap_window:
                self._episode_tap_count += 1
                if self._episode_tap_count >= 3:
                    self._episode_triple_tap = True
                elif self._episode_tap_count == 2:
                    self._episode_double_tap = True
            else:
                self._finalize_episode_button(timestamp=timestamp, force=True)
                self._episode_press_start = timestamp
                self._episode_tap_count = 1
                self._episode_double_tap = False
                self._episode_triple_tap = False
        else:
            self._episode_press_start = timestamp
            self._episode_double_tap = False
            self._episode_triple_tap = False
            self._episode_tap_count = 1

        self._episode_last_event = timestamp
        self._episode_press_pending = True

    def _finalize_episode_button(self, timestamp: float, force: bool = False) -> None:
        if not getattr(self, "_episode_press_pending", False):
            return
        if not force and (timestamp - self._episode_last_event) <= self._release_window:
            return

        held = max(0.0, timestamp - self._episode_press_start)
        since_last = timestamp - self._episode_last_event

        if self._episode_triple_tap:
            event = EpisodeEvent.TRIPLE_TAP
        elif self._episode_double_tap:
            if not force and since_last <= self._release_window:
                return
            event = EpisodeEvent.DOUBLE_TAP
        elif held >= self._long_press_threshold:
            event = EpisodeEvent.LONG_PRESS
        else:
            if not force and since_last <= self._release_window:
                return
            event = EpisodeEvent.SINGLE_TAP

        self._episode_controller.handle(event)
        self._episode_press_pending = False
        self._episode_double_tap = False
        self._episode_triple_tap = False
        self._episode_press_start = 0.0
        self._episode_last_event = 0.0
        self._episode_tap_count = 0

    def _handle_activation_button_press(self, timestamp: float) -> None:
        if self._activation_press_pending:
            if timestamp - self._activation_last_event <= self._double_tap_window:
                self._activation_tap_count += 1
            else:
                self._finalize_activation_button(timestamp=timestamp, force=True)
                self._activation_tap_count = 1
        else:
            self._activation_tap_count = 1

        self._activation_last_event = timestamp
        self._activation_press_pending = True

    def _finalize_activation_button(self, timestamp: float, force: bool = False) -> None:
        if not self._activation_press_pending:
            return
        if not force and (timestamp - self._activation_last_event) <= self._release_window:
            return

        tap_count = self._activation_tap_count
        self._activation_press_pending = False
        self._activation_tap_count = 0
        self._activation_last_event = 0.0

        if tap_count <= 0:
            return

        if tap_count >= 4:
            self._handle_gello_launch_request()
        else:
            target = self._activation_button_map.get(tap_count)
            if target:
                self._toggle_subsystem_activation(target)
            else:
                print(f"No activation mapping for {tap_count} tap(s)")

    # =========================================================================
    # Subsystem activation
    # =========================================================================

    def _apply_subsystem_activation(
        self,
        subsystem: str,
        new_state: bool,
        *,
        publish: bool = True,
        reason: str = "",
    ) -> None:
        spec = REGISTRY.get(subsystem)
        if spec and spec.activation_fn:
            spec.activation_fn(self, subsystem, new_state)
            return
        current = self._activated_subsystems.get(subsystem, False)
        if current == new_state:
            return
        self._activated_subsystems[subsystem] = new_state
        if reason:
            print(reason)
        command = "activate" if new_state else "deactivate"
        print(f"\n{subsystem.upper()} {'ACTIVATED' if new_state else 'DEACTIVATED'}")
        if publish:
            self.publish_activation_command(command, target=subsystem)

    def _toggle_subsystem_activation(self, subsystem: str) -> None:
        current = self._activated_subsystems.get(subsystem, False)
        self._apply_subsystem_activation(subsystem, not current)

    def _handle_gello_launch_request(self) -> None:
        if not self.active_systems.get("gello"):
            print("\nGELLO subsystem disabled; quadruple tap ignored.")
            return
        if not self._gello_launch_pending:
            print("\nGELLO already running; quadruple tap ignored.")
            return
        print("\nQuadruple tap detected - launching GELLO arms...")
        self._gello_launch_pending = False
        cfg = ProcessRunConfig.from_manager(self)
        gello_spec = REGISTRY.get("gello")
        if gello_spec and gello_spec.start_fn:
            if not gello_spec.start_fn(self, cfg, time.time()):
                print("Failed to start GELLO processes. Resolve the issue and quadruple tap again.")
        else:
            print("GELLO spec not found in registry.")

    # =========================================================================
    # Activation server / topics
    # =========================================================================

    def setup_activation_server(self):
        try:
            print("Setting up activation server...")
            self.activation_host = self.fixed_gello_ip
            self.activation_port = self.net.activation_port
            self.activation_server = robotmq.RMQServer(
                server_name="critical_activation_manager",
                server_endpoint=f"tcp://{self.net.bind_host}:{self.activation_port}",
            )
            self.activation_server.add_topic(Topics.ACTIVATION, message_remaining_time_s=30.0)
            self._activation_topics_registered = False
            return True
        except Exception as e:
            print(f"Failed to setup activation server: {e}")
            return False

    def _register_activation_topics(self):
        if not hasattr(self, "activation_server") or self._activation_topics_registered:
            return
        try:
            self.activation_server.add_topic(Topics.GELLO_PC_STATUS, message_remaining_time_s=30.0)
            self.activation_server.add_topic(Topics.ROBOT_PC_STATUS, message_remaining_time_s=30.0)
            self.activation_server.add_topic(Topics.EPISODE_STATE, message_remaining_time_s=30.0)
            # The bus topic set is a cross-PC contract: robot PC scripts publish to
            # these regardless of which modules are active here, so register them
            # unconditionally. Topics for inactive modules simply stay empty.
            self.activation_server.add_topic(Topics.VP_STATUS, message_remaining_time_s=30.0)
            self.activation_server.add_topic(Topics.VP_PUBLISH_ACTIVATION, message_remaining_time_s=30.0)
            self.activation_server.add_topic(Topics.NECK_IPHONE_STATUS, message_remaining_time_s=30.0)
            self.activation_server.add_topic(Topics.NECK_ACTIVATION, message_remaining_time_s=30.0)
            self.activation_server.add_topic(Topics.NECK_STATUS, message_remaining_time_s=30.0)
            self._activation_topics_registered = True
        except Exception as e:
            print(f"Failed to register activation topics: {e}")

    def publish_activation_command(self, command: str, target: str = "all"):
        try:
            io = getattr(self, "activation_server", None) or getattr(self, "activation_client", None)
            if not io:
                print(f"CRITICAL: Cannot send {command} - activation comms not ready")
                return
            activation_msg = {
                "command": command,
                "target": target,
                "timestamp": time.time(),
                "source": "system_manager",
            }
            serialized_msg = serialize(activation_msg)
            if target == "all":
                io.put_data(Topics.NECK_ACTIVATION, serialized_msg)
                io.put_data(Topics.VP_PUBLISH_ACTIVATION, serialized_msg)
            io.put_data(Topics.ACTIVATION, serialized_msg)
            if target == "all":
                print(f"System command sent to ALL: {command}")
            else:
                print(f"System command sent to {target.upper()}: {command}")
        except Exception as e:
            print(f"CRITICAL ERROR sending command: {e}")

    # ── Public API for module specs ──────────────────────────────────────────

    def publish_to_activation(self, topic: str, payload: bytes) -> None:
        """Publish bytes to a topic on the activation bus. For use by module specs."""
        io = getattr(self, "activation_server", None) or getattr(self, "activation_client", None)
        if io:
            io.put_data(topic, payload)

    def register_activation_topic(self, topic: str, ttl_s: float = 30.0) -> None:
        """Register a topic on the activation server. For use by module specs' setup_fn."""
        if getattr(self, "activation_server", None):
            self.activation_server.add_topic(topic, message_remaining_time_s=ttl_s)

    def register_data_topic(self, topic: str, ttl_s: float = 2.0) -> None:
        """Register a topic on the data server. For use by module specs' setup_fn."""
        if self.data_server:
            self.data_server.add_topic(topic, message_remaining_time_s=ttl_s)

    def _maybe_handle_robot_shutdown_activation(self):
        """Stop robot manager when ModPack broadcasts shutdown."""
        if self._remote_shutdown_triggered:
            return
        client = getattr(self, "activation_client", None)
        if client is None:
            return
        status = client.get_topic_status(Topics.ACTIVATION, 0.05)
        if status <= 0:
            return
        data_list, _ = client.peek_data(Topics.ACTIVATION, n=-1)
        if not data_list:
            return
        payload = data_list[0]
        try:
            activation_payload = (
                deserialize(payload) if isinstance(payload, (bytes, bytearray)) else payload
            )
            activation_msg = ActivationMessage.from_dict(activation_payload)
        except Exception as e:
            print(f"[coordinator] activation message parse error: {e}")
            return
        if self.node_id == "robot_pc":
            command = (getattr(activation_msg, "command", "") or "").lower()
            target = (getattr(activation_msg, "target", "") or "").lower()
            if command in ("activate", "deactivate"):
                for subsystem in ("base", "arx"):
                    if target in (subsystem, "all"):
                        desired_state = command == "activate"
                        self._apply_subsystem_activation(
                            subsystem,
                            desired_state,
                            publish=False,
                            reason=f"DEBUG: Robot PC synced {subsystem} activation to {desired_state}",
                        )
        command = (getattr(activation_msg, "command", "") or "").lower()
        target = (getattr(activation_msg, "target", "") or "").lower()
        if command not in ("shutdown", "manager_shutdown"):
            return
        if target not in ("all", "robot", self.node_id):
            return
        self._remote_shutdown_triggered = True
        print("Robot PC received remote shutdown command - stopping manager loop")
        self.running = False

    def _publish_local_readiness(self):
        io = getattr(self, "activation_server", None) or getattr(self, "activation_client", None)
        if not io:
            return
        ready, detail = self._collect_local_readiness()
        if self.node_id == "robot_pc":
            state_changed = ready != self._last_local_ready_state
            detail_changed = detail != self._last_local_ready_detail
            if not ready and (state_changed or detail_changed):
                print(f"[Robot PC readiness] Not ready: {detail or 'unspecified'}")
            elif ready and state_changed:
                print("[Robot PC readiness] All subsystems ready")
            self._last_local_ready_state = ready
            self._last_local_ready_detail = detail
        msg = self.message_factory.create_ready_message(node=self.node_id, ready=ready)
        msg.detail = detail
        payload = serialize_system_ready_message(msg)
        topic = Topics.GELLO_PC_STATUS if self.node_id == "gello_pc" else Topics.ROBOT_PC_STATUS
        io.put_data(topic, payload)

    # =========================================================================
    # Readiness
    # =========================================================================

    def _collect_local_readiness(self):
        """Aggregate subsystem readiness via registry ready_fn callbacks."""
        ready = True
        blockers = []
        for spec in REGISTRY:
            if spec.ready_fn is None:
                continue
            result = spec.ready_fn(self)
            if result is None:
                continue  # not active / not applicable on this PC
            ok, msg = result
            if not ok:
                ready = False
                blockers.append(msg)
        detail = ", ".join(b for b in blockers if b) or None
        return ready, detail

    # =========================================================================
    # Remote PC helpers
    # =========================================================================

    def launch_robot_pc_remotely(self):
        ok = self._remote_pc.launch()
        if ok:
            # Base runs on the robot PC; point the 'b'/'l' log views at its remote log
            remote_base_log = self._remote_pc.log_path_for("base_process")
            if remote_base_log is not None:
                self.base_log_path = remote_base_log
        return ok

    def shutdown_robot_pc_remotely(self):
        self._remote_pc.shutdown(
            publish_shutdown_fn=lambda: self.publish_activation_command("shutdown", target="all")
        )

    # =========================================================================
    # Process management
    # =========================================================================

    def start_processes(self) -> bool:
        """Start all active subsystem processes via registry start_fn callbacks."""
        try:
            print("Starting processes with multiprocessing...")
            timestamp = time.time()
            _cfg = ProcessRunConfig.from_manager(self)

            for spec in REGISTRY:
                # Skip subsystems that don't run on this PC (gello vs robot).
                if getattr(spec, "pc", "gello") not in (self.pc_id, "both"):
                    continue
                # Skip subsystems that are not enabled on this config
                if spec.active_key is not None and not self.active_systems.get(spec.active_key):
                    continue
                if spec.start_fn is None and spec.runner is None:
                    continue
                if spec.start_fn:
                    ok = spec.start_fn(self, _cfg, timestamp)
                else:
                    # Standard single-process spawn via SubsystemInstance
                    inst = SubsystemInstance(spec)
                    log_path = self.log_dir / f"{spec.name}_{timestamp}.log"
                    ok = inst.start(log_path, _cfg)
                    if not ok:
                        print(f"WARNING: {spec.log_label} process died immediately")
                    # Store instance for later shutdown
                    if not hasattr(self, "_subsystem_instances"):
                        self._subsystem_instances = {}
                    self._subsystem_instances[spec.name] = inst

                if not ok:
                    return False

            print("All processes started successfully")
            print("\n" + "=" * 60)
            print("CHECKING BASE LOGS FOR IPHONE URL...")
            print("=" * 60)
            self.show_recent_logs("base")
            return True

        except Exception as e:
            print(f"Error starting processes: {e}")
            return False

    def shutdown(self) -> None:
        """Shut down all subsystem processes via registry shutdown_fn callbacks."""
        if getattr(self, "_shutdown_started", False):
            print("Shutdown already in progress; ignoring duplicate request")
            return
        self._shutdown_started = True

        print("Shutting down manager...")
        self.running = False

        # Finalize any in-progress episode while the logger subprocess is still
        # alive. The logger has no SIGTERM handler and buffers the episode in
        # memory, so the teardown loop below would otherwise kill it before it
        # flushes to zarr, losing the recording. STOPPED no-ops if not recording.
        try:
            self._logging_handler(desired_state=LoggingState.STOPPED)
        except Exception as e:
            print(f"[coordinator] episode finalize on shutdown failed: {e}")

        self.publish_activation_command("emergency_stop", target="all")
        # Send shutdown to robot PC before manager_shutdown kills its activation loop
        if self.pc_id == "gello" and self.has_robot_pc:
            self.publish_activation_command("shutdown", target="all")
        self.publish_activation_command("manager_shutdown", target="all")

        # Shutdown all subsystems via registry (in reverse registration order).
        # shutdown_fn is a custom cleanup hook (status flags, broadcasts) — NOT a
        # substitute for killing a process the coordinator spawned. So always also
        # stop any generic SubsystemInstance we started for this spec. inst.stop()
        # is a no-op for start_fn-based specs (no instance stored) and idempotent.
        for spec in reversed(list(REGISTRY)):
            if spec.shutdown_fn:
                try:
                    spec.shutdown_fn(self)
                except Exception as e:
                    print(f"Warning: shutdown_fn for '{spec.name}' raised: {e}")
            inst = (getattr(self, "_subsystem_instances", {}) or {}).get(spec.name)
            if inst is not None:
                inst.stop()

        if self.pc_id == "gello" and self.has_robot_pc:
            self.shutdown_robot_pc_remotely()

        # Stop activation monitor threads
        self.activation_monitor.stop()
        self._monitor_thread = None

        # Close activation server / client
        if getattr(self, "activation_server", None):
            try:
                print("Shutting down activation server")
                act = self.activation_server
                for method in ("shutdown", "stop", "close"):
                    fn = getattr(act, method, None)
                    if fn:
                        fn()
                        break
            except Exception as e:
                print(f"Error shutting down activation server: {e}")

        if getattr(self, "activation_client", None):
            try:
                close_fn = getattr(self.activation_client, "close", None)
                if close_fn:
                    close_fn()
            except Exception as e:
                print(f"Error closing activation client: {e}")

        self._cleanup_input()
        time.sleep(0.2)
        print("Manager shutdown complete")

        # Print log paths via registry
        print(f"\nLog files preserved:")
        if self.has_robot_pc:
            for log_path in self._remote_pc.log_paths:
                formatted = self._format_log_path(f"Robot PC ({log_path.stem})", log_path)
                if formatted:
                    print(formatted)
        for spec in REGISTRY:
            try:
                if spec.log_entries_fn is not None:
                    for label, log_path in spec.log_entries_fn(self):
                        formatted = self._format_log_path(label, log_path)
                        if formatted:
                            print(formatted)
                else:
                    # Standard single-process subsystems store log_path on the instance
                    inst = (getattr(self, "_subsystem_instances", {}) or {}).get(spec.name)
                    if inst and inst.log_path:
                        formatted = self._format_log_path(spec.log_label, inst.log_path)
                        if formatted:
                            print(formatted)
            except Exception as e:
                print(f"[coordinator] log path error for {spec.name}: {e}")

    # =========================================================================
    # User input
    # =========================================================================

    def setup_user_input(self):
        try:
            if not sys.stdin.isatty():
                print("Standard input is not a TTY; keyboard controls disabled")
                self.keyboard_enabled = False
                return False
            self.old_settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setcbreak(sys.stdin.fileno())
            except Exception as exc:
                print(f"Failed to enter cbreak mode: {exc}")
                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
                except Exception as e:
                    print(f"[coordinator] tcsetattr restore failed: {e}")
                self.old_settings = None
                self.keyboard_enabled = False
                return False
            print("User input handling ready")
            self.keyboard_enabled = True
            return True
        except Exception as e:
            print(f"Failed to setup user input: {e}")
            self.keyboard_enabled = False
            return False

    def _process_input(self, char):
        now = time.monotonic()
        self._finalize_episode_button(now)
        self._finalize_activation_button(now)
        char = char.lower()
        if char == "s":
            self._handle_episode_button_press(time.monotonic())
        elif char == "v":
            self._handle_activation_button_press(time.monotonic())
        elif char == "q":
            print("\nQUIT requested - shutting down all nodes")
            self.publish_activation_command("shutdown", target="all")
            self.running = False

    # =========================================================================
    # Log display — registry-driven
    # =========================================================================

    def show_recent_logs(self, process_name: Optional[str] = None):
        """Show recent logs from all subsystems or a specific one."""
        print("\n" + "=" * 60)
        print(f"{process_name.upper()} PROCESS LOGS" if process_name else "RECENT PROCESS LOGS")
        print("=" * 60)
        for spec in REGISTRY:
            if process_name and spec.name != process_name:
                continue
            if spec.log_entries_fn is None:
                continue
            try:
                for label, log_path in spec.log_entries_fn(self):
                    print(f"\n--- {label} ---")
                    if log_path and log_path.exists():
                        with open(log_path, "r") as f:
                            for line in f.readlines()[-20:]:
                                print(f"  {line.rstrip()}")
                    else:
                        print("  No log file found")
            except Exception as e:
                print(f"[coordinator] log read error for {spec.name}: {e}")
        print("=" * 60)
        print("Use 'l' to refresh logs, 'b' for base logs only\n")

    # =========================================================================
    # Main loop
    # =========================================================================

    def run_input_loop(self):
        while self.running:
            LOOP_PERIOD = 1 / 30
            try:
                start = time.monotonic()
                if self.keyboard_enabled:
                    if select.select([sys.stdin], [], [], 0.0)[0]:
                        char = sys.stdin.read(1)
                        self._process_input(char)
                self._finalize_episode_button(time.monotonic())
                self._finalize_activation_button(time.monotonic())

                # Process death watchdog
                if hasattr(self, "gello_processes"):
                    for arm, process in self.gello_processes.items():
                        if process and not process.is_alive():
                            print(f"\nWARNING: GELLO {arm} process died")
                            self.publish_activation_command("shutdown", target="arx")

                elapsed = time.monotonic() - start
                time.sleep(max(0.0, LOOP_PERIOD - elapsed))
            except Exception as e:
                print(f"Input error: {e}")
                time.sleep(0.1)

    def start_system(self, mode=None):
        """Start the full system: monitor threads → processes → input loop."""
        print("Starting Message Queue System with Activation Control...")
        print("=" * 60)

        # Start activation monitor + status publisher threads
        self.activation_monitor.start()
        self._monitor_thread = self.activation_monitor._poll_thread  # back-compat

        if mode == "gello" and not getattr(self, "activation_server", None):
            print("Cannot start without activation server")
            return False

        if mode == "gello":
            self._register_activation_topics()
            self._activation_monitor_error_logged = False

        input_ready = self.setup_user_input()
        if not input_ready:
            print("Keyboard controls unavailable; continuing without interactive commands")

        if mode == "gello" and self.has_robot_pc:
            if not self.launch_robot_pc_remotely():
                print("Failed to launch ROBOT PC - aborting")
                self._cleanup_input()
                return False
            print("\nVerifying ROBOT PC process started...")
            time.sleep(3)
            crashed = [p for p in self._remote_pc.remote_processes if p.poll() is not None]
            if crashed:
                print(f"ERROR: {len(crashed)} ROBOT PC process(es) crashed (exit codes: {[p.returncode for p in crashed]})")
                self._cleanup_input()
                return False
            print("ROBOT PC process launched successfully")

        self.running = True
        print("\n" + "=" * 60)
        print("ACTIVATION SERVER READY - STARTING NODES")
        print("=" * 60)

        try:
            if not self.start_processes():
                print("Failed to start processes")
                return False
            self.run_input_loop()
        except Exception as e:
            print(f"Error during execution: {e}")
        finally:
            self.shutdown()
            self._logging_handler(desired_state=LoggingState.STOPPED)

        print("System control session ended")
        return True

    # =========================================================================
    # Utilities
    # =========================================================================

    def _cleanup_input(self):
        if self.keyboard_enabled and self.old_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
                print("Terminal settings restored")
            except Exception as e:
                print(f"Error restoring terminal settings: {e}")
                try:
                    subprocess.run(["stty", "sane"], check=False)
                    print("Fallback: restored terminal with 'stty sane'")
                except Exception as stty_exc:
                    print(f"Fallback terminal restore failed: {stty_exc}")
        self.keyboard_enabled = False
        self.old_settings = None

    def _format_log_path(self, label: str, log_path: Optional[Path]) -> Optional[str]:
        if not log_path:
            return None
        mounted_hint = ""
        if self.pc_id == "robot" and self._announce_robot_logs and self.robot_log_mount_root:
            try:
                mounted_path = Path(self.robot_log_mount_root).expanduser() / Path(log_path).name
                mounted_hint = f" (mounted: {mounted_path})"
            except Exception as e:
                print(f"[coordinator] log path format error: {e}")
                mounted_hint = ""
        return f"  {label}: {log_path}{mounted_hint}"


# =============================================================================
# CLI entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Message Queue System with Activation Control")
    parser.add_argument("--gello", action="store_true", help="Run in Modpack (GELLO) PC mode")
    parser.add_argument(
        "--pc-id",
        dest="pc_id",
        choices=["gello", "robot"],
        default="gello",
        help=(
            "Which PC this coordinator runs on. 'gello' (default) is the Modpack PC; "
            "'robot' runs the robot-side coordinator (hosts the camera server, syncs "
            "activation/episode state) and is launched on the Robot PC via robot_pc.scripts."
        ),
    )

    parser.add_argument(
        "--robot",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Robot manifest to load (e.g. 'rby1', 'arx5', or path to config.yaml). "
            "Derives active_systems and logging_streams from the manifest."
        ),
    )

    parser.add_argument("--vp-bypass-neck", action="store_true",
                        help="Bypass neck readiness for Vision Pro visualization testing")
    parser.add_argument("--announce-robot-logs", action="store_true",
                        help="Print robot PC log paths when created (use with sshfs mount)")
    parser.add_argument("--robot-log-mount-root", type=str,
                        help="Local mount point that mirrors /tmp/modpack_logs from robot PC")
    parser.add_argument("--arx-skip-gello-wait", action="store_true",
                        help="Skip waiting for GELLO topics on robot PC launch")
    parser.add_argument("--vp-stream-mode", choices=["pcd", "rgb"], default="pcd",
                        help="What to stream to Vision Pro via UDP: point cloud (pcd) or RGB (rgb)")
    parser.add_argument(
        "--modules",
        type=str,
        default=None,
        metavar="mod1,mod2,...",
        help=(
            "Override active modules (comma-separated). Only the listed modules will start. "
            "Example: --modules logger  (cameras only, no gello/base/vision_pro)"
        ),
    )

    args = parser.parse_args()

    # --gello stays a required safeguard for the Modpack PC; the robot-side
    # coordinator is selected with --pc-id robot and does not take --gello.
    if args.pc_id == "gello" and not args.gello:
        parser.error("--gello is required when running the GELLO PC (--pc-id gello)")

    robot_name = getattr(args, "robot", None)
    if robot_name:
        print(f"Robot: {robot_name}")
    else:
        print("Robot: (not specified — no modules will start; use --robot <name>)")

    _modules_arg = getattr(args, "modules", None)
    _modules_override = [m.strip() for m in _modules_arg.split(",") if m.strip()] if _modules_arg else None
    if _modules_override is not None:
        print(f"Modules override: {_modules_override}")

    manager = SimpleMessageQueueManager(
        pc_id=args.pc_id,
        manager_config=str(MODPACK_CONFIG_PATH),
        announce_robot_logs=getattr(args, "announce_robot_logs", False),
        robot_log_mount_root=getattr(args, "robot_log_mount_root", None),
        arx_skip_gello_wait=getattr(args, "arx_skip_gello_wait", False),
        vp_stream_mode=getattr(args, "vp_stream_mode", "pcd"),
        robot=robot_name,
        modules_override=_modules_override,
    )

    if getattr(args, "vp_bypass_neck", False):
        manager.vp_bypass_neck = True


    try:
        success = manager.start_system(mode=manager.pc_id)
        if not success:
            print("Failed to start system")
            sys.exit(1)
    except Exception as e:
        print(f"System error: {e}")
        manager.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()
