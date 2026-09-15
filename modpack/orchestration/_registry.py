"""Subsystem registry for SimpleMessageQueueManager.

A SubsystemSpec describes a single robot subsystem (gello, base, arx, neck,
iphone, vision_pro, backpack, …).  Register a spec once and the manager
handles startup, shutdown, readiness gating, and log display automatically.

Quickstart
----------
1. Write a run_my_process(log_path, cfg) function in process_runners.py.
2. Register a spec::

       from modpack.orchestration._registry import REGISTRY, SubsystemSpec
       REGISTRY.register(SubsystemSpec(
           name="my_module",
           active_key="my_module",   # key in active_systems YAML
           pc="robot",               # "gello" | "robot" | "both"
           log_label="My Module",
           runner=run_my_process,
       ))

3. Add my_module: true to active_systems in the relevant config YAML.

See docs/ADD_A_MODULE.md for the full guide.
"""
from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class SubsystemSpec:
    """Declarative description of one robot subsystem.

    Required fields
    ---------------
    name
        Unique identifier, e.g. "base", "arx".
    active_key
        Key in the active_systems YAML block.  Pass None for
        always-on subsystems such as the backpack logger.
    pc
        Which PC owns this subsystem — "gello", "robot", or
        "both".
    log_label
       Label used in log output and process listings.

    Optional hooks
    --------------
    runner
        Callable (log_path, cfg) -> None spawned inside an
        mp.Process.  Required for standard single-process subsystems.
    setup_fn
        (manager) -> None — called during __init__ to register RMQ
        topics and logger_endpoints entries.
    ready_fn
        (manager) -> Optional[Tuple[bool, str]] — returns
        (ok, blocker_message) or None when the subsystem is not
        active / not applicable on this PC.  Used by _collect_local_readiness.
    start_fn
        (manager, cfg, timestamp) -> bool — replaces the standard
        single-process spawn for multi-process or conditional subsystems
        (gello per-arm, arx two-process, backpack conditional).
    shutdown_fn
        (manager) -> None — custom cleanup hook (status flags, broadcasts,
        or killing processes the subsystem owns itself, e.g. start_fn-based
        multi-process subsystems). It is ADDITIVE, not a replacement: the
        coordinator still stops any generic SubsystemInstance it spawned for
        this spec (the runner path), so a spec with both a runner and a
        shutdown_fn does not need to terminate that process itself.
    log_entries_fn
        (manager) -> List[Tuple[str, Optional[Path]]] — returns
        (label, log_path) pairs consumed by show_recent_logs and
        the shutdown log-path summary.
    activation_fn
        (manager, subsystem, new_state) -> None - called by
        _apply_subsystem_activation instead of the generic
        publish-activation path.  Use for subsystems whose activation
        involves custom messaging (e.g. VP/neck sequencing).
    """

    name: str
    active_key: Optional[str]
    pc: str           # "gello" | "robot" | "both"
    log_label: str

    # Standard single-process runner — (log_path, cfg) -> None
    runner: Optional[Callable] = None

    # Optional lifecycle hooks
    setup_fn: Optional[Callable] = None        # (manager) -> None
    ready_fn: Optional[Callable] = None        # (manager) -> Optional[(bool, str)]
    start_fn: Optional[Callable] = None        # (manager, cfg, timestamp) -> bool
    shutdown_fn: Optional[Callable] = None     # (manager) -> None
    log_entries_fn: Optional[Callable] = None  # (manager) -> [(label, Path|None)]
    activation_fn: Optional[Callable] = None   # (manager, subsystem, new_state) -> None

    # Optional typed params schema (dataclass or TypedDict).  When provided,
    # the manager merges manifest.module_overrides[name] into a dict and passes
    # it to setup_fn / start_fn as manager.module_params[name].
    params_schema: Optional[type] = None


class SubsystemInstance:
    """Runtime state for one active subsystem (standard single-process case).

    For subsystems that use start_fn / shutdown_fn the instance is
    still created but its process / log_path may remain None; the
    custom hooks are responsible for managing process lifetime.
    """

    def __init__(self, spec: SubsystemSpec) -> None:
        self.spec = spec
        self.process: Optional[mp.Process] = None
        self.log_path: Optional[Path] = None

    def start(self, log_path: Path, cfg: Any) -> bool:
        """Spawn the runner in a new mp.Process.

        Returns True if the process is still alive after a 0.5 s
        sanity-check sleep.
        """
        if self.spec.runner is None:
            raise RuntimeError(
                f"SubsystemSpec '{self.spec.name}' has no runner; "
                "provide a start_fn instead."
            )
        self.log_path = log_path
        self.process = mp.Process(
            target=self.spec.runner,
            args=(log_path, cfg),
        )
        self.process.start()
        time.sleep(0.5)
        return self.process.is_alive()

    def stop(self) -> None:
        """Terminate then join the process (force-kill if needed)."""
        if self.process is None:
            return
        lbl = self.spec.log_label
        if self.process.is_alive():
            print(f"Terminating {lbl} process...")
            self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                print(f"Force killing {lbl}...")
                self.process.kill()

    def is_alive(self) -> bool:
        return self.process is not None and self.process.is_alive()


class SubsystemRegistry:
    """Central registry of :class:`SubsystemSpec` objects.

    Call :meth:`register` once at module level (or at import time) for each
    subsystem.  The manager reads the registry during __init__,
    start_processes, shutdown, _collect_local_readiness, and
    show_recent_logs.

    Example (in your own module or in builtin_subsystems.py)::

        from modpack.orchestration._registry import REGISTRY, SubsystemSpec
        from modpack.orchestration.process_runners import run_base_process

        REGISTRY.register(SubsystemSpec(
            name="base",
            active_key="base",
            pc="both",
            log_label="Base",
            runner=run_base_process,
        ))
    """

    def __init__(self) -> None:
        self._specs: Dict[str, SubsystemSpec] = {}

    def register(self, spec: SubsystemSpec) -> None:
        """Register a subsystem spec.  Raises ValueError if already taken."""
        if spec.name in self._specs:
            raise ValueError(f"Subsystem '{spec.name}' is already registered.")
        self._specs[spec.name] = spec

    def __iter__(self):
        return iter(self._specs.values())

    def get(self, name: str) -> Optional[SubsystemSpec]:
        return self._specs.get(name)

    def names(self) -> List[str]:
        return list(self._specs.keys())


# =============================================================================
# Module-level singleton
# =============================================================================
# Import this and call REGISTRY.register() to add new subsystems.
# builtin_subsystems.py registers the subsystems that ship with ModPack.
REGISTRY = SubsystemRegistry()
