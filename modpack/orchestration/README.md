# Orchestration

General-purpose lifecycle engine. Starts and stops modules, manages the activation bus, and owns the episode state machine.

No robot-specific code lives here. Robot runners connect via `modpack.bridge` independently.

---

## Key files

| File | Purpose |
|------|---------|
| `coordinator.py` | Main entry point which starts modules, manages keyboard input and watchdog loop for leader arms |
| `_registry.py` | `SubsystemSpec` + `REGISTRY`, i.e. module registration and dispatch |
| `robot_config.py` | Loads `config.yaml` for a robot; derives `modules:` and `roles:` |
| `episode_manager.py` | Episode start/stop/pause/resume state machine |
| `activation_monitor.py` | Background thread that polls activation RMQ topic |
| `process_runners.py` | `ProcessRunConfig` — stable snapshot passed to each module subprocess |
| `process_shutdown.py` | `ShutdownFlag` + signal handlers for coordinated graceful shutdown |
| `remote_pc.py` | SSH launch/shutdown of the Robot PC |
| `states.py` | `SystemState` enum (IDLE, ACTIVE, PAUSED) |
| `enums_and_events.py` | `LoggingState` and other shared enums |
| `message_formats.py` | Message format definitions for RMQ messages |

---

## How modules are started

Modules self-register at import time with `modpack/modules/<name>/spec.py`. The coordinator imports each spec module, which calls `REGISTRY.register(SubsystemSpec(...))`. Then `start_processes()` iterates through the registry and starts each active module.

A `SubsystemSpec` can supply optional functions that the coordinator uses. None are required, as a minimal spec just needs `name` and `runner`. Each function receives the coordinator instance as `manager` so the spec can call its API (`publish_to_activation`, `register_activation_topic`, `register_data_topic`) without reaching into private attributes. This keeps coordinator code free of robot- or module-specific branching.

| Field | When it runs | Typical use |
|-------|--------------|-------------|
| `setup_fn(manager)` | Once during `Coordinator.__init__`, before processes start | Initializes per-module state flags on the manager as well as registers activation/data topics |
| `ready_fn(manager) -> (bool, str)` | Polled while activation is pending | Gate activation on a precondition (e.g. iPhone ready, neck at init pose). Return `(False, reason)` to keep waiting. |
| `activation_fn(manager, subsystem, new_state)` | Each time the subsystem is toggled | Custom activation behaviour (e.g. publish a typed activation command); falls through to default behaviour when not provided |
| `shutdown_fn(manager)` | On coordinator shutdown | Reset module-owned state on the manager |

To add a new module: see [docs/ADD_A_MODULE.md](../../docs/ADD_A_MODULE.md).

---

## Episode lifecycle

```
IDLE    → [single tap s]  → ACTIVE  (start logging)
ACTIVE  → [single tap s]  → IDLE    (stop, episode saved)
ACTIVE  → [double tap s]  → PAUSED
PAUSED  → [double tap s]  → ACTIVE  (resume)
Any     → [triple tap s]  → delete last episode
```

Activation (button taps mapped via `activation_buttons:` in `config.yaml`) gates whether robot commands flow. Episodes are independent of activation state.

---

## Config source

Two config files exist:

| File | Owns |
|------|------|
| `modpack/modules/modpack_config.yaml` | Global infrastructure: RMQ host/ports, run metadata (`root_dir`, `project_name`), logging flags (`log_raw_gello_leader`). |
| `modpack/robots/<name>/config.yaml` | Everything robot-specific: topology, roles, modules, logging streams, gello hardware, activation buttons. |

`logging_streams` lives **only** in the robot config. The coordinator reads it from there and nowhere else. If no config is loaded, `_logging_streams` is empty.

---

## Robot config and logging_streams

The coordinator loads a `RobotConfig` from `modpack/robots/<name>/config.yaml` at startup. Three key fields drive orchestration:

**`modules:`** — list of subsystem names to activate (must be registered in `REGISTRY`).

**`logging_streams:`** — unified list of stream definitions for both proprio and camera data. Each entry is parsed into a `LoggingStreamEntry` or `CameraStreamEntry` depending on `logger_type`. The coordinator passes this list to `MainLogger` (which sends start/stop commands to all logger endpoints) and to the logger subprocess (which drives `BackpackStreamLogger` and `BackpackCameraLogger`).

**`activation_buttons:`** — maps button index → subsystem name. Numbered taps toggle subsystems at runtime.
