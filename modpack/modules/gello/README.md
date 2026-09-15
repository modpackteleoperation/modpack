# GELLO

## Components

| File | Role |
|------|------|
| `gello_process.py` | `GelloPublisher` |
| `runner.py` | `run_gello_process(arm, log_path, cfg)` — subprocess entry point |
| `spec.py` | module registration (`SubsystemSpec`) |
| `gello/` | vendored DynamixelRobot driver (`gello.robots.dynamixel`) |
| `utils/urdf_utils.py` | `resolve_urdf_path` |

## Lifecycle

Registered in `spec.py` (`SubsystemSpec(name="gello")`):

- `setup_fn` → `_gello_setup`
- `ready_fn` → `_gello_ready`
- `shutdown_fn` → `_gello_shutdown`

## RMQ topics

| Topic | Direction | Server (port) |
|-------|-----------|---------------|
| `<arm>` (e.g. `right`, `left`; role `command_topic`) | publish (leader joint state) | data (5555) |
| `right_arm_torque` / `left_arm_torque` (`Topics.torque_topic`) | consume (optional; only if `feedback_port` set) | feedback (5581) |

## Config

- `modules: [gello, ...]` — robot `config.yaml`
- `roles.*.command_topic` — per-arm publish topic
- gello hardware config (Dynamixel ids/ports) — robot `config.yaml`
- `debug.gello` — verbose logging toggle

## See also

- [`../../orchestration/README.md`](../../orchestration/README.md)
- robot `config.yaml`
