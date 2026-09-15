# Base

## Components

| File | Role |
|------|------|
| `base_process.py` | `BaseProcess`, `MockBase` |
| `policies.py` | `TeleopPolicy`, `TeleopController`, `WebServer` (iPhone WebXR) |
| `webxr_messages.py` | `UnifiedWebXRMessage` |
| `constants.py` | base constants |
| `runner.py` | `run_base_process(log_path, cfg)` — subprocess entry point |
| `spec.py` | module registration (`SubsystemSpec`) |

## Lifecycle

Registered in `spec.py` (`SubsystemSpec(name="base")`):

- `setup_fn` → `_base_setup`
- `ready_fn` → `_base_ready`
- `shutdown_fn` → `_base_shutdown`

## RMQ topics

| Topic | Direction | Server (port) |
|-------|-----------|---------------|
| `body` (`Topics.BASE`) | publish (measured pose, STATE) | data (5555) |
| `body_target` (`Topics.BASE_TARGET`) | publish (commanded target; gated by `publish_base_target`, default true) | data (5555) |
| `body_cmd` (`{Topics.BASE}_cmd`) | consume (incoming base command) | data (5555) |
| `robot_activation` (`Topics.ACTIVATION`) | consume (episode/activation msgs) | activation (5556) |

## Config

- `modules: [base, ...]` — robot `config.yaml`
- `roles.body: {type: mobile_base}`
- `module_overrides.base.use_mock_base` — mock-base toggle
- `debug.base` — verbose logging toggle

## See also

- [`../../orchestration/README.md`](../../orchestration/README.md)
- robot `config.yaml`
