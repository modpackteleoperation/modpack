# Adding a New GELLO-Side Module

A module is a self-contained process that runs on the Modpack PC (backpack). The orchestration coordinator starts and stops it automatically when listed in a robot's `config.yaml`.

Examples of existing modules: `gello` (reads GELLO device, publishes joint commands), `logger` (subscribes to robot state, writes zarr), `vision_pro` (receives VP pose, publishes neck commands).

---

## 3 steps

### 1. Create the runner

```python
# modpack/modules/my_module/runner.py
import os
import sys

from modpack.modules.my_module.my_module import MyModule


def run_my_module_process(log_path, cfg) -> None:
    sys.stdout = open(log_path, "w", buffering=1)
    sys.stderr = sys.stdout
    os.environ["PYTHONUNBUFFERED"] = "1"

    mod = MyModule(
        host=cfg.host,
        port=cfg.port,
        # ... pull what you need from cfg (ProcessRunConfig)
    )
    mod.run()
```

Imports go at the top of the file, not inside `run_my_module_process`.

| Rule | Reason |
|---|---|
| Imports at module top | `spec.py` imports `runner.py`, so the coordinator pulls in your module's deps at import. Declare them in `environment.yaml`. |
| Runtime env setup stays inside the function | `run_my_module_process` runs in the spawned child, so `os.environ[...]` there scopes to that child. At module level it would leak into the coordinator and every later child. |
| Exception: a dep that is not installed on both PCs | Import it inside the function. See `modpack/modules/base/base_process.py` (`base_server` pulls in phoenix6, Robot PC only). |

`cfg` is a `ProcessRunConfig` snapshot — see `modpack/orchestration/process_runners.py` for all available fields.

### 2. Create the spec

```python
# modpack/modules/my_module/spec.py
from modpack.orchestration._registry import REGISTRY, SubsystemSpec
from modpack.modules.my_module.runner import run_my_module_process

REGISTRY.register(SubsystemSpec(
    name="my_module",
    active_key="my_module",   # must match the key in active_systems YAML
    pc="gello",               # "gello" | "robot" | "both"
    log_label="My Module",
    runner=run_my_module_process,
))
```

For modules that need custom lifecycle (multi-process, readiness gating, setup), use `start_fn`, `shutdown_fn`, `ready_fn`, `setup_fn` instead of or in addition to `runner`. See `modpack/modules/gello/spec.py` for a full example.

### 3. Register it in the coordinator

In `modpack/orchestration/coordinator.py`, add one import line alongside the others:

```python
import modpack.modules.my_module.spec  # noqa: F401 — registers my_module
```

That's it — importing the spec module causes `REGISTRY.register()` to run, and the coordinator's startup, shutdown, and readiness loops pick it up automatically.

---

## 4. Enable it

Add the module to the robot's `config.yaml`:

```yaml
modules:
  - gello
  - logger
  - my_module    # ← add here
```

And add the active_systems flag to `modpack/modules/modpack_config.yaml`:

```yaml
active_systems:
  my_module: true
```

---

## SubsystemSpec hooks reference

| Hook | Signature | Purpose |
|------|-----------|---------|
| `runner` | `(log_path, cfg) → None` | Standard single-process spawn |
| `setup_fn` | `(manager) → None` | Called at manager init — register RMQ topics, init attributes |
| `start_fn` | `(manager, cfg, timestamp) → bool` | Replaces `runner` for custom spawn logic (multi-process, conditional) |
| `shutdown_fn` | `(manager) → None` | Additive cleanup hook (reset state, broadcast, stop self-owned processes); the coordinator still terminates any `runner`-spawned process |
| `ready_fn` | `(manager) → Optional[(bool, str)]` | Return `(ok, blocker_message)` or `None` if not applicable |
| `log_entries_fn` | `(manager) → [(label, Path\|None)]` | Log paths shown at shutdown |
