# Vision Pro

Apple Vision Pro head tracking module. Receives pose data over UDP from the Vision Pro app and republishes neck commands over RMQ for the robot bridge to consume.

## Components

| File | Role |
|------|------|
| `vision_pro_process.py` | `VisionPro` class. Main loop reads UDP pose packets, computes neck commands, and publishes `NECK_CMD` on the data RMQ. |
| `pose_receiver.py` | UDP listener for raw pose packets from the Swift app (port 5005) |
| `transformation_helper.py` | Pose math utilities: `mat2pose`, `neck_start_pose` constant |
| `runner.py` | Subprocess entry point, as `run_vision_pro_process`,is spawned by the coordinator |
| `spec.py` | Module registration|

## Lifecycle

`spec.py` declares this module to the coordinator by registering a `SubsystemSpec` (see [`modpack/orchestration/README.md`](../../orchestration/README.md) for the spec system). The coordinator calls the spec's functions when appropriate:

- **`setup_fn`** — runs once at coordinator startup. Initializes VP/neck state flags on the manager (`vision_pro_activated`, `vision_pro_ready`, `neck_at_init_flag`, `neck_ready_for_commands`, `iphone_ready_flag`, etc.) and registers the activation/data topics this module owns.
- **`ready_fn`** — polled while activation is pending. When `VP_BYPASS_NECK=1` it synthetically sets `neck_ready_for_commands = True`. Otherwise, it waits passively and prints a prompt once both `vision_pro_ready` and `neck_ready_for_commands` are set by the activation monitor. Returns `None` in all cases (readiness is signalled with state flags, not a return value).
- **`activation_fn`** — runs each time the user presses the activation button. Implements a two-step sequence: first press signals VP READY (publishes `VP_STATUS`); second press toggles publishing on/off (publishes `VP_PUBLISH_ACTIVATION` + `NECK_ACTIVATION` + neck bridge command). Each step checks prerequisites (neck at init, iPhone ready if required) and prints a diagnostic if they are not met.
- **`shutdown_fn`** — runs on coordinator shutdown. Clears `vision_pro_activated` and `vision_pro_ready`.

No VP-specific code lives in `coordinator.py`; `spec.py` is the only entry point.

## VP_BYPASS_NECK

Set `VP_BYPASS_NECK=1` to skip the neck-at-init readiness check. The bypass publishes synthetic neck and VP status messages so the coordinator sees the subsystem as ready without a physical neck. Also short-circuits the iPhone gate when iPhone is not in `active_systems`.

## Activation flow

The activation button triggers `_vp_activation` in `spec.py`, which implements a two-step toggle currently:

1. **First press** — if VP is not yet ready: checks `neck_at_init_flag` (or bypass) and `iphone_ready_flag` (if iPhone is active), then publishes `VP_STATUS(ready=True)`.
2. **Second press (and subsequent)** — if VP is ready but prerequisites for publishing aren't met, prints diagnostics. Otherwise toggles `vision_pro_activated`, sends `VP_PUBLISH_ACTIVATION` and (after a 5 s delay) `NECK_ACTIVATION`, and also publishes a neck bridge activation command.

Configure the activation button via `activation_buttons:` in the robot config.