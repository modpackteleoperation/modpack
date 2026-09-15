# Robots

Each robot is a package under `modpack/robots/<name>/` containing:

- `config.yaml` — topology, roles, and modules list
- `run.py` — bridge runner script(s) that connect to the Modpack PC and drive hardware

The bridge (`modpack/bridge.py`) is the only modpack API robot runners import.

---

## Adding a robot

See [docs/robot_integration/ADD_A_ROBOT.md](../../docs/robot_integration/ADD_A_ROBOT.md).

---

## Current robots

| Robot | Topology | Notes |
|-------|----------|-------|
| `arx5` | managed | ARX5 bimanual arm + Phoenix6 omnidirectional base. Launches robot pc as well as Modpack pc.
| `rby1` | unmanaged | Rainbow Robotics RBY1 whole-body robot. GELLO leader arms + Vision Pro head + mobile base. Modpack publishes to RMQ; control/inference runs on a separate machine modpack does not manage. |
| `mock` | unmanaged | Minimal mock robot for testing without hardware. |

## config.yaml structure

```yaml
name: <robot>
topology: managed | unmanaged

modules:
  - gello        # GELLO leader arms
  - logger       # episode logger (always add this)
  - vision_pro   # VisionPro head tracking
  - base         # mobile base

module_overrides:
  base:
    use_mock_base: true   # run MockBase instead of real hardware

activation_buttons:
  1: arx          # numbered tap → toggle subsystem
  2: base

logging_streams:
  - logger_name: left_arm
    logger_type: joint_state
    port: 5593
    topic: left_arm
    ...
  - logger_type: camera
    camera_key: camera_head_main_rgb
    port: 5571
    width: 960
    height: 720
    fps: 15.0
```

For `topology: managed`, add a `robot_pc:` block with `scripts:` listing commands to SSH-launch on the Robot PC.

The `modules:` list names entries in the `SubsystemSpec` registry, where each module under `modpack/modules/<name>/` registers itself via `spec.py`. To add a new module-side capability (or change an existing module's activation behaviour), edit the spec, not the coordinator. See [modpack/orchestration/README.md](../orchestration/README.md) for the spec lifecycle and [docs/ADD_A_MODULE.md](../../docs/ADD_A_MODULE.md) for a walkthrough.
