# Adding a New Robot

A robot integration is a `config.yaml` plus bridge calls in whatever script drives the hardware. The bridge contract is the same regardless of topology — only where scripts run differs.

---

## 1. Create the robot package

```
modpack/robots/<your_robot>/
    __init__.py
    config.yaml
```

---

## 2. Write `config.yaml`

```yaml
name: my_robot
topology: managed    # or unmanaged — see below

roles:
  right_arm: {type: joint_arm,  dof: 7}
  left_arm:  {type: joint_arm,  dof: 7}
  body:      {type: mobile_base}

modules:
  - gello
  - logger
  # - vision_pro

rmq:
  host: 192.168.0.225   # Modpack PC IP
  port: 5555
  activation_port: 5556
```

**Role types:** `joint_arm`, `cartesian_arm`, `mobile_base`, `sensor_stream`, `video_stream`

**`modules:`** lists the Modpack-side processes the orchestration layer starts. Your robot runner does not start these — they run on the Modpack PC.

---

## 3. Add bridge calls

ModPack only cares about two calls: `get_command` to receive GELLO targets and `publish_state` to log robot state. Where and how you add them depends on your topology.

### `unmanaged` — augment your existing control loop

The bridge runs on the same machine as orchestration. Add it directly to your control script:

```python
# my_robot/scripts/my_control_loop.py  (your existing script, augmented)
import time
from modpack.bridge import RobotBridge

bridge = RobotBridge.from_config("modpack/robots/my_robot/config.yaml")
bridge.connect()

# your existing SDK setup here...

try:
    while wbc.is_running():                      # your existing loop condition
        cmd = bridge.get_command("right_arm")    # None when inactive
        if cmd is not None:
            wbc.set_arm_target(cmd.joint_pos)    # pass to your SDK

        state = wbc.get_arm_state()
        bridge.publish_state("right_arm", {
            "positions":  state.joint_pos,       # ndarray from your SDK
            "gripper":    state.gripper,
            "timestamp":  time.monotonic(),
            "data_valid": True,
        })
finally:
    bridge.disconnect()
```

### `managed` — write a standalone runner on the Robot PC

The coordinator SSHes the Robot PC and launches each entry in `robot_pc.scripts` as a separate process. Typically one `run.py` with per-component modes (`--arm right`, `--arm left`, `--neck`) covers everything:

```python
# my_robot/run.py  (launched by coordinator via robot_pc.scripts)
import argparse, time
from modpack.bridge import RobotBridge
from my_robot_sdk import ArmController

_CONFIG = "modpack/robots/my_robot/config.yaml"

def run_arm(arm: str) -> None:
    bridge = RobotBridge.from_config(_CONFIG)
    robot  = ArmController(arm=arm)
    robot.wait_until_ready()
    bridge.connect()

    role = f"{arm}_arm"
    try:
        while True:
            cmd = bridge.get_command(role)       # None when inactive
            if cmd is not None:
                robot.move(cmd.joint_pos, cmd.gripper)

            bridge.publish_state(role, {
                "positions":  robot.get_joints(),
                "gripper":    robot.get_gripper(),
                "timestamp":  time.monotonic(),
                "data_valid": True,
            })
    finally:
        robot.close()
        bridge.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["right", "left"])
    args = parser.parse_args()
    run_arm(args.arm)
```

`bridge.is_active` is `False` until the operator activates — `get_command()` returns `None` automatically when inactive, so no explicit gating is required.

Call `publish_state` unconditionally in your loop. The logger decides what to write based on episode state — you do not need to gate it on `bridge.is_recording`. Use `is_recording` only if your own control logic needs to behave differently during a recorded episode.

---

## 4. Run it

**`unmanaged`:**

```bash
# terminal 1 — Modpack PC
python -m modpack --gello --robot my_robot

# terminal 2 — same machine
python my_robot/scripts/my_control_loop.py
```

**`managed`:** add a `robot_pc:` block to `config.yaml` — the coordinator SSHes the Robot PC and launches each script automatically:

```yaml
robot_pc:
  user: robot
  workspace: ~/modpack
  scripts:
    - python -m modpack.robots.my_robot.run --arm right
    - python -m modpack.robots.my_robot.run --arm left
```

Then on the Modpack PC:

```bash
python -m modpack --gello --robot my_robot   # SSHes Robot PC and launches the scripts above
```

---

## Bridge API reference

| Method | Description |
|--------|-------------|
| `RobotBridge.from_config(path)` | Load config, derive RMQ endpoints |
| `connect()` / `disconnect()` | Open/close RMQ connections |
| `get_command(role)` | Latest command for this role, or `None` |
| `publish_state(role, state)` | Send state to logger |
| `is_active` | `True` after operator activates |
| `is_recording` | `True` between episode start/stop |
| `activation_callback` | Constructor param — called on any activation message |

## Role type schemas

Defined in `modpack/schemas.py`. The `get_command()` column is enforced — the bridge
returns those dataclasses. The `publish_state()` column is **advisory**: the bridge
accepts a plain dict and does not validate it, so treat these as the baseline fields to
provide. You may include extra fields (e.g. `target_positions`, `gripper_target`).

| Role type | `get_command()` returns | `publish_state()` baseline fields (advisory) |
|-----------|------------------------|----------------------------------|
| `joint_arm` | `JointArmCommand(joint_pos: ndarray(dof,), gripper: float)` | `positions`, `gripper`, `timestamp`, `data_valid` |
| `cartesian_arm` | `CartesianArmCommand(eef_pose: ndarray(6,))` | `positions: ndarray(6,)`, `timestamp`, `data_valid` |
| `mobile_base` | `MobileBaseCommand(base_pose: ndarray(3,))` | `positions: ndarray(3,)`, `timestamp`, `data_valid` |
| `sensor_stream` | `None` — no commands | `data: ndarray`, `timestamp`, `data_valid` |
| `video_stream` | `None` — no commands | `frame: bytes`, `topic: str`, `timestamp`, `data_valid` |
