# RB-Y1 Gripper Networking

The RB-Y1 gripper hardware runs on the robot PC. Remote
workstations interact with it over TCP using the Python client/server utilities
in this package.

## Run the server on the robot PC

```bash
cd /path/to/rby1
python -m gripper.gripper_server --host 0.0.0.0 --port 5678
```

`gripper.gripper_server` initializes the hardware `Gripper`, optionally homes
it, and exposes a JSON-over-TCP API. Use the `--skip-*` flags if you need to
skip initialization steps.

## Control the gripper from a remote workstation

The CLI now runs an interactive shell so you can send multiple commands in one
session:

```bash
cd /path/to/rby1
python -m gripper.gripper_client --host <robot-pc-ip> --port 5678
```

Once connected you'll see a `gripper>` prompt. Type commands such as:

```
gripper> set-target 0.05 0.05
gripper> status
gripper> stop
```

Available commands: `set-target`, `status`, `ping`, `start`, `stop`,
`initialize`, `homing`, plus `help` and `quit`.

## Programmatic control

```python
from gripper.gripper_client import GripperClient

client = GripperClient(host="192.168.30.2", port=5678)
client.set_target([0.07, 0.07])
```

`GripperClient` mirrors the `Gripper` API, so existing control code can talk to
the hardware even when it runs on another machine.
