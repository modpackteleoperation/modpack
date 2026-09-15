# Examples

Minimal bridge integrations. These show the smallest working pattern for each role type.

| File | What it shows |
|------|---------------|
| `minimal_joint_arm.py` | Joint-space arm: read command, move robot, publish state |
| `minimal_sensor.py` | Sensor stream: publish F/T data at a fixed rate |

## Usage

Replace `MyRobotSDK` with your hardware SDK and point `from_config` at your `config.yaml`:

```python
bridge = RobotBridge.from_config("modpack/robots/my_robot/config.yaml")
```

See `modpack/robots/arx5/config.yaml` for a complete config example.
