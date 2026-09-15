# Logger

Backpack stream and camera logger. Runs on the Modpack PC, subscribes to RMQ topics, and writes zarr datasets to disk.

## Components

**`BackpackStreamLogger`** (`stream_logger.py`) — subscribes to joint state, base state, cartesian state, and torque feedback topics; writes to zarr via `robologger`. Skips entries with `logger_type: camera`.

**`BackpackCameraLogger`** (`stream_logger.py`) — reads `logger_type: camera` entries from `logging_streams`; receives JPEG frames from the camera server (port 5570), decodes them, and writes video via `VideoLogger`.

> **Note:** ARX5 wrist cameras are logged directly on the robot PC via `VideoLogger` inside `camera_process.py` and do not go through the camera server or `BackpackCameraLogger`.

## How it starts

The coordinator starts the logger automatically on every Modpack PC run.

Entry point: `runner.py:run_backpack_logger_process(log_path, cfg)`. Spawns `BackpackStreamLogger` and `BackpackCameraLogger` in two daemon threads.

## Logging streams

All stream definitions come from the robot's `config.yaml` via `RobotConfig.logging_streams`. The two logger classes filter by `logger_type`.

**Proprio entries** (`joint_state`, `base_state`, `cartesian_state`):
- `logger_name` — zarr dataset name
- `port` — RMQ port the logger's RMQ server listens on for start/stop commands
- `logger_type` — `joint_state` | `base_state` | `cartesian_state`
- `control_freq`, `joint_dof`, `log_eef` — passed to `RobotCtrlLogger` / `MobileBaseLogger`
- `gripper_port` / `gripper_logger_name` — optional companion gripper logger

**Companion EE gripper logger.** When a `joint_state` entry declares `gripper_port` + `gripper_logger_name`, `BackpackStreamLogger` builds a second `RobotCtrlLogger` (1-DOF) whose port is added to `MainLogger.logger_endpoints`. Its command queue is drained on every poll iteration so a `stop` at episode end always flushes buffered data to zarr.

**State message keys read** (`joint_state`):
- `positions` — required, written as `state_joint_pos`
- `eef_pose` — optional, written as `state_pos_xyz` + `state_quat_wxyz` when `log_eef: true`
- `gripper` — optional gripper opening (m) → companion EE logger `state_joint_pos`
- `gripper_target` — optional commanded gripper opening → companion EE logger `target_joint_pos`
- `target_positions` — optional commanded joint positions → `target_joint_pos`

**Camera entries** (`logger_type: camera`):
- `camera_key` — zarr dataset name and RMQ topic name on the camera server
- `port` — RMQ port the `VideoLogger`'s RMQ server listens on for start/stop commands
- `width`, `height`, `fps` — declared resolution; frames are resized to match if needed

## RMQ servers consumed

| Server | Port | Used by |
|--------|------|---------|
| Data server | 5555 | BackpackStreamLogger (proprio state topics) |
| Feedback server | 5581 | BackpackStreamLogger (torque topics) |
| Camera server | 5570 | BackpackCameraLogger (JPEG frame topics) |
