# Calibration

## Contents

| File | Role |
|------|------|
| `zero_calib.py` | GELLO joint-offset + gripper calibration (per arm) |

## Usage

```bash
python scripts/calibration/zero_calib.py <config_path> <arm> [--skip-gripper]
# examples
python scripts/calibration/zero_calib.py modpack/robots/arx5/config.yaml right
python scripts/calibration/zero_calib.py modpack/robots/rby1/config.yaml left --skip-gripper
```

## Arguments

| Arg | Values | Role |
|-----|--------|------|
| `config_path` | path | robot `config.yaml` (reads its `gello:` hardware + `port_configs`) |
| `arm` | `left` \| `right` | which GELLO arm to calibrate |
| `--skip-gripper` | flag | skip gripper open/close calibration |

## Steps

1. Run the command — connects to the GELLO arm, enables torque.
2. **Joint offsets:** move the arm to its neutral all-zero pose, press Enter. Reads raw joint angles → computes `joint_offsets`; prints a verification table.
3. **Gripper** (unless `--skip-gripper`): open the gripper fully → Enter; close fully → Enter. Captures `open_angle_deg` / `closed_angle_deg`.
4. Script prints a results block; torque is disabled on exit.
5. **Manual:** copy the printed values into the robot's `config.yaml` under `port_configs → <port>` (`joint_offsets` and the `gripper:` angles). The script does **not** write the config.
