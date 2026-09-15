# ARX5

## Overview

- Arms: 2 × 6-DoF
- Neck: ARX5 cartesian
- Base: mobile (holonomic)

## Topology

`managed` — ModPack SSH-launches the robot-side runners on a separate ModPack-owned PC; the two machines talk only over the RMQ bus.

| PC | Launch | Runs |
|----|--------|------|
| ModPack PC | `python -m modpack --gello --robot arx5` | `gello`, `logger`, `vision_pro`; RMQ bus; operator keys |
| Robot PC | SSH-launched by coordinator (`robot_pc:` in `config.yaml`) | bridge + runners: arms, neck, cameras, iPhone |

## Installation

`arx5_interface` is the only dep not in the conda env. Install it on the Robot PC:

- `arx5_interface` — Python binding built from the arx5 SDK (`arx5-sdk`). Checkout at `../arx5-sdk/`; runners add `../arx5-sdk/python/` and `../arx5-sdk/lib/` to `sys.path`.
- GELLO leader-arm build/wiring → [assembly guide](https://modpackteleoperation.github.io/docs/assembly/) · [BOM](https://modpackteleoperation.github.io/docs/bom/)

## Calibration

- `scripts/calibration/zero_calib.py` — required before teleop ([walkthrough](../../../scripts/calibration/README.md))

## Running

```bash
python -m modpack --gello --robot arx5
```

The coordinator SSH-launches the Robot-PC runners (see Topology). Entry point there is `run.py`, with per-part modes: `--arm right/left`, `--neck`, `--cameras`, `--iphone`.

## Config

- `config.yaml` — source of truth for this robot (roles, modules, logging streams, activation buttons, `robot_pc`)
- `neck_config.yaml` — neck IK / safety

## See also

- [`../README.md`](../README.md)
- [`../../../docs/robot_integration/ADD_A_ROBOT.md`](../../../docs/robot_integration/ADD_A_ROBOT.md)
