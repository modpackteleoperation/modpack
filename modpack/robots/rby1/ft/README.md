# FT utilities

Utilities and entrypoints for force/torque calibration and inspection.

## Modules
- `ft.calibrator`: runtime calibrator used by WBC to apply offsets, gravity compensation, and frame transforms from `config/ft_sensor.yaml`.
- `ft.calibrate_wbc`: drives the robot through tilts and solves for offsets/COM/gravity; prints YAML you can paste into `ft_sensor.yaml`.
- `ft.stream_calibrated_wrench`: replays orientations and streams/plots calibrated wrenches to validate a calibration.

## Usage
- Calibrate: `python ft/calibrate_wbc.py --arm left --wbc-config config/wbc.yaml`
- Stream/validate: `python ft/stream_calibrated_wrench.py --arm right --ft-config config/ft_sensor.yaml`

## Config snapshots
- `config/ft_sensor.yaml` (left/right arms):
  - Left offsets: fx -1.707501, fy 3.936528, fz -7.572591, tx -0.353983, ty -0.036272, tz 0.181269
  - Left gravity: x -0.041531, y -0.047670, z -9.028997; COM: x 0.001160, y 0.006207, z -0.024986
  - Left transforms: force/torque matrices are the same: `[[0, 1, 0], [1, 0, 0], [0, 0, -1]]`
  - Right offsets: fx 12.524427, fy 6.432185, fz -7.595952, tx 0.089976, ty 0.180715, tz 0.475939
  - Right gravity: x 0.609184, y 0.996855, z -9.283200; COM: x -0.001993, y -0.003849, z -0.023205
  - Right transforms: force/torque matrices match: `[[0, -1, 0], [-1, 0, 0], [0, 0, -1]]`

- `config/wbc.yaml` (selected FT-relevant fields):
  - `low_pass_freq_hz: 1`
  - `admittance.enabled: true`
  - Desired wrench: left/right `[0, 0, 0, 0, 0, 0]`
  - Controller: dt 0.01, stiffness `[300, 300, 300, 1, 1, 1]`, damping `[7, 7, 7, 0.2, 0.2, 0.2]`, inertia `[1, 1, 1, 0.005, 0.005, 0.005]`, stiction zeros, max spring force 50.0, torque 4.0, direct force-control gains all zero, I-limit zeros.
  - `force_controlled_axes.mode: translation_force`

## Dependency
These tools expect the `force_control` package to be installed (tested with the copy included in this repository at `../force_control`).
