# RBY1

## Installation
- Clone with submodules:
  ```bash
  git clone --recurse-submodules https://github.com/modpackteleoperation/modpack.git
  cd modpack/modpack/robots/rby1
  ```
- Python dependencies (venv example):
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  pip3 install -r requirements.txt
  ```
  Or create a conda env and `pip install -r requirements.txt` inside it.
- `robologger`: `pip install -e third_party/robologger` from the repository root
- From-source deps:
  - `rby1-sdk` → [`control/RBY1_SDK_BUILD.md`](control/RBY1_SDK_BUILD.md)
  - C++ control chain (`cpplibrary` → `force_control` → `control`) → [`control/README.md`](control/README.md)
  - GigE cameras (optional) → [`camera/README.md`](camera/README.md)
- Calibration (required before teleop): modpack `scripts/calibration/zero_calib.py` (in the parent modpack repo)
- GELLO leader-arm build/wiring → [assembly guide](https://modpackteleoperation.github.io/docs/assembly/) · [BOM](https://modpackteleoperation.github.io/docs/bom/)

## Directory Overview
- `camera/`: Aravis-based robot camera utilities.
- `config/`: Core configuration files.
- `control/`: Realtime driver and whole-body controller (`robot_policy`).
- `demo/`: Trajectory loader/recorder and saved trajectories.
- `ft/`: Compliance control utilities.
- `gripper/`: Real-robot hardware API for the gripper.
- `log/`: Logging output directory.
- `model/`: MJCF models.
- `rby1/`: Whole-body IK, state visualizer, and related modules.
- `scripts/`: Executable scripts.
- `teleop/`: Teleoperation modules (VR, iPhone).

## Running
### Simulator
Pull the simulator image:
```bash
docker pull rainbowroboticsofficial/rby1-sim:0.10.3-m_v1.0
```

Quick run:
```bash
xhost +
docker run --rm --gpus all --network host --ipc host \
  -e DISPLAY=${DISPLAY} -v /tmp/.X11-unix:/tmp/.X11-unix \
  rainbowroboticsofficial/rby1-sim:0.10.3-m_v1.0
```

Interactive run:
```bash
xhost +
docker run -it --name sim --gpus all --network host --ipc host \
  -e DISPLAY=${DISPLAY} -v /tmp/.X11-unix:/tmp/.X11-unix \
  rainbowroboticsofficial/rby1-sim:0.10.3-m_v1.0 bash
docker exec -it sim /bin/bash
./app_main
```

### Real Robot
- Turn on RPC / UPC (for camera and gripper).
- Open WebUI at [http://192.168.30.1:5173/](http://192.168.30.1:5173/) on the operator machine (wired to the robot).
- Power on and Servo On the robot.

### Scripts
```bash
python3 scripts/rby1_wbc_gui.py      # MuJoCo mocap-based control
python3 scripts/rby1_wbc_policy.py   # Diffusion Policy-based control
python3 scripts/rby1_wbc_teleop.py   # Teleoperation (VR/iPhone)
python3 scripts/rby1_wbc_traj.py     # Replay recorded trajectory
```
Check `config/` and adjust parameters as needed.

---

## Modpack integration

**Topology:** `unmanaged` — modpack publishes commands/state into RMQ; it does not launch or manage a robot PC. rby1's control/inference runs on a separate machine that connects on its own.

**Run:**
```bash
python -m modpack --gello --robot rby1
```

- Modules, roles, logging streams (ports/resolutions) → `config.yaml`
- Base runs mock by default (`module_overrides.base.use_mock_base: true`) as rby1 drives its own base
- Episode + activation controls (`s` / `v` / `q`) → modpack README (Usage)
