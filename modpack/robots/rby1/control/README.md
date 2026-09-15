# RBY1 Realtime Controller (pybind)

This directory contains a trimmed-down version of the realtime controller used
to command the RBY1 robot at ~500 Hz. The controller is implemented entirely in
C++, exposes a small C++ executable (`rby1_realtime_control_main`), and ships a
`pybind11` module (`rby1_controller`) for direct consumption from Python.

## Install rby1-sdk
Refer to [RBY1_SDK_BUILD.md](RBY1_SDK_BUILD.md) for build steps and notes.

## Install cpplibrary & force_control
Install dependencies:
```bash
sudo apt update
sudo apt install libeigen3-dev libyaml-cpp-dev
```

Build and install `cpplibrary`:
```bash
cd {PROJECT_HOME}/cpplibrary
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
sudo make install
```
If you do not have sudo, add `-DCMAKE_INSTALL_PREFIX=$HOME/.local` to the
`cmake ..` configure step and run `make install` without sudo.

Build and install `force_control`:
```bash
cd {PROJECT_HOME}/force_control
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
sudo make install
```
If you do not have sudo, add `-DCMAKE_INSTALL_PREFIX=$HOME/.local` to the
`cmake ..` configure step and run `make install` without sudo.

## Building
```bash
mkdir -p build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Debug \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DPython_EXECUTABLE="$(which python)" \
  -DRBY1_SDK_INCLUDE_DIR="/usr/local/include" \
  -DRBY1_SDK_LIBRARY="/usr/local/lib/librby1-sdk.so"
cmake --build .

cd ..
ln -s build/rby1_controller*.so rby1_controller.so
```
If you do not have sudo, add `-DCMAKE_INSTALL_PREFIX=$HOME/.local` to the
`cmake ..` configure step and run `make install` without sudo.  
If your SDK install uses non-standard paths, set `RBY1_SDK_INCLUDE_DIR` and
`RBY1_SDK_LIBRARY` explicitly.

## Rebuilding
If you have made the source code changes
```bash
cd build
cmake --build . -j
```

## Python Usage

```python
from rby1.control import RealtimeDriver, Config

config = Config()
config.robot_address = "192.168.30.1:50051"

driver = RealtimeDriver(config)
driver.start()
driver.wait_until_ready(10.0)
driver.set_body_position_targets([...])
driver.stop()
```

### Config (all optional; defaults are used when not set):
- `robot_address` (default: `"192.168.30.1:50051"`): gRPC control endpoint.
- `low_pass_freq_hz` (default: `1.0`): cutoff frequency (Hz) for state low-pass filtering.
- `expect_wheel_velocity` (default: `False`): set `True` when sending wheel velocity commands.
- `command_timeout_us` (default: `1000000`): command timeout in microseconds.
