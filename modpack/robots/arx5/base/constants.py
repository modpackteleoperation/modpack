"""Hardware constants for the ARX5 Phoenix6 omnidirectional base."""
import numpy as np

# Vehicle center to steer axis (m) — ARX5 geometry
h_x = 0.140150 * np.array([1.0, 1.0, -1.0, -1.0])
h_y = 0.120150 * np.array([-1.0, 1.0, 1.0, -1.0])

# CANcoder magnet calibration offsets
ENCODER_MAGNET_OFFSETS = [1580.0 / 4096, 905.0 / 4096, 25.0 / 4096, -360.0 / 4096]

# High-level policy control period (s)
POLICY_CONTROL_PERIOD = 0.1

# The 250 Hz Vehicle control loop runs in its own dedicated process (see
# base_server.BaseManager) so the Flask/SocketIO WebXR server and teleop policy
# cannot starve it of the GIL. BaseProcess talks to it over a local RPC proxy.
BASE_RPC_HOST = "127.0.0.1"
BASE_RPC_PORT = 50000
RPC_AUTHKEY = b"arx5-base-rpc"
