"""Stub policy server for smoke-testing rby1_wbc_policy_joint.py without a real model.

Responds to the ZMQ REQ/REP protocol:
  - "get_obs_keys" -> dict of required observation keys
  - obs dict       -> action chunk with timestamps

Run this before launching rby1_wbc_policy_joint.py --sim-only --state-only.
"""

import time
import numpy as np
import zmq

PORT = 8766
T = 8           # action chunk length
CONTROL_DT = 0.1

OBS_KEYS = {
    "left_arm_state_joint_pos":   (T, 7),
    "right_arm_state_joint_pos":  (T, 7),
    "head_state_joint_pos":       (T, 2),
    "left_eef_state_joint_pos":   (T, 1),
    "right_eef_state_joint_pos":  (T, 1),
    "body_state_pos_xyz":         (T, 3),
    "body_state_quat_wxyz":       (T, 4),
    # Uncomment to exercise the auto-enable torque path in
    # rby1_wbc_policy_joint.py. The bridge will then require --urdf-path and
    # populate a (16, 7) history sampled from its torque ring buffer.
    # "left_arm_state_joint_torque":  (16, 7),
    # "right_arm_state_joint_torque": (16, 7),
}

ctx = zmq.Context()
sock = ctx.socket(zmq.REP)
sock.bind(f"tcp://127.0.0.1:{PORT}")
print(f"[stub] listening on tcp://127.0.0.1:{PORT}")

while True:
    msg = sock.recv_pyobj()

    if msg == "get_obs_keys":
        sock.send_pyobj(OBS_KEYS)
        print("[stub] sent obs keys")
        continue

    # msg is an obs dict — print a summary and reply with a zero action chunk
    ts_arr = np.asarray(msg.get("timestamp", [0.0])).reshape(-1)
    print(f"[stub] got obs @ t={ts_arr[-1]:.3f}, keys={sorted(msg.keys())}")

    now = time.monotonic()
    action_ts = np.array([now + 0.3 + i * CONTROL_DT for i in range(T)])

    sock.send_pyobj({
        "actions": {
            "left_qpos":           np.zeros((T, 7),  dtype=np.float32),
            "right_qpos":          np.zeros((T, 7),  dtype=np.float32),
            "head_qpos":           np.zeros((T, 2),  dtype=np.float32),
            "left_gripper_width":  np.full(T, 0.04,  dtype=np.float32),
            "right_gripper_width": np.full(T, 0.04,  dtype=np.float32),
            "base_delta":          np.zeros((T, 3),  dtype=np.float32),
        },
        "timestamps": action_ts,
    })
