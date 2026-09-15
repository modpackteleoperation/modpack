"""Simplified streaming bridge that executes action chunks sequentially."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import zmq
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PROJECT_ROOT = str(PROJECT_ROOT)

from camera.camera_stream import AravisCameraStreamer
from control.rby1_policy import RBY1PolicyRobot

OBS_TF_KEYS = {
    "left": "gripper_left_tf",
    "right": "gripper_right_tf",
    "head": "head_tf",
}
PAYLOAD_TF_KEYS = {
    "left": "left_tf",
    "right": "right_tf",
    "head": "head_tf",
}
GRIPPER_WIDTH_LIMITS = (0.0, 0.085)


def _rpy_to_matrix(rpy: Sequence[float]) -> np.ndarray:
    return Rotation.from_euler("xyz", np.asarray(rpy, dtype=float)).as_matrix()


def _make_transform(rotation_rpy: Sequence[float], translation: Sequence[float]) -> np.ndarray:
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = _rpy_to_matrix(rotation_rpy)
    mat[:3, 3] = np.asarray(translation, dtype=float)
    return mat


MODEL_TO_TCP_FRAME = {
    "left": _make_transform([np.pi, 0.0, 0.0], [0.0, 0.0, -0.2]),
    "right": _make_transform([0.0, np.pi, 0.0], [0.0, 0.0, -0.2]),
    "head": _make_transform([-np.pi / 2.0, 0.0, -np.pi / 2.0], [0.04, 0.0, 0.0601]),
}
TCP_TO_MODEL_FRAME = {name: np.linalg.inv(mat) for name, mat in MODEL_TO_TCP_FRAME.items()}


def _apply_transform_tf(tf: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return tf @ transform


def _convert_robot_observations(
    robot_obs: Dict[str, np.ndarray],
    transform_map: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    converted: Dict[str, np.ndarray] = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in robot_obs.items()}
    for effector, key in OBS_TF_KEYS.items():
        if key not in robot_obs or effector not in transform_map:
            continue
        converted[key] = _apply_transform_tf(robot_obs[key], transform_map[effector])
    return converted


def _offset_gripper_obs(obs: Dict[str, np.ndarray], offset: float) -> Dict[str, np.ndarray]:
    if offset == 0.0:
        return obs
    adjusted = dict(obs)
    for key in ("gripper_left_gripper_width", "gripper_right_gripper_width"):
        if key in adjusted:
            adjusted[key] = np.asarray(adjusted[key], dtype=float) + offset
    return adjusted


def _offset_gripper_action(payload: Dict[str, np.ndarray], offset: float) -> Dict[str, np.ndarray]:
    if offset == 0.0:
        return payload
    adjusted = dict(payload)
    if "left_gripper_width" in adjusted:
        adjusted["left_gripper_width"] = float(
            np.clip(adjusted["left_gripper_width"] - offset, GRIPPER_WIDTH_LIMITS[0], GRIPPER_WIDTH_LIMITS[1])
        )
    if "right_gripper_width" in adjusted:
        adjusted["right_gripper_width"] = float(
            np.clip(adjusted["right_gripper_width"] - offset, GRIPPER_WIDTH_LIMITS[0], GRIPPER_WIDTH_LIMITS[1])
        )
    return adjusted


class PolicyClient:
    """Thin ZMQ wrapper that talks to the detached policy server."""

    def __init__(self, ip: str, port: int, timeout: float = 2.0) -> None:
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.connect(f"tcp://{ip}:{port}")
        self._socket.setsockopt(zmq.RCVTIMEO, int(max(timeout, 0.1) * 1000))
        self._socket.setsockopt(zmq.SNDTIMEO, int(max(timeout, 0.1) * 1000))

    def close(self) -> None:
        self._socket.close(0)

    def request_action_indexing(self) -> Dict[str, Tuple[int, int]]:
        while True:
            try:
                self._socket.send_string("get_action_indexing")
                reply = self._socket.recv_pyobj()
            except zmq.Again:
                continue
            if isinstance(reply, dict):
                return reply
            time.sleep(0.5)

    def infer(self, obs: Dict[str, np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
        try:
            self._socket.send_pyobj(obs)
            return self._socket.recv_pyobj()
        except (zmq.Again, zmq.ZMQError):
            return None


def build_action_sequence(actions_tf: Dict[str, np.ndarray]) -> List[Dict[str, np.ndarray]]:
    if not actions_tf:
        return []

    length = max(arr.shape[0] for arr in actions_tf.values() if arr is not None)
    if length == 0:
        return []

    sequence: List[Dict[str, np.ndarray]] = []
    for idx in range(length):
        payload: Dict[str, np.ndarray] = {}
        for effector, obs_key in OBS_TF_KEYS.items():
            if obs_key not in actions_tf:
                continue
            tf_series = np.asarray(actions_tf[obs_key], dtype=float)
            if idx >= tf_series.shape[0]:
                continue
            payload[PAYLOAD_TF_KEYS[effector]] = tf_series[idx]
        if "gripper_left_gripper_width" in actions_tf:
            payload["left_gripper_width"] = float(actions_tf["gripper_left_gripper_width"][idx].reshape(-1)[0])
        if "gripper_right_gripper_width" in actions_tf:
            payload["right_gripper_width"] = float(actions_tf["gripper_right_gripper_width"][idx].reshape(-1)[0])

        for effector, key in PAYLOAD_TF_KEYS.items():
            if key in payload and effector in TCP_TO_MODEL_FRAME:
                payload[key] = _apply_transform_tf(payload[key], TCP_TO_MODEL_FRAME[effector])
        if payload:
            sequence.append(payload)
    return sequence


def main() -> None:
    parser = argparse.ArgumentParser(description="Simplified RBY1 policy streaming bridge")
    parser.add_argument("--policy-ip", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8766)
    parser.add_argument("--control-dt", type=float, default=0.05)
    parser.add_argument("--robot-horizon", type=int, default=2)
    parser.add_argument("--robot-stride", type=int, default=3)
    parser.add_argument("--camera-horizon", type=int, default=2)
    parser.add_argument("--camera-stride", type=int, default=3)
    parser.add_argument("--obs-frequency", type=float, default=60.0)
    parser.add_argument("--policy-timeout", type=float, default=2.0)
    parser.add_argument("--state-only", action="store_true", help="Skip camera streaming")
    parser.add_argument("--mock-cameras", action="store_true", help="Generate synthetic camera images")
    parser.add_argument("--sim-only", action="store_true", help="Run without the realtime controller.")
    parser.add_argument("--sim-model", default=None, help="Optional custom MJCF path for --sim-only mode.")
    parser.add_argument("--sim-viewer", action="store_true", help="Open a Mujoco viewer when using --sim-only.")
    parser.add_argument("--gripper-width-offset", type=float, default=0.005)
    args = parser.parse_args()

    control_dt = max(args.control_dt, 1e-2)
    robot = RBY1PolicyRobot(
        config_path=PROJECT_ROOT + "/config/wbc.yaml",
        use_sim=args.sim_only,
        sim_model_path=args.sim_model,
        sim_viewer=args.sim_viewer,
    )
    robot.start()

    camera_streamer: Optional[AravisCameraStreamer] = None
    policy_client: Optional[PolicyClient] = None

    try:
        robot.wait_until_ready()
        required_robot_samples = max(int(args.robot_horizon * args.robot_stride / (robot.dt * args.obs_frequency)), 20)
        if not robot.wait_for_observations(required_robot_samples, timeout=5.0):
            raise TimeoutError("Timed out waiting for initial robot observations")

        if not args.state_only and not args.sim_only:
            camera_streamer = AravisCameraStreamer()
            camera_streamer.start()
            try:
                camera_streamer.wait_until_ready(
                    min_frames=max(args.camera_horizon * args.camera_stride, 1),
                    timeout=2.0,
                )
            except TimeoutError as exc:
                print(f"[camera] {exc}")
        elif args.sim_only and not args.state_only:
            print("[camera] Skipping camera streamer in simulation mode.")

        policy_client = PolicyClient(
            ip=args.policy_ip,
            port=args.policy_port,
            timeout=args.policy_timeout,
        )
        _ = policy_client.request_action_indexing()

        while True:
            loop_start = time.monotonic()
            try:
                robot_obs_model = robot.get_observation_window(
                    horizon=args.robot_horizon,
                    stride=args.robot_stride,
                    obs_frequency=args.obs_frequency,
                )
            except Exception as exc:  # pragma: no cover - runtime safeguard
                print(f"[robot] Failed to gather observation: {exc}")
                time.sleep(control_dt)
                continue

            policy_robot_obs = _convert_robot_observations(robot_obs_model, MODEL_TO_TCP_FRAME)
            policy_robot_obs = _offset_gripper_obs(policy_robot_obs, args.gripper_width_offset)
            obs_dict = {k: v for k, v in policy_robot_obs.items() if k != "timestamp"}
            timestamps = robot_obs_model["timestamp"]

            if camera_streamer is not None:
                try:
                    camera_obs = camera_streamer.get_observation_window(
                        horizon=args.camera_horizon,
                        stride=args.camera_stride,
                        obs_frequency=args.obs_frequency,
                    )
                    obs_dict.update(camera_obs)
                except RuntimeError as exc:
                    print(f"[camera] {exc}")

            obs_dict["timestamp"] = timestamps

            reply = policy_client.infer(obs_dict)
            if reply is None or "actions_tf" not in reply:
                print("[policy] Inference timeout or malformed reply")
                time.sleep(control_dt)
                continue

            action_sequence = build_action_sequence(reply["actions_tf"])
            if not action_sequence:
                print("[policy] No executable actions returned")
                time.sleep(control_dt)
                continue

            for payload in action_sequence:
                adjusted_payload = _offset_gripper_action(payload, args.gripper_width_offset)
                robot.apply_action(adjusted_payload, duration=control_dt)
                time.sleep(control_dt)

            elapsed = time.monotonic() - loop_start
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)

    except KeyboardInterrupt:
        print("[main] Interrupted, shutting down...")
    finally:
        if policy_client is not None:
            policy_client.close()
        if camera_streamer is not None:
            camera_streamer.stop()
        robot.stop()


if __name__ == "__main__":
    main()
