"""Background publisher: streams head camera frames as a JPEG UDP stream to the Vision Pro app,
and publishes the camera data + status heartbeat that vision_pro_process expects.

Protocol summary (matching read_iphone_camera.py UDPRgbSender):
  - Camera data → camera RMQ server (port 5570):
      Topics.IPHONE_RGB       raw uint8 bytes (H×W×3)
  - Status heartbeat → activation RMQ server (port 5556):
      Topics.NECK_IPHONE_STATUS  NeckiPhoneStatusMessage JSON (iphone_publishing=True)
      Required for vision_pro_process.check_status_messages() to set ready_to_stream=True
  - RGB image → Vision Pro app UDP (port 6007):
      UDPRgbSender packet format: 12-byte header (frame_id, chunk_idx, chunk_count) + JPEG bytes
      After resize, frames are rotated 90° CCW unless RBY1_VP_RGB_CCW90=0.
"""

from __future__ import annotations

import os
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Sequence

import logging
import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from modpack.orchestration.message_formats import (
    MessageFactory,
    NeckActivationMessage,
    Topics,
    serialize_message,
)

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


class UDPRgbSender:
    """Sends JPEG-compressed RGB frames over UDP in the format expected by the Vision Pro app.

    Packet layout per chunk (little-endian):
      [0:12]  header: frame_id, chunk_idx, chunk_count  (3× uint32)
      [12:]   JPEG payload bytes for this chunk
    """

    def __init__(self, host: str, port: int, mtu: int = 1400, jpeg_quality: int = 30):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.mtu = mtu
        self.jpeg_quality = jpeg_quality

    def send(self, rgb_u8: np.ndarray, frame_id: int) -> None:
        ret, buf = cv2.imencode('.jpg', rgb_u8, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ret:
            return
        data = buf.tobytes()
        payload_mtu = self.mtu - 12  # 12-byte header: frame_id, chunk_idx, chunk_count
        chunk_count = (len(data) + payload_mtu - 1) // payload_mtu
        for chunk_idx in range(chunk_count):
            s = chunk_idx * payload_mtu
            e = min(len(data), s + payload_mtu)
            header = struct.pack("<III", frame_id, chunk_idx, chunk_count)
            self.sock.sendto(header + data[s:e], self.addr)


class CameraStreamPublisher:
    STATUS_INTERVAL = 0.5  # seconds between NECK_IPHONE_STATUS heartbeats

    def __init__(
        self,
        bridge,
        vp_ip: str = "192.168.0.57",
        vp_rgb_port: int = 6007,
        camera_key: str = "camera_head_main_rgb",
        camera_keys: Optional[Sequence[str]] = None,
        fps: float = 15.0,
        camera_config_path: Optional[str] = None,
        camera_binning: Optional[Sequence[int]] = None,
        maximize_camera_resolution: bool = False,
        rgb_quality: int = 30,
        rgb_mtu: int = 1400,
    ):
        from camera.camera_stream import AravisCameraStreamer

        # camera_keys drives RMQ publishing; camera_key is the VP UDP stream camera
        resolved_keys = list(camera_keys) if camera_keys else [camera_key]

        config_path = camera_config_path or PROJECT_ROOT + "/config/camera.yaml"
        self._streamer = AravisCameraStreamer(
            config_path=config_path,
            camera_keys=resolved_keys,
            binning_override=camera_binning,
            expand_to_max_resolution=maximize_camera_resolution,
        )
        self._camera_key = camera_key
        self._fps = fps
        self._bridge = bridge
        self._factory = MessageFactory()
        self._rgb_sender = UDPRgbSender(vp_ip, vp_rgb_port, mtu=rgb_mtu, jpeg_quality=rgb_quality)

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._frame_id = 0
        self._last_status_ts = -999.0  # ensure first heartbeat fires immediately
        self._logged_shapes: set[str] = set()

    def start(self) -> None:
        camera_serial = self._streamer._camera_map.get(self._camera_key, "<unknown>")
        logging.info(
            "[CameraStreamPublisher] Starting — camera_key=%s serial=%s vp_rgb→%s:%d  fps=%.1f  mock=%s",
            self._camera_key, camera_serial,
            self._rgb_sender.addr[0], self._rgb_sender.addr[1],
            self._fps, self._streamer._mock_mode,
        )
        self._streamer.start()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._publish_loop, name="cam-publisher", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._streamer.stop()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _send_status_heartbeat(self) -> None:
        now = time.monotonic()
        if now - self._last_status_ts < self.STATUS_INTERVAL:
            return
        try:
            msg = self._factory.create_neck_iphone_status_message(iphone_publishing=True)
            self._bridge.publish_activation_raw(Topics.NECK_IPHONE_STATUS, serialize_message(msg))

            neck_activation_msg = NeckActivationMessage(
                timestamp=now,
                command="activate",
                target="neck",
                source="camera_stream_publisher",
            )
            self._bridge.publish_activation_raw(Topics.NECK_ACTIVATION, serialize_message(neck_activation_msg))

            self._last_status_ts = now
        except Exception as exc:
            logging.warning("[CameraStreamPublisher] Failed to send heartbeat: %s", exc)

    def _publish_loop(self) -> None:
        interval = 1.0 / self._fps
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                obs, _ = self._streamer.get_observation_window(
                    horizon=1, include_timestamps=True
                )

                # Publish each active camera as TSB1-wrapped JPEG to its own RMQ topic
                ts_ns = time.time_ns()
                tsb1_header = b"TSB1" + struct.pack("!Q", ts_ns)
                for key, frames in obs.items():
                    try:
                        frame = frames[0]
                        if key not in self._logged_shapes:
                            self._logged_shapes.add(key)
                            logging.info(
                                "[CameraStreamPublisher] First frame shape for %s: %s dtype=%s",
                                key,
                                tuple(frame.shape),
                                frame.dtype,
                            )
                        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if ret:
                            self._bridge.publish_camera_raw(key, tsb1_header + buf.tobytes())
                    except Exception as exc:
                        logging.warning("[CameraStreamPublisher] put_data(%s) error: %s", key, exc)

                # UDP stream to Vision Pro: primary camera only, resized to 960x720, then
                # 90° CCW to match expected headset orientation (iPhone path uses 90° CW on publish).
                # Set RBY1_VP_RGB_CCW90=0 to send without this rotation.
                if self._camera_key in obs:
                    vp_frame = cv2.resize(
                        obs[self._camera_key][0], (960, 720), interpolation=cv2.INTER_LINEAR
                    )
                    vp_frame = cv2.cvtColor(vp_frame, cv2.COLOR_BGR2RGB)
                    if os.environ.get("RBY1_VP_RGB_CCW90", "1").lower() not in (
                        "0",
                        "false",
                        "no",
                    ):
                        vp_frame = cv2.rotate(
                            vp_frame, cv2.ROTATE_90_COUNTERCLOCKWISE
                        )
                    self._rgb_sender.send(vp_frame, self._frame_id)

                self._frame_id += 1
                if self._frame_id == 1:
                    logging.info(
                        "[CameraStreamPublisher] First frame published — cameras: %s",
                        list(obs.keys()),
                    )
            except Exception as exc:
                logging.warning("[CameraStreamPublisher] Publish error: %s", exc)

            # Heartbeat is rate-limited independently of frame rate
            self._send_status_heartbeat()

            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, interval - elapsed))
