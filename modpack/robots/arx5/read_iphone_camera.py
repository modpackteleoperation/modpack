

#!/usr/bin/env python3
import numpy as np
import cv2
import hashlib
import time
import os
import sys
import yaml
import threading
import queue
import socket
import struct
from multiprocessing import Event
from record3d import Record3DStream
from robotmq.utils import deserialize
from modpack.orchestration.message_formats import *
from modpack.orchestration.process_shutdown import (
    ShutdownFlag,
    register_signals,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modpack.modules.vision_pro.transformation_helper import quaternion_camera_pose_to_extrinsic_matrix
from modpack.utils import DebugPrinter
from robologger.loggers.video_logger import VideoLogger


class UDPPcdSender:
    def __init__(self, host: str, port: int, mtu: int = 1200):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.mtu = mtu

    def send_pointcloud(self, xyz_f32, rgba_u8, frame_id: int, extrinsic_f32: np.ndarray):
        assert xyz_f32.dtype == np.float32 and xyz_f32.ndim == 2 and xyz_f32.shape[1] == 3
        assert rgba_u8.dtype == np.uint8 and rgba_u8.ndim == 2 and rgba_u8.shape[1] == 4
        assert extrinsic_f32.shape == (4, 4)
        n = xyz_f32.shape[0]
        extrinsic_payload = np.asarray(extrinsic_f32, dtype=np.float32).tobytes(order="C")

        header_bytes = 16
        extrinsic_bytes = 16 * 4
        bytes_per_point = 16
        pts_per_packet = max(1, (self.mtu - header_bytes - extrinsic_bytes) // bytes_per_point)
        chunk_count = (n + pts_per_packet - 1) // pts_per_packet

        for chunk_idx in range(chunk_count):
            s = chunk_idx * pts_per_packet
            e = min(n, s + pts_per_packet)
            pts = e - s

            header = struct.pack("<IIII", frame_id, chunk_idx, chunk_count, pts)
            payload = bytearray(header)
            payload += extrinsic_payload  # or order="F" for column-major
            payload += xyz_f32[s:e].tobytes(order="C")
            payload += rgba_u8[s:e].tobytes(order="C")
            self.sock.sendto(payload, self.addr)

class UDPRgbSender:
    def __init__(self, host: str, port: int, mtu: int = 1400, jpeg_quality: int = 30):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.mtu = mtu
        self.jpeg_quality = jpeg_quality

    def send_rgb(self, rgb_u8: np.ndarray, frame_id: int):
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

class ReadiPhone():
    def __init__(self, bridge, neck_config_path: str, logging_endpoint: str, debug: bool):
        self.event = Event()
        self._bridge = bridge
        self.neck_config_path = neck_config_path
        self.debug = debug
        self.debug_printer = DebugPrinter(enabled=debug)
        with open(neck_config_path, 'r') as f:
            self.neck_config = yaml.safe_load(f)

        self.camera_config = self.neck_config['camera_config']

        self.logger = VideoLogger(
            name="head_camera_0",
            endpoint=logging_endpoint,
            attr={"camera_configs": self.camera_config},
            codec="libx264"
        ) if logging_endpoint is not None else None


        self.session = None
        self.msg_factory = MessageFactory()
        self.frame_counter = 0
        self.obs_hash_debug = True
        self.obs_hash_debug_every = 30

        # Logging thread infrastructure
        self.log_queue = queue.Queue(maxsize=5)
        self.logging_thread = threading.Thread(target=self._logging_worker, daemon=True)
        self.logging_thread.start()

        # Pull resolutions from config so intrinsics match actual sensor sizes
        self.rgb_config = self.camera_config.get("iphone_rgb", {})
        self.depth_config = self.camera_config.get("iphone_depth", {})
        self.rgb_width = int(self.rgb_config.get("width", 960))
        self.rgb_height = int(self.rgb_config.get("height", 720))
        self.depth_width = int(self.depth_config.get("width", 192))
        self.depth_height = int(self.depth_config.get("height", 256))
        self.depth_indices = np.indices((self.depth_width, self.depth_height), dtype=np.float32)
        
        # ---- Vision Pro streaming ----
        self.vp_ip = os.environ.get("VP_IP", "192.168.0.57")
        self.vp_stream_mode = os.environ.get("VP_STREAM_MODE", "pcd")  # 'pcd' or 'rgb'

        if self.vp_stream_mode == 'rgb':
            self.vp_port = int(os.environ.get("VP_RGB_PORT", "6007"))
            self.rgb_sender = UDPRgbSender(self.vp_ip, self.vp_port, jpeg_quality=int(os.environ.get("VP_RGB_QUALITY", "30")))
            self.pcd_sender = None
        else:
            self.vp_port = int(os.environ.get("VP_PCD_PORT", "6006"))
            self.pcd_sender = UDPPcdSender(self.vp_ip, self.vp_port, mtu=1200)
            self.rgb_sender = None

        self.pcd_frame_id = 0
        self.max_pcd_points = int(os.environ.get("VP_MAX_PTS", "300000"))  # matches app cap by default
        self.send_pcd_every_n = int(os.environ.get("VP_EVERY_N", "3"))     # 1 = every frame, 2 = every other, etc.

        # State variables and objects
        self.running = False
        self.shutdown_flag = ShutdownFlag()
        register_signals(self.shutdown_flag)

        self.connect_to_device(dev_idx=0)
        self.desired_interval = 1 / 30.0

        self.status_interval = 1.0 # seconds
        self.last_sent_status = time.monotonic()
        self.activation_check_interval = 0.5  # seconds
        self.last_activation_check = 0.0
        self.pcd_points = np.empty((0, 3), dtype=np.float32)
        self.pcd_colors = np.empty((0, 3), dtype=np.float32)
        self.init_camera_pose = None
        

    def run(self):
        # try:
        # Ensure run loop executes
        self.running = True
        while self.running:
            now = time.monotonic()
            if now - self.last_activation_check >= self.activation_check_interval:
                self._check_activation_messages()
                self.last_activation_check = now
            if self.shutdown_flag.is_requested():
                break

            # Send readiness heartbeat early, independent of frame pipeline
            self.event.wait()
            self.event.clear()

            loop_start = time.monotonic()

            depth_frame = self.session.get_depth_frame()
            rgb_frame   = self.session.get_rgb_frame()
            
            # Get intrinsics as float32
            intrinsics = self.get_intrinsic_mat_from_coeffs(
                self.session.get_intrinsic_mat()
            ).astype(np.float32)

            depth_frame = np.asarray(depth_frame, dtype=np.float32)
            rgb_frame   = np.asarray(rgb_frame)
            self.debug_printer.log(lambda: f"[BEFORE shape]: {rgb_frame.shape}")

            self.debug_printer.log("depth shape", depth_frame.shape, "rgb shape", rgb_frame.shape)
            self.debug_printer.log("K cx/cy", intrinsics[0,2], intrinsics[1,2])
            extrinsics = quaternion_camera_pose_to_extrinsic_matrix(self.session.get_camera_pose())

            # Publish landscape-oriented frames so rollout consumers and logs share orientation.
            # Keep raw `depth_frame` / `rgb_frame` for geometry computations.
            depth_publish = cv2.rotate(depth_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            rgb_publish = cv2.rotate(rgb_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            self.frame_counter += 1

            # DEBUG: downsample?
            # rgb_small = cv2.resize(rgb_publish, (rgb_publish.shape[1]//2, rgb_publish.shape[0]//2))
            self.debug_printer.log(lambda: f"[Data type]: {type(rgb_publish[0])}")
            capture_ns = time.time_ns()
            rgb_msg = serialize_timestamped_bytes(rgb_publish.tobytes(), timestamp_ns=capture_ns)
            depth_msg = serialize_timestamped_bytes(depth_publish.tobytes(), timestamp_ns=capture_ns)

            if self.obs_hash_debug and (self.frame_counter % self.obs_hash_debug_every == 0):
                rgb_sha1 = hashlib.sha1(rgb_publish.tobytes()).hexdigest()[:16]
                depth_sha1 = hashlib.sha1(depth_publish.tobytes()).hexdigest()[:16]
                print(
                    "[obs_hash_send] "
                    f"frame={self.frame_counter} "
                    f"rgb_len={len(rgb_publish.tobytes())} rgb_sha1={rgb_sha1} rgb_shape={tuple(rgb_publish.shape)} "
                    f"depth_len={len(depth_publish.tobytes())} depth_sha1={depth_sha1} depth_shape={tuple(depth_publish.shape)}"
                )

            self._bridge.publish_camera_raw(Topics.IPHONE_RGB, rgb_msg)
            self._bridge.publish_camera_raw(Topics.IPHONE_DEPTH, depth_msg)
            
            # Offload logging to background thread to avoid blocking visualization
            try:
                self.log_queue.put_nowait(
                    (
                        time.monotonic(),
                        rgb_frame.copy(),
                        depth_frame.copy(),
                    )
                )
            except queue.Full:
                # Drop the frame if the logger is backed up to keep capture loop real-time
                pass

            if time.monotonic() - self.last_sent_status >= self.status_interval:
                self._bridge.publish_activation_raw(
                    Topics.NECK_IPHONE_STATUS,
                    serialize_message(
                        self.msg_factory.create_neck_iphone_status_message(
                            iphone_publishing=True
                        )
                    ),
                )
                self.last_sent_status = time.monotonic()

                if self.init_camera_pose is None:
                    self.debug_printer.log("None")
                    self.init_camera_pose = extrinsics

            # ---- Send to Vision Pro via UDP ----
            if (self.pcd_frame_id % self.send_pcd_every_n) == 0:
                if self.vp_stream_mode == 'rgb':
                    self.rgb_sender.send_rgb(rgb_publish, self.pcd_frame_id)
                else:
                    pts, cols = self.get_global_xyz(
                        depth_frame,
                        rgb_frame,
                        intrinsics,
                        extrinsics,
                        depth_scale=1000.0,
                        only_confident=False,
                    )
                    if pts.shape[0] > 0:
                        # Cap points to avoid overwhelming network / VP
                        if pts.shape[0] > self.max_pcd_points:
                            idx = np.random.choice(pts.shape[0], self.max_pcd_points, replace=False)
                            pts = pts[idx]
                            cols = cols[idx]

                        rgb_u8 = np.clip(cols * 255.0, 0, 255).astype(np.uint8)
                        rgba_u8 = np.empty((rgb_u8.shape[0], 4), dtype=np.uint8)
                        rgba_u8[:, :3] = rgb_u8
                        rgba_u8[:, 3] = 255
                        self.pcd_sender.send_pointcloud(pts, rgba_u8, self.pcd_frame_id, extrinsics)

            self.pcd_frame_id += 1
                                
            # Rate limiting
            elapsed = time.monotonic() - loop_start
            sleep_time = max(0, self.desired_interval - elapsed)
            time.sleep(sleep_time)

        # Ensure logging thread is signaled to exit
        self.log_queue.put(None)
        self.logging_thread.join(timeout=2)

    def cleanup(self):
        """Cleanup resources"""
        print("Cleaning up iPhone Publisher...")
        self.running = False

    def _logging_worker(self):
        """Background logger to keep capture loop responsive."""
        while True:
            item = self.log_queue.get()
            try:
                if item is None:
                    return

                timestamp, rgb_frame, depth_frame = item

                if self.logger is not None and self.logger.update_recording_state():
                    # Rotate to landscape (90° CW) so logged videos are oriented correctly.
                    visual_rgb = cv2.rotate(rgb_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    visual_depth = cv2.rotate(depth_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                    self.logger.log_frame(
                        camera_name="iphone_rgb",
                        timestamp=timestamp,
                        frame=visual_rgb
                    )

                    self.logger.log_frame(
                        camera_name="iphone_depth",
                        timestamp=timestamp,
                        frame=visual_depth
                    )
            except Exception as exc:
                print(f"[iPhone] Logging worker error: {exc}")
            finally:
                self.log_queue.task_done()
        
    def _check_activation_messages(self):
        """Listen for shutdown/emergency stop activation commands."""
        data_list = self._bridge.peek_activation_topic(Topics.NECK_ACTIVATION)
        if not data_list:
            return

        payload = deserialize(data_list[-1])
        if not payload:
            return

        activation_msg = ActivationMessage.from_dict(payload)
        target = (activation_msg.target or "").lower()
        command = (activation_msg.command or "").lower()

        if command in ("shutdown", "emergency_stop", "manager_shutdown"):
            if target in ("all", "neck", "iphone"):
                print(f"[iPhone] Received activation command '{command}' for target '{target}' - shutting down.")
                self.shutdown_flag.request()

    

    def get_global_xyz(self, depth, rgb, intrinsics, extrinsic, depth_scale=1000.0, only_confident=False):
        del only_confident  # Confidence map is not available in this stream.
        del extrinsic  # Keep PCD in camera optical frame (no world-frame transform).

        # Resize RGB to match depth dimensions so each depth sample has one color sample.
        depth_height, depth_width = depth.shape
        rgb_height, rgb_width = rgb.shape[:2]
        rgb_resized = cv2.resize(rgb, (depth_width, depth_height), interpolation=cv2.INTER_LINEAR)

        # Intrinsics may already be in depth coordinates (common after normalization)
        # or in RGB coordinates. Detect and convert only when needed.
        fx = float(intrinsics[0, 0])
        fy = float(intrinsics[1, 1])
        cx = float(intrinsics[0, 2])
        cy = float(intrinsics[1, 2])

        intrinsics_in_depth_coords = (
            0.0 <= cx < float(depth_width)
            and 0.0 <= cy < float(depth_height)
        )
        if not intrinsics_in_depth_coords:
            scale_x = depth_width / float(rgb_width)
            scale_y = depth_height / float(rgb_height)
            fx *= scale_x
            fy *= scale_y
            cx *= scale_x
            cy *= scale_y

        # Depth from Record3D is expected in meters. Keep compatibility with callers that
        # provide a non-default depth_scale by applying a safe scalar conversion.
        z = np.ascontiguousarray(depth).astype(np.float32)
        scale_to_m = float(depth_scale) / 1000.0
        if np.isfinite(scale_to_m) and scale_to_m > 0.0 and scale_to_m != 1.0:
            with np.errstate(over="ignore", invalid="ignore"):
                z = z * scale_to_m

        # Reject invalid/saturated values to keep projection stable.
        valid = np.isfinite(z) & (z > 0.0) & (z < 20.0)
        if not np.any(valid):
            self.pcd_points = np.empty((0, 3), dtype=np.float32)
            self.pcd_colors = np.empty((0, 3), dtype=np.float32)
            return self.pcd_points, self.pcd_colors

        v, u = self.depth_indices
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # Camera optical frame:
        # +x right, +y down, +z forward (aligned with image coordinates/intrinsics).
        points = np.stack((x, y, z), axis=-1)[valid].astype(np.float32)
        colors = (rgb_resized.reshape(-1, 3)[valid.reshape(-1)].astype(np.float32) / 255.0)
        self.pcd_points = points
        self.pcd_colors = colors
        return self.pcd_points, self.pcd_colors

    def on_stream_stopped(self):
        print('Stream stopped')
        
    def on_new_frame(self):
        self.event.set()  # Notify the main thread to stop waiting and process new frame.

    def connect_to_device(self, dev_idx):
        print('Searching for devices')
        devs = Record3DStream.get_connected_devices()
        print('{} device(s) found'.format(len(devs)))
        for dev in devs:
            print('\tID: {}\n\tUDID: {}\n'.format(dev.product_id, dev.udid))

        if len(devs) <= dev_idx:
            raise RuntimeError('Cannot connect to device #{}, try different index.'.format(dev_idx))

        dev = devs[dev_idx]
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(dev)  # Initiate connection and start capturing

    def get_intrinsic_mat_from_coeffs(self, coeffs):
        return np.array([[coeffs.fx,         0, coeffs.tx],
                         [        0, coeffs.fy, coeffs.ty],
                         [        0,         0,         1]])
    
