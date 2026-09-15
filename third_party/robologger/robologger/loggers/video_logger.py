import os
import subprocess
from typing import Any, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
import zarr
from loguru import logger

from robologger.loggers.base_logger import BaseLogger
from robologger.utils.huecodec import EncoderOpts, depth2logrgb


class VideoLogger(BaseLogger):
    """Logger for video data from multi-camera subsystems.

    Each VideoLogger handles multiple camera streams with custom names stored in zarr file attributes.
    Multiple VideoLoggers can be instantiated for each camera subsystem. For example, if you have
    3 iPhones mounted on the right wrist, each with 3 camera types (main_rgb, ultrawide_rgb, depth, etc.),
    you would create 3 VideoLoggers with names like: right_wrist_camera_0, right_wrist_camera_1,
    right_wrist_camera_2 - one VideoLogger for each iPhone on the same wrist.

    Naming Convention:
    - Must use CameraName enum values
    """

    def __init__(
        self,
        name: str,
        endpoint: str,
        attr: Dict[str, Any],
        codec: str = "h264_nvenc",  # XXX: can use av1_nvenc if supported
        depth_range: Tuple[float, float] = (0.0, 4.0),  # TODO: choose a proper range
        defer_encoding: bool = False,
    ):
        super().__init__(name, endpoint, attr)
        self.ffmpeg_processes: Dict[str, subprocess.Popen[bytes]] = {}
        self.spool_chunk_frames = 120
        self.spool_root: Optional[str] = None
        self.zarr_path: Optional[str] = None
        self._spool_states: Dict[str, Dict[str, Any]] = {}

        self._validate_camera_config(attr)
        self.depth_range = depth_range
        self.hue_opts = EncoderOpts(use_lut=True)
        self.codec = codec
        self.defer_encoding = defer_encoding

    def _validate_camera_config(self, attr: Dict[str, Any]) -> None:
        """Validate camera configuration dictionary."""
        if "camera_configs" not in attr:
            raise ValueError("Missing 'camera_configs' in attr")
        if not isinstance(attr["camera_configs"], dict):
            raise ValueError("'camera_configs' must be a dictionary")
        if not attr["camera_configs"]:
            raise ValueError("'camera_configs' cannot be empty")

        required_keys = ["width", "height", "fps", "type"]
        for cam_name, config in attr["camera_configs"].items():
            if not isinstance(config, dict):
                raise ValueError(f"Camera config for '{cam_name}' must be a dictionary")
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Missing required key '{key}' in camera config for '{cam_name}'")
            if config["type"] not in ["rgb", "depth"]:
                raise ValueError(f"Camera type for '{cam_name}' must be 'rgb' or 'depth', got '{config['type']}'")

    def _init_storage(self):
        """Initialize zarr storage. When defer_encoding is set, set up a per-camera raw frame
        spool for deferred encoding; otherwise spawn a live ffmpeg process per camera."""
        episode_dir = self.episode_dir
        if episode_dir is None:
            raise RuntimeError("episode_dir not set. start_recording() must be called first.")

        self.zarr_path = os.path.join(episode_dir, f"{self.name}.zarr")
        self.zarr_group = zarr.open_group(self.zarr_path, mode="w")
        logger.info(f"[{self.name}] Initialized zarr group: {self.zarr_path}")

        for cam_name in self.attr["camera_configs"].keys():
            self.data_lists[f"{cam_name}_timestamps"] = []

        if self.defer_encoding:
            self.spool_root = os.path.join(self.zarr_path, "_spool")
            os.makedirs(self.spool_root, exist_ok=True)
            self._spool_states = {}
            for cam_name, config in self.attr["camera_configs"].items():
                self._spool_states[cam_name] = {
                    "chunk_idx": 0,
                    "frame_count": 0,
                    "chunk_frame_count": 0,
                    "file": None,
                    "chunk_timestamps": [],
                    "frame_bytes": int(np.prod((config["height"], config["width"], 3))),
                }
            return

        try:
            for cam_name, config in self.attr["camera_configs"].items():
                self._spawn_ffmpeg_process(cam_name, config)
        except Exception as e:
            for cam_name, process in self.ffmpeg_processes.items():
                self._close_ffmpeg_process(cam_name, process)
            self.ffmpeg_processes.clear()
            raise RuntimeError(f"Failed to initialize FFmpeg processes: {e}") from e

    def _spawn_ffmpeg_process(self, cam_name: str, config: Dict[str, Any]) -> subprocess.Popen[bytes]:
        if self.zarr_path is None:
            raise RuntimeError(f"[{self.name}] zarr_path not initialized for ffmpeg spawn")

        mp4_file_path = os.path.join(self.zarr_path, f"{cam_name}.mp4")
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{config['width']}x{config['height']}",
            "-r",
            str(config["fps"]),
            "-i",
            "-",
            "-c:v",
            self.codec,
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "fast",
            "-b:v",
            "5M",
            mp4_file_path,
        ]
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )
        self.ffmpeg_processes[cam_name] = process
        logger.info(f"[{self.name}] Initialized ffmpeg process for camera: {cam_name}")
        return process

    def _open_spool_chunk(self, cam_name: str) -> None:
        if self.spool_root is None:
            raise RuntimeError(f"[{self.name}] spool_root not initialized")

        state = self._spool_states[cam_name]
        chunk_path = os.path.join(self.spool_root, f"{cam_name}_chunk_{state['chunk_idx']:06d}.bin")
        state["file"] = open(chunk_path, "wb")
        state["chunk_timestamps"] = []
        state["chunk_frame_count"] = 0

    def _finalize_spool_chunk(self, cam_name: str) -> None:
        state = self._spool_states[cam_name]
        chunk_file = state["file"]
        if chunk_file is None:
            return

        chunk_path = os.path.join(self.spool_root, f"{cam_name}_chunk_{state['chunk_idx']:06d}.bin")
        chunk_file.flush()
        chunk_file.close()

        if state["chunk_frame_count"] <= 0:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
        else:
            ts_path = os.path.join(self.spool_root, f"{cam_name}_chunk_{state['chunk_idx']:06d}_timestamps.npy")
            np.save(ts_path, np.asarray(state["chunk_timestamps"], dtype=np.float64))
            state["chunk_idx"] += 1

        state["file"] = None
        state["chunk_timestamps"] = []
        state["chunk_frame_count"] = 0

    def _spool_rgb_frame(self, cam_name: str, timestamp: float, frame_rgb: npt.NDArray[Any]) -> None:
        state = self._spool_states[cam_name]
        if state["file"] is None:
            self._open_spool_chunk(cam_name)

        frame_to_write = frame_rgb if frame_rgb.flags.c_contiguous else np.ascontiguousarray(frame_rgb)
        state["file"].write(memoryview(frame_to_write))
        state["chunk_timestamps"].append(timestamp)
        state["frame_count"] += 1
        state["chunk_frame_count"] += 1

        if state["chunk_frame_count"] >= self.spool_chunk_frames:
            self._finalize_spool_chunk(cam_name)

    def _close_ffmpeg_process(self, cam_name: str, process: subprocess.Popen[bytes], timeout: int = 10) -> None:
        """Gracefully close an FFmpeg process."""
        logger.info(f"Stopping FFmpeg process for '{cam_name}'...")
        if process.stdin:
            process.stdin.close()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning(f"FFmpeg process for '{cam_name}' did not terminate gracefully, forcing kill")
            process.kill()
            process.wait()
        logger.info(f"FFmpeg process for '{cam_name}' stopped.")

    def _close_storage(self):
        """Close storage. When defer_encoding is set, finalize and preserve the raw frame spool
        for later conversion; otherwise close the live ffmpeg processes."""
        if self.zarr_group is None:
            logger.error(f"[{self.name}] Zarr group is not initialized but _close_storage() is called in video logger")
            return

        if self.defer_encoding:
            for cam_name in list(self._spool_states.keys()):
                self._finalize_spool_chunk(cam_name)
            logger.info(f"[{self.name}] Deferred ffmpeg encoding; preserved raw frame spool at {self.spool_root}")
            self.spool_root = None
            self._spool_states = {}
        else:
            for cam_name, process in self.ffmpeg_processes.items():
                self._close_ffmpeg_process(cam_name, process)

        self.ffmpeg_processes.clear()
        self.zarr_group = None
        self.zarr_path = None

    def log_frame(
        self,
        *,
        camera_name: str,
        timestamp: float,
        frame: npt.NDArray[Any],
    ):
        """Log single video frame with timestamp."""
        if not self._is_recording:
            logger.warning(f"[{self.name}] Not recording, but received frame command")
            return

        if self.zarr_group is None:
            raise ValueError("Storage not initialized. Please call start_episode() before logging frames to make sure the zarr group is initialized.")

        if camera_name not in self.attr["camera_configs"]:
            raise ValueError(f"Camera '{camera_name}' not found in camera config")

        config: dict = self.attr["camera_configs"][camera_name]

        if config["type"] == "rgb":
            expected_shape = (config["height"], config["width"], 3)

            assert frame.dtype == np.uint8, f"RGB frame must be uint8, got {frame.dtype}"
            assert len(frame.shape) == 3, f"RGB frame must be HWC (3D), got shape {frame.shape}"
            assert frame.shape[2] == 3, f"RGB frame must have 3 channels, got {frame.shape[2]}"

            if frame.shape != expected_shape:
                raise ValueError(
                    f"RGB frame shape mismatch for camera '{camera_name}'. Expected {expected_shape}, got {frame.shape}"
                )

            frame_rgb = frame

        elif config["type"] == "depth":
            expected_shape = (config["height"], config["width"])

            assert frame.dtype in [np.float16, np.float32, np.float64], (
                f"Depth frame must be float16 or float32 or float64, got {frame.dtype}"
            )
            assert len(frame.shape) == 2, f"Depth frame must be HW (2D), got shape {frame.shape}"

            if frame.shape != expected_shape:
                raise ValueError(
                    f"Depth frame shape mismatch for camera '{camera_name}'. Expected {expected_shape}, got {frame.shape}"
                )

            frame_rgb = depth2logrgb(frame, self.depth_range, opts=self.hue_opts)

        else:
            raise ValueError(f"Unknown camera type: {config['type']}")

        self.data_lists[f"{camera_name}_timestamps"].append(timestamp)

        if self.defer_encoding:
            self._spool_rgb_frame(camera_name, timestamp, frame_rgb)
            return

        if camera_name in self.ffmpeg_processes:
            process = self.ffmpeg_processes[camera_name]
            if process.stdin:
                try:
                    process.stdin.write(frame_rgb.tobytes()) #frame_bgr
                    process.stdin.flush()
                except BrokenPipeError:
                    self._close_ffmpeg_process(camera_name, process, timeout=5)
                    del self.ffmpeg_processes[camera_name] # remove from active processes dict
                    raise RuntimeError(f"[{self.name}] FFmpeg process for '{camera_name}' closed unexpectedly")
        else:
            logger.warning(f"[{self.name}] FFmpeg process for '{camera_name}' not available")
            raise RuntimeError(f"[{self.name}] FFmpeg process for '{camera_name}' not available")

    def log_frames(self, frame_dict: Dict[str, Dict[str, Any]]):
        """Log frames for multiple cameras with individual timestamps.

        Expected frame_dict format:
        {
            "camera_name": {
                "frame": Union[npt.NDArray[np.uint8], npt.NDArray[np.float32]],
                "timestamp": float,
            }
        }
        """
        for cam_name in frame_dict.keys():
            if cam_name not in self.attr["camera_configs"]:
                raise ValueError(f"Camera '{cam_name}' not found in camera config")

        for cam_name, frame_data in frame_dict.items():
            self.log_frame(
                camera_name=cam_name,
                timestamp=frame_data["timestamp"],
                frame=frame_data["frame"],
            )
