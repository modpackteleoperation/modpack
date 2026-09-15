"""Camera streaming utilities backed by the Aravis GigE-Vision SDK.

The RBY1 policy stack relies on five FLIR IP cameras (three mounted on the
head, two on the wrists).  This module provides a small wrapper around the
`Aravis <https://github.com/AravisProject/aravis>`_ bindings so that the latest
RGB frames can be retrieved and marshalled into numpy arrays suitable for the
policy server.

The implementation is designed to be robust in simulation environments where
the Aravis bindings might not be available.  When mock_mode=True the
streamer simply produces placeholder images allowing the rest of the pipeline
to be exercised without hardware.
"""

from __future__ import annotations

import os
import site
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence

import numpy as np
import yaml


def _prepare_gi_import() -> None:
    """Expose system GI bindings/typelibs to compatible virtual environments."""

    dist_packages = Path("/usr/lib/python3/dist-packages")
    abi_tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
    gi_extension_matches = list(dist_packages.glob(f"gi/_gi.{abi_tag}-*.so"))
    if gi_extension_matches:
        site.addsitedir(str(dist_packages))

    typelib_dir = Path("/usr/local/lib/x86_64-linux-gnu/girepository-1.0")
    if typelib_dir.is_dir():
        existing = [p for p in os.environ.get("GI_TYPELIB_PATH", "").split(os.pathsep) if p]
        if str(typelib_dir) not in existing:
            os.environ["GI_TYPELIB_PATH"] = os.pathsep.join([str(typelib_dir), *existing])


_prepare_gi_import()

try:  # pragma: no cover - optional dependency
    import gi

    gi.require_version("Aravis", "0.8")
    from gi.repository import Aravis
    _ARAVIS_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - used in headless testing
    Aravis = None  # type: ignore
    _ARAVIS_IMPORT_ERROR = exc

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)

@dataclass(frozen=True)
class CameraFrame:
    """Container that stores a single RGB frame."""

    timestamp: float
    image: np.ndarray


class AravisCameraStreamer:
    """Stream and buffer frames from multiple GigE cameras.

    Parameters
    ----------
    camera_map:
        Mapping from observation key (e.g. "camera_left_main_rgb") to camera
        serial number.  The observation key is later used when composing the
        policy input dictionary.
    buffer_size:
        Maximum number of frames to retain per camera.
    mock_mode:
        When True the streamer generates placeholder images instead of
        connecting to real hardware.  This is useful when unit testing the
        policy pipeline without the cameras present.
    mock_resolution:
        Resolution of the synthetic frames used in mock_mode.
    """

    def __init__(
        self,
        config_path: str = PROJECT_ROOT + "/config/camera.yaml",
        camera_keys: Optional[Sequence[str]] = None,
        binning_override: Optional[Sequence[int]] = None,
        expand_to_max_resolution: bool = False,
    ) -> None:
        # Load Camera Config
        try:
            config_path = Path(config_path)
            with config_path.open("r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}
        except Exception as e:
            raise Exception(f"Exception while loading config file: {e}")
        if not isinstance(self.config, dict):
            raise ValueError(f"WBC config at {config_path} must be a mapping.")

        def require(name: str):
            if name not in self.config:
                raise KeyError(f"Missing required camera config key: {name}")
            return self.config[name]

        camera_map = self.config.get("camera_map")
        if not camera_map:
            raise KeyError("Missing required camera config key: camera_map")

        mock_resolution = self.config.get("mock_resolution", (720, 1280))
        if len(mock_resolution) != 2:
            raise ValueError("mock_resolution must be a pair of (height, width)")
        mock_resolution = (int(mock_resolution[0]), int(mock_resolution[1]))

        full_camera_map = dict(camera_map)
        if camera_keys is None:
            self._camera_map = full_camera_map
        else:
            selected_keys = [str(key) for key in camera_keys]
            if not selected_keys:
                raise ValueError("camera_keys must contain at least one camera key")
            missing_keys = [key for key in selected_keys if key not in full_camera_map]
            if missing_keys:
                raise KeyError(
                    f"Requested camera key(s) not found in camera_map: {missing_keys}. "
                    f"Available keys: {list(full_camera_map)}"
                )
            self._camera_map = {key: full_camera_map[key] for key in selected_keys}
        self._buffer_size = int(max(self.config.get("buffer_size", 16), 1))
        self._mock_mode = bool(self.config.get("mock_mode", False))
        self._mock_resolution = mock_resolution

        self._frame_rate = float(self.config.get("frame_rate", 30.0))
        binning = binning_override if binning_override is not None else self.config.get("binning", (4, 4))
        try:
            if binning is None or len(binning) != 2:
                raise ValueError
            self._binning = (int(binning[0]), int(binning[1]))
        except Exception:
            raise ValueError("binning must be a pair of integers") from None
        self._expand_to_max_resolution = bool(expand_to_max_resolution)
        self._pixel_format_config = self.config.get("pixel_format", "PIXEL_FORMAT_BGR_8_PACKED")

        self._buffers: Dict[str, Deque[CameraFrame]] = {
            key: deque(maxlen=self._buffer_size) for key in self._camera_map
        }
        self._buffers_lock = threading.Lock()

        self._threads: List[threading.Thread] = []
        self._stop = threading.Event()

        self._streams: Dict[str, tuple] = {}

        if Aravis is None and not self._mock_mode:  # pragma: no cover - env guard
            raise ImportError(
                "gi.repository.Aravis is required for camera streaming. "
                "Install the Aravis Python bindings or enable mock_mode. "
                f"Original import error: {_ARAVIS_IMPORT_ERROR!r}"
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._threads:
            return
        self._stop.clear()
        if self._mock_mode:
            for key in self._camera_map:
                thread = threading.Thread(
                    target=self._mock_worker,
                    args=(key,),
                    name=f"camera-mock-{key}",
                    daemon=True,
                )
                thread.start()
                self._threads.append(thread)
            return

        Aravis.update_device_list()
        pixel_format = self._resolve_pixel_format(self._pixel_format_config)
        for obs_key, serial in self._camera_map.items():
            camera = Aravis.Camera.new(serial)
            if camera is None:  # pragma: no cover - depends on hardware
                raise RuntimeError(f"Failed to create Aravis camera for serial {serial}")
            camera.set_binning(*self._binning)
            if self._expand_to_max_resolution:
                self._set_camera_to_max_resolution(camera)
            camera.set_frame_rate(self._frame_rate)
            camera.set_pixel_format(pixel_format)
            stream = camera.create_stream(None, None)
            payload = camera.get_payload()
            for _ in range(4):
                stream.push_buffer(Aravis.Buffer.new_allocate(payload))
            camera.start_acquisition()
            self._streams[obs_key] = (camera, stream)
            thread = threading.Thread(
                target=self._camera_worker,
                args=(obs_key,),
                name=f"camera-{serial}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._threads.clear()

        if not self._mock_mode:
            for camera, stream in self._streams.values():  # pragma: no cover - hardware cleanup
                try:
                    camera.stop_acquisition()
                finally:
                    stop_fn = getattr(stream, "stop_acquisition", None)
                    if stop_fn is not None:
                        stop_fn()
                    else:  # pragma: no cover - depends on Aravis version
                        stop_fn = getattr(stream, "stop", None)
                        if stop_fn is not None:
                            stop_fn()
            self._streams.clear()

        with self._buffers_lock:
            for buffer in self._buffers.values():
                buffer.clear()

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------
    def _mock_worker(self, obs_key: str) -> None:
        height, width = self._mock_resolution
        colour = np.random.randint(0, 255, size=(1, 1, 3), dtype=np.uint8)
        while not self._stop.is_set():
            image = np.tile(colour, (height, width, 1))
            frame = CameraFrame(timestamp=time.monotonic(), image=image)
            with self._buffers_lock:
                self._buffers[obs_key].append(frame)
            time.sleep(1 / 30.0)

    def _camera_worker(self, obs_key: str) -> None:  # pragma: no cover - hardware
        camera, stream = self._streams[obs_key]
        default_width = camera.get_integer("Width")
        default_height = camera.get_integer("Height")
        if hasattr(camera, "get_pixel_format_string"):
            default_pixel_format = camera.get_pixel_format_string()  # type: ignore[attr-defined]
        else:  # pragma: no cover - older/newer bindings
            default_pixel_format = camera.get_pixel_format_as_string()

        timeout_ns = 1_000_000  # 1 ms
        if hasattr(stream, "timeout"):  # pragma: no cover - depends on bindings
            try:
                setattr(stream, "timeout", max(int(timeout_ns // 1000), 1))
            except Exception:
                pass
        supports_timeout = True
        incomplete_count = 0
        first_frame_logged = False

        while not self._stop.is_set():
            if supports_timeout:
                try:
                    buffer = stream.pop_buffer(timeout_ns)
                except TypeError:
                    supports_timeout = False
                    continue
            if not supports_timeout:
                buffer = stream.pop_buffer()
            if buffer is None:
                continue
            try:
                if self._buffer_is_incomplete(buffer):
                    incomplete_count += 1
                    if incomplete_count == 1 or incomplete_count % 30 == 0:
                        print(
                            f"[camera:{obs_key}] Incomplete frame dropped "
                            f"(count={incomplete_count}, status={self._buffer_status_name(buffer)})",
                            file=sys.stderr,
                        )
                    continue

                data = buffer.get_data()
                width, height, pixel_format = self._buffer_frame_info(
                    buffer,
                    default_width,
                    default_height,
                    default_pixel_format,
                )
                expected_bytes = self._expected_payload_size(width, height, pixel_format)
                payload = memoryview(data)
                if len(payload) >= expected_bytes:
                    payload = payload[:expected_bytes]
                try:
                    np_image = self._decode_buffer(
                        payload,
                        width,
                        height,
                        pixel_format,
                    )
                except ValueError as exc:
                    print(
                        f"[camera:{obs_key}] {exc} "
                        f"(status={self._buffer_status_name(buffer)}, width={width}, height={height}, format={pixel_format})",
                        file=sys.stderr,
                    )
                    continue
                if not first_frame_logged:
                    first_frame_logged = True
                    print(
                        f"[camera:{obs_key}] First decoded frame: "
                        f"width={width} height={height} format={pixel_format} "
                        f"payload_bytes={len(data)} image_shape={tuple(np_image.shape)}",
                        file=sys.stderr,
                    )
                # Use host monotonic clock for timestamps so all downstream
                # consumers share a common time base (robot, cameras, policy).
                frame = CameraFrame(timestamp=time.monotonic(), image=np_image)
                with self._buffers_lock:
                    self._buffers[obs_key].append(frame)
            finally:
                stream.push_buffer(buffer)

    @staticmethod
    def _resolve_pixel_format(pixel_format_config: Any) -> Any:
        if isinstance(pixel_format_config, int):
            return pixel_format_config

        name = str(pixel_format_config).strip()
        attr_name = name if name.startswith("PIXEL_FORMAT_") else f"PIXEL_FORMAT_{name}"
        attr_name = attr_name.upper()
        if not hasattr(Aravis, attr_name):
            raise KeyError(f"Unknown Aravis pixel format constant '{name}'")
        return getattr(Aravis, attr_name)

    @staticmethod
    def _buffer_is_incomplete(buffer: Any) -> bool:
        status = AravisCameraStreamer._buffer_status_name(buffer)
        return status not in {"SUCCESS", "UNKNOWN", "N_A", "NA"}

    @staticmethod
    def _buffer_status_name(buffer: Any) -> str:
        status = None
        for attr in ("get_status", "status"):
            value = getattr(buffer, attr, None)
            if value is None:
                continue
            try:
                status = value() if callable(value) else value
                break
            except Exception:
                continue

        if status is None:
            return "UNKNOWN"

        name = getattr(status, "value_nick", None) or getattr(status, "name", None)
        if name:
            return str(name).upper()
        return str(status).split(".")[-1].upper()

    @staticmethod
    def _buffer_frame_info(
        buffer: Any,
        default_width: int,
        default_height: int,
        default_pixel_format: str,
    ) -> tuple[int, int, str]:
        width = default_width
        height = default_height
        pixel_format = default_pixel_format

        for getter_name, fallback in (
            ("get_image_width", default_width),
            ("get_width", default_width),
        ):
            getter = getattr(buffer, getter_name, None)
            if getter is not None:
                try:
                    width = int(getter())
                    break
                except Exception:
                    width = fallback

        for getter_name, fallback in (
            ("get_image_height", default_height),
            ("get_height", default_height),
        ):
            getter = getattr(buffer, getter_name, None)
            if getter is not None:
                try:
                    height = int(getter())
                    break
                except Exception:
                    height = fallback

        format_value = None
        for getter_name in (
            "get_image_pixel_format_as_string",
            "get_pixel_format_as_string",
            "get_image_pixel_format_string",
            "get_pixel_format_string",
        ):
            getter = getattr(buffer, getter_name, None)
            if getter is not None:
                try:
                    format_value = getter()
                    break
                except Exception:
                    continue

        if format_value:
            pixel_format = str(format_value)

        return width, height, pixel_format

    @staticmethod
    def _decode_buffer(data: memoryview | bytes, width: int, height: int, pixel_format: str) -> np.ndarray:
        array = np.frombuffer(data, dtype=np.uint8)
        fmt = pixel_format.upper()
        expected = AravisCameraStreamer._expected_payload_size(width, height, fmt)
        if array.size < expected:
            raise ValueError(
                f"Payload too small for format '{pixel_format}' ({array.size} bytes, expected {expected})"
            )
        if array.size > expected:
            array = array[:expected]

        if fmt.endswith("MONO8") or fmt == "MONO8":
            array = array.reshape(height, width)
            return np.repeat(array[..., None], 3, axis=-1)

        if "BAYER" in fmt:
            return AravisCameraStreamer._decode_bayer(array, width, height, fmt)

        image = array.reshape(height, width, 3)
        return image

    @staticmethod
    def _expected_payload_size(width: int, height: int, pixel_format: str) -> int:
        fmt = pixel_format.upper()
        if fmt.endswith("MONO8") or fmt == "MONO8":
            return width * height
        if "BAYER" in fmt:
            return width * height
        return width * height * 3

    @staticmethod
    def _set_camera_to_max_resolution(camera: Any) -> None:
        """Expand Width/Height to the maximum allowed by the current binning."""

        for name in ("OffsetX", "OffsetY"):
            try:
                camera.set_integer(name, 0)
            except Exception:
                pass

        for dim, dim_max in (("Width", "WidthMax"), ("Height", "HeightMax")):
            try:
                camera.set_integer(dim, int(camera.get_integer(dim_max)))
            except Exception:
                pass

    @staticmethod
    def _decode_bayer(data: np.ndarray, width: int, height: int, fmt: str) -> np.ndarray:
        """Decode Bayer-pattern images using OpenCV when available."""

        try:  # pragma: no cover - optional dependency
            import cv2
        except ImportError as exc:
            raise ValueError(
                f"Pixel format '{fmt}' requires OpenCV for debayering. Install opencv-python."
            ) from exc

        bayer = data.reshape(height, width)
        conversion_map = {
            "RG8": cv2.COLOR_BayerRG2BGR,
            "RG16": cv2.COLOR_BayerRG2BGR,
            "BG8": cv2.COLOR_BayerBG2BGR,
            "BG16": cv2.COLOR_BayerBG2BGR,
            "GB8": cv2.COLOR_BayerGB2BGR,
            "GB16": cv2.COLOR_BayerGB2BGR,
            "GR8": cv2.COLOR_BayerGR2BGR,
            "GR16": cv2.COLOR_BayerGR2BGR,
        }

        normalized = fmt.upper()
        if not normalized.startswith("BAYER"):
            raise ValueError(f"Unsupported Bayer pixel format '{fmt}'")
        suffix = normalized[len("BAYER") :]
        suffix = suffix.replace("_", "").replace("-", "")
        conversion = conversion_map.get(suffix)
        if conversion is None:
            raise ValueError(f"Unsupported Bayer pixel format '{fmt}'")
        bgr = cv2.cvtColor(bayer, conversion)
        rgb = bgr[..., ::-1]
        return rgb

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def wait_until_ready(self, min_frames: int = 1, timeout: float = 2.0) -> None:
        """Block until each camera buffer contains min_frames entries."""

        min_frames = int(max(min_frames, 1))
        deadline = time.monotonic() + max(timeout, 0.0)
        while time.monotonic() <= deadline:
            with self._buffers_lock:
                if all(len(buffer) >= min_frames for buffer in self._buffers.values()):
                    return
            time.sleep(0.01)
        raise TimeoutError("Timed out waiting for camera frames")

    def get_observation_window(
        self,
        horizon: int,
        stride: int = 1,
        obs_frequency: Optional[float] = None,
        *,
        include_timestamps: bool = False,
    ) -> Dict[str, np.ndarray] | tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Return stacked image observations.

        The returned dictionary maps observation keys (as passed in the
        constructor) to arrays with shape (T, H, W, 3) where T equals the
        requested horizon.  If insufficient frames are available the oldest frame
        is repeated.
        """

        horizon = int(max(horizon, 1))
        stride = int(max(stride, 1))

        with self._buffers_lock:
            buffers_copy = {k: list(v) for k, v in self._buffers.items()}

        observations: Dict[str, np.ndarray] = {}
        timestamps: Dict[str, np.ndarray] = {}
        for key, frames in buffers_copy.items():
            if not frames:
                raise RuntimeError(f"No frames available for camera '{key}'")
            if obs_frequency is None or len(frames) < 2:
                selected: List[CameraFrame] = frames[-horizon * stride :: stride]
                if len(selected) < horizon:
                    selected = [selected[0]] * (horizon - len(selected)) + selected
            else:
                latest_ts = frames[-1].timestamp
                step = float(stride) / max(obs_frequency, 1e-6)
                desired_ts = latest_ts - step * np.arange(horizon - 1, -1, -1, dtype=float)
                frame_ts = np.array([frame.timestamp for frame in frames], dtype=float)
                idx = np.abs(frame_ts[:, None] - desired_ts[None, :]).argmin(axis=0)
                selected = [frames[i] for i in idx]
            observations[key] = np.stack([frame.image for frame in selected], axis=0)
            timestamps[key] = np.array([frame.timestamp for frame in selected], dtype=float)

        if include_timestamps:
            return observations, timestamps
        return observations


__all__ = ["AravisCameraStreamer", "CameraFrame"]
