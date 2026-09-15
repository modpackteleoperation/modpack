#!/usr/bin/env python3

import cv2
import time
import queue
from threading import Thread
from modpack.orchestration.message_formats import Topics, serialize_timestamped_bytes
from modpack.orchestration.process_shutdown import (
    ShutdownFlag,
    register_signals,
)
from modpack.utils import DebugPrinter

# Define the paths to your cameras
right_paths = {
    # 'right': "/dev/video5"
    'right': "/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_766C49AE-video-index0",
}

left_paths = {
    # 'left': "/dev/video7"
    'left': "/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_608EF4FE-video-index0",
}

CAMERA_FOCUS = 0
CAMERA_TEMPERATURE = 3900
CAMERA_EXPOSURE = 156
CAMERA_GAIN = 10

# TODO: implement self.running as well as correct return values for self.run()
class CameraPublisher():
    def __init__(
        self,
        arms: list,
        debug: bool,
        bridge=None,
    ):
        self.desired_interval = 1 / 30.0
        self.debug = debug
        self.debug_printer = DebugPrinter(enabled=debug)
        self._bridge = bridge
        self.shutdown_flag = ShutdownFlag()

        register_signals(self.shutdown_flag)
        self.caps = {}
        if 'right' in arms:
            for key, path in right_paths.items():
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    print(f"Error: Could not open camera {key}")
                else:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Gives much better latency

                    cap.set(cv2.CAP_PROP_FPS, 30)
                
                    # Disable all auto
                    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                    cap.set(cv2.CAP_PROP_AUTO_WB, 0)  # White balance
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = off, 3 = on


                    # Read several frames to let settings (especially gain/exposure) stabilize
                    for _ in range(30):
                        cap.read()
                        cap.set(cv2.CAP_PROP_FOCUS, CAMERA_FOCUS)  # Fixed focus
                        cap.set(cv2.CAP_PROP_TEMPERATURE, CAMERA_TEMPERATURE)  # Fixed white balance
                        cap.set(cv2.CAP_PROP_EXPOSURE, CAMERA_EXPOSURE)  # Fixed exposure
                        cap.set(cv2.CAP_PROP_GAIN, CAMERA_GAIN)  # Fixed gain


                    # Check all settings match expected
                    assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 1280
                    assert cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == 720
                    assert cap.get(cv2.CAP_PROP_BUFFERSIZE) == 1
                    assert cap.get(cv2.CAP_PROP_AUTOFOCUS) == 0
                    assert cap.get(cv2.CAP_PROP_AUTO_WB) == 0
                    assert cap.get(cv2.CAP_PROP_AUTO_EXPOSURE) == 1
                    assert cap.get(cv2.CAP_PROP_FOCUS) == CAMERA_FOCUS
                    assert cap.get(cv2.CAP_PROP_TEMPERATURE) == CAMERA_TEMPERATURE
                    assert cap.get(cv2.CAP_PROP_EXPOSURE) == CAMERA_EXPOSURE
                    assert cap.get(cv2.CAP_PROP_GAIN) == CAMERA_GAIN

                    self.caps[key] = cap
                    time.sleep(1)


        time.sleep(1)
        if 'left' in arms:
            for key, path in left_paths.items():
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    print(f"Error: Could not open camera {key}")
                else:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Gives much better latency

                    cap.set(cv2.CAP_PROP_FPS, 30)
                
                    # Disable all auto
                    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                    cap.set(cv2.CAP_PROP_AUTO_WB, 0)  # White balance
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = off, 3 = on


                    # Read several frames to let settings (especially gain/exposure) stabilize
                    for _ in range(30):
                        cap.read()
                        cap.set(cv2.CAP_PROP_FOCUS, CAMERA_FOCUS)  # Fixed focus
                        cap.set(cv2.CAP_PROP_TEMPERATURE, CAMERA_TEMPERATURE)  # Fixed white balance
                        cap.set(cv2.CAP_PROP_EXPOSURE, CAMERA_EXPOSURE)  # Fixed exposure
                        cap.set(cv2.CAP_PROP_GAIN, CAMERA_GAIN)  # Fixed gain


                    # Check all settings match expected
                    assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 1280
                    assert cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == 720
                    assert cap.get(cv2.CAP_PROP_BUFFERSIZE) == 1
                    assert cap.get(cv2.CAP_PROP_AUTOFOCUS) == 0
                    assert cap.get(cv2.CAP_PROP_AUTO_WB) == 0
                    assert cap.get(cv2.CAP_PROP_AUTO_EXPOSURE) == 1
                    assert cap.get(cv2.CAP_PROP_FOCUS) == CAMERA_FOCUS
                    assert cap.get(cv2.CAP_PROP_TEMPERATURE) == CAMERA_TEMPERATURE
                    assert cap.get(cv2.CAP_PROP_EXPOSURE) == CAMERA_EXPOSURE
                    assert cap.get(cv2.CAP_PROP_GAIN) == CAMERA_GAIN
                    self.caps[key] = cap
                    time.sleep(1)  # <--- Add sleep for left cam
        time.sleep(1)
        # Create frame queueas for parallel capture
        self.frame_queues = {key: queue.Queue(maxsize=1) for key in self.caps.keys()}
        self.capture_threads = {}
        
        # Start continuous capture threads for each camera
        for key, cap in self.caps.items():
            thread = Thread(target=self._capture_loop, args=(key, cap), daemon=True)
            thread.start()
            self.capture_threads[key] = thread

    def _capture_loop(self, key, cap):
        """Continuous capture in separate thread"""
        while not self.shutdown_flag.is_requested():
            ret, frame = cap.read()
            if ret:
                try:
                    self.frame_queues[key].put_nowait(frame)
                except queue.Full:
                    try:
                        self.frame_queues[key].get_nowait()
                    except queue.Empty:
                        pass
                    self.frame_queues[key].put_nowait(frame)
        cap.release()

    def cleanup(self):
        """Release camera resources and stop threads."""
        for cap in self.caps.values():
            if cap.isOpened():
                cap.release()
        self.caps.clear()

    def run(self):
        """Main loop for publishing camera frames"""
        print("Starting capture threads...")
        print(f"Cameras opened: {list(self.caps.keys())}")
        time.sleep(2)
        print("Camera publisher running...")
        
        try:
            while not self.shutdown_flag.is_requested():
                loop_start = time.time()

                # Get latest frames from all cameras (non-blocking)
                frames = {}
                for key in self.caps.keys():
                    try:
                        frames[key] = self.frame_queues[key].get_nowait()
                    except queue.Empty:
                        continue
                                

                for key, frame in frames.items():
                    # Publish JPEG frames to the camera server so the gello-PC
                    # central logger (BackpackCameraLogger) can poll + record them.
                    # frame is BGR from VideoCapture; the logger decodes JPEG→BGR
                    # then swaps to RGB itself, so no conversion is needed here.
                    if self._bridge is not None:
                        ok, jpg = cv2.imencode(".jpg", frame)
                        if ok:
                            topic = Topics.RIGHT_RGB if key == "right" else Topics.LEFT_RGB
                            self._bridge.publish_camera_raw(
                                topic,
                                serialize_timestamped_bytes(jpg.tobytes(), timestamp_ns=time.time_ns()),
                            )


                # Rate limiting
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.desired_interval - elapsed)
                time.sleep(sleep_time)
                        
        except KeyboardInterrupt:
            print("\nShutting down camera publisher...")
        finally:
            self.cleanup()
