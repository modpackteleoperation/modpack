"""
Minimal sensor-stream bridge integration.

sensor_stream roles are read-only (get_command always returns None).
Your hardware loop reads the sensor and calls publish_state at its own rate.
"""
import time
import numpy as np
from modpack.bridge import RobotBridge


# Replace with your actual F/T sensor SDK
class MyFTSensor:
    def read(self) -> np.ndarray: return np.zeros(6)


bridge = RobotBridge.from_config("modpack/robots/my_robot/config.yaml")
sensor = MyFTSensor()

bridge.connect()
print("Publishing sensor stream...")

try:
    while True:
        bridge.publish_state("right_ft", {
            "data":       sensor.read(),    # shape (6,) — [Fx, Fy, Fz, Tx, Ty, Tz]
            "timestamp":  time.monotonic(),
            "data_valid": True,
        })
        time.sleep(1 / 100)                # 100 Hz
except KeyboardInterrupt:
    pass
finally:
    bridge.disconnect()
