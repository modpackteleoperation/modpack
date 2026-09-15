"""
Minimal joint-arm bridge integration.

Run on the Robot PC once the GELLO PC is running:
    python examples/minimal_joint_arm.py

The bridge connects to the GELLO PC, waits for activation,
then loops: read command → move robot → publish state.
"""
import time
import numpy as np
from modpack.bridge import RobotBridge


# Replace with your actual SDK
class MyRobotSDK:
    def move(self, joint_pos: np.ndarray, gripper: float) -> None: ...
    def get_joints(self) -> np.ndarray: return np.zeros(7)
    def get_gripper(self) -> float: return 0.0


bridge = RobotBridge.from_config("modpack/robots/my_robot/config.yaml")
robot  = MyRobotSDK()

bridge.connect()
print("Waiting for activation...")

try:
    while True:
        if not bridge.is_active:
            time.sleep(0.01)
            continue

        cmd = bridge.get_command("right_arm")   # JointArmCommand or None
        if cmd is not None:
            robot.move(cmd.joint_pos, cmd.gripper)

        bridge.publish_state("right_arm", {
            "positions":  robot.get_joints(),
            "gripper":    robot.get_gripper(),
            "timestamp":  time.monotonic(),
            "data_valid": True,
        })
except KeyboardInterrupt:
    pass
finally:
    bridge.disconnect()
