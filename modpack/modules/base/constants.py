POLICY_CONTROL_FREQ = 100
POLICY_CONTROL_PERIOD = 1.0 / POLICY_CONTROL_FREQ

CTRL_FREQ = 10

# Rate at which the Robot PC republishes measured STATE (Topics.BASE) and the
# commanded TARGET (Topics.BASE_TARGET) over RMQ for remote consumers (the gello
# logger, rby1 teleop, vision_pro). This runs on a dedicated publisher thread,
# OFF the command hot path, so the blocking cross-PC put_data round-trips never
# delay policy.step -> execute_action. CTRL_FREQ matches the logger cadence.
PUBLISH_FREQ = CTRL_FREQ
PUBLISH_PERIOD = 1.0 / PUBLISH_FREQ
