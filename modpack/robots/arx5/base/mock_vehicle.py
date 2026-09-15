import numpy as np

class MockVehicle:
    """Mock base controller - interface matches real Vehicle class"""
    
    def __init__(self, max_vel=(0.5, 0.5, 1.57), max_accel=(0.25, 0.25, 0.79)):
        self.max_vel = np.array(max_vel)
        self.max_accel = np.array(max_accel)
        self.x = np.zeros(3)  # [x, y, theta]
        self.dx = np.zeros(3)
        self.control_loop_running = False
        print(f"MockVehicle initialized: max_vel={max_vel}")
        
    def start_control(self):
        self.control_loop_running = True
        print("Mock base control started")
        
    def stop_control(self):
        self.control_loop_running = False
        print("Mock base control stopped")
        
    def set_target_velocity(self, velocity):
        """Interface matches real Vehicle.set_target_velocity()"""
        velocity = np.clip(velocity, -self.max_vel, self.max_vel)
        self.dx = velocity
        self.x += self.dx * 0.02  # 50Hz simulation
        print(f"Base velocity: [{velocity[0]:.2f}, {velocity[1]:.2f}, {velocity[2]:.2f}]")
        
    def set_target_position(self, position):
        """Interface matches real Vehicle.set_target_position()"""
        self.x = np.array(position)
        print(f"Base position: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]")