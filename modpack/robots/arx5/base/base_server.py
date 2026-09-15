# Author: Jimmy Wu
# Date: October 2024
#
# RPC server around the mobile base low-level controller (Vehicle). The 250 Hz
# real-time Vehicle.control_loop runs in THIS process; BaseProcess talks to it
# over a multiprocessing.managers proxy from a separate process. Keeping the
# control loop in its own process (its own GIL) is what prevents the Flask /
# SocketIO WebXR server and the teleop policy from starving it — co-locating
# them in one process caused chronic 250 Hz step-time violations (median ~8.7 ms
# vs the 4 ms budget) and the base lag/coast regression.
#
# Note: Operations that are not time-sensitive must run in a separate,
# non-real-time process to avoid interfering with low-level control.

import time
from multiprocessing.managers import BaseManager as MPBaseManager

from .base_controller import Vehicle
from .constants import BASE_RPC_HOST, BASE_RPC_PORT, RPC_AUTHKEY


class Base:
    def __init__(self, max_vel=(0.25, 0.25, 1), max_accel=(0.5, 0.5, 1.79)):
        self.max_vel = max_vel
        self.max_accel = max_accel
        self.vehicle = None

        # Activation flag for message queue system
        self.activated = False

    def reset(self):
        # Stop low-level control
        if self.vehicle is not None:
            if self.vehicle.control_loop_running:
                self.vehicle.stop_control()

        # Create new instance of vehicle
        self.vehicle = Vehicle(max_vel=self.max_vel, max_accel=self.max_accel)

        # Start low-level control
        self.vehicle.start_control()
        while not self.vehicle.control_loop_running:
            time.sleep(0.01)

    def execute_action(self, action):
        # Check activation flag
        if not self.activated:
            return False

        self.vehicle.set_target_position(action['base_pose'])
        return True

    def get_state(self):
        # Return a plain list so the value pickles cleanly across the RPC proxy
        # (avoids any numpy ABI mismatch between the server and client processes).
        pose = self.vehicle.x.tolist() if self.vehicle is not None else None
        return {'base_pose': pose, 'activated': self.activated}

    # Activation control methods
    def activate(self):
        self.activated = True

    def deactivate(self):
        if self.vehicle is not None:
            # Stop at current position
            self.vehicle.set_target_position(self.vehicle.x.copy())
        self.activated = False

    def emergency_stop(self):
        if self.vehicle is not None:
            # Stop at current position
            self.vehicle.set_target_position(self.vehicle.x.copy())
        self.activated = False

    def close(self):
        if self.vehicle is not None:
            if self.vehicle.control_loop_running:
                self.vehicle.stop_control()


class BaseManager(MPBaseManager):
    pass


BaseManager.register('Base', Base)


def serve_base_manager(host=BASE_RPC_HOST, port=BASE_RPC_PORT, authkey=RPC_AUTHKEY):
    """Run the Base RPC server forever (entry point for the dedicated process)."""
    manager = BaseManager(address=(host, port), authkey=authkey)
    server = manager.get_server()
    print(f'Base manager server started at {host}:{port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    serve_base_manager()
