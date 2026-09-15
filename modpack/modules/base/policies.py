import logging
import math
import socket
import threading
import time
from pathlib import Path
from queue import Queue

import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from scipy.spatial.transform import Rotation as R

from modpack.modules.base.webxr_messages import UnifiedWebXRMessage


DEVICE_CAMERA_OFFSET = np.array([0.0, 0.02, -0.04])
DEVICE_CENTER_OFFSET = 0.37
# DEVICE_Y_CENTER_OFFSET = 0.059
DEVICE_Y_CENTER_OFFSET = 0.075
DEVICE_CENTER_OFFSET_LOCAL = np.array([DEVICE_CENTER_OFFSET, DEVICE_Y_CENTER_OFFSET, 0])
DEVICE_MOUNT_ROT_Z_DEG = 180
DEVICE_MOUNT_ROT_Z_RAD = math.radians(DEVICE_MOUNT_ROT_Z_DEG)
TWO_PI = 2 * math.pi


def _rotation_matrix_x(angle_degrees):
    angle_rad = math.radians(angle_degrees)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def convert_webxr_pose(pos, quat):
    # Step 1: change of basis from WebXR to robot convention 
    pos = np.array([-pos['z'], -pos['x'], pos['y']], dtype=np.float64)
    rot = R.from_quat([-quat['z'], -quat['x'], quat['y'], quat['w']])
    
    # Step 2: rotate to account for phone direction and mount rotation
    rot_x = R.from_matrix(_rotation_matrix_x(90))
    rot_mount = R.from_euler('z', DEVICE_MOUNT_ROT_Z_DEG, degrees=True)
    rot = rot * rot_x * rot_mount
    pos = rot_mount.apply(pos)

    # Step 3: account for center of rotation of iPhone
    pos = pos + rot.apply(DEVICE_CAMERA_OFFSET)
    return pos, rot


class WebServer:
    def __init__(self, input_queue):
        _templates = str(Path(__file__).parent / "templates")
        _static = str(Path(__file__).parent / "static")
        self.app = Flask(__name__, template_folder=_templates, static_folder=_static)
        self.socketio = SocketIO(self.app)
        self.input_queue = input_queue
        self.port = None
        self.address = None

        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/debug', methods=['GET', 'POST'])
        def debug_route():
            self.input_queue.put({'debug': True, 'timestamp': int(1000 * time.time())})
            return "Debug route hit"

        @self.socketio.on('message')
        def handle_message(data):
            msg = UnifiedWebXRMessage(**data)
            emit('echo', msg.timestamp)
            self.input_queue.put(msg.to_dict())

        logging.getLogger('werkzeug').setLevel(logging.WARNING)

        @self.app.route('/base_pose', methods=['POST'])
        def receive_data():
            data = request.json if request.is_json else request.form.to_dict()
            self.socketio.emit('base_pose', data)
            return jsonify({"message": "Data received successfully!"}), 200

    def broadcast_to_iphone(self, message):
        self.socketio.emit('message', message)

    def run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            s.connect(('8.8.8.8', 1))
            self.address = s.getsockname()[0]
        except Exception:
            self.address = '127.0.0.1'
        finally:
            s.close()
        self.port = 5000
        print(f'Starting base web server at {self.address}:{self.port}')
        self.socketio.run(self.app, host='0.0.0.0', port=self.port, allow_unsafe_werkzeug=True)


class TeleopController:
    def __init__(self):
        self.primary_device_id = None
        self.base_pose = np.zeros(3)
        self.targets_initialized = False
        self.base_target_pose = np.zeros(3)
        self.base_xr_ref_pos = None
        self.base_xr_ref_rot_inv = None
        self.base_ref_pose = None
        self.device_offset_local = DEVICE_CENTER_OFFSET_LOCAL.copy()
        self.original_iphone_pos = None
        self.original_iphone_orientation = None
        self.new_iphone_pos = None
        self.new_iphone_orientation = None
        # Ground-plane yaw that maps the iPhone-forward (as held at the reference
        # latch) onto base-forward. Captured at init/rezero so that "push the
        # phone forward" drives the base straight along +x regardless of the
        # angle the phone is held at. 0 when the phone happens to point along
        # base-forward, so it's a no-op in the aligned case.
        self.iphone_to_base_yaw = 0.0

    @staticmethod
    def _iphone_to_base_yaw(iphone_rot, base_theta):
        """Yaw that rotates the iPhone-forward (held orientation) onto base-forward.

        convert_webxr_pose bakes the 180 deg device mount into the returned
        orientation, so the iPhone's ground heading reads ~180 deg when the
        operator points "forward". We subtract that mount before comparing to
        the base heading, so an aligned phone yields a 0 offset.
        """
        fwd = iphone_rot.apply([1.0, 0.0, 0.0])
        iphone_heading = math.atan2(fwd[1], fwd[0])
        return base_theta - (iphone_heading - DEVICE_MOUNT_ROT_Z_RAD)

    def process_message(self, data):
        if not self.targets_initialized:
            return

        device_id = data.get('device_id')
        if self.primary_device_id is None:
            self.primary_device_id = device_id
            print(f"[TeleopController] Primary device registered: {device_id}")

        if self.primary_device_id is not None and data.get('teleop_mode') != 'none':
            pos, rot = convert_webxr_pose(data['position'], data['orientation'])

            if data['teleop_mode'] == 'base':
                pivot_pos = pos - rot.apply(self.device_offset_local)

                if self.base_xr_ref_pos is None:
                    if self.base_pose is None:
                        self.base_pose = np.zeros(3)
                    self.base_ref_pose = self.base_pose.copy()
                    self.base_xr_ref_pos = pivot_pos[:2]
                    self.base_xr_ref_rot_inv = rot.inv()
                    self.original_iphone_pos = pivot_pos[:2]
                    self.original_iphone_orientation = rot
                    self.new_iphone_pos = pivot_pos[:2]
                    self.new_iphone_orientation = rot
                    # Define base-forward as the phone-forward at the FIRST latch.
                    # rot_rel below always maps the push into this original frame,
                    # so the alignment offset is tied to original_iphone_orientation
                    # and is captured once here (not recomputed on rezero, which
                    # would double-correct).
                    self.iphone_to_base_yaw = self._iphone_to_base_yaw(
                        self.original_iphone_orientation, self.base_ref_pose[2]
                    )

                if data['rezero']:
                    self.new_iphone_pos = pivot_pos[:2]
                    self.new_iphone_orientation = rot
                    self.base_ref_pose = self.base_pose.copy()

                delta_pos_3d = np.array([*(pivot_pos[:2] - self.new_iphone_pos), 0.0])
                rot_rel = self.original_iphone_orientation * self.new_iphone_orientation.inv()
                delta_in_old_iphone_frame = rot_rel.apply(delta_pos_3d)[:2]
                # Rotate the push delta so phone-forward maps to base +x; no-op
                # when the phone was latched pointing along base-forward.
                align = R.from_euler("z", self.iphone_to_base_yaw)
                delta_in_base_frame = align.apply([*delta_in_old_iphone_frame, 0.0])[:2]
                self.base_target_pose[:2] = self.base_ref_pose[:2] + delta_in_base_frame

                base_fwd_vec_rotated = (rot * self.new_iphone_orientation.inv()).apply([1.0, 0.0, 0.0])
                base_target_theta = self.base_ref_pose[2] + math.atan2(base_fwd_vec_rotated[1], base_fwd_vec_rotated[0])
                self.base_target_pose[2] += (base_target_theta - self.base_target_pose[2] + math.pi) % TWO_PI - math.pi

        elif self.primary_device_id is None:
            self.base_target_pose = self.base_pose

    def step(self, obs):
        self.base_pose = obs['base_pose']

        if not self.targets_initialized:
            self.base_target_pose = obs['base_pose']
            self.targets_initialized = True

        if self.primary_device_id is None:
            return None

        return {'base_pose': self.base_target_pose.copy()}


class TeleopPolicy:
    def __init__(self):
        self.web_server_queue = Queue()
        self.teleop_controller = None
        self.teleop_state = None
        self.episode_ended = False

        self.server = WebServer(self.web_server_queue)
        threading.Thread(target=self.server.run, daemon=True).start()
        while self.server.address is None or self.server.port is None:
            time.sleep(0.01)

        self.url = f'http://{self.server.address}:{self.server.port}/base_pose'
        self.page_url = f'http://{self.server.address}:{self.server.port}/'

        threading.Thread(target=self.listener_loop, daemon=True).start()

    def reset(self):
        self.teleop_controller = TeleopController()
        self.episode_ended = False
        self.teleop_state = None

    def step(self, obs):
        if not self.episode_ended and self.teleop_state == 'episode_ended':
            self.episode_ended = True
            return 'end_episode'
        if self.teleop_state == 'reset_env':
            return 'reset_env'
        if self.teleop_controller is None:
            return None
        return self._step(obs)

    def _step(self, obs):
        return self.teleop_controller.step(obs)

    def listener_loop(self):
        while True:
            if not self.web_server_queue.empty():
                data = self.web_server_queue.get()
                if 'state_update' in data:
                    self.teleop_state = data['state_update']
                self._process_message(data)
            time.sleep(0.001)

    def process_activation_message(self, msg):
        if not isinstance(msg, dict):
            msg = msg.to_dict()
        command = msg.get("command")

        if command == "start_episode":
            self.teleop_state = "episode_started"
            self.episode_ended = False
            self.teleop_controller = TeleopController()
            self.teleop_controller.targets_initialized = False
            self.server.broadcast_to_iphone({'rezero': True, 'episode_active': True})
            print("[TeleopPolicy] Episode started with rezero")

        elif command == "end_episode":
            self.teleop_state = "episode_ended"
            self.server.broadcast_to_iphone({'episode_active': False})
            print("[TeleopPolicy] Episode ended")

        elif command == "pause_episode":
            self.server.broadcast_to_iphone({'episode_active': False})
            print("[TeleopPolicy] Episode paused")

        elif command == "resume_episode":
            self.server.broadcast_to_iphone({'rezero': True, 'episode_active': True})
            print("[TeleopPolicy] Episode resumed - rezero command sent")

    def _process_message(self, data):
        if self.teleop_controller:
            self.teleop_controller.process_message(data)
