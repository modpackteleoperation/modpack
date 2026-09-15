#!/usr/bin/env python3

import numpy as np
np.set_printoptions(precision=2, suppress=True)
import time
import sys
import signal
from scipy.spatial.transform import Rotation as R
from multiprocessing import Event
from queue import Queue
from robotmq import RMQClient
from pathlib import Path
from robotmq.utils import deserialize

# Repo root on path for `import modpack` when run as a script
sys.path.append(str(Path(__file__).resolve().parents[3]))
from modpack.orchestration.message_formats import *
from modpack.modules.vision_pro.pose_receiver import PoseReceiver
from modpack.modules.vision_pro.transformation_helper import mat2pose, neck_start_pose
from modpack.orchestration.message_formats import deserialize_base_proprio_message
from modpack.utils import DebugPrinter

class VisionPro():
    def __init__(self,
                data_host: str,
                data_port: str,
                activation_host: str,
                activation_port: str,
                debug: bool,
                ):
        # Actions go to separate server
        self.debug = debug
        self.debug_printer = DebugPrinter(enabled=debug)
        self.system_activated = False
        self.activation_host = activation_host
        self.activation_port = activation_port

        # Setup clients
        self.activation_client = RMQClient(
            client_name="vision_pro_activation_client",
            server_endpoint=f"tcp://{self.activation_host}:{self.activation_port}"
        )

        self.data_server_client = RMQClient(
            client_name="data_server_client", 
            server_endpoint=f"tcp://{data_host}:{data_port}"
        )
        self.pose_receiver = PoseReceiver(bind_ip="0.0.0.0", port=5005, timeout_s=1.0)
        self.pose_receiver.start()
        print(f"[VisionPro] Listening for headset pose on UDP 0.0.0.0:5005")
        print(f"[VisionPro] Swift app must send pose to: {data_host}:5005")

        self.factory = MessageFactory()

        self.tv = None
        self.init_camera_pose = None
        self.desired_interval = 1 / 30.0
        self.status_poll_timeout = 2
        self.camera_pop_timeout = 3.0
        # Queue and Event
        self.image_queue = Queue()
        self.toggle_streaming = Event()

        # State variables
        self.running = False
        self.shutdown_requested = False
        self.sent_ready = False
        self.publishing_neck_pose = False
        self.neck_at_init = False
        self.vr_start_pose = None
        self.vr_start_matrix = None
        self.vr_start_pose_ts = None
        self.base_start_matrix = None
        self._last_processed_pose_ts = None
        self._prev_head_quat = None
        self._vr_to_neck_axes_legacy = np.array(
            [
                [0.0, 0.0, -1.0],  # x_neck = -z_vr
                [-1.0, 0.0, 0.0],  # y_neck = -x_vr
                [0.0, 1.0, 0.0],   # z_neck =  y_vr
            ],
            dtype=np.float64,
        )
        self._vr_to_neck_axes = self._vr_to_neck_axes_legacy.copy()
        self._prev_base_world_pos = None
        self._prev_neck_cmd_world_uncomp_pos = None
       
        # Optional fixed extrinsic from mobile-base frame to neck-base frame.
        # Defaults to identity if not provided.
        self._T_mb_nb = np.eye(4, dtype=np.float64)
        self.exe_queue_head = Queue(maxsize=10)
        self._neck_cmd_send_count = 0
        self._last_neck_cmd_log_time = 0.0
        self._last_neckbase_log_time = 0.0
        self._neck_cmd_log_interval = 5.0
        self._last_vp_activation_topic_status = None
        self.status_interval = 2.0 # seconds
        self.last_sent_status = time.monotonic()

        self._last_valid_D_nb = np.eye(4, dtype=np.float64)

        signal.signal(signal.SIGTERM, self._handle_termination_signal)
        signal.signal(signal.SIGINT, self._handle_termination_signal)

    def _reset_reference_frames(self) -> None:
        """Reset per-activation reference transforms."""
        self.vr_start_pose = None
        self.vr_start_matrix = None
        self.vr_start_pose_ts = None
        self.base_start_matrix = None
        self._last_processed_pose_ts = None
        self._prev_head_quat = None
        self._recv_correction_rot = np.eye(3, dtype=np.float64)
        self._recv_correction_active = False
        self._last_valid_D_nb = np.eye(4, dtype=np.float64)

    @staticmethod
    def _build_transform(pos_xyz: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
        print(f'z is {pos_xyz[-1]}')
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
        T[:3, 3] = pos_xyz
        return T

    # Parse base pose message into world transform. 
    # Accept both [x, y, yaw] and [x, y, z, qx, qy, qz, qw] formats
    # By default- uses quat_xyzw 
    def _parse_base_world_transform(self, base_msg) -> np.ndarray:
        if base_msg is None or getattr(base_msg, "pose", None) is None:
            return None
        raw = np.asarray(base_msg.pose, dtype=np.float64)
        if raw.shape[0] == 3:
            pos = np.array([raw[0], raw[1], 0.0], dtype=np.float64)
            quat = R.from_euler("z", raw[2], degrees=False).as_quat()
            print("raw is [x, y, yaw]")
            return self._build_transform(pos, quat)
        if raw.shape[0] >= 7:
            print(raw[:3])
            pos = raw[:3]
            quat = raw[3:7]
            print("raw is [x, y, z, qx, qy, qz, qw]")
            return self._build_transform(pos, quat)
        return None

    def _get_latest_base_world_transform(self) -> np.ndarray:
        try:
            base_list, _ = self.data_server_client.peek_data(Topics.BASE, timeout_s=0.05, n=-1)
            if not base_list:
                return None
            base_msg = deserialize_base_proprio_message(base_list[0])
            return self._parse_base_world_transform(base_msg)
        except Exception as exc:
            print(f"[VisionPro] Failed to parse base pose: {exc}")
            return None

    @staticmethod
    def _pose_from_transform(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xyz = np.asarray(T[:3, 3], dtype=np.float64)
        rpy = R.from_matrix(np.asarray(T[:3, :3], dtype=np.float64)).as_euler("xyz", degrees=False)
        return xyz, rpy

    def _handle_termination_signal(self, signum, frame) -> None:
        sig_name = signal.Signals(signum).name
        print(f"[VisionPro] Received {sig_name}")
        self.shutdown_requested = True
        self.running = False

    def run(self):
        self.running = True
        iteration_count = 0
        last_activation_check = time.monotonic()

        # try:
        while self.running:
            if self.shutdown_requested:
                print("Shutdown requested - exiting Vision Pro loop")
                break

            loop_start = time.time()

            # Check activation commands periodically
            if time.monotonic() - last_activation_check >= self.status_interval:
                self.check_activation_messages()
                last_activation_check = time.monotonic()

            should_publish = True

            if should_publish and not self.system_activated:
                should_publish = False

            if should_publish:
                pose_pkt = self.pose_receiver.get_latest()
                if pose_pkt is None or (not pose_pkt.valid):
                    if not getattr(self, "_logged_no_pose", False):
                        print("[VisionPro] WARNING: system activated but no valid pose from headset yet")
                        self._logged_no_pose = True
                    should_publish = False
                else:
                    self._logged_no_pose = False

            if should_publish:
                if self._last_processed_pose_ts is not None and pose_pkt.ts <= self._last_processed_pose_ts:
                    # Avoid reprocessing the same/stale packet.
                    should_publish = False

            if should_publish:
                # make sure it's float32/float64 as your downstream expects
                head_matrix = np.asarray(pose_pkt.Tw_device, dtype=np.float32)

                if self.vr_start_pose is None:
                    self.vr_start_pose = np.asarray(mat2pose(head_matrix), dtype=np.float64)
                    self.vr_start_matrix = np.asarray(head_matrix, dtype=np.float64).copy()
                    self.vr_start_pose_ts = float(pose_pkt.ts)
                    self._last_processed_pose_ts = float(pose_pkt.ts)
                    # Align neck-forward with base-forward at activation.
                    #
                    # VR-forward (vr_start_matrix) and base-forward (base odometry)
                    # are latched from two unrelated devices, so the angle between
                    # the operator's gaze and the base's physical heading at
                    # activation becomes a permanent offset: driving the base
                    # straight forward then pushes the neck command along the
                    # VR-latched forward instead, and the head wanders left/right.
                    #
                    # We DEFINE base-forward as "operator looking straight ahead at
                    # activation" by latching the base's actual world transform here
                    # (instead of identity). The base delta the compensator uses,
                    # D_mb = inv(base_start) @ base_current, is then motion since
                    # activation expressed in the base frame as it was oriented at
                    # activation — which coincides with neck-forward.
                    base_at_start = self._get_latest_base_world_transform()
                    self.base_start_matrix = (
                        base_at_start.copy() if base_at_start is not None
                        else np.eye(4, dtype=np.float64)
                    )

                    # Diagnostics: confirm what each device reports at activation.
                    # If the base hasn't moved, base_start is expected to be ~zero
                    # (identity) — meaning any forward mismatch lives on the VR side
                    # (the latched headset heading), not the base side.
                    vr_yaw = float(self.vr_start_pose[5])
                    if base_at_start is not None:
                        base_xyz, base_rpy = self._pose_from_transform(self.base_start_matrix)
                        print(
                            "[VisionPro] ACTIVATION LATCH | "
                            f"base_start xyz={np.round(base_xyz, 4)} "
                            f"yaw={np.round(np.rad2deg(base_rpy[2]), 2)} deg | "
                            f"VR-forward yaw={np.round(np.rad2deg(vr_yaw), 2)} deg "
                            f"(from vr_start_pose)",
                            flush=True,
                        )
                    else:
                        print(
                            "[VisionPro] ACTIVATION LATCH | base pose UNAVAILABLE, "
                            "base_start=identity | "
                            f"VR-forward yaw={np.round(np.rad2deg(vr_yaw), 2)} deg",
                            flush=True,
                        )
                else:
                    # Ensure first relative sample uses a frame strictly after latch.
                    if self.vr_start_pose_ts is not None and pose_pkt.ts <= self.vr_start_pose_ts:
                        should_publish = False
                    else:
                        base_current_world = self._get_latest_base_world_transform()

                        # Always call filter_head_pose (with or without base data)
                        execute_smooth_head = self._filter_head_pose(
                            vr_start_pose=self.vr_start_pose,
                            pos_speed=1,
                            ori_speed=1,
                            exe_queue_head=self.exe_queue_head,
                            head_mat=head_matrix,
                            base_current=base_current_world,
                            base_start=self.base_start_matrix,
                            neck_start_pose=neck_start_pose
                        )

                        # self.debug_printer.log(lambda: f"[VR Pose] smooth head: {execute_smooth_head}")

                        exe_action = execute_smooth_head

                        # self.debug_printer.log(lambda: f"[VR Pose] exe action: {exe_action}")

                        neck_cmd = self.factory.create_neck_message(list(map(float, exe_action)))
                        neck_cmd_bytes = serialize_neck_message(neck_cmd)
                        self.data_server_client.put_data(Topics.NECK_CMD, neck_cmd_bytes)
                        self._neck_cmd_send_count += 1
                        self._last_processed_pose_ts = float(pose_pkt.ts)
                        if self._neck_cmd_send_count == 1:
                            print("[VisionPro] First neck command published")
                        elif self._neck_cmd_send_count % 100 == 0:
                            print(f"[VisionPro] Neck commands sent: {self._neck_cmd_send_count}")

            iteration_count += 1

            elapsed = time.time() - loop_start
            sleep_time = max(0, self.desired_interval - elapsed)
            time.sleep(sleep_time)

        # except KeyboardInterrupt:
        #     print("Received keyboard interrupt")
        # except Exception as e:
        #     print(f"ERROR in main loop: {e}")
        # finally:
        #     self.running = False
        #     self.shutdown()
        
        self.shutdown()
        print("Terminating vision pro process")
        return True

    def _filter_head_pose(self, vr_start_pose, pos_speed, ori_speed, exe_queue_head, head_mat, base_current=None, base_start=None, neck_start_pose=None):
        head_mat = np.asarray(head_mat, dtype=np.float64)

        # Relative VR motion from episode start
        if self.vr_start_matrix is not None:
            start_mat = np.asarray(self.vr_start_matrix, dtype=np.float64)
        else:
            start_pose = np.asarray(vr_start_pose, dtype=np.float64)
            start_mat = np.eye(4, dtype=np.float64)
            start_mat[:3, :3] = R.from_euler("xyz", start_pose[3:], degrees=False).as_matrix()
            start_mat[:3, 3] = start_pose[:3]

        rel_mat_vr = np.linalg.inv(start_mat) @ head_mat
        rel_pos_vr = rel_mat_vr[:3, 3]
        rel_rot_vr = rel_mat_vr[:3, :3]

        # Relative neck motion from VR in neck-command axes
        rel_pos_neck = self._vr_to_neck_axes @ rel_pos_vr * float(pos_speed)
        remapped_rot = self._vr_to_neck_axes @ rel_rot_vr @ self._vr_to_neck_axes.T
        rel_rot_neck = remapped_rot
        
        # Compute base delta in neck frame
        D_nb = np.asarray(self._last_valid_D_nb, dtype=np.float64).copy()
        if base_current is not None and base_start is not None:
            base_current = np.asarray(base_current, dtype=np.float64)
            base_start = np.asarray(base_start, dtype=np.float64)
            if base_current.shape == (4, 4) and base_start.shape == (4, 4):
                D_mb = np.linalg.inv(base_start) @ base_current
                D_nb = D_mb
                self._last_valid_D_nb = D_nb.copy()

                # Diagnostics: show the EXECUTED base delta (odometry) the
                # compensator is applying, alongside the COMMANDED base delta
                # (the base target). This answers "am I driving at an angle vs is
                # the base veering off the commanded heading":
                #   - commanded path angle ~0 but executed angle nonzero
                #       => real commanded-vs-executed mismatch (base veers)
                #   - both angles match
                #       => the input itself is at an angle (operator/teleop)
                # Path angle = atan2(dy, dx) of the translation since activation.
                # Rate-limited to ~2 Hz.
                now = time.time()
                if now - self._last_neck_cmd_log_time >= 0.5:
                    self._last_neck_cmd_log_time = now
                    d_xyz, d_rpy = self._pose_from_transform(D_mb)
                    exec_ang = float(np.rad2deg(np.arctan2(d_xyz[1], d_xyz[0])))

                    cmd_str = "cmd=UNAVAILABLE"
                    try:
                        tgt_list, _ = self.data_server_client.peek_data(
                            Topics.BASE_TARGET, timeout_s=0.05, n=-1
                        )
                        if tgt_list:
                            tgt_msg = deserialize_base_proprio_message(tgt_list[0])
                            T_cmd = self._parse_base_world_transform(tgt_msg)
                            if T_cmd is not None:
                                D_cmd = np.linalg.inv(base_start) @ T_cmd
                                c_xyz, c_rpy = self._pose_from_transform(D_cmd)
                                cmd_ang = float(np.rad2deg(np.arctan2(c_xyz[1], c_xyz[0])))
                                cmd_str = (
                                    f"cmd dx={c_xyz[0]:+.4f} dy={c_xyz[1]:+.4f} "
                                    f"dyaw={np.rad2deg(c_rpy[2]):+.2f} ang={cmd_ang:+.2f}"
                                )
                    except Exception as exc:
                        cmd_str = f"cmd ERR {exc}"

                    print(
                        "[VisionPro] BASE DELTA | "
                        f"exec dx={d_xyz[0]:+.4f} dy={d_xyz[1]:+.4f} "
                        f"dyaw={np.rad2deg(d_rpy[2]):+.2f} ang={exec_ang:+.2f} | "
                        f"{cmd_str}",
                        flush=True,
                    )

        # Convert relative command to absolute neck-frame
        T_nb_ee_start = np.eye(4, dtype=np.float64)
        T_nb_ee_start[:3, :3] = R.from_euler("xyz", neck_start_pose[3:], degrees=False).as_matrix()
        T_nb_ee_start[:3, 3] = np.asarray(neck_start_pose[:3], dtype=np.float64)

        # Before compensation
        T_nb_ee_cmd = np.eye(4, dtype=np.float64)
        T_nb_ee_cmd[:3, :3] = rel_rot_neck @ T_nb_ee_start[:3, :3]
        T_nb_ee_cmd[:3, 3] = rel_rot_neck @ T_nb_ee_start[:3, 3] + rel_pos_neck
        T_nb0_ee_cmd = T_nb_ee_cmd.copy()

        # Compensate rotation in ee space
        T_nb_ee_cmd = np.linalg.inv(D_nb) @ T_nb0_ee_cmd

        cmd_rot = T_nb_ee_cmd[:3, :3]
        cmd_pos = T_nb_ee_cmd[:3, 3]
        cmd_quat = R.from_matrix(cmd_rot).as_quat()

        # Diagnostic: is NECK-forward aligned with BASE-forward?
        #
        # Isolates the mount-yaw between the base frame (where D_mb lives) and
        # the neck-base frame (where the compensation is applied). The base
        # forward delta is dx = D_nb[0,3]; the compensation shifts the neck
        # command by (compensated - uncompensated). If the two frames share an
        # axis, a pure-forward base motion produces neck shift along -x ONLY, so
        # the lateral (y) shift is ~0. A nonzero ratio dy_shift/dx_base = the
        # tangent of the mount yaw -> the angle we must put in to align them.
        # Drive the base straight forward and read theta off this line.
        comp_shift = T_nb_ee_cmd[:3, 3] - T_nb0_ee_cmd[:3, 3]
        dx_base = float(D_nb[0, 3])
        now_n = time.time()
        if abs(dx_base) > 0.02 and now_n - self._last_neckbase_log_time >= 0.5:
            self._last_neckbase_log_time = now_n
            theta_deg = float(np.rad2deg(np.arctan2(comp_shift[1], -comp_shift[0])))
            print(
                "[VisionPro] NECK-vs-BASE FWD | "
                f"base dx={dx_base:+.4f} | neck comp shift "
                f"x={comp_shift[0]:+.4f} y={comp_shift[1]:+.4f} | "
                f"mount yaw est={theta_deg:+.2f} deg "
                f"(0 = neck-fwd aligned with base-fwd)",
                flush=True,
            )
      
        execute_pose = np.concatenate([cmd_pos, cmd_quat])

        if (exe_queue_head.maxsize > 0 and exe_queue_head._qsize() == exe_queue_head.maxsize):
            exe_queue_head._get()
        exe_queue_head.put_nowait(execute_pose)

        # Update previous-state caches for next timestep.
        if base_current is not None and np.asarray(base_current).shape == (4, 4):
            self._prev_base_world_pos = np.asarray(base_current, dtype=np.float64)[:3, 3].copy()
        self._prev_neck_cmd_world_uncomp_pos = T_nb0_ee_cmd[:3, 3].copy()
        
        return execute_pose

    def shutdown(self):
        """Gracefully shutdown the Vision Pro process"""
        print("Starting Vision Pro shutdown...")

        # Set flags to stop main loop and prevent new operations
        self.running = False
        self.shutdown_requested = True
        self.system_activated = False
        
        # Release the UDP pose socket so a restart can rebind it.
        self.pose_receiver.stop()

        # Close RMQ clients so the underlying ZMQ context can tear down without
        # blocking process exit.
        for _name in ("activation_client", "data_server_client"):
            _client = getattr(self, _name, None)
            _close = getattr(_client, "close", None)
            if callable(_close):
                try:
                    _close()
                except Exception as exc:
                    print(f"[VisionPro] Error closing {_name}: {exc}")

        print("Vision Pro shutdown complete")

    def check_activation_messages(self):
        """Check for activation messages from message queue manager"""
        try:
            if not self.activation_client:
                return
            
            status = self.activation_client.get_topic_status(Topics.VP_PUBLISH_ACTIVATION, 0.1)
            if status != self._last_vp_activation_topic_status:
                self.debug_printer.log(lambda: f"[VisionPro] VP_PUBLISH_ACTIVATION topic status={status}")
                self._last_vp_activation_topic_status = status
            if status > 0:
                data_list, _ = self.activation_client.peek_data(Topics.VP_PUBLISH_ACTIVATION, n=-1)
            else:
                return
            if not data_list:
                return
            
            payload = deserialize(data_list[0])
            if not payload:
                return
            activation_msg = ActivationMessage.from_dict(payload)
            
            if activation_msg.command and activation_msg.target in ('all', 'vision_pro'):
                command = activation_msg.command
                self.handle_activation_command(command)
                
        except Exception as e:
            print(e)

    def handle_activation_command(self, command: str):
        """Handle activation commands"""
        if command == "activate":
            if not self.system_activated:
                print("SYSTEM ACTIVATED via message queue")
                self.system_activated = True
                self._reset_reference_frames()
                self.pose_receiver.clear_latest()
        elif command == "deactivate":
            if self.system_activated:
                print("SYSTEM DEACTIVATED via message queue")
                self.system_activated = False
                self._reset_reference_frames()
                
        elif command == "emergency_stop":
            print("EMERGENCY STOP via message queue")
            self.system_activated = False
            self._reset_reference_frames()
            self.shutdown_requested = True
            
        elif command == "shutdown":
            print("SHUTDOWN command via message queue")
            self.system_activated = False
            self._reset_reference_frames()
            self.shutdown_requested = True
            
        elif command == "manager_shutdown":
            print("Manager shutdown - VisionPro continuing independently")
            self.system_activated = False
            self._reset_reference_frames()
            self.shutdown_requested = True
            
        else:
            print(f"Unknown activation command: {command}")
