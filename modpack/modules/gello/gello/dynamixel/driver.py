import time
from threading import Event, Lock, Thread
from typing import Protocol, Sequence, Dict, Any
import numpy as np
from dynamixel_sdk.group_sync_read import GroupSyncRead
from dynamixel_sdk.group_sync_write import GroupSyncWrite
from dynamixel_sdk.packet_handler import PacketHandler
from dynamixel_sdk.port_handler import PortHandler
from dynamixel_sdk.robotis_def import (
    COMM_SUCCESS,
    DXL_HIBYTE,
    DXL_LOBYTE,
)

# Dynamixel register addresses
ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11
ADDR_GOAL_POSITION = 116
ADDR_GOAL_CURRENT = 102
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_VELOCITY = 128
ADDR_HOMING_OFFSET = 20
ADDR_PRESENT_CURRENT = 126

# Data lengths
LEN_GOAL_POSITION = 4
LEN_GOAL_CURRENT = 2
LEN_PRESENT_POSITION = 4
LEN_PRESENT_VELOCITY = 4
LEN_HOMING_OFFSET = 4
LEN_PRESENT_CURRENT = 2

# Control modes
CURRENT_CONTROL_MODE = 0
POSITION_CONTROL_MODE = 3
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

class DynamixelDriverProtocol(Protocol):
    def set_joints(self, joint_angles: Sequence[float]):
        """Set the joint angles for position-controlled servos."""
        ...

    def set_torques(self, torques: Sequence[float]):
        """Set the torques for torque-controlled servos."""
        ...

    def torque_enabled(self) -> bool:
        """Check if torque is enabled for the servos."""
        ...

    def enable_torque(self, enable: bool):
        """Set the torque enable/disable for all servos."""
        ...

    def get_joints(self) -> np.ndarray:
        """Get the current joint angles in radians."""
        ...

    def get_velocities(self) -> np.ndarray:
        """Get the current joint velocities in rad/s."""
        ...

    def close(self):
        """Close the driver."""
        ...


class DynamixelDriver(DynamixelDriverProtocol):
    def __init__(
    self, 
    motor_config: Dict[int, Dict[str, Any]], 
    port: str = "/dev/ttyACM0", 
    baudrate: int = 57600,
):
        """Initialize the DynamixelDriver with health monitoring."""
        self.motor_config = motor_config
        self._ids = list(motor_config.keys())
        self._joint_angles_raw = None
        self._joint_velocities_raw = None
        self._lock = Lock()
        
        # Write-in-progress flag for read throttling
        self._writing_in_progress = False
        
        # Communication health monitoring
        self._last_successful_read = time.time()
        self._consecutive_comm_failures = 0
        self._comm_healthy = True
        self._responsive_motors = set(self._ids)
        self._last_data_update = time.time()
        self._data_freshness_threshold = 0.1  # 100ms
        
        # Constants
        self.COUNTS_PER_REV = 4096
        
        # Initialize communication
        self._portHandler = PortHandler(port)
        self._packetHandler = PacketHandler(2.0)
        
        # Group sync read for positions and velocities
        self._groupSyncReadPos = GroupSyncRead(
            self._portHandler,
            self._packetHandler,
            ADDR_PRESENT_POSITION,
            LEN_PRESENT_POSITION,
        )
        self._groupSyncReadVel = GroupSyncRead(
            self._portHandler,
            self._packetHandler,
            ADDR_PRESENT_VELOCITY,
            LEN_PRESENT_VELOCITY,
        )
        
        # Group sync write for positions and torques
        self._groupSyncWritePos = GroupSyncWrite(
            self._portHandler,
            self._packetHandler,
            ADDR_GOAL_POSITION,
            LEN_GOAL_POSITION,
        )
        self._groupSyncWriteCurrent = GroupSyncWrite(
            self._portHandler,
            self._packetHandler,
            ADDR_GOAL_CURRENT,
            LEN_GOAL_CURRENT,
        )

        self._groupSyncReadCurrent = GroupSyncRead(
            self._portHandler,
            self._packetHandler,
            ADDR_PRESENT_CURRENT,
            LEN_PRESENT_CURRENT,
        )
        
        # Open port and set baudrate
        if not self._portHandler.openPort():
            raise RuntimeError("Failed to open the port")
        if not self._portHandler.setBaudRate(baudrate):
            raise RuntimeError(f"Failed to change the baudrate to {baudrate}")

        # Add parameters for group sync reads
        for motor_id in self._ids:
            if not self._groupSyncReadPos.addParam(motor_id):
                raise RuntimeError(f"Failed to add position read param for motor {motor_id}")
            if not self._groupSyncReadVel.addParam(motor_id):
                raise RuntimeError(f"Failed to add velocity read param for motor {motor_id}")
            if not self._groupSyncReadCurrent.addParam(motor_id):
                raise RuntimeError(f"Failed to add current read param for motor {motor_id}")
        # Setup motors
        self._setup_motors()
        
        # Read homing offsets
        self.homing_offsets = self._read_homing_offsets()
        
        # Start reading thread
        self._stop_thread = Event()
        self._torque_enabled = False
        self._start_reading_thread()

    def _setup_motors(self):
        """Setup each motor with appropriate control mode"""
        
        for motor_id, config in self.motor_config.items():
            # Test communication
            try:
                pos, comm_result, error = self._packetHandler.read4ByteTxRx(
                    self._portHandler, motor_id, ADDR_PRESENT_POSITION)

                if comm_result != COMM_SUCCESS or error != 0:
                    print(f"WARNING: Communication failed with Motor {motor_id}")
                    self._responsive_motors.discard(motor_id)
                    continue
                else:
                    print(f"Motor {motor_id} communication OK")
            except Exception as e:
                print(f"Error communicating with Motor {motor_id}: {e}")
                self._responsive_motors.discard(motor_id)
                continue

            # Set control mode
            control_mode = CURRENT_CONTROL_MODE if config['control'] == 'torque' else POSITION_CONTROL_MODE
            success = self._set_motor_mode(motor_id, control_mode)
            if success:
                print(f"Motor {motor_id} set to {config['control'].upper()} control")
            else:
                print(f"Motor {motor_id} FAILED to set {config['control']} control")
                self._responsive_motors.discard(motor_id)

    def _set_motor_mode(self, motor_id: int, mode_value: int, max_retries: int = 3) -> bool:
        """Set motor control mode with verification"""
        for attempt in range(max_retries):
            # Disable torque
            res, err = self._packetHandler.write1ByteTxRx(self._portHandler, motor_id, ADDR_TORQUE_ENABLE, 0)
            if res != COMM_SUCCESS or err != 0:
                continue
            time.sleep(0.2)
            
            # Set mode
            res, err = self._packetHandler.write1ByteTxRx(self._portHandler, motor_id, ADDR_OPERATING_MODE, mode_value)
            if res != COMM_SUCCESS or err != 0:
                continue
            time.sleep(0.2)
            
            # Verify mode was set
            actual_mode, comm_result, error = self._packetHandler.read1ByteTxRx(self._portHandler, motor_id, ADDR_OPERATING_MODE)
            if comm_result == COMM_SUCCESS and error == 0 and actual_mode == mode_value:
                # Re-enable torque
                self._packetHandler.write1ByteTxRx(self._portHandler, motor_id, ADDR_TORQUE_ENABLE, 1)
                return True
            
            time.sleep(0.5)

        return False

    def _read_homing_offsets(self) -> Dict[int, int]:
        """Read homing offsets from all motors"""
        offsets = {}
        for motor_id in self._ids:
            try:
                homing_offset, comm_result, error = self._packetHandler.read4ByteTxRx(
                    self._portHandler, motor_id, ADDR_HOMING_OFFSET)
                if comm_result == COMM_SUCCESS and error == 0:
                    offsets[motor_id] = homing_offset
                else:
                    offsets[motor_id] = 0
                    print(f"Motor {motor_id}: failed to read homing offset")
            except Exception as e:
                offsets[motor_id] = 0
                print(f"Motor {motor_id}: error reading homing offset: {e}")
        
        return offsets

    def _convert_radians_to_raw(self, radians: float) -> int:
        """Convert radians to raw encoder counts"""
        raw_counts = int(round(radians * (self.COUNTS_PER_REV / (2.0 * np.pi))))
        return raw_counts

    def _convert_raw_velocity_to_rad_per_sec(self, raw_velocity: int) -> float:
        """Convert raw velocity to rad/s"""
        return raw_velocity * 0.229 * np.pi / 30

    def _start_reading_thread(self):
        """Start the continuous reading thread"""
        self._reading_thread = Thread(target=self._read_joint_data)
        self._reading_thread.daemon = True
        self._reading_thread.start()

    def _read_joint_data(self):
        """Continuously read joint positions and velocities with health monitoring and write throttling"""
        while not self._stop_thread.is_set():
            time.sleep(0.00333)  # 300 Hz
            
            # Skip read cycle if control is writing
            if self._writing_in_progress:
                continue
            
            current_time = time.time()
            read_successful = False
            
            with self._lock:
                try:
                    # Read positions (store as raw counts)
                    _joint_angles_raw = np.zeros(len(self._ids), dtype=int)
                    dxl_comm_result = self._groupSyncReadPos.txRxPacket()
                    
                    if dxl_comm_result == COMM_SUCCESS:
                        position_read_count = 0
                        for i, motor_id in enumerate(self._ids):
                            if self._groupSyncReadPos.isAvailable(motor_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
                                raw_pos = self._groupSyncReadPos.getData(motor_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                                raw_pos = np.int32(np.uint32(raw_pos))
                                _joint_angles_raw[i] = raw_pos
                                position_read_count += 1
                        
                        # Check if we got data from most motors
                        if position_read_count >= len(self._ids) * 0.8:  # 80% success rate
                            read_successful = True
                    # Read velocities (store as raw values)
                    self._joint_velocities_raw = np.zeros(len(self._ids), dtype=int)
                    dxl_comm_result = self._groupSyncReadVel.txRxPacket()
                    
                    if dxl_comm_result == COMM_SUCCESS:
                        for i, motor_id in enumerate(self._ids):
                            if self._groupSyncReadVel.isAvailable(motor_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY):
                                raw_vel = self._groupSyncReadVel.getData(motor_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY)
                                raw_vel = np.int32(np.uint32(raw_vel))
                                self._joint_velocities_raw[i] = raw_vel
                    # Update health status
                    if read_successful:
                        self._joint_angles_raw = _joint_angles_raw
                        self._joint_velocities_raw = self._joint_velocities_raw
                        self._last_successful_read = current_time
                        self._last_data_update = current_time
                        self._consecutive_comm_failures = 0
                        
                        if not self._comm_healthy:
                            self._comm_healthy = True
                    else:
                        self._consecutive_comm_failures += 1
                        
                        if self._consecutive_comm_failures >= 10 and self._comm_healthy:
                            print(f"DRIVER: Communication degraded - {self._consecutive_comm_failures} consecutive failures")
                            self._comm_healthy = False
                    
                except Exception as e:
                    self._consecutive_comm_failures += 1
                    if self._consecutive_comm_failures % 100 == 0:
                        print(f"Error in reading thread (#{self._consecutive_comm_failures}): {e}")
                    
                    if self._consecutive_comm_failures >= 10 and self._comm_healthy:
                        print("DRIVER: Communication failure detected")
                        self._comm_healthy = False
                    continue

    def get_communication_status(self) -> dict:
        """Get real-time communication health status"""
        current_time = time.time()
        
        with self._lock:
            data_age = current_time - self._last_data_update
            time_since_success = current_time - self._last_successful_read
            
            return {
                'communication_healthy': self._comm_healthy,
                'last_successful_read': self._last_successful_read,
                'time_since_success': time_since_success,
                'consecutive_failures': self._consecutive_comm_failures,
                'data_age_ms': data_age * 1000,
                'data_fresh': data_age < self._data_freshness_threshold,
                'responsive_motors': list(self._responsive_motors),
                'total_motors': len(self._ids)
            }

    def is_communication_healthy(self) -> bool:
        """Quick check if communication is healthy"""
        current_time = time.time()
        with self._lock:
            data_age = current_time - self._last_data_update
            return (self._comm_healthy and 
                    data_age < self._data_freshness_threshold and 
                    self._consecutive_comm_failures < 5)

    def test_motor_communication(self, motor_id: int) -> bool:
        """Test communication with a specific motor"""
        try:
            pos, comm_result, error = self._packetHandler.read4ByteTxRx(
                self._portHandler, motor_id, ADDR_PRESENT_POSITION)
            return comm_result == COMM_SUCCESS and error == 0
        except Exception:
            return False

    def get_responsive_motors(self) -> list:
        """Get list of currently responsive motors"""
        responsive = []
        for motor_id in self._ids:
            if self.test_motor_communication(motor_id):
                responsive.append(motor_id)
        
        # Update internal tracking
        with self._lock:
            self._responsive_motors = set(responsive)
            
        return responsive

    # TODO update to groupSyncWrite
    def set_joints(self, joint_angles: Sequence[float]):
        """Set joint angles for position-controlled motors using direct writes"""
        if len(joint_angles) != len(self._ids):
            raise ValueError("The length of joint_angles must match the number of servos")
        if not self._torque_enabled:
            raise RuntimeError("Torque must be enabled to set joint angles")
        if not self.is_communication_healthy():
            return

        self._writing_in_progress = True
        try:
            for motor_id, angle in zip(self._ids, joint_angles):
                if self.motor_config[motor_id]['control'] != 'position':
                    continue
                if motor_id not in self._responsive_motors:
                    continue

                position_value = self._convert_radians_to_raw(angle)
                # Direct write to goal position using the packet handler and port
                self._packetHandler.write4ByteTxRx(
                    self._portHandler,
                    motor_id,
                    ADDR_GOAL_POSITION,
                    position_value
                )
        finally:
            self._writing_in_progress = False

    def set_torques(self, torques: Sequence[float]):
        """Set torques for torque-controlled motors with direct writes"""
        if not self.is_communication_healthy():
            return
        try:
            self._writing_in_progress = True            
            torque_motors = [(motor_id, torque) for motor_id, torque in zip(self._ids, torques) 
                            if self.motor_config[motor_id]['control'] == 'torque']
            if not torque_motors:
                return
            
            for motor_id, torque_nm in torque_motors:
                if motor_id not in self._responsive_motors:
                    continue

                config = self.motor_config[motor_id]
                current_limit = config['current_limit']  # mA
                model = config['model']

                # Torque constants
                if model == 'XM540':
                    torque_constant = 2.4
                elif model == 'XM430':
                    torque_constant = 1.57
                elif model == 'XL430':
                    torque_constant = 1.15
                else:
                    torque_constant = 1.79

                # Convert torque to raw current units
                required_current_A = torque_nm / torque_constant if abs(torque_constant) > 1e-6 else 0.0
                required_current_raw = required_current_A / 2.69e-3
                max_current_raw = current_limit / 2.69
                current_value = int(np.clip(required_current_raw, -max_current_raw, max_current_raw))
                
                # Convert signed to unsigned 16-bit
                if current_value < 0:
                    unsigned_value = (1 << 16) + current_value  # Two's complement conversion
                else:
                    unsigned_value = current_value
            
                unsigned_value = int(unsigned_value) & 0xFFFF  # Ensure it's within 16-bit range

                param_goal_current = [
                    DXL_LOBYTE(unsigned_value),
                    DXL_HIBYTE(unsigned_value),
                ]

                # Add to sync write
                if not self._groupSyncWriteCurrent.addParam(motor_id, param_goal_current):
                    raise RuntimeError(f"Failed to add current param for motor {motor_id}")

            # Execute sync write
            dxl_comm_result = self._groupSyncWriteCurrent.txPacket()
            if dxl_comm_result != COMM_SUCCESS:
                print(f"Failed to sync write currents: {dxl_comm_result}")

            # Clear parameters
            self._groupSyncWriteCurrent.clearParam()

  
        finally:
            self._writing_in_progress = False

    def torque_enabled(self) -> bool:
        """Check if torque is enabled"""
        return self._torque_enabled

    def enable_torque(self, enable: bool, max_retries=3):
        """Enable or disable torque for torque-controlled motors only."""

        with self._lock:
            for motor_id in self._ids:
                # Skip non-responsive motors
                if motor_id not in self._responsive_motors:
                    continue
                    
                # Check if this motor should have torque enabled, we currently set position motors to passive
                should_enable_torque = (enable and 
                                    self.motor_config[motor_id]['control'] == 'torque')
                
                target_state = TORQUE_ENABLE if should_enable_torque else TORQUE_DISABLE
                
                for _ in range(max_retries):
                    # Set torque state
                    dxl_comm_result, dxl_error = self._packetHandler.write1ByteTxRx(
                        self._portHandler, motor_id, ADDR_TORQUE_ENABLE, target_state
                    )
                    if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                        continue
                    time.sleep(0.2)

                    # Verify torque status
                    actual_torque, dxl_comm_result, dxl_error = self._packetHandler.read1ByteTxRx(
                        self._portHandler, motor_id, ADDR_TORQUE_ENABLE
                    )
                    if dxl_comm_result == COMM_SUCCESS and dxl_error == 0 and actual_torque == target_state:
                        if should_enable_torque:
                            print(f"Torque enabled for motor {motor_id} (torque control)")
                        else:
                            print(f"Torque disabled for motor {motor_id}")
                        break
                    time.sleep(0.5)
                else:
                    print(f"Failed to set torque state for motor {motor_id} after {max_retries} attempts")
                    self._responsive_motors.discard(motor_id)  # Mark as non-responsive

            self._torque_enabled = enable

    def get_joints(self) -> np.ndarray:
        """Get current joint angles in radians with communication health check"""
        
        if not self.is_communication_healthy():
            print("WARNING: Joint data may be stale due to communication issues")
        
        if self._joint_angles_raw is None:
            return np.zeros(len(self._ids))
        
        with self._lock:
            _j = self._joint_angles_raw.copy()
        
        # Wrap positions to 0-4095 range (one revolution)
        _j = _j % 4096

        # Convert to radians
        return _j / 4096.0 * 2 * np.pi


    def _convert_raw_current_to_torque(self, motor_id: int, raw_current: int) -> float:
        """Convert raw current reading to torque in Nm using ROBOTIS specs"""
        # At this point, raw current is already signed based on the handling of the groupSyncRead, no conversion to signed
        
        # X-series current unit: 2.69 mA per unit (from ROBOTIS e-Manual)
        current_A = raw_current * 2.69e-3
        
        # Get motor model from config
        config = self.motor_config[motor_id]
        model = config['model']

        # Official torque constants from ROBOTIS specifications (Nm/A)
        if model == 'XM540':
            torque_constant = 2.4  # Nm/A
        elif model == 'XM430':
            torque_constant = 1.57  # Nm/A  
        elif model == 'XL430':
            torque_constant = 1.15  # Nm/A
        else:
            torque_constant = 1.79  # Default fallback
        
        # Convert current to torque: tau = K_t x I
        torque_nm = current_A * torque_constant
        
        return torque_nm


    def get_torques(self) -> np.ndarray:
        """Get current actual torques in Nm with communication health check"""
        
        # Check data freshness
        if not self.is_communication_healthy():
            print("WARNING: Torque data may be stale due to communication issues")
        
        # Fallback if current data not yet initialized
        if self._joint_currents_raw is None:
            return np.zeros(len(self._ids))
        
        # Copy raw current values
        with self._lock:
            _c_raw = self._joint_currents_raw.copy()
        
        # Convert raw currents to torques for each motor
        _torques_nm = np.zeros(len(self._ids))
        for i, motor_id in enumerate(self._ids):
            _torques_nm[i] = self._convert_raw_current_to_torque(motor_id, _c_raw[i])
        
        return _torques_nm

    def get_torque_by_id(self, motor_id: int) -> float:
        """Get actual torque for specific motor ID"""
        try:
            idx = self._ids.index(motor_id)
            return self.get_torques()[idx]
        except ValueError:
            raise ValueError(f"Motor ID {motor_id} not found")
    
    def get_velocities(self) -> np.ndarray:
        """Get current joint velocities in rad/s with communication health check"""
        
        # Check data freshness
        if not self.is_communication_healthy():
            print("WARNING: Velocity data may be stale due to communication issues")
            
        while self._joint_velocities_raw is None:
            time.sleep(0.1)
        
        # Convert raw values to rad/s at read time
        with self._lock:
            _v_raw = self._joint_velocities_raw.copy()
        
        # Apply conversion for each velocity
        _v_rad_per_sec = np.zeros(len(self._ids))
        for i, raw_vel in enumerate(_v_raw):
            _v_rad_per_sec[i] = self._convert_raw_velocity_to_rad_per_sec(raw_vel)
        
        return _v_rad_per_sec

    def get_joint_by_id(self, motor_id: int) -> float:
        """Get joint angle for specific motor ID"""
        try:
            idx = self._ids.index(motor_id)
            return self.get_joints()[idx]
        except ValueError:
            raise ValueError(f"Motor ID {motor_id} not found")

    def get_velocity_by_id(self, motor_id: int) -> float:
        """Get joint velocity for specific motor ID"""
        try:
            idx = self._ids.index(motor_id)
            return self.get_velocities()[idx]
        except ValueError:
            raise ValueError(f"Motor ID {motor_id} not found")

    def close(self):
        """Close the driver and cleanup"""

        # Stop reading thread
        self._stop_thread.set()
        if hasattr(self, '_reading_thread'):
            self._reading_thread.join()
        
        # Disable torque for all motors
        try:
            self.enable_torque(False)
        except Exception as e:
            print(f"Error disabling torque: {e}")
        
        # Close port
        if hasattr(self, '_portHandler'):
            self._portHandler.closePort()
        
        print("Enhanced DynamixelDriver shutdown complete")


class FakeDynamixelDriver(DynamixelDriverProtocol):
    """Fake driver for testing GELLO methods without hardware"""
    def __init__(self, motor_config):
        # Allow passing either a list of IDs or a dict like real driver
        if isinstance(motor_config, dict):
            self._ids = list(motor_config.keys())
        else:
            self._ids = motor_config
        self.motor_config = {i: {"control": "position", "offset": 0.0} for i in self._ids}
        
        self._joint_angles_raw = np.zeros(len(self._ids))  # raw counts
        self._joint_velocities_raw = np.zeros(len(self._ids))
        # Start with torque off (matches hardware bring-up; tests expect False until enabled)
        self._torque_enabled = False
        self._lock = Lock()
        
        # Add health monitoring for fake driver too
        self._comm_healthy = True
        self._last_successful_read = time.time()
        
        ##print(f"FakeDynamixelDriver initialized with {len(self._ids)} motors")

    def get_communication_status(self) -> dict:
        """Fake communication status - always healthy"""
        return {
            'communication_healthy': True,
            'last_successful_read': self._last_successful_read,
            'time_since_success': 0.0,
            'consecutive_failures': 0,
            'data_age_ms': 0.0,
            'data_fresh': True,
            'responsive_motors': self._ids,
            'total_motors': len(self._ids)
        }

    def is_communication_healthy(self) -> bool:
        """Fake driver is always healthy"""
        return True

    def get_joints(self) -> np.ndarray:
        with self._lock:
            self._last_successful_read = time.time()
            return self._joint_angles_raw / 2048.0 * np.pi

    def get_velocities(self) -> np.ndarray:
        with self._lock:
            return self._joint_velocities_raw.copy()

    def set_joints(self, joint_angles: Sequence[float]):
        if len(joint_angles) != len(self._ids):
            raise ValueError(
                f"expected {len(self._ids)} joint angles, got {len(joint_angles)}"
            )
        if not self._torque_enabled:
            raise RuntimeError("torque is disabled; enable torque before set_joints")
        with self._lock:
            for i, motor_id in enumerate(self._ids):
                if self.motor_config[motor_id]["control"] == "position":
                    # Convert radians to raw counts
                    self._joint_angles_raw[i] = int(joint_angles[i] / np.pi * 2048)

    def set_torques(self, torques: Sequence[float]):
        # Fake driver ignores torque but allows torque_enabled check
        pass

    def torque_enabled(self) -> bool:
        return self._torque_enabled

    def enable_torque(self, enable: bool):
        self._torque_enabled = enable

    def close(self):
        pass