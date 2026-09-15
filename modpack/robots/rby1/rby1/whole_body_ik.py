"""RBY1 Whole-Body IK solver using Mink library.

This solver optimizes both base movement and joint positions to reach end-effector targets,
while maintaining stability and upright posture constraints.
"""
import copy
from pathlib import Path
from typing import Optional, Tuple, Dict

import mujoco
import numpy as np
import yaml

import mink
from mink import Limit, Constraint

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)

class FreeJointVelocityLimit(Limit):
    model: mujoco.MjModel
    ang_max: np.ndarray
    lin_max: np.ndarray
    indices: np.ndarray | None = None
    limit:   np.ndarray | None = None
    P:       np.ndarray | None = None

    def __init__(self, model: mujoco.MjModel,
                 joint_id: int,
                 ang_max=(np.inf, np.inf, np.inf),
                 lin_max=(np.inf, np.inf, np.inf)):
        object.__setattr__(self, "model", model)
        ang_max = np.asarray(ang_max, dtype=float).reshape(3)
        lin_max = np.asarray(lin_max, dtype=float).reshape(3)
        object.__setattr__(self, "ang_max", ang_max)
        object.__setattr__(self, "lin_max", lin_max)

        indices = []
        vmaxs   = []
        assert model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE, "Joint must be free"
        dof0 = model.jnt_dofadr[joint_id]
        idx = np.arange(dof0, dof0 + 6, dtype=int)   # [wx, wy, wz, vx, vy, vz]
        indices.append(idx)
        vmaxs.append(np.concatenate([lin_max, ang_max], axis=0))
        # vmaxs.append(np.concatenate([ang_max, lin_max], axis=0))

        indices = np.concatenate(indices, axis=0)
        vmaxs   = np.concatenate(vmaxs,   axis=0)

        P = np.zeros((indices.size, model.nv), dtype=float)
        P[np.arange(indices.size), indices] = 1.0

        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "limit",   vmaxs)
        object.__setattr__(self, "P",       P)

    def compute_qp_inequalities(self, configuration, dt: float) -> Constraint:
        if self.P is None or self.P.shape[0] == 0:
            return Constraint(None, None)
        vmax_dt = self.limit * float(dt)
        G = np.vstack(( self.P, -self.P ))
        h = np.hstack(( vmax_dt,  vmax_dt ))
        return Constraint(G, h)

class RBY1WholeBodyIK:
    """Whole-body IK solver for RBY1 robot using Mink optimization library.
    
    This solver handles:
    - Base movement (X, Y, theta) - optimized together with joints
    - 6 DOF torso chain
    - Dual 7 DOF arms
    
    Optimization priorities:
    1. End-effector target positions and orientations (highest priority)
    2. Base Z stays on ground (hard constraint)
    3. Upper body upright orientation (weak regularization)
    4. COM stability within base support polygon (medium regularization)
    """
    
    def __init__(self, config_path: str = PROJECT_ROOT + "/config/wbik.yaml"):
        """Initialize RBY1 whole-body IK solver."""
        try:
            config_path = Path(config_path)
            with config_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        except Exception as e:
            raise Exception(f"Exception while loading IK config file: {e}")
        if not isinstance(cfg, dict):
            raise ValueError(f"IK config at {config_path} must be a mapping.")

        def require(name: str):
            if name not in cfg:
                raise KeyError(f"Missing required IK config key: {name}")
            return cfg[name]

        self.model_path = PROJECT_ROOT + str(require("model_path"))
        self.ee_pos_cost = float(require("ee_pos_cost"))
        self.ee_ori_cost = float(require("ee_ori_cost"))
        self.head_pos_cost = float(require("head_pos_cost"))
        self.head_ori_cost = np.asarray(require("head_ori_cost"), dtype=float)
        self.torso_upright_ori_cost = float(require("torso_upright_ori_cost"))
        self.nominal_posture_cost_main = float(require("nominal_posture_cost_main"))
        self.nominal_posture_cost_head = float(require("nominal_posture_cost_head"))
        self.current_posture_cost_main = float(require("current_posture_cost_main"))
        self.current_posture_cost_head = float(require("current_posture_cost_head"))
        self.com_over_base_pos_cost = float(require("com_over_base_pos_cost"))
        self.com_target_height = float(require("com_target_height"))
        self.base_ground_position_cost = np.asarray(require("base_ground_position_cost"), dtype=float)
        self.base_ground_orientation_cost = np.asarray(require("base_ground_orientation_cost"), dtype=float)
        self.nominal_torso_angles = np.asarray(require("nominal_torso_rad"), dtype=float)
        self.nominal_right_arm_angles = np.asarray(require("nominal_right_arm_rad"), dtype=float)
        self.nominal_left_arm_angles = np.asarray(require("nominal_left_arm_rad"), dtype=float)
        self.nominal_head_angles = np.asarray(require("nominal_head_rad"), dtype=float)
        self.safety_distance = float(require("safety_distance"))
        self.influence_distance = float(require("influence_distance"))
        self.velocity_limit_scale = float(require("velocity_limit_scale"))
        self.base_xy_velocity_limit = float(require("base_xy_velocity_limit"))
        self.base_rz_velocity_limit = float(require("base_rz_velocity_limit"))
        self.joint_velocity_limits = copy.deepcopy(require("joint_velocity_limits"))
        for name, limit in self.joint_velocity_limits.items():
            self.joint_velocity_limits[name] = float(limit)

        self.solver = require("solver")
        self.damping = float(require("damping"))

        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        
        # Store joint indices for different parts
        self._setup_joint_indices()
        
        # Fixed unprefixed names consistent with the loaded XML
        self.base_name = "base"
        self.torso5_name = "link_torso_5"
        self.left_ee_name = "end_effector_l"
        self.right_ee_name = "end_effector_r"
        self.head_name = "head"
        self.head_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.head_name
        )
        # Wheel link names for stability check
        self.wheel_names = [
            "link_wheel_fr",
            "link_wheel_fl",
            "link_wheel_rr",
            "link_wheel_rl",
        ]

        assert len(self.torso_qpos_indices) == self.nominal_torso_angles.size
        assert len(self.right_arm_qpos_indices) == self.nominal_right_arm_angles.size
        assert len(self.left_arm_qpos_indices) == self.nominal_left_arm_angles.size
        assert len(self.head_qpos_indices) == self.nominal_head_angles.size

        self.nominal_posture_cost_vector = np.full(self.model.nv, self.nominal_posture_cost_main, dtype=float)
        for dof_idx in self.head_dof_indices:
            self.nominal_posture_cost_vector[dof_idx] = self.nominal_posture_cost_head

        self.current_posture_cost_vector = np.full(self.model.nv, self.current_posture_cost_main, dtype=float)
        for dof_idx in self.head_dof_indices:
            self.current_posture_cost_vector[dof_idx] = self.current_posture_cost_head

        self.environment_geoms = None
        # Limits cache (built once and reused to avoid per-iteration overhead)
        self._cached_limits = None
        self._cached_tasks = None
        self.configuration = mink.Configuration(self.model, self.data.qpos.copy())
        self._build_limits_cache()
        self._build_tasks_cache()
        self._build_reusable_tasks()
    
    def _setup_joint_indices(self):
        """Setup joint indices for different robot parts."""
        # Base joint (now controlled by IK)
        self.base_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "world_j")
        self.base_qpos_indices = [0, 1, 2]  # X, Y, Z positions
        self.base_quat_indices = [3, 4, 5, 6]  # Quaternion (w, x, y, z)
        
        # Wheel joints (not modified by IK)
        self.wheel_joint_names = [
            "wheel_fr",
            "wheel_fl",
            "wheel_rr",
            "wheel_rl",
        ]
        self.wheel_qpos_indices = []
        for name in self.wheel_joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id >= 0:
                qpos_adr = self.model.jnt_qposadr[joint_id]
                self.wheel_qpos_indices.append(qpos_adr)
        
        # Torso joints (controlled by IK)
        self.torso_joint_names = [f"torso_{i}" for i in range(6)]
        self.torso_qpos_indices = []
        for name in self.torso_joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id >= 0:
                qpos_adr = self.model.jnt_qposadr[joint_id]
                self.torso_qpos_indices.append(qpos_adr)
        
        # Left arm joints (controlled by IK)
        self.left_arm_joint_names = [f"left_arm_{i}" for i in range(7)]
        self.left_arm_qpos_indices = []
        for name in self.left_arm_joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id >= 0:
                qpos_adr = self.model.jnt_qposadr[joint_id]
                self.left_arm_qpos_indices.append(qpos_adr)
        
        # Right arm joints (controlled by IK)
        self.right_arm_joint_names = [f"right_arm_{i}" for i in range(7)]
        self.right_arm_qpos_indices = []
        for name in self.right_arm_joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id >= 0:
                qpos_adr = self.model.jnt_qposadr[joint_id]
                self.right_arm_qpos_indices.append(qpos_adr)

        # Head joints
        self.head_joint_names = [f"head_{i}" for i in range(2)]
        self.head_qpos_indices = []
        self.head_dof_indices = []
        for name in self.head_joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_adr = self.model.jnt_qposadr[joint_id]
            self.head_qpos_indices.append(qpos_adr)
            dof_adr = self.model.jnt_dofadr[joint_id]
            self.head_dof_indices.append(dof_adr)
        
        # All IK-controlled indices (including base now)
        self.ik_controlled_indices = (
            self.base_qpos_indices +  # Base X, Y, Z
            self.torso_qpos_indices + 
            self.left_arm_qpos_indices + 
            self.right_arm_qpos_indices +
            self.head_qpos_indices
        )

    def _get_nominal_posture(self, base_qpos: np.ndarray) -> np.ndarray:
        """Return a copy of qpos with torso and arm joints set to nominal angles."""
        reference = base_qpos.copy()
        for idx, angle in zip(self.torso_qpos_indices, self.nominal_torso_angles):
            reference[idx] = angle
        for idx, angle in zip(self.left_arm_qpos_indices, self.nominal_left_arm_angles):
            reference[idx] = angle
        for idx, angle in zip(self.right_arm_qpos_indices, self.nominal_right_arm_angles):
            reference[idx] = angle
        for idx, angle in zip(self.head_qpos_indices, self.nominal_head_angles):
            reference[idx] = angle
        return reference
    
    def solve(
        self,
        left_target_pos: Optional[np.ndarray] = None,
        left_target_quat: Optional[np.ndarray] = None,
        right_target_pos: Optional[np.ndarray] = None,
        right_target_quat: Optional[np.ndarray] = None,
        head_target_pos: Optional[np.ndarray] = None,
        head_target_quat: Optional[np.ndarray] = None,
        current_qpos: Optional[np.ndarray] = None,
        dt: float = 1e-3,
    ) -> Tuple[np.ndarray, np.ndarray, bool, Dict]:
        """Solve whole-body IK for given end-effector targets.
        
        The solver optimizes base position (X, Y, theta) along with joint positions
        to reach the targets while maintaining stability constraints.
        
        Args:
            left_target_pos: Left end effector target position (3D)
            left_target_quat: Left end effector target orientation (quaternion wxyz)
            right_target_pos: Right end effector target position (3D)
            right_target_quat: Right end effector target orientation (quaternion wxyz)
            head_target_pos: Head target position (3D)
            head_target_quat: Head target orientation (quaternion wxyz)
            current_qpos: Current joint positions (if None, uses data.qpos)
            dt: Integration timestep (default 1ms)
        Returns:
            solution_qpos: Optimized generalized positions (base + joints).
            solution_qvel: Generalized velocity used for the integration step.
            success: True if the solver found a feasible solution.
            info_dict: Additional diagnostics.
        """
        # Use current configuration if not provided
        if current_qpos is None:
            current_qpos = self.data.qpos.copy()
        
        # Update MuJoCo data with initial configuration
        self.data.qpos[:] = current_qpos
        mujoco.mj_forward(self.model, self.data)
        
        # Reuse the persistent Mink configuration
        configuration = self.configuration
        configuration.update(q=current_qpos)
        
        # Create task and limit list
        tasks = []
        limits = []
        
        # End-effector tasks (highest priority)
        if left_target_pos is not None:
            left_ee_task = self._left_ee_task
            left_ee_task.set_position_cost(self.ee_pos_cost)
            if left_target_quat is not None:
                left_ee_task.set_orientation_cost(self.ee_ori_cost)
                target_quat = left_target_quat
            else:
                left_ee_task.set_orientation_cost(0.0)
                target_quat = np.array([1, 0, 0, 0])

            target_matrix = self._pose_to_matrix(left_target_pos, target_quat)
            left_ee_task.set_target(mink.SE3.from_matrix(target_matrix))
            tasks.append(left_ee_task)
        
        if right_target_pos is not None:
            right_ee_task = self._right_ee_task
            right_ee_task.set_position_cost(self.ee_pos_cost)
            if right_target_quat is not None:
                right_ee_task.set_orientation_cost(self.ee_ori_cost)
                target_quat = right_target_quat
            else:
                right_ee_task.set_orientation_cost(0.0)
                target_quat = np.array([1, 0, 0, 0])

            target_matrix = self._pose_to_matrix(right_target_pos, target_quat)
            right_ee_task.set_target(mink.SE3.from_matrix(target_matrix))
            tasks.append(right_ee_task)

        head_pos_specified = head_target_pos is not None
        head_quat_specified = head_target_quat is not None
        head_target_pos_used = None
        head_target_quat_used = None

        if head_pos_specified or head_quat_specified:
            head_task = self._head_task
            head_task.set_position_cost(self.head_pos_cost if head_pos_specified else 0.0)
            head_task.set_orientation_cost(self.head_ori_cost if head_quat_specified else 0.0)
            current_head_pos = self.data.site_xpos[self.head_site_id].copy()
            current_head_quat = np.zeros(4)
            mujoco.mju_mat2Quat(current_head_quat, self.data.site_xmat[self.head_site_id])

            head_target_pos_used = head_target_pos if head_pos_specified else current_head_pos
            head_target_quat_used = head_target_quat if head_quat_specified else current_head_quat

            target_matrix = self._pose_to_matrix(head_target_pos_used, head_target_quat_used)
            
            head_task.set_target(mink.SE3.from_matrix(target_matrix))
            tasks.append(head_task)
        
        # Base ground constraint (very high priority - base must stay on ground)
        # Constrain base Z position to 0 and only allow yaw rotation
        base_ground_task = self._base_ground_task
        # Set target to current X,Y but Z=0 and upright orientation with current yaw
        base_target_matrix = np.eye(4)
        base_target_matrix[0, 3] = current_qpos[0]  # Current X
        base_target_matrix[1, 3] = current_qpos[1]  # Current Y  
        base_target_matrix[2, 3] = 0.0  # Z must be 0 (ground)
        # Extract yaw from current quaternion and create upright rotation with that yaw
        current_quat = current_qpos[3:7]
        yaw = np.arctan2(2*(current_quat[0]*current_quat[3] + current_quat[1]*current_quat[2]),
                         1 - 2*(current_quat[2]**2 + current_quat[3]**2))
        # Create rotation matrix for yaw-only rotation
        c_yaw = np.cos(yaw)
        s_yaw = np.sin(yaw)
        base_target_matrix[:3, :3] = np.array([
            [c_yaw, -s_yaw, 0],
            [s_yaw, c_yaw, 0],
            [0, 0, 1]
        ])
        base_ground_task.set_target(mink.SE3.from_matrix(base_target_matrix))
        tasks.append(base_ground_task)
        
        # Nominal posture task (keep robot near reference pose)
        nominal_posture_task = self._nominal_posture_task
        nominal_posture_task.set_target(self._get_nominal_posture(current_qpos))
        tasks.append(nominal_posture_task)

        # Current posture task (acts like velocity damping)
        current_posture_task = self._current_posture_task
        current_posture_task.set_target(current_qpos.copy())
        tasks.append(current_posture_task)

        # Extend with cached tasks
        tasks.extend(self._cached_tasks)

        # Limits (cached to avoid per-call construction overhead)
        limits.extend(self._cached_limits)

        # Solver parameters
        solver = self.solver
        damping = self.damping
        
        try:
            vel = mink.solve_ik(configuration, tasks, dt, solver, damping, limits=limits)
            solution_vel = vel.copy()
            configuration.integrate_inplace(vel, dt)
            # Get solution
            solution_qpos = configuration.q.copy()
            success = True
        except mink.NoSolutionFound:
            solution_qpos = current_qpos.copy()
            solution_vel = np.zeros(self.model.nv, dtype=float)
            success = False
        else:
            if solution_vel.shape[0] != self.model.nv:
                # Ensure velocity has consistent dimension (pad if necessary).
                padded = np.zeros(self.model.nv, dtype=float)
                count = min(solution_vel.shape[0], self.model.nv)
                padded[:count] = solution_vel[:count]
                solution_vel = padded
        
        info = {
            "success": success,
            "base_position": solution_qpos[:3].copy(),
        }
        
        return solution_qpos, solution_vel, success, info
    
    def _pose_to_matrix(self, position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        """Convert position and quaternion to 4x4 transformation matrix.
        
        Args:
            position: 3D position
            quaternion: Quaternion [w, x, y, z]
            
        Returns:
            4x4 transformation matrix
        """
        matrix = np.eye(4)
        matrix[:3, :3] = self._quat_to_rotmat(quaternion)
        matrix[:3, 3] = position
        return matrix
    
    def _quat_to_rotmat(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix.
        
        Args:
            quat: Quaternion [w, x, y, z]
            
        Returns:
            3x3 rotation matrix
        """
        w, x, y, z = quat
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        return R
    
    def _get_environment_geoms(self) -> set:
        """Get all environment collision geoms (non-robot geoms).
        
        Returns:
            Set of environment geom names
        """
        if self.environment_geoms is None:
            environment_geoms = set()
            
            # Get all geoms in the model
            for geom_id in range(self.model.ngeom):
                # Only check collision geoms (skip visual geoms)
                contype = self.model.geom_contype[geom_id]
                conaffinity = self.model.geom_conaffinity[geom_id]
                
                # Skip geoms that don't participate in collisions
                if contype == 0 or conaffinity == 0:
                    continue
                    
                geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)

                if geom_name is None or geom_name == 'floor':
                    continue
                    
                # Get the body this geom belongs to
                body_id = self.model.geom_bodyid[geom_id]
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                
                if body_name is None:
                    continue
                
                # Check if it's a known robot body name (model uses unprefixed names)
                robot_body_names = [
                    "base", "wheel_fr_link", "wheel_fl_link", "wheel_rr_link", "wheel_rl_link",
                    "link_torso_0", "link_torso_1", "link_torso_2", "link_torso_3",
                    "link_torso_4", "link_torso_5", "link_head_1", "link_head_2",
                    "link_right_arm_0", "link_right_arm_1", "link_right_arm_2", "link_right_arm_3",
                    "link_right_arm_4", "link_right_arm_5", "link_right_arm_6", "FT_SENSOR_R", "EE_BODY_R",
                    "link_left_arm_0", "link_left_arm_1", "link_left_arm_2", "link_left_arm_3",
                    "link_left_arm_4", "link_left_arm_5", "link_left_arm_6", "FT_SENSOR_L", "EE_BODY_L"
                ]
                is_robot_body = body_name in robot_body_names
                
                # If it's not a robot body, it's an environment collision geom
                if not is_robot_body:
                    environment_geoms.add(geom_name)
            
            self.environment_geoms = environment_geoms

        return self.environment_geoms

    # Namespace detection/resolution not needed; model uses unprefixed names exclusively

    def _build_limits_cache(self) -> None:
        """Build once and cache all Mink limits to avoid per-solve construction overhead."""
        # Collision avoidance limits using Mink's built-in functionality
        base_group = {"base_col_0", "base_col_1"}

        torso_0_group = {"torso_0_col_0", "torso_0_col_1"}
        torso_1_group = {"torso_1_col_0", "torso_1_col_1", "torso_1_col_2", "torso_1_col_3", "torso_1_col_4", "torso_1_col_5", "torso_1_col_6", "torso_1_col_7", "torso_1_col_8", "torso_1_col_9", "torso_1_col_10"}
        torso_2_group = {"torso_2_col_0", "torso_2_col_1", "torso_2_col_2", "torso_2_col_3", "torso_2_col_4", "torso_2_col_5", "torso_2_col_6", "torso_2_col_7", "torso_2_col_8", "torso_2_col_9", "torso_2_col_10"}
        torso_4_group = {"torso_4_col_0", "torso_4_col_1", "torso_4_col_2", "torso_4_col_3"}
        torso_5_group = {"torso_5_col_0", "torso_5_col_1", "torso_5_col_2", "torso_5_col_3", "torso_5_col_4"}
        head_group = {"head_col_0"}

        right_arm_0_group = {"right_arm_0_col_0", "right_arm_0_col_1", "right_arm_0_col_2"}
        right_arm_1_group = {"right_arm_1_col_0"}
        right_arm_2_group = {"right_arm_2_col_0", "right_arm_2_col_1", "right_arm_2_col_2", "right_arm_2_col_3", "right_arm_2_col_4", "right_arm_2_col_5", "right_arm_2_col_6", "right_arm_2_col_7"}
        right_arm_3_group = {"right_arm_3_col_0", "right_arm_3_col_1", "right_arm_3_col_2", "right_arm_3_col_3"}
        right_arm_4_group = {"right_arm_4_col_0", "right_arm_4_col_1", "right_arm_4_col_2", "right_arm_4_col_3", "right_arm_4_col_4"}
        right_arm_5_group = {"right_arm_5_col_0", "right_arm_5_col_1", "right_arm_5_col_2"}
        right_arm_6_group = {"right_arm_6_col_0"}
        right_arm_7_group = {"right_arm_7_col_0", "right_wrist_cam_col_0", "right_wrist_cam_col_1", "right_wrist_cam_col_2"}
        right_ee_group = {"right_ee_col_0", "right_ee_col_1", "right_ee_col_2", "right_ee_col_3", "right_ee_col_4", "right_finger_col_0", "right_finger_col_1"}

        left_arm_0_group = {"left_arm_0_col_0", "left_arm_0_col_1", "left_arm_0_col_2"}
        left_arm_1_group = {"left_arm_1_col_0"}
        left_arm_2_group = {"left_arm_2_col_0", "left_arm_2_col_1", "left_arm_2_col_2", "left_arm_2_col_3", "left_arm_2_col_4", "left_arm_2_col_5", "left_arm_2_col_6", "left_arm_2_col_7"}
        left_arm_3_group = {"left_arm_3_col_0", "left_arm_3_col_1", "left_arm_3_col_2", "left_arm_3_col_3"}
        left_arm_4_group = {"left_arm_4_col_0", "left_arm_4_col_1", "left_arm_4_col_2", "left_arm_4_col_3", "left_arm_4_col_4"}
        left_arm_5_group = {"left_arm_5_col_0", "left_arm_5_col_1", "left_arm_5_col_2"}
        left_arm_6_group = {"left_arm_6_col_0"}
        left_arm_7_group = {"left_arm_7_col_0", "left_wrist_cam_col_0", "left_wrist_cam_col_1", "left_wrist_cam_col_2"}
        left_ee_group = {"left_ee_col_0", "left_ee_col_1", "left_ee_col_2", "left_ee_col_3", "left_ee_col_4", "left_finger_col_0", "left_finger_col_1"}

        base_torso_group = base_group | torso_0_group | torso_1_group | torso_2_group | torso_4_group | torso_5_group | head_group
        left_arm_group = left_arm_0_group | left_arm_1_group | left_arm_2_group | left_arm_3_group | left_arm_4_group | left_arm_5_group | left_arm_6_group | left_arm_7_group | left_ee_group
        right_arm_group = right_arm_0_group | right_arm_1_group | right_arm_2_group | right_arm_3_group | right_arm_4_group | right_arm_5_group | right_arm_6_group | right_arm_7_group | right_ee_group

        # Environment collision group - all robot collision geoms
        # robot_collision_group = base_torso_group | left_arm_group | right_arm_group

        # Get environment collision geoms (non-robot geoms)
        # environment_geom_group = self._get_environment_geoms()

        geom_pairs = [
            (base_torso_group, left_arm_group),
            (base_torso_group, right_arm_group),
            (left_arm_group, right_arm_group),
            # (robot_collision_group, environment_geom_group),
        ]

        collision_avoidance_limit = mink.CollisionAvoidanceLimit(
            model=self.model,
            geom_pairs=geom_pairs,
            minimum_distance_from_collisions=self.safety_distance,
            collision_detection_distance=self.influence_distance,
        )

        # Configuration limit
        configuration_limit = mink.ConfigurationLimit(self.model)

        # Joint velocity limit
        joint_velocity_limits = copy.deepcopy(self.joint_velocity_limits)
        for name, limit in joint_velocity_limits.items():
            joint_velocity_limits[name] = float(limit) * self.velocity_limit_scale
        joint_velocity_limit = mink.VelocityLimit(self.model, joint_velocity_limits)

        # Base velocity limit
        lin_max = [self.base_xy_velocity_limit * self.velocity_limit_scale,
                   self.base_xy_velocity_limit * self.velocity_limit_scale,
                   0]
        ang_max = [0, 0, self.base_rz_velocity_limit * self.velocity_limit_scale]
        free_joint_velocity_limit = FreeJointVelocityLimit(
            self.model,
            self.base_joint_id,
            ang_max=ang_max,
            lin_max=lin_max,
        )

        self.collision_avoidance_limit = collision_avoidance_limit
        self.configuration_limit = configuration_limit
        self.joint_velocity_limit = joint_velocity_limit
        self.base_velocity_limit = free_joint_velocity_limit

        self._cached_limits = [
            self.collision_avoidance_limit,
            self.configuration_limit,
            self.joint_velocity_limit,
            self.base_velocity_limit,
        ]

    def _build_tasks_cache(self) -> None:
        """Build once and cache all Mink tasks to avoid per-solve construction overhead."""
        # Upper body upright orientation (STRONG constraint for stability)
        # Constrain torso_5 link to point upward - CRITICAL for preventing falls
        torso_upright_task = mink.FrameTask(
            frame_name=self.torso5_name,
            frame_type="body",
            position_cost=0.0,  # Don't constrain position
            orientation_cost=[self.torso_upright_ori_cost, self.torso_upright_ori_cost, 0],  # STRONG constraint to maintain upright posture
            lm_damping=1e-4,
        )
        # Set target to upright orientation (identity rotation)
        upright_matrix = np.eye(4)
        upright_matrix[:3, 3] = [0, 0, 1.0]  # Dummy position (not used due to position_cost=0)
        torso_upright_task.set_target(mink.SE3.from_matrix(upright_matrix))
        
        # COM stability constraint (medium regularization)
        # This is approximated by keeping torso_5 position within base support polygon
        # We use a relative position task between torso and base
        com_stability_task = mink.RelativeFrameTask(
            frame_name=self.torso5_name,
            frame_type="body",
            root_name=self.base_name,
            root_type="body",
            position_cost=[self.com_over_base_pos_cost, self.com_over_base_pos_cost, 0.0],
            orientation_cost=0.0,
            lm_damping=1e-4,
        )
        # Torso should be above base center with some tolerance
        relative_matrix = np.eye(4)
        relative_matrix[:3, 3] = [0, 0, self.com_target_height]  # Torso approximately 0.8m above base
        com_stability_task.set_target(mink.SE3.from_matrix(relative_matrix))

        self._cached_tasks = [
            torso_upright_task,
            com_stability_task,
        ]
        
    def _build_reusable_tasks(self) -> None:
        """Create task objects that are re-targeted each solve."""
        self._left_ee_task = mink.FrameTask(
            frame_name=self.left_ee_name,
            frame_type="site",
            position_cost=self.ee_pos_cost,
            orientation_cost=self.ee_ori_cost,
            lm_damping=1e-5,
        )
        self._right_ee_task = mink.FrameTask(
            frame_name=self.right_ee_name,
            frame_type="site",
            position_cost=self.ee_pos_cost,
            orientation_cost=self.ee_ori_cost,
            lm_damping=1e-5,
        )
        self._head_task = mink.FrameTask(
            frame_name=self.head_name,
            frame_type="site",
            position_cost=0.0,
            orientation_cost=0.0,
            lm_damping=1e-5,
        )
        self._base_ground_task = mink.FrameTask(
            frame_name=self.base_name,
            frame_type="body",
            position_cost=self.base_ground_position_cost,
            orientation_cost=self.base_ground_orientation_cost,
            lm_damping=1e-6,
        )
        self._nominal_posture_task = mink.PostureTask(
            model=self.model,
            cost=self.nominal_posture_cost_vector,
        )
        self._current_posture_task = mink.PostureTask(
            model=self.model,
            cost=self.current_posture_cost_vector,
        )
        
