#!/usr/bin/env python3
"""
Inverse Dynamics
"""
import numpy as np
import PyKDL as kdl
from urdf_parser_py.urdf import URDF
from typing import Sequence

from modpack.modules.gello.utils.urdf_utils import preprocess_urdf_replace_package_paths, kdl_tree_from_urdf_model
from modpack.utils import DebugPrinter

class InverseDynamicsCalculator:
    """Handles inverse dynamics computation using KDL"""
    
    def __init__(self, urdf_path: str, base_link: str, tip_link: str, debug: bool = False):
        self.debug = debug
        self.debug_printer = DebugPrinter(enabled=debug)
        # Load and process URDF
        processed_urdf_path = preprocess_urdf_replace_package_paths(urdf_path)
        with open(processed_urdf_path, 'rb') as f:
            self.robot = URDF.from_xml_string(f.read())

        # Create KDL tree and chain
        self.kdl_tree = kdl_tree_from_urdf_model(self.robot)
        root_link = self.robot.get_root()
        self.chain = self.kdl_tree.getChain(root_link, tip_link)
        self.debug_printer.log("\n=== KDL Chain Structure ===")
        self.debug_printer.log(lambda: f"Root link: {root_link}")
        for i in range(self.chain.getNrOfSegments()):
            seg = self.chain.getSegment(i)
            joint = seg.getJoint()
            self.debug_printer.log(
                lambda i=i, seg=seg, joint=joint: (
                    f"Segment {i}: {seg.getName()}, Joint: {joint.getName()}, "
                    f"Type: {joint.getType()}, Mass: {seg.getInertia().getMass():.4f} kg"
                )
            )
        # Setup solvers
        self.gravity_vector = kdl.Vector(0, 0, -9.81)
        self.dyn_param_solver = kdl.ChainDynParam(self.chain, self.gravity_vector)
        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)
        for i in range(self.chain.getNrOfSegments()):
            seg = self.chain.getSegment(i)
            #print(f"Segment {i}: {seg.getName()}, Joint: {seg.getJoint().getName()}, Mass: {seg.getInertia().getMass():.4f} kg")
        # Pre-allocate arrays
        self.n_joints = self.chain.getNrOfJoints()
        self.q_kdl = kdl.JntArray(self.n_joints)
        self.qdot_kdl = kdl.JntArray(self.n_joints)
        self.gravity_torques = kdl.JntArray(self.n_joints)
        self.coriolis_torques = kdl.JntArray(self.n_joints)
        self.mass_matrix = kdl.JntSpaceInertiaMatrix(self.n_joints)
        
        self.debug_printer.log("\n=== KDL Joint Axes (from KDL chain) ===")
        for i in range(self.chain.getNrOfSegments()):
            seg = self.chain.getSegment(i)
            joint = seg.getJoint()
            
            if joint.getType() != joint.Fixed:
                axis = joint.JointAxis()
                origin = joint.JointOrigin()
                
                self.debug_printer.log(lambda i=i, joint=joint: f"KDL Joint {i} ('{joint.getName()}'):")
                self.debug_printer.log(
                    lambda axis=axis: f"  Axis: [{axis.x():.6f}, {axis.y():.6f}, {axis.z():.6f}]"
                )
                self.debug_printer.log(
                    lambda origin=origin: f"  Origin: [{origin.x():.6f}, {origin.y():.6f}, {origin.z():.6f}]"
                )
        
        self.debug_printer.log("\n=== Link Inertial Properties ===")
        for i in range(self.chain.getNrOfSegments()):
            seg = self.chain.getSegment(i)
            inertia = seg.getInertia()
            
            mass = inertia.getMass()
            cog = inertia.getCOG()
            
            self.debug_printer.log(lambda i=i, seg=seg: f"Link {i} ('{seg.getName()}'):")
            self.debug_printer.log(lambda mass=mass: f"  Mass: {mass:.6f} kg")
            self.debug_printer.log(
                lambda cog=cog: f"  COG: [{cog.x():.6f}, {cog.y():.6f}, {cog.z():.6f}]"
            )


    def compute_feedforward_torque(
        self,
        current_positions: np.ndarray,
        current_velocities: np.ndarray,
        kp_gains: Sequence[float] | np.ndarray,
        kd_gains: Sequence[float] | np.ndarray,
        actual_torques: np.ndarray = None,
    ) -> np.ndarray:
        """Compute feedforward torque with PD control and optional RLS orientation update"""

        n_actuated = min(self.n_joints, len(current_positions))
        torque_commands = np.zeros(n_actuated)

        # Convert to KDL arrays
        for i in range(n_actuated):
            self.q_kdl[i] = current_positions[i]
            self.qdot_kdl[i] = 0.0
            
        # Compute dynamics
        ret_gravity = self.dyn_param_solver.JntToGravity(self.q_kdl, self.gravity_torques)

        if ret_gravity != 0:
            print("Warning: Dynamics computation failed")
            return np.zeros(n_actuated)
        
        # Compute control torques for actuated joints
        for i in range(n_actuated-1):
            # Base link is included in computation, which we need to skip. TODO add this to config, or skip base link in code
            gravity_term = self.gravity_torques[i+1]
            coriolis_term = self.coriolis_torques[i]
            
            if abs(coriolis_term) > 50.0:
                coriolis_term = 0.0
            
            velocity_error = current_velocities[i] if i < len(current_velocities) else 0.0
            kp = kp_gains[i] if i < len(kp_gains) else 0.0
            kd = kd_gains[i] if i < len(kd_gains) else 0.0
            total_torque = (gravity_term + kd * velocity_error)
            
            torque_commands[i] = total_torque
            # Empirically, motor 4 is flipped sign.
            if i == 3:
                torque_commands[i] *=-1

        return torque_commands

