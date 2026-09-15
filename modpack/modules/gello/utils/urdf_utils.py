import os
import numpy as np
import xml.etree.ElementTree as ET
import PyKDL as kdl
from pathlib import Path
from urdf_parser_py.urdf import URDF
from scipy.spatial.transform import Rotation as R
from typing import Dict, Optional


def resolve_urdf_path(urdf_path: Optional[str]) -> Optional[str]:
    """Resolve URDF path: absolute, cwd-relative, or relative to repo root (parent of modpack)."""
    if not urdf_path:
        return None
    raw = os.path.expanduser(os.path.expandvars(urdf_path.strip()))
    p = Path(raw)
    if p.is_file():
        return str(p.resolve())
    repo_root = Path(__file__).resolve().parents[4]
    under_repo = (repo_root / p).resolve()
    if under_repo.is_file():
        return str(under_repo)
    return raw


def resolve_package_uri(uri, package_name='gello_description', package_path='my-robot_new'):
    if uri.startswith(f"package://{package_name}/"):
        relative_path = uri[len(f"package://{package_name}/"):]
        # Convert to absolute path
        absolute_path = os.path.abspath(os.path.join(package_path, relative_path))
        return absolute_path
    return uri


def preprocess_urdf_replace_package_paths(urdf_file, package_name='gello_description', package_path='my-robot_new'):
    tree = ET.parse(urdf_file)
    root = tree.getroot()
    for mesh in root.findall('.//mesh'):
        filename = mesh.attrib.get('filename')
        if filename and filename.startswith(f'package://{package_name}/'):
            mesh.attrib['filename'] = resolve_package_uri(filename, package_name, package_path)
    temp_urdf_path = '/tmp/temp_robot_new.urdf'
    tree.write(temp_urdf_path)
    return temp_urdf_path

def euler_to_quat(r, p, y):
    sr, sp, sy = np.sin(r/2.0), np.sin(p/2.0), np.sin(y/2.0)
    cr, cp, cy = np.cos(r/2.0), np.cos(p/2.0), np.cos(y/2.0)
    return [sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy,
            cr*cp*cy + sr*sp*sy]

def urdf_pose_to_kdl_frame(pose):
    pos = [0., 0., 0.]
    rot = [0., 0., 0.]
    if pose is not None:
        if pose.position is not None:
            pos = pose.position
        if pose.rotation is not None:
            rot = pose.rotation
    return kdl.Frame(kdl.Rotation.Quaternion(*euler_to_quat(*rot)),
                     kdl.Vector(*pos))


def urdf_inertial_to_kdl_rbi(i):
    origin = urdf_pose_to_kdl_frame(i.origin)
    rbi = kdl.RigidBodyInertia(i.mass, origin.p,
                               kdl.RotationalInertia(i.inertia.ixx,
                                                     i.inertia.iyy,
                                                     i.inertia.izz,
                                                     i.inertia.ixy,
                                                     i.inertia.ixz,
                                                     i.inertia.iyz))
    return origin.M * rbi

def urdf_joint_to_kdl_joint(jnt):
    origin_frame = urdf_pose_to_kdl_frame(jnt.origin)
    if jnt.joint_type == 'fixed':
        return kdl.Joint(jnt.name, kdl.Joint.Fixed)
    axis = kdl.Vector(*jnt.axis)
    if jnt.joint_type == 'revolute':
        return kdl.Joint(jnt.name, origin_frame.p,
                         origin_frame.M * axis, kdl.Joint.RotAxis)
    if jnt.joint_type == 'continuous':
        return kdl.Joint(jnt.name, origin_frame.p,
                         origin_frame.M * axis, kdl.Joint.RotAxis)
    if jnt.joint_type == 'prismatic':
        return kdl.Joint(jnt.name, origin_frame.p,
                         origin_frame.M * axis, kdl.Joint.TransAxis)
    print("Unknown joint type: %s." % jnt.joint_type)
    return kdl.Joint(jnt.name, kdl.Joint.Fixed)

def kdl_tree_from_urdf_model(urdf):
    root = urdf.get_root()
    tree = kdl.Tree(root)
    def add_children_to_tree(parent):
        if parent in urdf.child_map:
            for joint, child_name in urdf.child_map[parent]:
                child = urdf.link_map[child_name]
                if child.inertial is not None:
                    kdl_inert = urdf_inertial_to_kdl_rbi(child.inertial)
                else:
                    kdl_inert = kdl.RigidBodyInertia()
                kdl_jnt = urdf_joint_to_kdl_joint(urdf.joint_map[joint])
                kdl_origin = urdf_pose_to_kdl_frame(urdf.joint_map[joint].origin)
                kdl_sgm = kdl.Segment(child_name, kdl_jnt,
                                      kdl_origin, kdl_inert)
                tree.addSegment(kdl_sgm, parent)
                add_children_to_tree(child_name)
    add_children_to_tree(root)
    return tree


# ── Extract URDF joint axes ──────────────────────────────────────────────────
def get_urdf_axes(robot: URDF):
    base_link = robot.get_root()
    def compute_transform(link_name, parent=np.eye(4)):
        transforms = {}
        children = [j for j in robot.joints if j.parent == link_name]
        for joint in children:
            origin_xyz = joint.origin.xyz
            origin_rot = joint.origin.rpy
            xyz = origin_xyz if origin_xyz is not None else [0,0,0]
            Rmat = R.from_euler('xyz', origin_rot).as_matrix() if origin_rot is not None else np.eye(3)
            T = np.eye(4)
            T[:3,:3] = Rmat
            T[:3,3] = xyz
            T_world = parent @ T
            transforms[joint.name] = T_world
            transforms.update(compute_transform(joint.child, T_world))
        return transforms
    joint_transforms = compute_transform(base_link)
    axes_world = {}
    for joint in robot.joints:
        if joint.joint_type != 'revolute':
            continue
        T = joint_transforms[joint.name]
        axis = np.array(joint.axis if joint.axis is not None else [0,0,1])
        axis_world = T[:3,:3] @ axis
        axis_world /= np.linalg.norm(axis_world)
        axes_world[joint.name] = axis_world
    return axes_world

# ── Compute relative rotation between joint axis and motor axis ───────────────
def compute_relative_joint_to_motor_axes(robot: URDF) -> Dict[str, np.ndarray]:
    """
    Computes the rotation from each joint axis (in URDF) to the actual motor frame.
    Returns a dict mapping joint name -> axis in motor frame (unit vector).
    """
    axes_motor_frame = {}
    base_link = robot.get_root()

    # Recursively compute transforms from base to each joint
    def compute_transform(link_name, parent=np.eye(4)):
        transforms = {}
        children = [j for j in robot.joints if j.parent == link_name]
        for joint in children:
            # joint origin
            T = np.eye(4)
            if joint.origin is not None:
                T[:3, :3] = R.from_euler('xyz', joint.origin.rpy).as_matrix()
                T[:3, 3] = joint.origin.xyz
            T_world = parent @ T
            transforms[joint.name] = T_world
            transforms.update(compute_transform(joint.child, T_world))
        return transforms

    joint_transforms = compute_transform(base_link)

    for joint in robot.joints:
        if joint.joint_type != 'revolute':
            continue

        T_joint = joint_transforms[joint.name]
        urdf_axis = np.array(joint.axis if joint.axis is not None else [0, 0, 1])
        axis_world = T_joint[:3, :3] @ urdf_axis
        axis_world /= np.linalg.norm(axis_world)

        # Assume motor rotation axis is along local z-axis of the joint child link
        # Transform motor axis into world frame
        motor_axis_local = np.array([0, 0, 1])
        R_child = T_joint[:3, :3]  # child rotation in world
        motor_axis_world = R_child @ motor_axis_local
        motor_axis_world /= np.linalg.norm(motor_axis_world)

        # Compute relative rotation (dot product gives cosine of angle)
        cosine = np.clip(np.dot(axis_world, motor_axis_world), -1.0, 1.0)
        angle = np.arccos(cosine)
        axes_motor_frame[joint.name] = {
            'urdf_axis_world': axis_world,
            'motor_axis_world': motor_axis_world,
            'angle_rad': angle,
            'flip_needed': np.dot(axis_world, motor_axis_world) < 0
        }

    return axes_motor_frame


if __name__ == "__main__":
    # Load URDF
    urdf_path = preprocess_urdf_replace_package_paths("my-robot_new/robot.urdf")
    with open(urdf_path, 'rb') as f:
        robot = URDF.from_xml_string(f.read())

    # Build KDL tree
    tree = kdl_tree_from_urdf_model(robot)

    # Compute world axes
    axes_world = get_urdf_axes(robot)

    print("Revolute joint axes in world frame:")
    for name, axis in axes_world.items():
        print(f"Joint {name}: {axis}")

    # Compute relative axes to motor
    rel_axes = compute_relative_joint_to_motor_axes(robot)
    print("\nJoint-to-motor relative axes:")
    for name, info in rel_axes.items():
        print(f"Joint {name}: flip_needed={info['flip_needed']}, angle_deg={np.degrees(info['angle_rad']):.2f}")
        print(f"    urdf_axis_world: {info['urdf_axis_world']}")
        print(f"    motor_axis_world: {info['motor_axis_world']}")