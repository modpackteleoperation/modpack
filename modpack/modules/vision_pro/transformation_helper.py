import numpy as np
from quaternion import as_rotation_matrix, quaternion

neck_start_pose = np.array([0.35, -0.0405, 0.025, 0, 0.523599, 0])


def quaternion_camera_pose_to_extrinsic_matrix(camera_pose):
    extrinsic_matrix = np.eye(4)
    qx, qy, qz, qw = camera_pose.qx, camera_pose.qy, camera_pose.qz, camera_pose.qw
    px, py, pz = camera_pose.tx, camera_pose.ty, camera_pose.tz
    extrinsic_matrix[:3, :3] = as_rotation_matrix(quaternion(qw, qx, qy, qz))
    extrinsic_matrix[:3, -1] = [px, py, pz]
    return extrinsic_matrix


def mat2pose(matrix):
    rot = matrix[:3, :3]
    translation = matrix[:3, 3]
    theta = np.arcsin(-rot[2, 0])
    cos_theta = np.cos(theta)
    if cos_theta != 0:
        psi = np.arctan2(rot[1, 0] / cos_theta, rot[0, 0] / cos_theta)
        phi = np.arctan2(rot[2, 1] / cos_theta, rot[2, 2] / cos_theta)
    else:
        psi = 0
        phi = 0
    return [translation[0], translation[1], translation[2], np.array(psi), np.array(theta), np.array(phi)]
