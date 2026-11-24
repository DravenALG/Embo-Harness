"""RoboLab Panda joint-to-end-effector conversion used by act()."""

import numpy as np
from scipy.spatial.transform import Rotation

from src.agent.executor import finite_vector


# Extracted from third_party/RoboLab/assets/robots/franka_robotiq_2f_85_flattened.usd:
# panda_joint1..7, panda_joint8, and panda_hand_joint. After normalizing the
# USD joint-frame quaternions, the arm chain is Trans(origin) Rx(roll) Rz(q).
# Preserve the asset's dimensions and its custom Robotiq mount; a stock Panda
# hand/TCP transform (or DroidCfg's rotated eef_frame) is a different endpoint.
_JOINT_ORIGINS = (
    (0.0, 0.0, 0.33299997448921204),
    (0.0, 0.0, 0.0),
    (0.0, -0.31599998474121094, 0.0),
    (0.08249995857477188, 0.0, 0.0),
    (-0.08249994367361069, 0.38399988412857056, 0.0),
    (0.0, 0.0, 0.0),
    (0.08799993246793747, 0.0, 0.0),
)
_JOINT_ROLLS = np.array([0, -1, 1, 1, -1, 1, 1]) * (np.pi / 2)
_JOINT_ROTATIONS = Rotation.from_euler("x", _JOINT_ROLLS[:, None]).as_matrix()
_FLANGE_POSITION = np.array([0.0, 0.0, 0.10699999332427979])
# panda_hand_joint's parent and child localRot, in SciPy's xyzw order.
_MOUNT_ROTATION = (
    Rotation.from_quat([0.0, 0.0, -0.3826834559440613, 0.9238795042037964])
    * Rotation.from_quat([0.6532813906669617, 0.27059805393218994,
                          0.6532815098762512, 0.2705981731414795]).inv()
).as_matrix()
_MOUNT_POSITION = -_MOUNT_ROTATION @ np.array(
    [-0.018174022436141968, -7.564315146479927e-11, -2.115828579007939e-8])


def droid_forward_kinematics(joint_positions) -> tuple[list[float], list[float]]:
    """Return absolute robot-root xyz meters and a unit wxyz quaternion.

    Input is panda_joint1..7 in radians, in the same order as model actions and
    public proprioception. This evaluates geometry only, without simulation.
    """
    joints = finite_vector(joint_positions, 7, "joint_positions")
    position, rotation = np.zeros(3), np.eye(3)
    for origin, fixed_rotation, joint_rotation in zip(
            _JOINT_ORIGINS, _JOINT_ROTATIONS, Rotation.from_euler("z", joints[:, None]).as_matrix()):
        position += rotation @ origin
        rotation = rotation @ fixed_rotation @ joint_rotation
    position += rotation @ (_FLANGE_POSITION + _MOUNT_POSITION)
    rotation = rotation @ _MOUNT_ROTATION
    quaternion = Rotation.from_matrix(rotation).as_quat(canonical=True)[[3, 0, 1, 2]]
    return position.tolist(), quaternion.tolist()
