moves to seven absolute joint_positions in radians,
  ordered panda_joint1 through panda_joint7, matching robot_state.joint_positions.
  Joint angles are linearly interpolated from the measured joints at actual
  execution start, without angle wrapping. Gripper is a separate absolute command.
