moves by seven delta_joint_positions in radians,
  ordered panda_joint1 through panda_joint7, matching robot_state.joint_positions.
  At actual execution start, q_target = q_measured + delta_joint_positions.
  Joint angles are linearly interpolated to that target without angle wrapping.
  These are total changes over the whole action, never repeated per-frame increments.
  During asynchronous inference, the baseline is the measured state after the
  previous tail finishes, not the earlier observation used for planning.
  Gripper is a separate absolute command, not a delta.
