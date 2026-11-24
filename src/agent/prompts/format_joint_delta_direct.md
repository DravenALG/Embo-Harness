moves by one seven-joint increment per listed frame in radians, ordered panda_joint1
  through panda_joint7. Starting from the measured execution state, each
  `delta_joint_positions` is accumulated from the preceding frame without angle
  wrapping. Gripper is an absolute per-frame command. No interpolation is applied.
