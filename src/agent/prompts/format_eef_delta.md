moves the Robotiq base_link (mount flange) by a total
  delta_position [dx, dy, dz] in meters and delta_rotation [rx, ry, rz] in radians.
  Both are expressed in the robot-root frame, not the gripper's local frame.
  delta_rotation is an axis-angle rotation vector: its direction is the axis
  and its norm is the angle, not Euler angles. A zero vector means no rotation.
  At actual execution start, p_target = p_measured + delta_position and
  R_target = Exp(delta_rotation) * R_measured. Position is linearly interpolated
  and orientation follows shortest-path SLERP to that target. These are total
  changes over the whole action, never repeated per-frame increments.
  During asynchronous inference, the baseline is the measured state after the
  previous tail finishes, not the earlier observation used for planning.
