moves the Robotiq base_link (mount flange) by one robot-root-frame increment per
  listed frame. `delta_position` is [dx, dy, dz] in meters and `delta_rotation` is
  an axis-angle vector [rx, ry, rz] in radians, not Euler angles. Starting from the
  measured execution state, each increment is accumulated from the preceding frame
  with R_next = Exp(delta_rotation) * R_previous. Gripper is an absolute
  per-frame command. No interpolation is applied.
