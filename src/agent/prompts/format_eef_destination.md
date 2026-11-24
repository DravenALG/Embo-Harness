Move toward an absolute final Robotiq base_link pose in the robot-root frame:
`position` is xyz meters and `quaternion` is wxyz. The harness computes the
number of interpolation frames from the measured pose, the control period, and
the configured limits ({{EEF_DESTINATION_TRANSLATION_SPEED}} m/s translation and
{{EEF_DESTINATION_ROTATION_SPEED}} rad/s rotation). It executes at most
`{{ACTION_CHUNK_SIZE}}` frames, then returns a new observation so you can
correct this target or choose another one. Do not provide `steps`; gripper is
held at the requested value for the executed prefix.
