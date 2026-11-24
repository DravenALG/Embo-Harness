- act({{ACTION_PARAMETERS}}, steps): {{ACTION_DESCRIPTION}}
  steps is the number of control frames for this action, setting its
  duration to steps * control_dt seconds; more steps make the same motion slower.
  steps must be a positive integer <= remaining_steps.
  Adjust the target and steps together: use larger motions with more steps
  for travel, and smaller motions with fewer steps for fine manipulation.
  Hold gripper at 0 (open) or 1 (closed) throughout; observed closedness
  may be anywhere in [0, 1].
  Returns the new observation and actual executed_steps, which may be fewer
  than requested if the episode ends.
