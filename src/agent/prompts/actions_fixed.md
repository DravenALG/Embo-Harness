- act({{ACTION_PARAMETERS}}): {{ACTION_DESCRIPTION}}
  Each action takes {{ACTION_CHUNK_SIZE}} control frames ({{ACTION_CHUNK_SIZE}} * control_dt seconds).
  If fewer than {{ACTION_CHUNK_SIZE}} frames remain, the motion is interpolated to reach the target
  within remaining_steps. Do not supply a steps argument.
  Hold gripper at 0 (open) or 1 (closed) throughout; observed closedness
  may be anywhere in [0, 1].
  Returns the new observation and actual executed_steps, which may be fewer
  if the episode ends early.
