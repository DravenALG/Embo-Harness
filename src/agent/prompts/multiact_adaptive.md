- multiact(actions): executes a nonempty list of act targets in order, starting
  each from the latest measured state. Each target has {{ACTION_PARAMETERS}},
  and steps. Total steps must not exceed remaining_steps.
  Returns the final state, total/per-action executed steps, and each segment's
  state and both camera images in order, all together after the batch ends.
  Early episode termination discards remaining targets and returns done=true.
  Batch motions whose next actions are clear; split at points requiring
  feedback, such as confirming a grasp.
