- multiact(actions): executes a nonempty list of act targets in order, starting
  each from the latest measured state. Each target has only {{ACTION_PARAMETERS}},
  with the same fixed {{ACTION_CHUNK_SIZE}}-frame duration as act.
  The entire list is validated before any motion. A batch may exceed the remaining
  budget: the last executed target uses any remaining frames to reach its target,
  and subsequent targets are discarded.
  Returns the final state, total/per-action executed steps (including zeros for
  unexecuted targets), and each executed segment's state and both camera images
  in order, all together after the batch ends.
  Early episode termination discards remaining targets and returns done=true.
  Batch motions whose next actions are clear; split at points requiring
  feedback, such as confirming a grasp.
