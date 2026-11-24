- vla(steps, instruction=None): delegates control to fast pi0.5 for steps control
  frames, replanning from fresh observations every 15 frames. steps must be a
  positive integer <= remaining_steps. Omit instruction or pass null to use
  the episode task; otherwise provide a nonempty subtask instruction.
  Returns the final robot state, both camera images, requested_steps, actual
  executed_steps, and done. Execution may stop early if the episode ends or
  VLA becomes unavailable.
  Choose VLA for simple, clearly specified subtasks. Use {{EEF_ACTION_TOOLS}} for complex
  scenes, precise contact, or recovery. Inspect the returned state and images
  before continuing. If VLA reports unavailable, continue with {{EEF_ACTION_TOOLS}};
  do not retry VLA.
