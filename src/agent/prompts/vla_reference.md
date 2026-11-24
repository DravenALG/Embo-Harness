- vla(instruction=None): returns one suggested action from pi0.5 without
  moving the robot. Omit instruction or pass null to use the episode task.
  The returned action uses the current refine_act argument format, without steps.
  The returned reference_state and reference_control_step identify the query observation
  used to generate the suggestion. Inspect or refine the suggestion, then call
  refine_act to execute it. Do not pass steps to vla.
  To discard the suggestion and generate an independent action, use
  {{EEF_ACTION_TOOLS}} with their normal fixed or adaptive durations.
  If VLA is unavailable, use {{EEF_ACTION_TOOLS}}; do not retry VLA.
- refine_act({{ACTION_PARAMETERS}}): executes one refined reference target in the
  selected action format over 15 control frames.
  If fewer frames remain at execution start, reach the target within that budget.
  Do not supply steps. Requires the latest successful vla query in this decision;
  a new query replaces the reference, and a failed query clears it.
  Each reference is consumed once. Any motion or goal change discards it;
  observe does not. Returns the resulting observation and actual executed_steps,
  which may be fewer if the episode ends early. Wait for the result before
  another refine_act or {{EEF_ACTION_TOOLS}} call. Query again for a new reference.
