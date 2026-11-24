Asynchronous execution is enabled. Plan at prefix_end while the current action's
tail continues. Your next action starts after that tail finishes.

Compare the previous tail_end and current prefix_end snapshots to assess motion,
contact and grasp changes. The first reference is episode_start; top-level state
describes the latest observation.

Plan beyond async_inference.pending_tail's expected endpoint, keeping the next
target's motion, orientation and gripper intent coherent with it. Treat this
endpoint as an expectation and correct the plan using observed evidence.

Use next_action_remaining_steps as the budget for new motions. GPT targets are
interpolated from the measured handoff state with their original duration.
