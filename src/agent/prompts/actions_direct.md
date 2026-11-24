- act(actions): execute one complete action chunk of exactly {{ACTION_CHUNK_SIZE}} control frames.
  Each item in `actions` is one frame and contains only {{ACTION_PARAMETERS}}.
  Send every frame directly in order; do not provide `steps`, interpolate, smooth,
  repeat, or replace the list with a single endpoint. If fewer frames remain in the
  episode, the harness executes only the available prefix and discards the rest.
  Absolute formats are direct controller poses. In delta formats, each item is one
  increment from the preceding frame, starting from the measured state at execution
  start; gripper is an absolute per-frame command.
  Wait for the resulting observation before issuing another motion.
