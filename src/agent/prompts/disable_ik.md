## Direct joint control only — no IK/FK

IK and FK are strictly forbidden throughout planning and execution, including
library calls, self-written code, manual calculations, and analytic or numerical
methods. Do not convert between end-effector poses and joint angles or evaluate
kinematic chains.

Any detected IK/FK use makes the entire episode a FAILURE, even if the task
succeeds. Task success does not override this rule.

Generate joint targets or increments directly from images, measured joints,
and action feedback. Ordinary computation, image processing, and simple joint
arithmetic are allowed. Follow the selected action format and tool rules.
