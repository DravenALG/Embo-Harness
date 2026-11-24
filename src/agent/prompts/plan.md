Plan the episode autonomously. Start with observe(); use the task and public
observations. No human input is available.
Do not move the robot or create goals.
Do not query VLA during planning.

Return only one proposed_plan block containing this JSON structure:
<proposed_plan>
{"subgoals": ["Actionable subgoal and its observable completion condition", "Next subgoal and its observable completion condition"]}
</proposed_plan>

Use a nonempty ordered list of concise, nonempty strings, without Markdown
fences or extra keys. Together, the subgoals must cover the task and fit the
remaining control-step budget.
