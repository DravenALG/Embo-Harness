"""Split execution and prefetch one decision outside the model/tool loop."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
import time

import numpy as np

from src.agent.embodiment import get_executor, action_dim, gripper_columns

from src.agent.async_context import AsyncInferenceContext, ObservationSnapshot, PrefetchCancelled
from src.agent.executor import (
    ActionChunk, DEFAULT_ACTION_CHUNK_SIZE, DEFAULT_EEF_DESTINATION_ROTATION_SPEED, DEFAULT_EEF_DESTINATION_TRANSLATION_SPEED,
    positive_int, validate_action_format,
)
from src.agent.errors import validate_policy_output
from src.agent.vla import JointActionChunk


def diagnostics(backend):
    return deepcopy({
        "model_output": backend.last_response_text, "usage": backend.last_usage,
        "codex_events_path": str(backend.events_path), "codex_stderr_path": str(backend.stderr_path),
    })


@dataclass
class PreparedAction:
    command: dict | JointActionChunk | ActionChunk
    context: AsyncInferenceContext
    metadata: dict
    diagnostics: dict
    infer_ms: float


class AsyncActionScheduler:
    def __init__(self, backend, *, tail_steps, max_steps, hybrid, episode_id, control_dt=1 / 15):
        self.benchmark = getattr(backend, "benchmark", "robolab")
        self.executor = get_executor(self.benchmark)
        self.backend = backend
        self.action_format = validate_action_format(getattr(backend, "action_format", "eef_absolute"))
        self._frame_width = action_dim(self.benchmark, self.action_format)
        self._gripper_columns = gripper_columns(self.benchmark, self.action_format)
        self.tail_steps, self.max_steps, self.hybrid = tail_steps, max_steps, hybrid
        self.control_dt = control_dt
        self.episode_id = episode_id
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="harness-prefetch")
        self.future = None
        self.tail = None
        self.boundary = None
        self.chunk_id = -1
        self.chunk_end = 0
        self.chunk_record = {}
        self.last_diagnostics = {}

    def _prepare(self, context, *, refresh=False):
        started = time.monotonic()
        method = self.backend.refresh_vla if refresh else self.backend.act
        try:
            command = method(context.current.observation, control_step=context.current.control_step,
                             async_context=context)
            if isinstance(command, JointActionChunk):
                raw = np.asarray(command.actions, dtype=np.float32)
                if (not self.hybrid or raw.ndim != 2 or raw.shape[1] != 8 or not 1 <= len(raw) <= 15
                        or not np.isfinite(raw).all() or not np.isin(raw[:, 7], [0, 1]).all()):
                    raise ValueError("Invalid VLA joint action chunk")
            elif isinstance(command, ActionChunk):
                raw = np.asarray(command.actions, dtype=np.float32)
                if (raw.ndim != 2 or raw.shape[1] != self._frame_width
                        or command.mode not in (0, 1) or not 1 <= len(raw)
                        or not np.isfinite(raw).all() or not np.isin(raw[:, self._gripper_columns], [0, 1]).all()):
                    raise ValueError("Invalid direct action chunk")
            if isinstance(command, (JointActionChunk, ActionChunk)):
                if command.execution_limit is not None:
                    positive_int(command.execution_limit, "execution_limit")
        except BaseException:
            # Tail-only replies do not read diagnostics. A failed worker is joined
            # before an error/end reply consumes this snapshot.
            self.last_diagnostics = diagnostics(self.backend)
            raise
        return PreparedAction(deepcopy(command), context,
                              deepcopy(self.backend.last_action_metadata or {}),
                              diagnostics(self.backend), (time.monotonic() - started) * 1000)

    def next(self, observation, control_step):
        if self.tail is not None:
            tail, self.tail = self.tail, None
            current = ObservationSnapshot.capture(observation, control_step, self.chunk_id,
                                                  "prefix_end", self.episode_id)
            endpoint = tail[-1, 1:] if self.hybrid else tail[-1]
            mode = int(tail[-1, 0]) if self.hybrid else int(self.action_format.startswith("joint_"))
            context = AsyncInferenceContext(self.boundary, current, self.chunk_end, {
                "steps": len(tail), "mode": "joints" if mode else "eef",
                "expected_endpoint": endpoint.tolist(), "endpoint_is_measured": False,
            })
            self.future = self.pool.submit(self._prepare, context)
            record = {**deepcopy(self.chunk_record), "segment": "tail", "new_decision": False,
                      "infer_ms": 0.0, "wait_ms": 0.0, "prefetch_observation_control_step": control_step}
            return tail, record

        current = ObservationSnapshot.capture(
            observation, control_step, self.chunk_id if self.chunk_id >= 0 else None,
            "tail_end" if self.chunk_id >= 0 else "episode_start", self.episode_id)
        wait_started = time.monotonic()
        if self.future is not None:
            future, self.future = self.future, None
            prepared = future.result()
        else:
            context = AsyncInferenceContext(self.boundary or current, current, control_step)
            prepared = self._prepare(context)
        wait_ms = (time.monotonic() - wait_started) * 1000
        self.boundary = current
        command = prepared.command
        skipped = control_step - prepared.context.current.control_step
        if isinstance(command, JointActionChunk) and skipped >= len(command.actions):
            # A new observation, never a resend of the stale inference request.
            context = AsyncInferenceContext(current, current, control_step)
            fresh = self._prepare(context, refresh=True)
            fresh.infer_ms += prepared.infer_ms
            prepared, command, skipped = fresh, fresh.command, 0
            wait_ms = (time.monotonic() - wait_started) * 1000
        remaining = self.max_steps - control_step
        record = {"chunk_id": self.chunk_id + 1, "new_decision": True,
                  "observation_control_step": prepared.context.current.control_step,
                  "reference_control_step": prepared.context.previous.control_step,
                  "execution_start_control_step": control_step,
                  "infer_ms": prepared.infer_ms, "wait_ms": wait_ms,
                  "action_metadata": prepared.metadata}
        if isinstance(command, (JointActionChunk, ActionChunk)):
            raw = np.asarray(command.actions, dtype=np.float32)
            if isinstance(command, ActionChunk) and command.relative_actions is not None:
                _, raw = validate_policy_output(
                    self.executor.prepare_action_chunk, observation, list(command.relative_actions),
                    action_chunk_size=len(command.relative_actions),
                    action_format=command.action_format or self.action_format,
                )
            limit = min(remaining, command.execution_limit if command.execution_limit is not None else remaining)
            # A direct chunk is issued for the actual execution start carried
            # by its async context, so every listed frame remains available at
            # handoff. VLA predictions are sampled continuously and retain the
            # existing stale-prefix skip behavior.
            offset = 0 if isinstance(command, ActionChunk) else skipped
            actions, mode = raw[offset:][:limit].copy(), 1
            if isinstance(command, ActionChunk):
                mode = command.mode
            record["skipped_steps"] = 0 if isinstance(command, ActionChunk) else skipped
        else:
            self.executor.robot_state(observation)
            if self.action_format == "eef_destination" and prepared.metadata.get("tool") != "refine_act":
                target = validate_policy_output(
                    self.executor.validate_fixed_target,
                    command, action_format=self.action_format,
                    action_chunk_size=getattr(self.backend, "action_chunk_size", DEFAULT_ACTION_CHUNK_SIZE),
                )
                resolved, actions = validate_policy_output(
                    self.executor.prepare_eef_destination_actions, observation, target,
                    control_dt=self.control_dt,
                    action_chunk_size=getattr(self.backend, "action_chunk_size", DEFAULT_ACTION_CHUNK_SIZE),
                    translation_speed=getattr(self.backend, "eef_destination_translation_speed",
                                              DEFAULT_EEF_DESTINATION_TRANSLATION_SPEED),
                    rotation_speed=getattr(self.backend, "eef_destination_rotation_speed",
                                           DEFAULT_EEF_DESTINATION_ROTATION_SPEED),
                    remaining_steps=remaining,
                )
            else:
                target = validate_policy_output(self.executor.validate_target, command, action_format=self.action_format)
                target["steps"] = min(target["steps"], remaining)
                resolved, actions = validate_policy_output(self.executor.prepare_actions, observation, target,
                                                          action_format=self.action_format)
            mode = int(self.action_format.startswith("joint_"))
            record.update(target=target, resolved_target=resolved, skipped_steps=0)
        if not len(actions):
            raise ValueError("Async adaptation produced an empty action chunk")
        self.backend.commit_action(control_step=control_step, steps=len(actions))
        if self.hybrid:
            actions = np.column_stack([np.full(len(actions), mode, dtype=np.float32), actions])
        self.chunk_id += 1
        self.chunk_end = control_step + len(actions)
        # Do not ask for an observation solely to prefetch beyond the episode budget.
        reserved = min(self.tail_steps, len(actions) - 1) if self.chunk_end < self.max_steps else 0
        if reserved:
            self.tail = actions[-reserved:].copy()
            actions = actions[:-reserved]
        record.update(segment="prefix" if reserved else "whole", tail_steps=reserved)
        self.chunk_record = deepcopy(record)
        self.last_diagnostics = prepared.diagnostics
        return actions, record

    def end(self, observation, *, control_step, reason):
        if self.future is not None:
            future, self.future = self.future, None
            cancelled = not future.done()
            if cancelled:
                self.backend.cancel_prefetch()
            try:
                future.result()
            except PrefetchCancelled:
                cancelled = True
            if cancelled:
                self.backend.close()
                self.last_diagnostics = diagnostics(self.backend)
                return "cancelled" if reason != "execution_error" else "aborted"
        result = self.backend.end_episode(observation, control_step=control_step, reason=reason)
        self.last_diagnostics = diagnostics(self.backend)
        return result

    def close(self):
        self.backend.abort()
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.future = self.tail = self.boundary = None
