"""Episode protocol and lifecycle for the frontier-model-as-policy harness."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import time
import threading

import numpy as np

from src.agent.embodiment import get_executor, action_dim, gripper_columns

from src.agent.executor import (
    ActionChunk, DEFAULT_ACTION_CHUNK_SIZE, DEFAULT_EEF_DESTINATION_ROTATION_SPEED, DEFAULT_EEF_DESTINATION_TRANSLATION_SPEED,
    positive_int, validate_action_format,
)
from src.agent.errors import PolicyExecutionError, PolicyValidationError, validate_policy_output
from src.agent.client import validate_views
from src.agent.vla import JointActionChunk
from src.evaluation.robolab_bench.episode_stats import EpisodeStats
from src.evaluation.robolab_bench.async_scheduler import AsyncActionScheduler


ACTION_REPRESENTATION = "robolab_base_link_absolute_xyz_quat_wxyz_gripper"
HYBRID_ACTION_REPRESENTATION = "robolab_mode_absolute_xyz_quat_wxyz_or_jointpos_gripper"
PROTOCOL_VERSION = 2
logger = logging.getLogger(__name__)


class HarnessPolicy:
    def __init__(self, backend, log_path: Path, *, image_size: int = 512,
                 async_inference: bool = False, async_tail_steps: int = 5) -> None:
        if type(async_inference) is not bool:
            raise ValueError("async_inference must be a boolean")
        if getattr(backend, "async_inference", async_inference) != async_inference:
            raise ValueError("Policy and backend async_inference settings must agree")
        self.async_tail_steps = positive_int(async_tail_steps, "async_tail_steps")
        self.async_inference = async_inference
        self._scheduler = None
        self.benchmark = getattr(backend, "benchmark", "robolab")
        self.executor = get_executor(self.benchmark)
        self.backend = backend
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.stats_path = self.log_path.with_name(f"{self.log_path.stem}.episodes.jsonl")
        self._episode_stats = None
        self.action_format = validate_action_format(getattr(backend, "action_format", "eef_absolute"))
        self._frame_width = action_dim(self.benchmark, self.action_format)
        self._gripper_columns = gripper_columns(self.benchmark, self.action_format)
        self.vla_enabled = getattr(backend, "vla", None) is not None
        self.hybrid = self.vla_enabled or self.action_format.startswith("joint_")
        self.metadata = {
            "policy": "harness", "protocol_version": PROTOCOL_VERSION,
            "model": backend.model,
            "backend": backend.backend,
            "reasoning_effort": backend.reasoning_effort,
            "auth": backend.auth,
            "plan": getattr(backend, "plan", False),
            "action_format": self.action_format,
            "direct_action_chunk": getattr(backend, "direct_action_chunk", False),
            "action_representation": ACTION_REPRESENTATION, "action_dim": 8,
            "image_size": image_size, "views": validate_views(getattr(backend, "views", "multi")),
            "supports_batch_inference": False,
        }
        if self.action_format == "eef_destination":
            self.metadata.update(
                eef_destination_translation_speed=getattr(backend, "eef_destination_translation_speed", DEFAULT_EEF_DESTINATION_TRANSLATION_SPEED),
                eef_destination_rotation_speed=getattr(backend, "eef_destination_rotation_speed", DEFAULT_EEF_DESTINATION_ROTATION_SPEED),
            )
        if self.hybrid:
            self.metadata.update(protocol_version=3, action_dim=9,
                                 action_representation=HYBRID_ACTION_REPRESENTATION)
            # Request VLA images only when a service is present, including when
            # protocol 3 is already needed for ordinary joint control.
            self.metadata["vla_enabled"] = self.vla_enabled
        self.metadata.update(async_inference=async_inference, async_tail_steps=async_tail_steps)
        self._episode_id = None
        self._next_step = 0
        self._last_control_step = self._expected_control_step = 0
        self._failed = self._ended = False
        self._episode_count = 0
        self._disconnected = threading.Event()
        backend.connection_cancelled = self._disconnected

    def connect(self) -> None:
        self._disconnected.clear()

    def disconnect(self) -> None:
        self._disconnected.set()
        self.abort()

    def infer(self, request: dict) -> dict:
        if not isinstance(request, dict):
            raise PolicyValidationError("Harness request must be an object")
        started = time.monotonic()
        record = {key: request.get(key) for key in (
            "type", "episode_id", "decision_step", "control_step", "instruction", "reason")}
        record.update({key: self.metadata[key] for key in ("model", "backend", "reasoning_effort", "auth", "action_format")})
        called_backend = False
        confirmed_control_step = None
        try:
            if self._disconnected.is_set():
                raise RuntimeError("Harness client disconnected")
            ending = request.get("type") == "end_episode"
            if request.get("type") not in ("infer", "end_episode"):
                raise PolicyValidationError("Expected infer or end_episode request")
            required = {"episode_id", "decision_step", "control_step", "observation"}
            required |= {"reason"} if ending else {"instruction", "max_steps", "control_dt"}
            if not required <= request.keys():
                raise PolicyValidationError("Missing required harness request fields")
            episode_id = request["episode_id"]
            if not isinstance(episode_id, str) or not episode_id:
                raise PolicyValidationError("A nonempty episode_id is required")
            if request.get("env_id") != 0:
                raise PolicyValidationError("The minimal harness supports only env_id=0")
            if episode_id != self._episode_id:
                if ending or request["decision_step"] != 0 or request["control_step"] != 0:
                    raise PolicyValidationError("A new episode must start at decision_step=0 and control_step=0")
                if self._episode_id is not None and not (self._ended or self._failed):
                    raise PolicyValidationError("End the previous episode before starting another")
                instruction = request["instruction"]
                if not isinstance(instruction, str) or not instruction.strip():
                    raise PolicyValidationError("A nonempty task instruction is required")
                self._max_steps = validate_policy_output(positive_int, request["max_steps"], "max_steps")
                self._control_dt = validate_policy_output(float, request["control_dt"])
                if not math.isfinite(self._control_dt) or self._control_dt <= 0:
                    raise PolicyValidationError("control_dt must be positive and finite")
                self._episode_id, self._instruction = episode_id, instruction
                self._next_step = self._last_control_step = self._expected_control_step = 0
                self._failed = self._ended = False
                self._episode_count += 1
                self._record_stats("start", started=started)
                self._start_episode()
            if self._failed:
                raise RuntimeError("This episode already had an execution error; no retry is allowed")
            if self._ended:
                raise PolicyValidationError("This episode has already ended")
            if request["decision_step"] != self._next_step:
                raise PolicyValidationError("Out-of-order or repeated decision; no resampling is allowed")
            control_step = request["control_step"]
            if isinstance(control_step, bool) or not isinstance(control_step, int):
                raise PolicyValidationError("control_step must be an integer")
            if ending:
                if not self._last_control_step <= control_step <= self._expected_control_step:
                    raise PolicyValidationError("Invalid final control_step")
                if request["reason"] not in ("native_terminal", "time_limit", "execution_error"):
                    raise PolicyValidationError("Invalid episode end reason")
                confirmed_control_step = control_step
                called_backend = True
                result = self._end_episode(request)
                self._ended = True
                record.update(status="ended", executed_steps=control_step - self._last_control_step, **result)
                return {"ended": True, **result}
            if control_step != self._expected_control_step or control_step >= self._max_steps:
                raise PolicyValidationError("Unexpected control_step or episode step budget exhausted")
            if request["instruction"] != self._instruction:
                raise PolicyValidationError("The task instruction cannot change within an episode")
            if request["max_steps"] != self._max_steps or request["control_dt"] != self._control_dt:
                raise PolicyValidationError("Episode limits cannot change within an episode")
            confirmed_control_step = control_step
            called_backend = True
            if self._scheduler is not None:
                actions, details = self._scheduler.next(request["observation"], control_step)
                record.update(details)
                called_backend = details["new_decision"]
            else:
                command = self._act(request)
                remaining = self._max_steps - control_step
                if isinstance(command, JointActionChunk):
                    actions = np.asarray(command.actions, dtype=np.float32)
                    if (actions.ndim != 2 or actions.shape[1] != 8
                            or not self.vla_enabled or not 1 <= len(actions) <= min(15, remaining)
                            or not np.isfinite(actions).all() or not np.isin(actions[:, 7], [0, 1]).all()):
                        raise ValueError("Invalid VLA joint action chunk")
                    action_mode = 1
                elif isinstance(command, ActionChunk):
                    actions = np.asarray(command.actions, dtype=np.float32)
                    if (actions.ndim != 2 or actions.shape[1] != self._frame_width
                            or command.mode not in (0, 1)
                            or not 1 <= len(actions) <= remaining
                            or not np.isfinite(actions).all() or not np.isin(actions[:, self._gripper_columns], [0, 1]).all()):
                        raise ValueError("Invalid direct action chunk")
                    action_mode = command.mode
                else:
                    observation = request["observation"]
                    self.executor.robot_state(observation)
                    if (self.action_format == "eef_destination"
                            and (getattr(self.backend, "last_action_metadata", None) or {}).get("tool") != "refine_act"):
                        target = validate_policy_output(
                            self.executor.validate_fixed_target, command, action_format=self.action_format,
                            action_chunk_size=getattr(self.backend, "action_chunk_size", DEFAULT_ACTION_CHUNK_SIZE),
                        )
                        resolved, actions = validate_policy_output(
                            self.executor.prepare_eef_destination_actions, observation, target,
                            control_dt=self._control_dt,
                            action_chunk_size=getattr(self.backend, "action_chunk_size", DEFAULT_ACTION_CHUNK_SIZE),
                            translation_speed=getattr(self.backend, "eef_destination_translation_speed",
                                                      DEFAULT_EEF_DESTINATION_TRANSLATION_SPEED),
                            rotation_speed=getattr(self.backend, "eef_destination_rotation_speed",
                                                   DEFAULT_EEF_DESTINATION_ROTATION_SPEED),
                            remaining_steps=remaining,
                        )
                        # eef_destination's actual prefix can be shorter than the
                        # configured chunk when the velocity-limited target is
                        # reached early; update the client's motion boundary.
                        commit_action = getattr(self.backend, "commit_action", None)
                        if commit_action is not None:
                            commit_action(control_step=control_step, steps=len(actions))
                    else:
                        target = validate_policy_output(self.executor.validate_target, command, remaining_steps=remaining,
                                                        action_format=self.action_format)
                        resolved, actions = validate_policy_output(self.executor.prepare_actions, observation, target,
                                                                  action_format=self.action_format)
                    record["target"] = target
                    record["resolved_target"] = resolved
                    action_mode = int(self.action_format.startswith("joint_"))
                if self.hybrid:
                    actions = np.column_stack([np.full(len(actions), action_mode, dtype=np.float32), actions])
                if (metadata := getattr(self.backend, "last_action_metadata", None)) is not None:
                    record["action_metadata"] = dict(metadata)
            record.update(status="ok", actions=actions.tolist(),
                          executed_steps=control_step - self._last_control_step)
            self._last_control_step = control_step
            self._expected_control_step = control_step + len(actions)
            self._next_step += 1
            return {"actions": actions}
        except BaseException as error:
            self._failed = True
            self.abort()
            record.update(status="execution_error", error_type=type(error).__name__, error=str(error))
            if isinstance(error, PolicyExecutionError):
                record["error_kind"] = "policy_execution"
            raise
        finally:
            finished = time.monotonic()
            record["elapsed_ms"] = (finished - started) * 1000
            if called_backend:
                record.update(self._diagnostics())
            with self.log_path.open("a") as log:
                log.write(json.dumps(record, allow_nan=False) + "\n")
            self._record_stats("request", record=record, started=started, finished=finished, called_backend=called_backend,
                               confirmed_control_step=confirmed_control_step)

    def _record_stats(self, operation: str, **values) -> None:
        """Statistics never participate in model input or execution control."""
        try:
            if operation == "start":
                self._episode_stats = None
                self._episode_stats = EpisodeStats(
                    log_path=self.log_path, stats_path=self.stats_path, metadata=self.metadata,
                    episode_id=self._episode_id, episode_index=self._episode_count,
                    instruction=self._instruction, max_steps=self._max_steps,
                    control_dt=self._control_dt, **values,
                )
            elif self._episode_stats is not None and not self._episode_stats.finalized:
                if operation == "request":
                    self._episode_stats.request(disconnected=self._disconnected.is_set(),
                                                **values)
                elif operation == "close":
                    self._episode_stats.finish(
                        "interrupted", "disconnected" if self._disconnected.is_set() else "server_closed",
                        time.monotonic(),
                    )
        except Exception:
            # A broken accumulator must not later publish an apparently complete
            # result, nor affect returned actions, original errors, or cleanup.
            self._episode_stats = None
            try:
                logger.warning("Episode statistics unavailable for %s (%s)", self._episode_id, operation,
                               exc_info=True)
            except Exception:
                pass  # Even a failing logging handler must not affect execution.

    def _start_episode(self) -> None:
        if self._scheduler is not None:
            self._scheduler.close()
            self._scheduler = None
        directory = self.log_path.with_suffix("") / f"episode-{self._episode_count:04d}"
        self.backend.start_episode(self._instruction, max_steps=self._max_steps,
                                   control_dt=self._control_dt, log_dir=directory)
        if self.async_inference:
            self._scheduler = AsyncActionScheduler(self.backend, tail_steps=self.async_tail_steps,
                                                   max_steps=self._max_steps, hybrid=self.hybrid,
                                                   episode_id=self._episode_id, control_dt=self._control_dt)

    def _act(self, request: dict) -> dict | JointActionChunk | ActionChunk:
        return self.backend.act(request["observation"], control_step=request["control_step"])

    def _end_episode(self, request: dict) -> dict:
        if self._scheduler is not None:
            return {"codex_finish": self._scheduler.end(
                request["observation"], control_step=request["control_step"], reason=request["reason"])}
        return {"codex_finish": self.backend.end_episode(
            request["observation"], control_step=request["control_step"], reason=request["reason"])}

    def _diagnostics(self) -> dict:
        if self._scheduler is not None:
            return dict(self._scheduler.last_diagnostics)
        return {"model_output": self.backend.last_response_text, "usage": self.backend.last_usage,
                "codex_events_path": str(self.backend.events_path),
                "codex_stderr_path": str(self.backend.stderr_path)}

    def abort(self) -> None:
        """Wake an in-flight request when its WebSocket disconnects."""
        self.backend.abort()

    def close(self) -> None:
        self.abort()
        if self._scheduler is not None:
            self._scheduler.close()
        if self._episode_id is not None and not (self._ended or self._failed):
            self._failed = True
            with self.log_path.open("a") as log:
                log.write(json.dumps({"episode_id": self._episode_id, "status": "interrupted"}) + "\n")
        self.backend.close()
        self._record_stats("close")
