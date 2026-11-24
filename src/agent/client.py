"""Isolated Codex robot episodes (Linux, tested with Codex 0.168.0)."""

from __future__ import annotations

import base64
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import io
import json
import math
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import time

import numpy as np
from PIL import Image

from src.agent.errors import PolicyProtocolError, PolicyValidationError, validate_policy_output
from src.agent.executor import (
    ActionChunk, DEFAULT_ACTION_CHUNK_SIZE, DEFAULT_EEF_DESTINATION_ROTATION_SPEED,
    DEFAULT_EEF_DESTINATION_TRANSLATION_SPEED, action_schema, positive_float, positive_int,
    reference_target, validate_action_format,
)
from src.agent.embodiment import get_executor, camera_names, prompt_path as robot_prompt_path
from src.agent.kinematics import droid_forward_kinematics
from src.agent.prompts import load_system_prompt, load_thread_prompt, validate_disable_ik, validate_note_file
from src.agent.vla import JointActionChunk, validate_joint_actions
from src.agent.async_context import AsyncInferenceContext, PrefetchCancelled


VIEW_MODES = ("single", "multi")


def is_claude_model(model: str) -> bool:
    return isinstance(model, str) and model.startswith("claude-")


def _finite_json_number(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Codex event contains a non-finite number")
    return number


def _resolve_codex_executable(source: str, local_path: str | Path | None) -> str:
    if source not in ("system", "local"):
        raise ValueError("codex_source must be 'system' or 'local'")
    if source == "system":
        executable = shutil.which("codex")
        if executable is None:
            raise RuntimeError("Codex CLI is missing from PATH; install the official Codex CLI (tested with 0.168.0)")
        return os.path.abspath(executable)
    if not isinstance(local_path, (str, Path)) or not str(local_path).strip():
        raise ValueError("codex_local_path must be a non-empty path when codex_source is 'local'")
    repo = Path(__file__).resolve().parents[2]
    executable = (repo / Path(local_path).expanduser()).resolve()
    if not executable.is_file():
        raise RuntimeError(f"Local Codex executable is not a file: {executable}. "
                           "Build Codex first or set codex_local_path to a compiled binary.")
    if not os.access(executable, os.X_OK):
        raise RuntimeError(f"Local Codex file is not executable: {executable}")
    host = executable.with_name("codex-code-mode-host")
    if not host.is_file() or not os.access(host, os.X_OK):
        raise RuntimeError(f"Local Codex code-mode host is missing or not executable: {host}. "
                           "Build it with 'cargo build --locked -p codex-code-mode-host "
                           "--bin codex-code-mode-host' in the same codex-rs checkout using the V8 "
                           "setup required by that checkout, then install it beside codex.")
    return str(executable)


def prepare_image(image: np.ndarray, image_size: int) -> Image.Image:
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("Images must be nonempty H x W x 3 uint8 RGB arrays")
    pil_image = Image.fromarray(image)
    pil_image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
    return pil_image


def image_data_url(image: np.ndarray, image_size: int) -> str:
    buffer = io.BytesIO()
    prepare_image(image, image_size).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def validate_reasoning_effort(value: str) -> str:
    if value not in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
        raise ValueError(f"Unsupported reasoning_effort: {value!r}")
    return value


def validate_views(value: str) -> str:
    if not isinstance(value, str) or value not in VIEW_MODES:
        raise ValueError("views must be 'single' or 'multi'")
    return value


def validate_codex_stream_idle_timeout(value: int) -> int:
    """Validate seconds before converting to a signed 64-bit TOML millisecond value."""
    value = positive_int(value, "codex_stream_idle_timeout")
    max_seconds = ((1 << 63) - 1) // 1000
    if value > max_seconds:
        raise ValueError(f"codex_stream_idle_timeout must be at most {max_seconds} seconds")
    return value


def robot_tools(*, adaptive_action_chunk: bool, multiact: bool = False,
                action_format: str = "eef_absolute",
                action_chunk_size: int = DEFAULT_ACTION_CHUNK_SIZE,
                direct_action_chunk: bool = False, benchmark: str = "robolab") -> list[dict]:
    """Build independent schemas for the selected action format and duration."""
    positive_int(action_chunk_size, "action_chunk_size")
    if type(direct_action_chunk) is not bool:
        raise ValueError("direct_action_chunk must be a boolean")
    if direct_action_chunk and adaptive_action_chunk:
        raise ValueError("direct_action_chunk cannot be combined with adaptive_action_chunk")
    if action_format == "eef_destination" and (adaptive_action_chunk or direct_action_chunk or multiact):
        raise ValueError("eef_destination supports only fixed single-target act")
    target_schema = get_executor(benchmark).action_schema(action_format, adaptive_action_chunk=adaptive_action_chunk)
    if adaptive_action_chunk:
        duration = "steps control frames (steps * control_dt seconds)"
        budget = "Each action requires steps; the sum of steps must fit the remaining episode budget. "
    else:
        duration = f"{action_chunk_size} control frames, shortened to remaining_steps if fewer remain"
        budget = ("Do not supply steps. Each target is reached within the available frames; "
                  "when the episode ends, remaining targets are discarded and report zero executed steps. ")
    if action_format == "eef_destination":
        duration = f"at most {action_chunk_size} control frames, using the configured velocity limits"
        budget = "Do not supply steps. The harness returns an observation after this prefix for target correction. "
    if direct_action_chunk:
        frame_schema = get_executor(benchmark).action_schema(action_format, adaptive_action_chunk=False)
        act_schema = {"type": "object", "properties": {
            "actions": {"type": "array", "items": frame_schema,
                        "minItems": action_chunk_size, "maxItems": action_chunk_size},
        }, "required": ["actions"], "additionalProperties": False}
        act_description = (
            f"Execute one complete {action_format} action chunk of {action_chunk_size} control frames. "
            "Each list item is sent directly as one frame; do not provide steps or an endpoint. "
            "If fewer frames remain, only the available prefix is executed without interpolation."
        )
    else:
        act_schema = target_schema
        act_description = f"Execute one {action_format} target over {duration}; wait for the resulting observation."
    tools = [
        {"type": "function", "name": "observe", "description": "Observe the robot without advancing simulation.",
         "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
        {"type": "function", "name": "act", "description": act_description,
         "inputSchema": deepcopy(act_schema)},
    ]
    if multiact:
        tools.append({"type": "function", "name": "multiact", "description": (
            f"Execute {action_format} targets in order, starting each from its measured state. "
            f"The number of actions is len(actions); each action lasts {duration}. "
            f"{budget}After the batch finishes or the episode ends, "
            "returns a final execution summary followed by each executed action's measured state and camera images "
            "in order, labeled with its zero-based action_index, control_step, and executed_steps. "
            "Split batches when the next action depends on visual feedback; wait for the result before acting again."),
         "inputSchema": {"type": "object", "properties": {
             "actions": {"type": "array", "items": target_schema, "minItems": 1},
         }, "required": ["actions"], "additionalProperties": False}})
    return tools


VLA_TOOL = {
    "type": "function", "name": "vla", "description": (
        "Delegate a clear subtask to fast pi0.5 for steps control frames. Replans from new observations "
        "every 15 frames. Omit instruction or pass null to use the episode task. Steps must fit the remaining "
        "budget. Returns final state, images, requested/executed steps and done; wait before acting again."),
    "inputSchema": {"type": "object", "properties": {
        "steps": {"type": "integer", "minimum": 1},
        "instruction": {"type": ["string", "null"], "minLength": 1},
    }, "required": ["steps"], "additionalProperties": False},
}

VLA_REFERENCE_TOOL = {
    "type": "function", "name": "vla", "description": (
        "Query pi0.5 for one suggested action without moving the robot. "
        "Uses prediction frame min(15, remaining_steps), converted to the selected action format. "
        "Returns action in the current refine_act argument format and reference_steps. "
        "The returned reference_state and reference_control_step identify the query observation "
        "used to generate the suggestion. Inspect or refine action, then call refine_act to execute it "
        "over 15 frames, shortened to the remaining execution budget. Use the ordinary action tools for independent "
        "actions instead; they retain their own durations. Omit instruction or pass null to use "
        "the episode task. Wait for the result before acting; do not supply steps."),
    "inputSchema": {"type": "object", "properties": {
        "instruction": {"type": ["string", "null"], "minLength": 1},
    }, "required": [], "additionalProperties": False},
}


def refine_act_tool(action_format: str) -> dict:
    return {
        "type": "function", "name": "refine_act", "description": (
            f"Execute one refined {action_format} target from the latest successful vla reference query. "
            "Takes 15 control frames, shortened to the remaining execution budget, even in adaptive mode. "
            "Do not supply steps. Each reference can be used once, before any other motion or goal change; "
            "a new query replaces it. Wait for the resulting observation before acting again."),
        "inputSchema": action_schema(action_format, adaptive_action_chunk=False),
    }


VLA_TOOLS = {"vla"}
MOTION_TOOLS = {"act", "multiact", "refine_act", *VLA_TOOLS}


@dataclass(frozen=True)
class _VLAReference:
    call_id: str
    control_step: int
    execution_step: int
    steps: int


@dataclass
class _PendingMotion:
    request_id: int | str
    call_id: str
    tool: str
    targets: list[dict]
    start_control_step: int
    action_index: int = 0
    observations: dict[int, list[dict]] = field(default_factory=dict)
    issued_steps: int = 0
    requested_steps: int = 0
    instruction: str | None = None
    reference: _VLAReference | None = None
    direct_actions: np.ndarray | None = None
    direct_payload: tuple[dict, ...] | None = None
    direct_mode: int | None = None


class CodexClient:
    backend = "codex"
    auth = "api_key"
    finish_timeout = 30.0

    def __init__(self, model: str, *, timeout: float = 300, image_size: int = 512,
                 views: str = "multi",
                 reasoning_effort: str = "xhigh", codex_max_retries: int = 5,
                 codex_stream_idle_timeout: int = 300,
                 codex_source: str = "system", codex_local_path: str | Path | None = None,
                 system_prompt: Path | None = None, vla=None, enable_vla: bool | None = None,
                 include_note: bool = False,
                 note_file: str = "note.md", multiact: bool = False,
                 action_format: str = "eef_absolute", disable_ik: bool = False,
                 adaptive_action_chunk: bool = False, vla_as_reference: bool = False,
                 action_chunk_size: int = DEFAULT_ACTION_CHUNK_SIZE,
                 direct_action_chunk: bool = False,
                 eef_destination_translation_speed: float = DEFAULT_EEF_DESTINATION_TRANSLATION_SPEED,
                 eef_destination_rotation_speed: float = DEFAULT_EEF_DESTINATION_ROTATION_SPEED,
                 async_inference: bool = False, include_async_prompt: bool = True,
                 plan: bool = False, benchmark: str = "robolab") -> None:
        get_executor(benchmark)
        self.benchmark = benchmark
        self.action_format = validate_action_format(action_format)
        self.views = validate_views(views)
        self.disable_ik = validate_disable_ik(disable_ik, action_format=self.action_format)
        self.action_chunk_size = positive_int(action_chunk_size, "action_chunk_size")
        self.eef_destination_translation_speed = positive_float(
            eef_destination_translation_speed, "eef_destination_translation_speed")
        self.eef_destination_rotation_speed = positive_float(
            eef_destination_rotation_speed, "eef_destination_rotation_speed")
        if type(direct_action_chunk) is not bool:
            raise ValueError("direct_action_chunk must be a boolean")
        if direct_action_chunk and adaptive_action_chunk:
            raise ValueError("direct_action_chunk cannot be combined with adaptive_action_chunk")
        if not model or not 0 < timeout < float("inf") or image_size < 1:
            raise ValueError("model, positive finite timeout, and positive image_size are required")
        if isinstance(codex_max_retries, bool) or not isinstance(codex_max_retries, int) or codex_max_retries < 0:
            raise ValueError("codex_max_retries must be a non-negative integer")
        self.codex_stream_idle_timeout = validate_codex_stream_idle_timeout(codex_stream_idle_timeout)
        if not isinstance(adaptive_action_chunk, bool):
            raise ValueError("adaptive_action_chunk must be a boolean")
        if not isinstance(multiact, bool):
            raise ValueError("multiact must be a boolean")
        if self.action_format == "eef_destination":
            if adaptive_action_chunk or direct_action_chunk:
                raise ValueError("eef_destination supports only fixed single-target act")
            multiact = False
        if not isinstance(vla_as_reference, bool):
            raise ValueError("vla_as_reference must be a boolean")
        if not isinstance(async_inference, bool):
            raise ValueError("async_inference must be a boolean")
        if not isinstance(include_async_prompt, bool):
            raise ValueError("include_async_prompt must be a boolean")
        if not isinstance(plan, bool):
            raise ValueError("plan must be a boolean")
        self.plan = plan
        self.note_file = validate_note_file(note_file)
        self.async_inference = async_inference
        self._async_context = None
        self._action_committed = True
        self._prefetch_cancelled = threading.Event()
        if sys.platform != "linux":
            raise RuntimeError("The isolated Codex harness currently requires Linux")
        self.executable = self._resolve_executable(codex_source, codex_local_path)
        self.codex_source = codex_source
        self.model, self.timeout, self.image_size = model, timeout, image_size
        self.codex_max_retries = codex_max_retries
        self.reasoning_effort = validate_reasoning_effort(reasoning_effort)
        self.adaptive_action_chunk = adaptive_action_chunk
        self.direct_action_chunk = direct_action_chunk
        self.multiact = multiact
        self.tools = robot_tools(adaptive_action_chunk=adaptive_action_chunk, multiact=multiact,
                                 action_format=self.action_format, action_chunk_size=self.action_chunk_size,
                                 direct_action_chunk=direct_action_chunk, benchmark=benchmark)
        if enable_vla is not None and type(enable_vla) is not bool:
            raise ValueError("enable_vla must be a boolean")
        self.vla = vla
        self.enable_vla = vla is not None if enable_vla is None else enable_vla
        self.vla_as_reference = self.enable_vla and vla_as_reference
        if self.enable_vla and vla is None:
            raise ValueError("VLA tools require a resident VLA service")
        self._vla_error = None
        self._vla_reference: _VLAReference | None = None
        prompt_path = robot_prompt_path(benchmark, system_prompt)
        prompt_options = dict(benchmark=benchmark, enable_vla=self.enable_vla,
                              action_format=self.action_format, disable_ik=self.disable_ik,
                              vla_as_reference=self.vla_as_reference,
                              include_note=include_note, note_file=self.note_file,
                              adaptive_action_chunk=adaptive_action_chunk, multiact=multiact,
                              action_chunk_size=self.action_chunk_size,
                              direct_action_chunk=direct_action_chunk,
                              eef_destination_translation_speed=self.eef_destination_translation_speed,
                              eef_destination_rotation_speed=self.eef_destination_rotation_speed,
                              async_inference=async_inference, include_async_prompt=include_async_prompt)
        self.system = load_system_prompt(prompt_path, **prompt_options)
        self._plan_prompt = load_system_prompt(prompt_path, phase="plan", **prompt_options) if plan else None
        self._thread_prompt = load_thread_prompt(prompt_path, plan=plan, **prompt_options)
        self._prompt_options = {"system_prompt": str(Path(prompt_path).resolve()), "plan": plan, **prompt_options}
        self._phase = None
        self.env = dict(os.environ)
        self._configure_environment()
        self.last_response_text = self.last_usage = None
        self.last_action_metadata = None
        self._pending_motion = None
        self.events_path = self.stderr_path = None
        self._process = self._directory = self._selector = self._stderr = None
        self._process_lock = threading.Lock()
        self._cancelled = threading.Event()
        self.connection_cancelled = threading.Event()

    def _resolve_executable(self, source: str, local_path: str | Path | None) -> str:
        return _resolve_codex_executable(source, local_path)

    def _configure_environment(self) -> None:
        for name in ("CODEX_SESSION_ID", "CODEX_THREAD_ID", "CODEX_API_KEY"):
            self.env.pop(name, None)
        if not self.env.get("OPENAI_API_KEY"):
            raise RuntimeError("Codex harness requires OPENAI_API_KEY")

    def _config(self) -> list[str]:
        # Sandbox aliases are symlinks: bwrap also needs their real executable,
        # which npm may install outside the minimal system-readable paths.
        codex_dir = Path(self.env.get("CODEX_HOME", str(Path.home() / ".codex")))
        filesystem_rules = {":minimal": "read", ":workspace_roots": "write",
                            str(codex_dir / "tmp/arg0"): "read"}
        for executable in self._sandbox_executables():
            filesystem_rules[str(executable)] = "read"
        filesystem = "{" + ",".join(f"{json.dumps(path)}={json.dumps(access)}"
                                    for path, access in filesystem_rules.items()) + "}"
        settings = {
            "model_provider": "robot_harness",
            "model_reasoning_effort": self.reasoning_effort,
            # Persistent robot rules are supplied by thread/start.developerInstructions.
            "developer_instructions": "",
            "include_collaboration_mode_instructions": False,
            "approval_policy": "never",
            "default_permissions": "robot_harness",
            "permissions.robot_harness.network.enabled": False,
            "web_search": "disabled",
            "project_doc_max_bytes": 0,
            "allow_login_shell": False,
            "shell_environment_policy.inherit": "core",
            "shell_environment_policy.filters.PATH": "include",
            "model_providers.robot_harness.name": "Robot harness",
            "model_providers.robot_harness.wire_api": "responses",
            "model_providers.robot_harness.request_max_retries": self.codex_max_retries,
            "model_providers.robot_harness.stream_max_retries": self.codex_max_retries,
            "model_providers.robot_harness.stream_idle_timeout_ms": self.codex_stream_idle_timeout * 1000,
            "model_providers.robot_harness.requires_openai_auth": False,
            "model_providers.robot_harness.env_key": "OPENAI_API_KEY",
            "model_providers.robot_harness.base_url": self.env.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        }
        for feature in ("apps", "plugins", "remote_plugin", "hooks", "memories", "multi_agent",
                        "browser_use", "computer_use", "image_generation", "shell_snapshot",
                        "skill_search", "skill_mcp_dependency_install", "unbounded_connection_retries"):
            settings[f"features.{feature}"] = False
        settings["features.skip_host_skill_discovery"] = True
        settings["features.goals"] = True
        args = ["-c", "permissions.robot_harness.filesystem=" + filesystem]
        for key, value in settings.items():
            args.extend(["-c", f"{key}={json.dumps(value)}"])
        return args

    def _sandbox_executables(self) -> list[Path]:
        """Expose runtime files without granting access to their install directories."""
        entrypoint = Path(self.executable).resolve()
        executables = {entrypoint}
        if entrypoint.name == "codex.js":
            # npm can nest or hoist the platform package. Resolve symlinks for
            # pnpm/bun layouts too; older bundles keep vendor beside bin/.
            vendor_roots = [entrypoint.parent.parent / "vendor"]
            for parent in entrypoint.parents:
                vendor_roots.extend((parent / "node_modules/@openai").glob("codex-linux-*/vendor"))
            for vendor in vendor_roots:
                for directory in ("bin", "codex"):
                    executables.update(path.resolve() for path in vendor.glob(f"*/{directory}/codex")
                                       if path.is_file())
        return sorted(executables)

    def start_episode(self, instruction: str, *, max_steps: int, control_dt: float, log_dir: Path) -> None:
        self._prefetch_cancelled.clear()
        self._async_context = None
        self._action_committed = True
        self._instruction = instruction
        self._max_steps, self._control_dt = max_steps, control_dt
        deadline = time.monotonic() + self.timeout
        try:
            self._start_session(log_dir=log_dir, deadline=deadline)
            self._set_phase("plan" if self.plan else "execute", deadline)
            self._subgoals = []
            self._subgoal_index = -1
            if self.vla is not None and hasattr(self.vla, "reconnect_cancelled"):
                try:
                    self.vla.reconnect_cancelled(self._check_cancelled)
                except Exception as error:
                    self._handle_vla_error(error, phase="reconnect")
                    self._log("vla_reconnect_error", {"error": self._vla_error})
            if self._phase == "execute":
                self._advance_goal(deadline)
        except BaseException:
            self.close()
            raise

    @property
    def executor(self):
        return get_executor(getattr(self, "benchmark", "robolab"))

    def _prepare_session(self, log_dir: Path) -> Path:
        """Reset shared robot state and allocate a private episode workspace."""
        self.close()
        self._cancelled.clear()
        self.last_response_text = self.last_usage = None
        self._queue = deque()
        self._buffer = b""
        self._request_id = 0
        self._pending_motion = None
        self.last_action_metadata = None
        self._seen_calls = set()
        self._last_control_step = 0
        self._goal_status = "active"
        self._turn_started = False
        self._turn_id = None
        self._phase = None
        log_dir = Path(log_dir).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        self.events_path, self.stderr_path = log_dir / "events.jsonl", log_dir / "stderr.log"
        (log_dir / "prompt.md").write_text(self._thread_prompt, encoding="utf-8")
        self._log("prompt", {"path": "prompt.md", "sha256": hashlib.sha256(
            self._thread_prompt.encode("utf-8")).hexdigest(), "options": self._prompt_options})
        # Codex refuses to create sandbox helpers under the system temp directory.
        # Use the experiment output directory for this disposable private store.
        self._directory = TemporaryDirectory(prefix="runtime_", dir=log_dir)
        root = Path(self._directory.name)
        work = root / "workspace"
        work.mkdir()
        return work

    def _dynamic_tools(self) -> list[dict]:
        vla_tools = ([VLA_REFERENCE_TOOL, refine_act_tool(self.action_format)]
                     if self.vla_as_reference else [VLA_TOOL]) if self.enable_vla else []
        return [{**tool, "description": tool["description"].replace(
            "episode task", "current goal's task")} for tool in [*self.tools, *vla_tools]]

    def _start_session(self, *, log_dir: Path, deadline: float) -> None:
        work = self._prepare_session(log_dir)
        root = work.parent
        try:
            codex_home = root / "codex"
            codex_home.mkdir(mode=0o700)
            # Native goals require persisted threads, outside the readable workspace.
            self.env["CODEX_HOME"] = str(codex_home)
            config = self._config()
            self._check_sandbox(work, config, deadline=deadline)
            with self._process_lock:
                self._check_cancelled()
                self._stderr = self.stderr_path.open("ab")
                self._process = subprocess.Popen(
                    [self.executable, "app-server", "--stdio", "--strict-config", *config],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                    env=self.env,
                    cwd=work, start_new_session=True,
                )
            self._selector = selectors.DefaultSelector()
            self._selector.register(self._process.stdout, selectors.EVENT_READ)
            self._rpc("initialize", {"clientInfo": {"name": "robot_harness", "version": "2"},
                                     "capabilities": {"experimentalApi": True}}, deadline)
            self._send({"method": "initialized", "params": {}})
            result = self._rpc("thread/start", {
                "model": self.model, "modelProvider": "robot_harness", "cwd": str(work),
                "ephemeral": False, "approvalPolicy": "never", "permissions": "robot_harness",
                "developerInstructions": self._thread_prompt,
                "dynamicTools": self._dynamic_tools(),
            }, deadline)
            self._thread_id = result["thread"]["id"]
        except BaseException:
            self.close()
            raise

    def act(self, observation: dict, *, control_step: int,
            async_context: AsyncInferenceContext | None = None) -> dict | JointActionChunk | ActionChunk:
        self._async_context = async_context
        self._check_cancelled()
        self.last_action_metadata = None
        deadline = time.monotonic() + self.timeout
        self.executor.robot_state(observation)
        if self._phase == "plan":
            self._act_plan(observation, control_step=control_step, deadline=deadline)
        return self._act_execute(observation, control_step=control_step, deadline=deadline)

    def _set_phase(self, phase: str, deadline: float) -> None:
        """Wait for the native mode to apply before a turn or goal can start."""
        if phase not in ("plan", "execute"):
            raise ValueError("phase must be 'plan' or 'execute'")
        if self._phase == phase:
            return
        mode = "plan" if phase == "plan" else "default"
        self._rpc("thread/settings/update", {
            "threadId": self._thread_id, "collaborationMode": {
                "mode": mode,
                "settings": {
                    "model": self.model, "reasoning_effort": self.reasoning_effort,
                    "developer_instructions": None,
                },
            },
        }, deadline)
        deferred = []
        try:
            while True:
                event = self._next_event(deadline)
                if event.get("method") == "thread/settings/updated":
                    params = event.get("params")
                    if not isinstance(params, dict) or not isinstance(params.get("threadId"), str):
                        raise PolicyProtocolError("Malformed Codex settings notification")
                    if params["threadId"] == self._thread_id:
                        settings = params.get("threadSettings")
                        applied = settings.get("collaborationMode") if isinstance(settings, dict) else None
                        if not isinstance(applied, dict) or not isinstance(applied.get("mode"), str):
                            raise PolicyProtocolError("Malformed Codex collaboration mode notification")
                        if applied["mode"] == mode:
                            break
                elif event.get("method") == "error":
                    self._check_event(event, require_goal=False)
                deferred.append(event)
        except TimeoutError as error:
            raise TimeoutError(f"Timed out waiting for Codex {mode} settings") from error
        finally:
            # _rpc may already have queued notifications before its acknowledgement.
            # Preserve every unrelated event, including its order relative to later events.
            self._queue.extendleft(reversed(deferred))
        self._phase = phase

    def _act_plan(self, observation: dict, *, control_step: int, deadline: float) -> None:
        """Plan from public observations without creating a goal or moving the robot."""
        content = self._observation(observation, control_step, done=False)
        result = self._rpc("turn/start", {
            "threadId": self._thread_id,
            "input": [{"type": "text", "text":
                "Plan this robot episode.\n"
                f"Task instruction: {self._instruction}\nControl period: {self._control_dt} seconds. "
                f"Step budget: {self._max_steps}."}],
        }, deadline)
        plan_turn = result["turn"]["id"]
        plan_text = None
        while True:
            event = self._next_event(deadline)
            self._check_event(event, require_goal=False)
            params = event.get("params", {})
            if params.get("threadId", self._thread_id) != self._thread_id:
                raise PolicyProtocolError("Unexpected thread during planning")
            if event.get("method") == "item/tool/call":
                tool = self._tool_params(event)
                if tool["tool"] != "observe" or tool["arguments"] != {}:
                    raise PolicyProtocolError("Planning only allows observe(); robot actions are unavailable")
                self._tool_result(event["id"], content)
            elif event.get("method") == "thread/goal/updated":
                raise PolicyProtocolError("Codex must not create a goal during planning")
            elif params.get("turnId") == plan_turn and event.get("method") == "item/completed":
                item = params.get("item", {})
                if item.get("type") == "plan":
                    plan_text = item.get("text")
            elif event.get("method") == "turn/completed" and params["turn"].get("id") == plan_turn:
                break
        try:
            document = json.loads(plan_text)
        except (TypeError, ValueError) as error:
            raise PolicyValidationError("Codex plan must contain a JSON subgoals list") from error
        if (not isinstance(document, dict) or set(document) != {"subgoals"}
                or not isinstance(document["subgoals"], list) or not document["subgoals"]
                or any(not isinstance(goal, str) or not goal.strip() for goal in document["subgoals"])):
            raise PolicyValidationError("Codex plan must contain a nonempty list of nonempty subgoal strings")
        self._subgoals = [goal.strip() for goal in document["subgoals"]]
        self.last_response_text = plan_text
        self._log("plan", {"subgoals": self._subgoals})
        # Setting an active goal can immediately start a turn. Switch modes first.
        self._set_phase("execute", deadline)
        self._turn_started = True  # Native goals start and continue execution turns.
        self._advance_goal(deadline)

    @property
    def _executing_subgoal(self) -> bool:
        return 0 <= self._subgoal_index < len(self._subgoals)

    @property
    def _execution_instruction(self) -> str:
        return self._subgoals[self._subgoal_index] if self._executing_subgoal else self._instruction

    def _advance_goal(self, deadline: float) -> None:
        """Activate the next subgoal, or the original episode task when none remain."""
        if self._pending_motion is not None:
            raise PolicyProtocolError("Cannot switch goals before the pending motion returns")
        self._vla_reference = None
        self._subgoal_index += 1
        self._turn_id = None
        if self._executing_subgoal:
            objective = (f"Execute subgoal {self._subgoal_index + 1}/{len(self._subgoals)}: "
                         f"{self._execution_instruction}\nStart with observe(). Wait for each motion result. "
                         "Mark this goal complete once public measured observations show the subgoal is achieved, "
                         "then finish the turn without further actions. The harness selects the next goal. "
                         "If a robot tool returns done=true, stop acting and complete the goal instead.")
        else:
            objective = ("Execute this robot episode toward the supplied task until a robot tool returns done=true. "
                         "Then stop acting and complete the goal. Task: " + self._instruction[:3000])
        result = self._rpc("thread/goal/set", {
            "threadId": self._thread_id, "status": "active", "objective": objective,
        }, deadline)
        self._goal_status = result["goal"]["status"]
        if self.plan:
            self._log("execution_goal", {
                "subgoal_index": self._subgoal_index if self._executing_subgoal else None,
                "instruction": self._execution_instruction,
            })

    def _act_execute(self, observation: dict, *, control_step: int,
                     deadline: float) -> dict | JointActionChunk | ActionChunk:
        execution_step = self._execution_step(control_step)
        # Freeze each segment's observation before issuing the next target;
        # the model receives the collected observations only when the batch ends.
        self._check_motion_boundary(control_step, deadline, done=False)
        content = self._observation(observation, control_step, done=False)
        pending = self._pending_motion
        if (pending is not None and pending.tool in VLA_TOOLS
                and execution_step < pending.start_control_step + pending.requested_steps):
            pending.action_index += 1
            chunk = self._issue_vla(observation, control_step)
            if chunk is not None:
                return chunk
            # An inference timeout must leave GPT its own recovery budget.
            deadline = time.monotonic() + self.timeout
            content = self._observation(observation, control_step, done=False)
        elif pending is not None and pending.action_index + 1 < len(pending.targets):
            pending.action_index += 1
            return self._issue_target(execution_step)
        if not self._turn_started:
            self._turn_started = True
            self._rpc("turn/start", {"threadId": self._thread_id, "input": [{
                "type": "text", "text": "Execute the current episode autonomously. Start with observe().\n"
                f"Task instruction: {self._instruction}\nControl period: {self._control_dt} seconds. "
                f"Step budget: {self._max_steps}.",
            }]}, deadline)
        self._complete_action(content)
        while True:
            try:
                event = self._next_event(deadline)
            except TimeoutError:
                if not (self._executing_subgoal and self._goal_status == "complete"):
                    self._require_active_goal()
                raise
            self._check_event(event, require_goal=True, allow_subgoal_completion=self._executing_subgoal)
            if self._executing_subgoal and self._goal_status == "complete":
                if (event.get("method") == "turn/completed"
                        and event["params"]["turn"].get("id") == self._turn_id):
                    self._advance_goal(deadline)
                    continue
                if event.get("method") != "item/tool/call":
                    continue  # Wait for the completed subgoal's turn to finish.
            if self._goal_status != "active":
                # A transport failure can block the goal before emitting its
                # error event. Preserve that diagnostic without allowing motion.
                deadline = min(deadline, time.monotonic() + 1)
            if event.get("method") != "item/tool/call":
                continue
            self._require_active_goal()
            params = self._tool_params(event)
            if params["tool"] == "observe":
                if params["arguments"] != {}:
                    raise PolicyValidationError("observe takes no arguments")
                self._tool_result(event["id"], content)
            elif params["tool"] == "vla" and self.vla_as_reference:
                arguments = params["arguments"]
                if not isinstance(arguments, dict) or not set(arguments) <= {"instruction"}:
                    raise PolicyValidationError(f"{params['tool']} takes only an optional instruction")
                instruction = arguments.get("instruction")
                if instruction is not None and (not isinstance(instruction, str) or not instruction.strip()):
                    raise PolicyValidationError("VLA instruction must be a nonempty string or null")
                was_available = self._vla_error is None
                self._query_vla_reference(event["id"], params["callId"], observation, control_step,
                                          instruction or self._execution_instruction)
                if was_available and self._vla_error is not None:
                    deadline = time.monotonic() + self.timeout
                content = self._observation(observation, control_step, done=False)
            elif params["tool"] in MOTION_TOOLS:
                if params["tool"] == "multiact" and not self.multiact:
                    raise PolicyValidationError("multiact is not enabled")
                remaining = self._max_steps - execution_step
                reference = None
                if params["tool"] in VLA_TOOLS:
                    if not self.enable_vla:
                        raise PolicyValidationError("VLA is not enabled")
                    arguments = params["arguments"]
                    allowed = {"steps", "instruction"}
                    required = {"steps"}
                    if not isinstance(arguments, dict) or not required <= set(arguments) <= allowed:
                        raise PolicyValidationError("vla takes steps and an optional instruction")
                    steps = validate_policy_output(positive_int, arguments["steps"], "steps")
                    if steps > remaining:
                        raise PolicyValidationError("steps cannot exceed the remaining episode steps")
                    instruction = arguments.get("instruction")
                    if instruction is not None and (not isinstance(instruction, str) or not instruction.strip()):
                        raise PolicyValidationError("VLA instruction must be a nonempty string or null")
                    self._pending_motion = _PendingMotion(
                        event["id"], params["callId"], params["tool"], [], execution_step,
                        requested_steps=steps, instruction=instruction or self._execution_instruction,
                    )
                    self._vla_reference = None
                    was_available = self._vla_error is None
                    chunk = self._issue_vla(observation, control_step)
                    if chunk is not None:
                        return chunk
                    if was_available and self._vla_error is not None:
                        deadline = time.monotonic() + self.timeout
                    content = self._observation(observation, control_step, done=False)
                    continue
                elif params["tool"] == "act" and self.direct_action_chunk:
                    arguments = params["arguments"]
                    if not isinstance(arguments, dict) or set(arguments) != {"actions"}:
                        raise PolicyValidationError("direct act takes only an actions list")
                    normalized, actions = validate_policy_output(
                        self.executor.prepare_action_chunk, observation, arguments["actions"],
                        action_chunk_size=self.action_chunk_size, action_format=self.action_format,
                    )
                    self._pending_motion = _PendingMotion(
                        event["id"], params["callId"], params["tool"], [], execution_step,
                        direct_actions=actions,
                        direct_payload=tuple(normalized) if self.action_format.endswith("_delta") else None,
                        direct_mode=int(self.action_format.startswith("joint_")),
                    )
                    self._vla_reference = None
                    return self._issue_direct_action_chunk(actions, execution_step)
                elif params["tool"] == "refine_act":
                    if not self.vla_as_reference:
                        raise PolicyValidationError("refine_act requires VLA reference mode")
                    reference = self._vla_reference
                    if (reference is None or reference.control_step != control_step
                            or reference.execution_step != execution_step):
                        raise PolicyValidationError("refine_act requires a fresh VLA reference for this execution start")
                    # Refinement has a fixed interpolation horizon even when
                    # ordinary eef_destination actions use velocity-limited prefixes.
                    refinement_format = "eef_absolute" if self.action_format == "eef_destination" else self.action_format
                    targets = [validate_policy_output(self.executor.validate_fixed_target, params["arguments"],
                                                      action_format=refinement_format, action_chunk_size=15)]
                elif params["tool"] == "act":
                    targets = [validate_policy_output(self.executor.validate_target, params["arguments"], remaining_steps=remaining,
                                               action_format=self.action_format)
                               if self.adaptive_action_chunk else validate_policy_output(self.executor.validate_fixed_target,
                                   params["arguments"], action_format=self.action_format,
                                   action_chunk_size=self.action_chunk_size)]
                else:
                    arguments = params["arguments"]
                    if not isinstance(arguments, dict) or set(arguments) != {"actions"}:
                        raise PolicyValidationError("multiact takes only an actions list")
                    targets = validate_policy_output(self.executor.validate_actions, arguments["actions"], remaining_steps=remaining,
                                               adaptive_action_chunk=self.adaptive_action_chunk,
                                               action_format=self.action_format, action_chunk_size=self.action_chunk_size)
                self._pending_motion = _PendingMotion(
                    event["id"], params["callId"], params["tool"], targets, execution_step,
                    reference=reference,
                )
                self._vla_reference = None
                return self._issue_target(execution_step)
            else:
                raise PolicyValidationError(f"Unknown robot tool: {params['tool']}")

    def end_episode(self, observation: dict, *, control_step: int, reason: str) -> str:
        self._vla_reference = None
        deadline = time.monotonic() + self.finish_timeout
        try:
            if reason == "execution_error":
                return "aborted"
            self._check_motion_boundary(control_step, deadline, done=True)
            self._async_context = None
            content = self._observation(observation, control_step, done=True, reason=reason)
            self._complete_action(content)
            while True:
                event = self._next_event(deadline)
                self._check_event(event, require_goal=False)
                if event.get("method") == "item/tool/call":
                    params = self._tool_params(event)
                    if params["tool"] == "observe" and params["arguments"] == {}:
                        self._tool_result(event["id"], content)
                    else:
                        self._tool_result(event["id"], content if self.async_inference else [{
                            "type": "inputText", "text": "Episode ended; do not act again."}], success=False)
                elif event.get("method") == "turn/completed":
                    # Stop any goal continuation even if the agent forgot to
                    # mark it complete. This records completion of execution,
                    # never a benchmark success claim.
                    self._rpc("thread/goal/set", {"threadId": self._thread_id, "status": "complete"}, deadline)
                    return "completed"
        except TimeoutError:
            return "timeout"
        except Exception as error:
            self._log("finish_error", {"type": type(error).__name__, "message": str(error)})
            return "error"
        finally:
            self.close()

    def _observation(self, observation: dict, control_step: int, *, done: bool, reason=None) -> list[dict]:
        state = {"robot_state": self.executor.robot_state(observation), "control_step": control_step,
                 "action_format": self.action_format,
                 "remaining_steps": max(0, self._max_steps - control_step), "control_dt": self._control_dt,
                 "executed_steps": max(0, control_step - self._last_control_step), "done": done}
        if reason is not None:
            state["end_reason"] = reason
        if self.vla is not None:
            state["vla_available"] = self._vla_error is None
        images = self._image_content(observation)
        pending = self._pending_motion
        if pending is not None and pending.tool in VLA_TOOLS:
            state.update(executed_steps=max(0, control_step - pending.start_control_step))
            state["requested_steps"] = pending.requested_steps
        if pending is not None and pending.direct_actions is not None:
            state["direct_action_chunk"] = True
            state["action_chunk_size"] = len(pending.direct_actions)
        if pending is not None and pending.tool == "multiact":
            state["executed_steps"] = max(0, control_step - pending.start_control_step)
            remaining = state["executed_steps"]
            per_action = []
            last_executed_index = None
            for index, target in enumerate(pending.targets):
                executed = min(remaining, target["steps"])
                per_action.append(executed)
                remaining -= executed
                if executed:
                    last_executed_index = index
            state["action_executed_steps"] = per_action
            snapshot = {
                "action_index": last_executed_index, "control_step": control_step,
                "executed_steps": per_action[last_executed_index] if last_executed_index is not None else 0,
                "robot_state": state["robot_state"],
            }
            # Encoding here freezes mutable simulator arrays. Replacing the same
            # control step preserves order and the latest final measurement,
            # without inventing a snapshot for an issued but unexecuted target.
            pending.observations[control_step] = [
                {"type": "inputText", "text": json.dumps(snapshot, allow_nan=False)}, *images,
            ]
        context = self._async_context
        if context is not None and not done:
            state["async_inference"] = {
                "observation_control_step": control_step,
                "reference_control_step": context.previous.control_step,
                "execution_start_control_step": context.execution_start,
                "pending_tail": context.tail,
            }
            state["next_action_remaining_steps"] = max(0, self._max_steps - context.execution_start)
            snapshots = {}
            samples = {sample.control_step: sample for sample in (context.previous, context.current)}
            for sample in samples.values():
                snapshot = {"episode_id": sample.episode_id, "phase": sample.phase, "chunk_id": sample.chunk_id,
                            "control_step": sample.control_step, "robot_state": self.executor.robot_state(sample.observation)}
                if pending is not None and pending.tool == "multiact":
                    elapsed = max(0, sample.control_step - pending.start_control_step)
                    counts = []
                    for target in pending.targets:
                        count = min(elapsed, target["steps"])
                        counts.append(count)
                        elapsed -= count
                    indices = [i for i, count in enumerate(counts) if count]
                    index = indices[-1] if indices else None
                    snapshot.update(action_index=index, executed_steps=counts[index] if index is not None else 0)
                items = [{"type": "inputText", "text": json.dumps(snapshot, allow_nan=False)}]
                items.extend(images if sample.control_step == control_step else self._image_content(sample.observation))
                snapshots[sample.control_step] = items
            if pending is not None and pending.tool == "multiact":
                pending.observations.update(snapshots)
                pending.observations = dict(sorted(pending.observations.items()))
            return [{"type": "inputText", "text": json.dumps(state, allow_nan=False)},
                    *(item for step in sorted(snapshots) for item in snapshots[step])]
        return [{"type": "inputText", "text": json.dumps(state, allow_nan=False)}, *images]

    def _image_content(self, observation):
        images = []
        names = camera_names(getattr(self, "benchmark", "robolab"), self.views)
        for name in names:
            images.extend([{"type": "inputText", "text": f"Camera: {name}"},
                           {"type": "inputImage", "imageUrl": image_data_url(observation[name], self.image_size)}])
        return images

    def _execution_step(self, control_step: int) -> int:
        return self._async_context.execution_start if self._async_context is not None else control_step

    def commit_action(self, *, control_step: int, steps: int) -> None:
        """Called only after speculative inference has returned and adaptation is complete."""
        self._last_control_step = control_step
        self._pending_motion.issued_steps = steps
        self._action_committed = True

    def _issue_direct_action_chunk(self, actions: np.ndarray, control_step: int) -> ActionChunk:
        """Return a validated direct chunk without endpoint interpolation."""
        pending = self._pending_motion
        if pending is None or pending.direct_actions is None:
            raise PolicyProtocolError("No direct action chunk is pending")
        execution_step = self._execution_step(control_step)
        remaining = self._max_steps - execution_step
        limit = min(len(actions), remaining)
        if limit < 1:
            raise PolicyValidationError("Episode step budget exhausted")
        prepared = np.asarray(actions[:limit], dtype=np.float32).copy()
        pending.issued_steps = len(prepared)
        self._last_control_step = execution_step
        self._action_committed = self._async_context is None
        self.last_action_metadata = {
            "tool": pending.tool, "tool_call_id": pending.call_id,
            "direct_action_chunk": True, "action_format": self.action_format,
            "action_chunk_size": len(actions), "issued_steps": len(prepared),
        }
        return ActionChunk(
            prepared, mode=int(pending.direct_mode), execution_limit=remaining,
            relative_actions=pending.direct_payload, action_format=self.action_format,
        )

    def cancel_prefetch(self) -> None:
        self._prefetch_cancelled.set()

    def refresh_vla(self, observation, *, control_step, async_context):
        """Replace an entirely expired prediction without advancing its delegation."""
        self._async_context = async_context
        chunk = self._issue_vla(observation, control_step)
        if chunk is not None:
            return chunk
        return self.act(observation, control_step=control_step, async_context=async_context)

    def _issue_target(self, control_step: int) -> dict:
        self._check_cancelled()
        pending = self._pending_motion
        self.last_action_metadata = {
            "tool_call_id": pending.call_id, "tool": pending.tool,
            "action_format": self.action_format,
            "action_index": pending.action_index, "action_count": len(pending.targets),
        }
        if pending.reference is not None:
            self.last_action_metadata.update(
                reference_tool_call_id=pending.reference.call_id,
                reference_control_step=pending.reference.control_step,
                reference_steps=pending.reference.steps,
            )
        self._last_control_step = control_step
        target = pending.targets[pending.action_index]
        # Keep nominal durations in pending.targets so batch summaries include
        # zero-step entries for targets discarded at the episode boundary.
        if self.action_format == "eef_destination" and pending.tool != "refine_act":
            # The executor computes the velocity-limited horizon from the
            # measured state, then the policy returns only one fixed prefix.
            pending.issued_steps = min(self.action_chunk_size, self._max_steps - control_step)
            self._action_committed = self._async_context is None
            return target
        if not self.adaptive_action_chunk or pending.tool == "refine_act":
            target = {**target, "steps": min(target["steps"], self._max_steps - control_step)}
        pending.issued_steps = target["steps"]
        self._action_committed = self._async_context is None
        return target

    def _handle_vla_error(self, error: Exception, *, phase: str) -> None:
        self._vla_reference = None
        self._check_cancelled()  # Cancellation is not a service failure.
        if isinstance(error, PrefetchCancelled):
            raise error
        self._vla_error = self._vla_error or f"{type(error).__name__}: {error}"

    def _query_vla_reference(self, request_id, call_id, observation: dict, control_step: int,
                             instruction: str) -> None:
        self._vla_reference = None
        remaining = self._max_steps - self._execution_step(control_step)
        skipped = self._execution_step(control_step) - control_step
        horizon = min(15, skipped + remaining)
        result = {"control_step": control_step, "remaining_steps": self._max_steps - control_step,
                  "control_dt": self._control_dt,
                  "action_format": self.action_format, "reference_state": self.executor.robot_state(observation),
                  "reference_control_step": control_step,
                  "instruction": instruction, "executed_steps": 0, "done": False}
        if self._async_context is not None:
            result["next_action_remaining_steps"] = remaining
        started = time.monotonic()
        try:
            self._check_cancelled()
            if self._vla_error is not None:
                raise RuntimeError(f"VLA unavailable until harness restart: {self._vla_error}")
            if horizon < 1:
                raise ValueError("Episode step budget exhausted")
            actions = validate_joint_actions(self.vla.infer(
                observation, instruction, check_cancelled=self._check_cancelled))
            self._check_cancelled()
            prediction = actions[horizon - 1]
            if self.action_format.startswith("joint_"):
                endpoint = {"joint_positions": prediction[:7].tolist()}
            else:
                position, quaternion = droid_forward_kinematics(prediction[:7])
                endpoint = {"position": position, "quaternion": quaternion}
            action = reference_target({**endpoint, "gripper": int(prediction[7]), "steps": min(15, remaining)},
                                      observation, action_format=self.action_format)
            del action["steps"]
            result.update(action=action, reference_steps=horizon)
            self._vla_reference = _VLAReference(call_id, control_step, self._execution_step(control_step), horizon)
        except Exception as error:
            self._handle_vla_error(error, phase="reference")
            result.update(error=self._vla_error, return_reason="service_error")
        result.update(vla_available=self._vla_error is None, infer_ms=(time.monotonic() - started) * 1000)
        self._log("vla_reference_query", {"tool_call_id": call_id, **result})
        self._tool_result(request_id, [{"type": "inputText", "text": json.dumps(result, allow_nan=False)}],
                          success=self._vla_error is None)


    def _issue_vla(self, observation: dict, control_step: int) -> JointActionChunk | None:
        pending = self._pending_motion
        started = time.monotonic()
        try:
            self._check_cancelled()
            if self._vla_error is not None:
                raise RuntimeError(f"VLA unavailable until harness restart: {self._vla_error}")
            actions = validate_joint_actions(self.vla.infer(
                observation, pending.instruction, check_cancelled=self._check_cancelled,
            ))
        except Exception as error:
            self._handle_vla_error(error, phase=pending.tool)
            content = self._observation(observation, control_step, done=False)
            state = json.loads(content[0]["text"])
            state["error"] = self._vla_error
            state["return_reason"] = "service_error"
            content[0] = {"type": "inputText", "text": json.dumps(state, allow_nan=False)}
            self._log("vla_error", {"tool_call_id": pending.call_id, "instruction": pending.instruction,
                                    "elapsed_ms": (time.monotonic() - started) * 1000, **state})
            self._complete_action(content, success=False)
            self._last_control_step = control_step
            return None
        self._check_cancelled()
        execution_step = self._execution_step(control_step)
        remaining = pending.requested_steps - (execution_step - pending.start_control_step)
        if self._async_context is None:
            actions = actions[:min(15, remaining)].copy()
        pending.issued_steps = len(actions)
        self._last_control_step = execution_step
        self._action_committed = self._async_context is None
        self.last_action_metadata = {
            "tool": pending.tool, "tool_call_id": pending.call_id, "chunk_index": pending.action_index,
            "instruction": pending.instruction, "requested_steps": pending.requested_steps,
            "infer_ms": (time.monotonic() - started) * 1000,
        }
        return JointActionChunk(actions, execution_limit=remaining)

    def _check_motion_boundary(self, control_step: int, deadline: float, *, done: bool) -> None:
        pending = self._pending_motion
        if pending is None:
            return
        expected = self._last_control_step + pending.issued_steps
        early = (self._async_context is not None and not done
                 and self._async_context.execution_start == expected)
        lower = 0 if done and not self._action_committed else self._last_control_step
        if not lower <= control_step <= expected or (not done and not early and control_step != expected):
            raise PolicyValidationError("Unexpected control_step for pending motion")
        # Check every segment boundary without completing the pending tool.
        # This collects overlapping calls and failures before any next target.
        goal = self._rpc("thread/goal/get", {"threadId": self._thread_id}, deadline)["goal"]
        if goal is None:
            raise PolicyProtocolError("Codex cleared its goal before the environment ended")
        self._goal_status = goal["status"]
        self._require_active_goal()
        deferred = deque()
        while self._queue:
            event = self._queue.popleft()
            self._check_event(event, require_goal=True)
            if event.get("method") == "item/tool/call":
                if event["params"]["tool"] in MOTION_TOOLS:
                    raise PolicyProtocolError("Wait for each act or multiact or refine_act or vla result before issuing another motion")
                deferred.append(event)
        self._queue.extend(deferred)
        self._require_active_goal()

    def _complete_action(self, content: list[dict], *, success: bool = True) -> None:
        if self._pending_motion is not None:
            pending = self._pending_motion
            if pending.tool == "multiact":
                content = [content[0], *(item for step in sorted(pending.observations)
                                         for item in pending.observations[step])]
            self._tool_result(pending.request_id, content, success=success)
            self._pending_motion = None

    def _tool_params(self, event: dict) -> dict:
        params = event.get("params")
        if (not isinstance(params, dict) or not {"threadId", "callId", "tool", "arguments"} <= params.keys()
                or not isinstance(params["callId"], str) or not isinstance(params["tool"], str)):
            raise PolicyProtocolError("Malformed Codex tool call")
        if params["threadId"] != self._thread_id or params["callId"] in self._seen_calls:
            raise PolicyProtocolError("Unexpected thread or repeated tool call; no replay is allowed")
        self._seen_calls.add(params["callId"])
        return params

    def _check_event(self, event: dict, *, require_goal: bool, allow_subgoal_completion: bool = False) -> None:
        method, params = event.get("method"), event.get("params", {})
        if not isinstance(params, dict):
            raise PolicyProtocolError("Malformed Codex event parameters")
        if method == "item/tool/call" and (
                type(event.get("id")) not in (int, str)
                or not {"threadId", "callId", "tool", "arguments"} <= params.keys()
                or any(not isinstance(params[key], str) or not params[key] for key in ("threadId", "callId", "tool"))):
            raise PolicyProtocolError("Malformed Codex tool call")
        if method == "item/completed" and not isinstance(params.get("item"), dict):
            raise PolicyProtocolError("Malformed Codex item event")
        if method in ("turn/started", "turn/completed") and (
                not isinstance(params.get("turn"), dict)
                or ((method == "turn/started" or "id" in params["turn"])
                    and not isinstance(params["turn"].get("id"), str))
                or (method == "turn/completed" and not isinstance(params["turn"].get("status"), str))):
            raise PolicyProtocolError("Malformed Codex turn event")
        if method == "thread/goal/updated" and (
                not isinstance(params.get("goal"), dict) or not isinstance(params["goal"].get("status"), str)):
            raise PolicyProtocolError("Malformed Codex goal event")
        if method == "turn/started":
            self._turn_id = params["turn"]["id"]
        elif method == "thread/goal/updated":
            self._goal_status = params["goal"]["status"]
        elif method == "thread/goal/cleared" and require_goal:
            raise PolicyProtocolError("Codex cleared its goal before the environment ended")
        elif method == "thread/tokenUsage/updated":
            self.last_usage = params.get("tokenUsage")
        elif method == "item/completed" and params.get("item", {}).get("type") == "agentMessage":
            self.last_response_text = params["item"].get("text")
        elif method == "turn/completed":
            turn = params["turn"]
            if turn["status"] != "completed":
                raise RuntimeError(f"Codex turn {turn['status']}: {turn.get('error')}")
            if require_goal and not (allow_subgoal_completion and self._goal_status == "complete"):
                self._require_active_goal()
            # Native goals may continue after an intermediate completed turn.
        elif method == "error" and not params.get("willRetry", False):
            raise RuntimeError(f"Codex error: {params.get('error', params)}")
        elif "id" in event and method and method != "item/tool/call":
            self._send({"id": event["id"], "error": {"code": -32601, "message": "Interactive requests are unavailable"}})
            raise PolicyProtocolError(f"Unexpected Codex request: {method}")

    def _require_active_goal(self) -> None:
        if self._goal_status != "active":
            raise PolicyProtocolError(f"Codex stopped its goal before the environment ended ({self._goal_status})")

    def _rpc(self, method: str, params: dict, deadline: float) -> dict:
        self._request_id += 1
        request_id = self._request_id
        self._send({"id": request_id, "method": method, "params": params})
        while True:
            event = self._read(deadline)
            if event.get("id") == request_id and "method" not in event:
                if "error" in event:
                    raise RuntimeError(f"{method}: {event['error']}")
                if not isinstance(event.get("result"), dict):
                    raise PolicyProtocolError(f"Malformed Codex RPC result: {method}")
                result = event["result"]
                field = {"thread/start": "thread", "turn/start": "turn",
                         "thread/goal/set": "goal", "thread/goal/get": "goal"}.get(method)
                if field is not None:
                    value = result.get(field)
                    nullable = method == "thread/goal/get" and field in result and value is None
                    if not nullable and (not isinstance(value, dict)
                                         or not isinstance(value.get("status" if field == "goal" else "id"), str)):
                        raise PolicyProtocolError(f"Malformed Codex RPC result: {method}")
                return result
            self._queue.append(event)

    def _tool_result(self, request_id, content: list[dict], *, success: bool = True) -> None:
        self._send({"id": request_id, "result": {"contentItems": content, "success": success}})

    def _send(self, event: dict) -> None:
        self._check_cancelled()
        self._process.stdin.write((json.dumps(event, allow_nan=False) + "\n").encode())
        self._process.stdin.flush()
        self._log("send", event)

    def _next_event(self, deadline: float) -> dict:
        self._check_cancelled()
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for Codex")
        return self._queue.popleft() if self._queue else self._read(deadline)

    def _read(self, deadline: float) -> dict:
        while b"\n" not in self._buffer:
            self._check_cancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for Codex")
            if not self._selector.select(min(remaining, 0.2)):
                continue
            data = os.read(self._process.stdout.fileno(), 65536)
            if not data:
                raise RuntimeError(f"Codex app-server closed its output; see {self.stderr_path}")
            self._buffer += data
        line, self._buffer = self._buffer.split(b"\n", 1)
        try:
            event = json.loads(line, parse_float=_finite_json_number, parse_constant=_finite_json_number)
        except (ValueError, UnicodeError) as error:
            raise PolicyProtocolError("Malformed Codex JSON event") from error
        if not isinstance(event, dict):
            raise PolicyProtocolError("Codex event must be an object")
        self._log("receive", event)
        return event

    def _log(self, direction: str, event: dict) -> None:
        if self.events_path is not None:
            with self.events_path.open("a") as stream:
                stream.write(json.dumps({"direction": direction, "event": event}, allow_nan=False) + "\n")

    def _check_cancelled(self) -> None:
        if self._cancelled.is_set() or self.connection_cancelled.is_set():
            raise RuntimeError("Codex session interrupted")
        if self._prefetch_cancelled.is_set():
            raise PrefetchCancelled("Unused asynchronous prediction cancelled at episode end")

    def abort(self) -> None:
        """Thread-safe cancellation; resource disposal waits for the reader to exit."""
        with self._process_lock:
            self._cancelled.set()
            if self._process is not None:
                try:
                    os.killpg(self._process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def close(self) -> None:
        self.abort()
        self._pending_motion = None
        self._vla_reference = None
        self._async_context = None
        self.last_action_metadata = None
        if self._process is not None:
            self._process.wait()
            self._process.stdin.close()
            self._process.stdout.close()
            self._process = None
        if self._selector is not None:
            self._selector.close()
            self._selector = None
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None

    def _check_sandbox(self, work: Path, config: list[str], *, deadline: float | None = None) -> None:
        """Fail before inference if the required tool isolation cannot run."""
        repo = Path(__file__).resolve().parents[2]
        codex_dir = Path(self.env.get("CODEX_HOME", str(Path.home() / ".codex")))
        probe = (
            "import os,socket; from pathlib import Path\n"
            "assert not any(name in os.environ for name in ('OPENAI_API_KEY','CODEX_API_KEY'))\n"
            f"assert not os.access({str(repo / 'README.md')!r},os.R_OK)\n"
            f"assert not os.access({str(codex_dir / 'auth.json')!r},os.R_OK)\n"
            "Path('.sandbox_probe').write_text('ok')\n"
            "Path('.sandbox_probe').unlink()\n"
            "try:\n s=socket.socket(); s.close()\n"
            "except PermissionError:\n pass\n"
            "else:\n raise RuntimeError('Tool networking is not isolated')\n"
        )
        result = subprocess.run(
            [self.executable, "sandbox", "-C", str(work), "-P", "robot_harness", *config,
             "--", "/usr/bin/python3", "-c", probe],
            env=self.env, capture_output=True, text=True,
            timeout=20 if deadline is None else min(20, max(0.001, deadline - time.monotonic())),
        )
        self.stderr_path.write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f"Codex tool isolation is unavailable: {result.stderr[-2000:]}")
