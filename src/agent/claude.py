"""Claude Code transport for the shared robot episode state machine.

The CLI owns model requests; the harness hosts its MCP tools over the SDK
control protocol. Native host tools are disabled. Robot motion is acknowledged
only after the environment returns its measured observation.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import subprocess
from tempfile import TemporaryFile
import time
from uuid import uuid4

from src.agent.client import CodexClient
from src.agent.errors import PolicyProtocolError, PolicyValidationError


logger = logging.getLogger(__name__)
EFFORTS = ("low", "medium", "high", "xhigh", "max")
CONTRACT_POLICY = "claude_bounded_recovery_v1"
CONTRACT_STEP_LIMIT = 3
CONTRACT_EPISODE_LIMIT = 10
CONTROL_TOOLS = [
    {"name": "get_goal", "description": "Read the current harness-assigned goal. Pass exactly {}.",
     "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "update_goal", "description": (
        "Acknowledge the episode goal with status=complete only after a robot result says done=true. "
        "Visual success is not permission to end the episode; blocked cannot end it early either. "
        "Exception: a harness-assigned planning subgoal can be completed once measured observations "
        "show that subgoal is achieved. Wait for every motion result first, then finish the turn."),
     "inputSchema": {"type": "object", "properties": {
         "status": {"type": "string", "enum": ["complete", "blocked"]}},
         "required": ["status"], "additionalProperties": False}},
    {"name": "exec_command", "description": (
        "Run a shell command for calculations or scratch files in the isolated episode workspace. "
        "System Python is available. No host files, credentials, or networking are accessible. "
        "Commands must finish within 60 seconds; output is truncated to 32000 bytes."),
     "inputSchema": {"type": "object", "properties": {"command": {"type": "string"}},
                     "required": ["command"], "additionalProperties": False}},
]


class ClaudeClient(CodexClient):
    """Reuse robot behavior, replacing only Codex's session/transport boundary."""

    backend = "claude"
    auth = "claude_settings"

    def _resolve_executable(self, source, local_path):
        executable = shutil.which("claude")
        fallback = Path.home() / ".local/bin/claude"
        if executable is None and fallback.is_file() and os.access(fallback, os.X_OK):
            executable = str(fallback)
        if executable is None:
            raise RuntimeError("Claude Code is missing; install claude in PATH or ~/.local/bin/claude")
        return os.path.abspath(executable)

    def _configure_environment(self):
        if self.reasoning_effort not in EFFORTS:
            raise ValueError(f"Claude reasoning_effort must be one of {', '.join(EFFORTS)}")
        self.codex_source = "system"
        self._settings_path = Path(self.env.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude")) / "settings.json"
        for key in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SESSION_ID"):
            self.env.pop(key, None)
        # Model traffic inherits the caller's proxies. Local robot traffic
        # must remain local even when all HTTP(S) proxy variants are exported.
        for key in ("NO_PROXY", "no_proxy"):
            entries = [v for v in self.env.get(key, "").split(",") if v]
            self.env[key] = ",".join(dict.fromkeys([*entries, "localhost", "127.0.0.1", "::1"]))
        self._compute_process = None
        self._mcp_pending = {}
        self._ended = False
        self._goal = None
        self._prompt_options["claude_contract_policy"] = {
            "version": CONTRACT_POLICY, "per_control_step_limit": CONTRACT_STEP_LIMIT,
            "per_episode_limit": CONTRACT_EPISODE_LIMIT,
        }
        self._thread_prompt += (
            "\nClaude transport: all robot and goal tools are provided by the robot MCP server. "
            "Use exec_command for isolated calculations and scratch files. "
            "Use get_goal to inspect the harness-assigned goal.\n"
            "Claude execution contract:\n"
            "- Call observe and get_goal with exactly {}. Never add _call, explanations, or other keys. "
            "Use only the parameters in each registered tool schema.\n"
            "- Only done=true in an actual robot observation permits episode completion. Your visual "
            "judgment, remaining_steps estimate, or text saying 'finished' cannot end the episode. "
            "While done=false, keep executing; neither complete nor blocked may stop the episode goal.\n"
            "- If placement appears finished, verify release and stability and withdraw the gripper "
            "as appropriate. If needed, advance simulation with an act holding the measured pose and "
            "the appropriate legal gripper command. observe() does not advance simulation: repeated "
            "observations or a text-only 'waiting' response cannot make the environment end.\n"
            "- Wait for each motion result before issuing another motion. Never replay a motion after "
            "a transport error. For a recoverable tool error, correct the indicated call in this same "
            "episode; do not interpret that error as episode termination. Recovery is bounded.\n"
            "- When done=true, stop acting and acknowledge complete, then finish your response. "
            "done means the episode ended, not that it succeeded.\n"
            "- Only a harness-assigned planning subgoal may be completed before done=true, once "
            "measured observations establish its objective; then finish the turn and let the harness "
            "select the next goal. Completing a subgoal never completes the episode.\n")
        if not self.adaptive_action_chunk:
            self._thread_prompt += "- Fixed action chunks: never add a steps argument to act or multiact targets.\n"

    def _runtime_settings(self):
        # These settings override the user file for this process only. Keep the
        # user's API endpoint, token and model settings without logging secrets.
        settings = json.loads(self._settings_path.read_text()) if self._settings_path.is_file() else {}
        if not isinstance(settings, dict) or not isinstance(settings.get("env", {}), dict):
            raise ValueError("Claude settings must be an object with an env object")
        allowed = {key: settings[key] for key in ("env", "model", "modelSettings", "apiKeyHelper") if key in settings}
        allowed["env"] = {**allowed.get("env", {}),
            "CLAUDE_CODE_MAX_RETRIES": str(self.codex_max_retries),
            "CLAUDE_STREAM_IDLE_TIMEOUT_MS": str(self.codex_stream_idle_timeout * 1000),
            "CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS": str(self.codex_stream_idle_timeout * 1000),
            "API_TIMEOUT_MS": str(max(1, int(self.timeout * 1000))),
            "CLAUDE_CODE_RETRY_WATCHDOG": "0",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_AUTOUPDATER": "1",
        }
        return allowed

    def _start_session(self, *, log_dir: Path, deadline: float):
        work = self._prepare_session(log_dir)
        self._work = work
        self._thread_id = str(uuid4())
        self._mcp_pending = {}
        self._mcp_seen = set()
        self._goal = None
        self._ended = False
        self._contract_step = 0
        self._contract_step_count = 0
        self._contract_episode_count = 0
        self._wire_turn = None
        self._turn_number = 0
        self._message_number = 0
        self._usage_by_message = {}
        self._finished_usage = {}
        self._current_usage = {}
        self._claude_initialized = False
        self._mcp_tools = [
            {key: tool[key] for key in ("name", "description", "inputSchema")}
            for tool in self._dynamic_tools()
        ] + CONTROL_TOOLS
        self._tool_names = {tool["name"] for tool in self._mcp_tools}
        for tool in self._mcp_tools:
            if tool["name"] == "observe":
                tool["description"] += " Pass exactly {}; do not add _call or any other argument. Does not advance simulation."
        try:
            self._check_sandbox(work, [], deadline=deadline)
            config_dir = work.parent / "claude"
            config_dir.mkdir(mode=0o700)
            settings = self._runtime_settings()
            settings_path = config_dir / "settings.json"
            settings_path.touch(mode=0o600)
            settings_path.write_text(json.dumps(settings))
            args = [self.executable, "-p", "--input-format", "stream-json", "--output-format", "stream-json",
                    "--verbose", "--include-partial-messages", "--model", self.model,
                    "--effort", self.reasoning_effort, "--tools", "", "--strict-mcp-config",
                    "--disable-slash-commands", "--no-session-persistence", "--setting-sources", "",
                    "--settings", str(settings_path), "--system-prompt", self._thread_prompt,
                    "--permission-mode", "dontAsk", "--allowedTools",
                    *[f"mcp__robot__{name}" for name in sorted(self._tool_names)]]
            with self._process_lock:
                self._check_cancelled()
                self._stderr = self.stderr_path.open("ab")
                self._process = subprocess.Popen(
                    args, cwd=work, env={**self.env, "CLAUDE_CONFIG_DIR": str(config_dir)},
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr, start_new_session=True)
            self._selector = selectors.DefaultSelector()
            self._selector.register(self._process.stdout, selectors.EVENT_READ)
            self._send({"type": "control_request", "request_id": "harness-initialize", "request": {
                "subtype": "initialize", "sdkMcpServers": ["robot"],
                "sdkMcpServerConfigs": {"robot": {"timeout": max(86400000, int(self.timeout * 1000)),
                                                   "disableAutoBackground": True}},
            }})
            while not self._claude_initialized:
                self._receive_wire(deadline)
            if self.codex_max_retries > 15:
                logger.warning("Claude Code may cap codex_max_retries=%d at its native limit (15); "
                               "the harness never replays motions to extend retries", self.codex_max_retries)
        except BaseException:
            self.close()
            raise

    def _emit(self, method, params, *, request_id=None):
        event = {"method": method, "params": {"threadId": self._thread_id, **params}}
        if request_id is not None:
            event["id"] = request_id
        self._log("receive", event)
        self._queue.append(event)

    def _start_turn(self, text):
        self._turn_number += 1
        self._turn_start_control_step = self._contract_step
        self._wire_turn = f"claude-turn-{self._turn_number}"
        self._current_usage = {}
        self._usage_by_message = {}
        self._emit("turn/started", {"turn": {"id": self._wire_turn}})
        self._send({"type": "user", "message": {"role": "user", "content": text}})
        return {"turn": {"id": self._wire_turn}}

    def _rpc(self, method, params, deadline):
        self._check_cancelled()
        if method == "thread/settings/update":
            mode = params["collaborationMode"]["mode"]
            self._emit("thread/settings/updated", {"threadSettings": {"collaborationMode": {"mode": mode}}})
            return {}
        if method == "thread/goal/set":
            self._goal = {**(self._goal or {}), **{k: params[k] for k in ("status", "objective") if k in params}}
            if params["status"] == "active" and self._turn_started:
                self._start_turn(self._goal["objective"])
            return {"goal": dict(self._goal)}
        if method == "thread/goal/get":
            # Collect pending concurrent calls before the next motion segment.
            while b"\n" in self._buffer or self._selector.select(0):
                self._receive_wire(deadline)
            return {"goal": dict(self._goal) if self._goal is not None else None}
        if method == "turn/start":
            text = "\n".join(item["text"] for item in params["input"])
            if self._phase != "plan" and self._goal:
                text += "\nCurrent goal: " + self._goal["objective"]
            return self._start_turn(text)
        raise PolicyProtocolError(f"Unsupported Claude session operation: {method}")

    def _read(self, deadline):
        while not self._queue:
            self._receive_wire(deadline)
        return self._queue.popleft()

    def _receive_wire(self, deadline):
        event = super()._read(min(deadline, time.monotonic() + self.codex_stream_idle_timeout))
        kind = event.get("type")
        if kind == "control_request":
            self._control_request(event, deadline)
        elif kind == "control_response":
            response = event.get("response", {})
            if response.get("request_id") == "harness-initialize":
                if response.get("subtype") != "success":
                    raise RuntimeError("Claude initialization failed: " + str(response.get("error")))
                self._claude_initialized = True
        elif kind == "assistant":
            message = event.get("message", {})
            if event.get("error"):
                raise RuntimeError(f"Claude API error: {event['error']}")
            self._record_message_usage(message)
            text = "\n".join(block.get("text", "") for block in message.get("content", [])
                             if block.get("type") == "text")
            if text:
                self._message_number += 1
                self._emit("item/completed", {"turnId": self._wire_turn,
                           "item": {"type": "agentMessage", "text": text,
                                    "id": event.get("uuid") or f"claude-message-{self._message_number}"}})
        elif kind == "result":
            if event.get("is_error") or event.get("subtype") != "success":
                raise RuntimeError(f"Claude turn failed ({event.get('subtype')}): "
                                   f"{event.get('errors', event.get('result', ''))}")
            if self._mcp_pending:
                raise PolicyProtocolError("Claude ended a turn with unanswered tool calls")
            self._current_usage = self._normalize_usage(event.get("usage", {}))
            self._publish_usage()
            self._finished_usage = self._total_usage()
            self._current_usage = {}
            if self._phase == "plan":
                text = event.get("result", "")
                plans = re.findall(r"<proposed_plan>\s*(.*?)\s*</proposed_plan>", text, re.S)
                self._emit("item/completed", {"turnId": self._wire_turn,
                           "item": {"type": "plan", "id": f"{self._wire_turn}-plan",
                                    "text": plans[0] if len(plans) == 1 else None}})
            self._emit("turn/completed", {"turn": {"id": self._wire_turn, "status": "completed"}})
            if self._phase == "execute" and self._goal and self._goal["status"] == "active" and not self._ended:
                # A text-only response is not episode termination. Continue the
                # same conversation without resubmitting any motion request.
                message = (
                    "The environment has not ended: done=false. Continue the current goal in this same episode. "
                    "Use observe({}) if needed, then act on the measured state. observe does not advance simulation. "
                    "If placement appears complete, verify release/stability and withdraw as appropriate; "
                    "use a pose-holding act if needed to advance simulation. "
                    "Only a completed planning subgoal may be acknowledged before done=true."
                )
                if self._contract_step == self._turn_start_control_step:
                    feedback = self._contract_error("turn", "no_progress_turn", message)
                    message = feedback["content"][0]["text"]
                else:
                    state = self._contract_state()
                    self._log("contract_continuation", {"policy": CONTRACT_POLICY, **state})
                    message += "\n" + json.dumps(state)
                self._start_turn(message)

    def _observation(self, observation, control_step, *, done, reason=None):
        content = super()._observation(observation, control_step, done=done, reason=reason)
        # Only measured environment progress resets the local allowance. Tool
        # acknowledgements, goal changes and new conversation turns do not.
        if control_step > self._contract_step:
            self._contract_step = control_step
            self._contract_step_count = 0
        self._ended |= done
        return content

    def _contract_state(self):
        return {
            "done": self._ended, "control_step": self._contract_step,
            "remaining_steps": max(0, self._max_steps - self._contract_step),
            "goal_status": self._goal["status"] if self._goal else None,
            "goal_kind": "subgoal" if self._executing_subgoal else "episode",
        }

    def _contract_error(self, tool, code, message):
        """Reject a side-effect-free call without mutating the goal or replaying motion."""
        self._contract_step_count += 1
        self._contract_episode_count += 1
        recoverable = (self._contract_step_count <= CONTRACT_STEP_LIMIT
                       and self._contract_episode_count <= CONTRACT_EPISODE_LIMIT)
        record = {
            "policy": CONTRACT_POLICY, "tool": tool, "code": code, "message": message,
            **self._contract_state(), "step_count": self._contract_step_count,
            "episode_count": self._contract_episode_count, "recoverable": recoverable,
        }
        self._log("contract_violation", record)
        if not recoverable:
            raise PolicyProtocolError(
                f"Claude contract recovery limit exceeded ({code}; "
                f"control step {self._contract_step}: {self._contract_step_count}/{CONTRACT_STEP_LIMIT}, "
                f"episode: {self._contract_episode_count}/{CONTRACT_EPISODE_LIMIT})")
        return {"content": [{"type": "text", "text": json.dumps(record)}], "isError": True}

    def _control_reply(self, request_id, response):
        self._send({"type": "control_response", "response": {
            "subtype": "success", "request_id": request_id, "response": response}})

    def _mcp_reply(self, request_id, message_id, result):
        self._control_reply(request_id, {"mcp_response": {"jsonrpc": "2.0", "id": message_id, "result": result}})

    def _control_request(self, event, deadline):
        request = event.get("request", {})
        request_id = event.get("request_id")
        if not isinstance(request_id, str) or not isinstance(request, dict):
            raise PolicyProtocolError("Malformed Claude control request")
        if request.get("subtype") == "can_use_tool":
            # Only the explicitly registered MCP tools may run without a human.
            self._control_reply(request_id, {"behavior": "deny", "message": "Only harness MCP tools are permitted"})
            return
        if request.get("subtype") != "mcp_message" or request.get("server_name") != "robot":
            raise PolicyProtocolError("Unexpected Claude control request")
        message = request.get("message", {})
        method, message_id = message.get("method"), message.get("id", 0)
        if method == "initialize":
            result = {"protocolVersion": message.get("params", {}).get("protocolVersion", "2024-11-05"),
                      "capabilities": {"tools": {}}, "serverInfo": {"name": "robot_harness", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": self._mcp_tools}
        elif method == "ping" or method == "notifications/initialized":
            result = {}
        elif method == "tools/call":
            if request_id in self._mcp_seen or message_id in self._mcp_seen:
                raise PolicyProtocolError("Repeated Claude tool call; no replay is allowed")
            self._mcp_seen.update((request_id, message_id))
            params = message.get("params", {})
            tool, arguments = params.get("name"), params.get("arguments", {})
            if tool not in self._tool_names or not isinstance(arguments, dict):
                raise PolicyValidationError(f"Unknown or malformed Claude tool: {tool}")
            if self._phase == "plan" and tool != "observe":
                raise PolicyProtocolError("Planning only allows observe(); robot actions are unavailable")
            if tool in ("observe", "get_goal") and arguments:
                result = self._contract_error(tool, "invalid_arguments",
                    f"{tool} takes no arguments. Call {tool}({{}}), without _call or any other keys. "
                    "The rejected call did not advance simulation or change the goal.")
                self._mcp_reply(request_id, message_id, result)
                return
            if tool in ("get_goal", "update_goal"):
                if tool == "update_goal":
                    if self._pending_motion is not None or self._mcp_pending:
                        raise PolicyProtocolError("Wait for motion results before updating the goal")
                    if not self._goal:
                        raise PolicyProtocolError("No harness-assigned goal")
                    if set(arguments) != {"status"} or arguments["status"] not in ("complete", "blocked"):
                        result = self._contract_error(tool, "invalid_arguments",
                            'update_goal accepts only {"status":"complete"} or {"status":"blocked"}. '
                            "The episode goal must remain active until done=true; only a measured, achieved "
                            "planning subgoal may be completed earlier. The goal was not changed.")
                        self._mcp_reply(request_id, message_id, result)
                        return
                    if not self._ended and not (self._executing_subgoal and arguments["status"] == "complete"):
                        result = self._contract_error(tool, "premature_goal_update",
                            f"Rejected status={arguments['status']}: the environment has not ended (done=false). "
                            "The goal remains active. Continue acting from measured observations; "
                            "visual success is not episode termination. observe({}) does not advance simulation. "
                            "Only an achieved planning subgoal may be completed before done=true.")
                        self._mcp_reply(request_id, message_id, result)
                        return
                    self._goal["status"] = arguments["status"]
                    self._emit("thread/goal/updated", {"goal": dict(self._goal)})
                result = {"content": [{"type": "text", "text": json.dumps({"goal": self._goal})}]}
            elif tool == "exec_command":
                if set(arguments) != {"command"} or not isinstance(arguments["command"], str):
                    raise PolicyValidationError("exec_command requires a command string")
                code, output = self._run_sandbox(["/bin/sh", "-c", arguments["command"]],
                                               min(deadline, time.monotonic() + 60))
                result = {"content": [{"type": "text", "text": f"Exit code: {code}\n{output}"}], "isError": code != 0}
            else:
                self._mcp_pending[request_id] = message_id
                self._emit("item/tool/call", {"callId": request_id, "tool": tool, "arguments": arguments},
                           request_id=request_id)
                return
        else:
            raise PolicyProtocolError(f"Unsupported robot MCP operation: {method}")
        self._mcp_reply(request_id, message_id, result)

    def _tool_result(self, request_id, content, *, success=True):
        if request_id not in self._mcp_pending:
            raise PolicyProtocolError("Claude tool response has no pending request")
        converted = []
        for item in content:
            if item["type"] == "inputText":
                converted.append({"type": "text", "text": item["text"]})
            elif item["type"] == "inputImage":
                header, data = item["imageUrl"].split(",", 1)
                converted.append({"type": "image", "mimeType": header[5:].split(";", 1)[0], "data": data})
            else:
                raise PolicyProtocolError("Unsupported Claude tool result content")
        self._log("send", {"id": request_id, "result": {"contentItems": content, "success": success}})
        self._mcp_reply(request_id, self._mcp_pending.pop(request_id), {"content": converted, "isError": not success})

    @staticmethod
    def _normalize_usage(usage):
        usage = usage if isinstance(usage, dict) else {}
        def count(key):
            value = usage.get(key)
            return value if type(value) is int and value >= 0 else None
        read, write, plain, output = (count(k) for k in (
            "cache_read_input_tokens", "cache_creation_input_tokens", "input_tokens", "output_tokens"))
        inputs = sum((plain, read, write)) if all(v is not None for v in (plain, read, write)) else None
        details = usage.get("output_tokens_details")
        thinking = details.get("thinking_tokens") if isinstance(details, dict) else None
        return {"inputTokens": inputs, "cachedInputTokens": read, "cacheWriteInputTokens": write,
                "outputTokens": output, "totalTokens": inputs + output if inputs is not None and output is not None else None,
                "reasoningOutputTokens": thinking if type(thinking) is int and thinking >= 0 else None}

    def _record_message_usage(self, message):
        if isinstance(message.get("usage"), dict) and isinstance(message.get("id"), str):
            self._usage_by_message[message["id"]] = self._normalize_usage(message["usage"])
            self._current_usage = {key: sum(values) if all(v is not None for v in values) else None
                                   for key in next(iter(self._usage_by_message.values()))
                                   for values in [[v[key] for v in self._usage_by_message.values()]]}
            self._publish_usage()

    def _publish_usage(self):
        self._emit("thread/tokenUsage/updated", {"tokenUsage": {"total": self._total_usage()}})

    def _total_usage(self):
        return {key: value + self._finished_usage.get(key, 0)
                if value is not None and self._finished_usage.get(key, 0) is not None else None
                for key, value in self._current_usage.items()}

    def _sandbox_command(self, command):
        binary = shutil.which("bwrap")
        if binary is None:
            raise RuntimeError("Claude harness tool isolation requires bubblewrap (bwrap)")
        args = [binary, "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL",
                "--ro-bind", "/usr", "/usr"]
        for name in ("bin", "sbin", "lib", "lib64"):
            path = Path("/") / name
            if path.is_symlink():
                args += ["--symlink", os.readlink(path), str(path)]
            elif path.exists():
                args += ["--ro-bind", str(path), str(path)]
        # Do not mount host /proc; this also works in containers that prohibit
        # mounting a fresh procfs in a child namespace.
        return [*args, "--dir", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                "--bind", str(self._work), "/workspace", "--chdir", "/workspace",
                "--clearenv", "--setenv", "PATH", "/usr/bin:/bin", "--", *command]

    def _run_sandbox(self, command, deadline):
        with TemporaryFile() as output:
            with self._process_lock:
                self._check_cancelled()
                process = subprocess.Popen(self._sandbox_command(command), stdin=subprocess.DEVNULL,
                                           stdout=output, stderr=subprocess.STDOUT,
                                           env={"PATH": "/usr/bin:/bin"}, start_new_session=True)
                self._compute_process = process
            try:
                while process.poll() is None:
                    self._check_cancelled()
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Claude scratch command timed out")
                    try:
                        process.wait(timeout=min(.1, max(.001, deadline - time.monotonic())))
                    except subprocess.TimeoutExpired:
                        pass
                output.seek(0)
                data = output.read(32001)
                return process.returncode, data[:32000].decode("utf-8", "replace") + ("\n[truncated]" if len(data) > 32000 else "")
            finally:
                self._kill_compute()
                process.wait()
                self._compute_process = None

    def _check_sandbox(self, work, config, *, deadline=None):
        self._work = work
        probe = (
            "import os,socket; from pathlib import Path; "
            f"assert not Path({str(Path(__file__).resolve().parents[2] / 'README.md')!r}).exists(); "
            f"assert not Path({str(self._settings_path)!r}).exists(); "
            "assert set(os.environ) <= {'PATH','PWD','LC_CTYPE'}; "
            "Path('.probe').write_text('ok'); Path('.probe').unlink(); "
            "s=socket.socket(); s.settimeout(1); assert s.connect_ex(('1.1.1.1',443)) != 0; s.close()")
        code, output = self._run_sandbox(["/usr/bin/python3", "-c", probe],
                                         min(deadline or float('inf'), time.monotonic() + 20))
        if code:
            raise RuntimeError(f"Claude tool isolation is unavailable: {output}")

    def _kill_compute(self):
        process = getattr(self, "_compute_process", None)
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def abort(self):
        super().abort()
        with self._process_lock:
            self._kill_compute()
