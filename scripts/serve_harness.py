"""Serve the RoboLab Codex or Claude harness."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path
import signal
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.client import CodexClient, is_claude_model, validate_codex_stream_idle_timeout, validate_reasoning_effort, validate_views
from src.agent.claude import ClaudeClient
from src.agent.executor import (
    ACTION_FORMATS, DEFAULT_ACTION_CHUNK_SIZE, DEFAULT_EEF_DESTINATION_ROTATION_SPEED,
    DEFAULT_EEF_DESTINATION_TRANSLATION_SPEED, positive_float, positive_int, validate_action_format,
)
from src.agent.prompts import NOTE_FILES, validate_disable_ik, validate_note_file
from src.agent.vla import VLAConfig, VLAService
from src.evaluation.robolab_bench.harness_policy import HarnessPolicy
from src.evaluation.robolab_bench.websocket_server import WebsocketPolicyServer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("robolab",), default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "config/harness/robolab.yaml")
    parser.add_argument("--log-dir", "--log_dir", type=Path,
                        help="Output directory; overrides config log_dir. Relative paths use the repository root.")
    parser.add_argument("--model", help="Model name; claude-* selects Claude Code, otherwise Codex; overrides config")
    parser.add_argument("--reasoning-effort", type=validate_reasoning_effort,
                        help="Reasoning effort (e.g. xhigh or max); overrides config")
    parser.add_argument("--codex-max-retries", type=int,
                        help="Model CLI HTTP/stream retry budget; overrides config (0 disables retries; CLI limits apply)")
    parser.add_argument("--codex-stream-idle-timeout", type=int, metavar="SECONDS",
                        help="Positive integer seconds without model stream events before timeout; "
                             "overrides config (default: 300 when absent from config)")
    parser.add_argument("--codex-source", choices=("system", "local"),
                        help="Use Codex from PATH or a local compiled binary; overrides config (default: system Codex from PATH)")
    parser.add_argument("--codex-local-path",
                        help="Compiled Codex binary; overrides config codex_local_path. Relative paths use the repository root; only used with source local.")
    parser.add_argument("--timeout", type=float,
                        help="Frontier-model action timeout in seconds, including retries; overrides config")
    parser.add_argument("--include-note", action=argparse.BooleanOptionalAction, default=None,
                        help="Append notice reminders distilled from failure cases (default: config, otherwise off)")
    parser.add_argument("--note-file", choices=NOTE_FILES,
                        help="Bundled notice-reminder file; overrides config note_file (default: note.md). Does not enable reminders.")
    parser.add_argument("--plan", action=argparse.BooleanOptionalAction, default=None,
                        help="Explicit subgoal planning; otherwise use implicit planning (default: config, otherwise off)")
    parser.add_argument("--multiact", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable short multi-action sequences in one tool call (default: config, otherwise off)")
    parser.add_argument("--action-format", choices=ACTION_FORMATS,
                        help="Frontier-model action representation; overrides YAML action_format (default: eef_absolute)")
    parser.add_argument("--adaptive-action-chunk", action=argparse.BooleanOptionalAction, default=None,
                        help="Adaptive chunking: model-selected act/multiact duration; otherwise use YAML action_chunk_size "
                             "(default: 15 frames). Adaptive mode defaults to config, otherwise off.")
    parser.add_argument("--direct-action-chunk", action=argparse.BooleanOptionalAction, default=None,
                        help="Frame-wise action generation (trajectory-level bias): act returns action_chunk_size frames")
    parser.add_argument("--async-inference", action=argparse.BooleanOptionalAction, default=None,
                        help="Asynchronous execution: overlap next-action inference with motion (default: off)")
    parser.add_argument("--async-tail-steps", type=int,
                        help="Control steps reserved for overlapping inference (default: 5)")
    parser.add_argument("--include-async-prompt", action=argparse.BooleanOptionalAction, default=None,
                        help="Append async_inference.md when async inference is enabled (default: config, otherwise on)")
    parser.add_argument("--enable-vla", action="store_true", default=None,
                        help="Enable pi0.5 for VLA selection or refinement (default: config, otherwise off)")
    parser.add_argument("--vla-as-reference", action=argparse.BooleanOptionalAction, default=None,
                        help="VLA refinement via refine_act (15 frames); otherwise VLA selection. Requires VLA enabled (default: off)")
    parser.add_argument("--vla-checkpoint", help="pi05_droid_jointpos checkpoint directory; overrides config")
    parser.add_argument("--vla-port", type=int, help="Local pi0.5 port; overrides config (default: 9000)")
    parser.add_argument("--vla-startup-timeout", type=float, help="pi0.5 loading and warmup timeout; overrides config")
    parser.add_argument("--vla-timeout", type=float, help="pi0.5 inference timeout; overrides config")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    try:
        action_format = validate_action_format(args.action_format if args.action_format is not None
                                               else config.get("action_format", "eef_absolute"))
        disable_ik = validate_disable_ik(config.get("disable_ik", False), action_format=action_format)
        benchmark = args.benchmark or config.get("benchmark", "robolab")
        if benchmark != "robolab":
            parser.error("Only the RoboLab harness is supported")
        views = validate_views(config.get("views", "multi"))
    except ValueError as error:
        parser.error(str(error))
    plan = args.plan if args.plan is not None else config.get("plan", False)
    if not isinstance(plan, bool):
        parser.error("plan must be a boolean")
    multiact = args.multiact if args.multiact is not None else config.get("multiact", False)
    if not isinstance(multiact, bool):
        parser.error("multiact must be a boolean")
    async_inference = args.async_inference if args.async_inference is not None else config.get("async_inference", False)
    async_tail_steps = args.async_tail_steps if args.async_tail_steps is not None else config.get("async_tail_steps", 5)
    include_async_prompt = (args.include_async_prompt if args.include_async_prompt is not None
                            else config.get("include_async_prompt", True))
    if type(async_inference) is not bool:
        parser.error("async_inference must be a boolean")
    if type(include_async_prompt) is not bool:
        parser.error("include_async_prompt must be a boolean")
    try:
        positive_int(async_tail_steps, "async_tail_steps")
        action_chunk_size = positive_int(config.get("action_chunk_size", DEFAULT_ACTION_CHUNK_SIZE), "action_chunk_size")
        eef_destination_translation_speed = positive_float(
            config.get("eef_destination_translation_speed", DEFAULT_EEF_DESTINATION_TRANSLATION_SPEED),
            "eef_destination_translation_speed")
        eef_destination_rotation_speed = positive_float(
            config.get("eef_destination_rotation_speed", DEFAULT_EEF_DESTINATION_ROTATION_SPEED),
            "eef_destination_rotation_speed")
    except ValueError as error:
        parser.error(str(error))
    if "action_horizon" in config:
        parser.error("Remove action_horizon from config; use action_chunk_size for fixed frames "
                     "or adaptive_action_chunk for model-selected steps")
    model = args.model or config["model"]
    options = dict(timeout=args.timeout if args.timeout is not None else config["timeout"], image_size=config["image_size"],
                   views=views,
                   reasoning_effort=args.reasoning_effort if args.reasoning_effort is not None else config.get("reasoning_effort", "xhigh"))
    retries = args.codex_max_retries if args.codex_max_retries is not None else config.get("codex_max_retries", 5)
    try:
        codex_stream_idle_timeout = validate_codex_stream_idle_timeout(
            args.codex_stream_idle_timeout if args.codex_stream_idle_timeout is not None
            else config.get("codex_stream_idle_timeout", 300))
    except ValueError as error:
        parser.error(str(error))
    codex_source = args.codex_source if args.codex_source is not None else config.get("codex_source", "system")
    codex_local_path = (args.codex_local_path if args.codex_local_path is not None
                        else config.get("codex_local_path"))
    include_note = args.include_note if args.include_note is not None else config.get("include_note", False)
    try:
        note_file = validate_note_file(args.note_file if args.note_file is not None
                                      else config.get("note_file", "note.md"))
    except ValueError as error:
        parser.error(str(error))
    adaptive_action_chunk = (args.adaptive_action_chunk if args.adaptive_action_chunk is not None
                             else config.get("adaptive_action_chunk", False))
    if not isinstance(adaptive_action_chunk, bool):
        parser.error("adaptive_action_chunk must be a boolean")
    direct_action_chunk = (args.direct_action_chunk if args.direct_action_chunk is not None
                           else config.get("direct_action_chunk", False))
    if not isinstance(direct_action_chunk, bool):
        parser.error("direct_action_chunk must be a boolean")
    if direct_action_chunk and adaptive_action_chunk:
        parser.error("direct_action_chunk cannot be combined with adaptive_action_chunk")
    if action_format == "eef_destination":
        if adaptive_action_chunk or direct_action_chunk:
            parser.error("eef_destination supports only fixed single-target act")
        multiact = False
    vla_as_reference = (args.vla_as_reference if args.vla_as_reference is not None
                        else config.get("vla_as_reference", False))
    if type(vla_as_reference) is not bool:
        parser.error("vla_as_reference must be a boolean")
    try:
        vla_options = config.get("vla", {})
        if not isinstance(vla_options, dict):
            raise ValueError("vla must be a mapping")
        vla_options = dict(vla_options)
        for key, value in (("enabled", args.enable_vla), ("checkpoint", args.vla_checkpoint),
                           ("port", args.vla_port), ("startup_timeout", args.vla_startup_timeout),
                           ("timeout", args.vla_timeout)):
            if value is not None:
                vla_options[key] = value
        vla_config = VLAConfig(**vla_options)
        if vla_config.enabled and vla_config.port == args.port:
            raise ValueError("VLA and harness must use different ports")
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    log_dir = args.log_dir if args.log_dir is not None else Path(config["log_dir"])
    log_path = ROOT / log_dir / f"{timestamp}_eval.jsonl"
    vla = (VLAService(vla_config, root=ROOT, log_path=log_path.with_suffix(".vla.log"))
           if vla_config.enabled else None)
    client_type = ClaudeClient if is_claude_model(model) else CodexClient
    llm = client_type(model, benchmark=benchmark, codex_max_retries=retries,
                      codex_stream_idle_timeout=codex_stream_idle_timeout,
                      codex_source=codex_source, codex_local_path=codex_local_path,
                      system_prompt=ROOT / config.get("codex_prompt", "src/agent/prompts/system.md"),
                      include_note=include_note, note_file=note_file, plan=plan,
                      action_format=action_format, disable_ik=disable_ik,
                      adaptive_action_chunk=adaptive_action_chunk, multiact=multiact,
                      action_chunk_size=action_chunk_size,
                      direct_action_chunk=direct_action_chunk,
                      eef_destination_translation_speed=eef_destination_translation_speed,
                      eef_destination_rotation_speed=eef_destination_rotation_speed,
                      async_inference=async_inference, include_async_prompt=include_async_prompt,
                      vla=vla, enable_vla=vla_config.enabled, vla_as_reference=vla_as_reference, **options)
    policy = None
    try:
        logging.info("%s source: %s; executable: %s", llm.backend.capitalize(), llm.codex_source, llm.executable)
        logging.info("Harness action timeout: %g seconds; %s stream idle timeout: %d seconds",
                     llm.timeout, llm.backend.capitalize(), llm.codex_stream_idle_timeout)
        if vla is not None:
            vla.start()
        policy = HarnessPolicy(llm, log_path, image_size=config["image_size"],
                             async_inference=async_inference, async_tail_steps=async_tail_steps)
        logging.info("Backend: %s; model: %s; effort: %s; decision log: %s",
                     llm.backend, llm.model, llm.reasoning_effort, log_path)
        logging.info("Episode statistics: %s", policy.stats_path)
        logging.info("Plan before execution: %s", plan)
        logging.info("Multi-action execution: %s", multiact)
        logging.info("Frontier-model action format: %s", action_format)
        logging.info("Camera views sent to model: %s", views)
        if disable_ik:
            logging.info("Model IK/FK prohibited by prompt; joint control required (disable_ik=true)")
        logging.info("Manipulation notes included: %s; selected file: %s", include_note, note_file)
        logging.info("Adaptive action chunk: %s", adaptive_action_chunk)
        logging.info("Frame-wise action generation: %s", direct_action_chunk)
        logging.info("Configured fixed action chunk size: %d frames", action_chunk_size)
        logging.info("Async inference: %s; reserved tail steps: %d", async_inference, async_tail_steps)
        logging.info("Async prompt included: %s", async_inference and include_async_prompt)
        logging.info("VLA enabled: %s; reference mode: %s", vla_config.enabled, llm.vla_as_reference)
        logging.info("Serving harness on ws://%s:%d", args.host, args.port)
        WebsocketPolicyServer(policy, host=args.host, port=args.port).serve_forever()
    finally:
        try:
            if policy is not None:
                policy.close()
        finally:
            try:
                llm.close()
            finally:
                if vla is not None:
                    vla.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Harness server stopped")
