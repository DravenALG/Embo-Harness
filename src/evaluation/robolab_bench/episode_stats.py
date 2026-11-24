"""Passive episode accounting. This module has no access to a model or process."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
EPISODE_SCHEMA_VERSION = 5

TOKEN_FIELDS = {
    "input_tokens": "inputTokens",
    "output_tokens": "outputTokens",
    "total_tokens": "totalTokens",
    "cached_input_tokens": "cachedInputTokens",
    "cache_write_input_tokens": "cacheWriteInputTokens",
    "reasoning_output_tokens": "reasoningOutputTokens",
}


def normalize_usage(usage: object) -> dict:
    """Keep unavailable counters distinct from reported zeros; never add details."""
    source = usage.get("total") if isinstance(usage, dict) else None
    result = {}
    for name, key in TOKEN_FIELDS.items():
        value = source.get(key) if isinstance(source, dict) else None
        result[name] = value if type(value) is int and value >= 0 else None
    return result


def read_codex_usage(path: Path) -> tuple[dict | None, bool]:
    """Scan only already recorded notifications, including those queued by an RPC.

    A damaged/missing log is a statistics problem, never a reason to ask Codex
    for more data. Preserve the last usable snapshot on a partial read.
    """
    latest = None
    complete = True
    try:
        with path.open("rb") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    event = row.get("event", {})
                    if row.get("direction") != "receive" or event.get("method") != "thread/tokenUsage/updated":
                        continue
                    usage = normalize_usage(event.get("params", {}).get("tokenUsage"))
                    if any(value is not None for value in usage.values()):
                        latest = usage
                    else:
                        complete = False
                except (ValueError, UnicodeError, AttributeError, TypeError):
                    complete = False
    except OSError:
        complete = False
    if not complete:
        logger.warning("Codex statistics log is missing, unreadable, or damaged: %s; usage is partial", path)
    return latest, complete


class EpisodeStats:
    def __init__(self, *, log_path: Path, stats_path: Path, metadata: dict,
                 episode_id: str, episode_index: int, instruction: str,
                 max_steps: int, control_dt: float, started: float) -> None:
        self.stats_path = stats_path
        self.events_path = log_path.with_suffix("") / f"episode-{episode_index:04d}" / "events.jsonl"
        self.started = started
        self.finalized = False
        self.backend = metadata["backend"]
        self._last_confirmed_control_step = 0
        self._pending_tool = None
        self._execution_finished = None
        self.data = {
            "schema_version": EPISODE_SCHEMA_VERSION,
            "episode_id": episode_id, "episode_index": episode_index,
            **{key: metadata[key] for key in ("backend", "model", "reasoning_effort")},
            "async_inference": metadata.get("async_inference", False),
            "plan": metadata.get("plan", False),
            "action_format": metadata.get("action_format", "eef_absolute"),
            "async_tail_steps": metadata.get("async_tail_steps", 5),
            "instruction": instruction, "max_steps": max_steps, "control_dt": control_dt,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "harness_seconds": 0.0, "finish_seconds": 0.0,
            "decision_count": 0, "issued_control_steps": 0, "executed_control_steps": 0,
            "control_steps_complete": False,
            "executed_vla_steps": 0, "vla_delegation_ratio": 0.0,
            "tokens": dict.fromkeys(TOKEN_FIELDS), "usage_complete": True,
        }

    def request(self, *, record: dict, started: float, finished: float, called_backend: bool,
                confirmed_control_step: int | None, disconnected: bool) -> None:
        if self.finalized:
            return
        data = self.data
        ending = record["type"] == "end_episode"
        if ending:
            if self._execution_finished is None:
                self._execution_finished = started
            data["finish_seconds"] += max(0.0, finished - started)
        else:
            data["harness_seconds"] += max(0.0, finished - started)
        if confirmed_control_step is not None:
            if self._pending_tool == "vla":
                data["executed_vla_steps"] += max(0, confirmed_control_step - self._last_confirmed_control_step)
            self._last_confirmed_control_step = confirmed_control_step
            data["executed_control_steps"] = confirmed_control_step
            data["control_steps_complete"] = ending
        if record["status"] == "ok":
            self._pending_tool = record.get("action_metadata", {}).get("tool")
            data["decision_count"] += int(record.get("new_decision", True))
            data["issued_control_steps"] += len(record["actions"])
        if called_backend:
            usage = normalize_usage(record.get("usage"))
            if any(value is not None for value in usage.values()):
                # Fallback if the current episode's event log cannot be read.
                data["tokens"] = usage
        if ending and confirmed_control_step is not None and record.get("reason") in ("native_terminal", "time_limit"):
            # Rollout has already ended. A failed/cancelled wrap-up only affects
            # its own status and usage completeness, never benchmark outcome.
            if disconnected or record.get("error_type") in ("KeyboardInterrupt", "SystemExit"):
                record = {**record, "codex_finish": "cancelled"}
            elif record["status"] == "execution_error":
                record = {**record, "codex_finish": "error"}
            self.finish("ended", record["reason"], finished, record)
        elif disconnected or record.get("error_type") in ("KeyboardInterrupt", "SystemExit"):
            self.finish("interrupted", "disconnected" if disconnected else "interrupted", finished, record)
        elif record["status"] == "execution_error":
            self.finish("execution_error", "execution_error", finished, record)
        elif record["status"] == "ended":
            failed = record.get("reason") == "execution_error"
            self.finish("execution_error" if failed else "ended", record.get("reason"), finished, record)

    def finish(self, status: str, reason: str, finished: float, record: dict | None = None) -> None:
        if self.finalized:
            return
        # Freeze all times before scanning/writing logs. Mark first so failures
        # cannot cause a duplicate summary on a later close or request.
        self.finalized = True
        data = self.data
        data.update(ended_at=datetime.now(timezone.utc).isoformat(),
                    status=status, end_reason=reason,
                    simulated_seconds=data["executed_control_steps"] * data["control_dt"])
        execution_end = finished if self._execution_finished is None else self._execution_finished
        data["wall_seconds"] = max(0.0, execution_end - self.started)
        data["vla_delegation_ratio"] = data["executed_vla_steps"] / max(1, data["executed_control_steps"])
        if record is not None:
            for key in ("error_type", "error", "error_kind", "codex_finish"):
                if key in record:
                    data[key] = record[key]
        usage, complete = read_codex_usage(self.events_path)
        if usage is not None:
            data["tokens"] = usage
        data["usage_complete"] &= complete and usage is not None
        data["usage_complete"] &= (status == "ended" and (record or {}).get("codex_finish", "completed") == "completed"
                                   and all(value is not None for value in data["tokens"].values()))
        try:
            with self.stats_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(data, allow_nan=False) + "\n")
        finally:
            logger.info(
                "Episode %s (%s/%s) %s: wall=%ss harness=%ss tokens=%s "
                "usage_complete=%s decisions=%d executed_steps=%d steps_complete=%s; statistics: %s",
                data["episode_id"], self.backend, data["model"], status,
                *(f"{data[key]:.3f}" if data[key] is not None else "N/A"
                  for key in ("wall_seconds", "harness_seconds")), data["tokens"]["total_tokens"],
                data["usage_complete"], data["decision_count"], data["executed_control_steps"],
                data["control_steps_complete"], self.stats_path,
            )
