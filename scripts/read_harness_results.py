"""Read offline harness episode statistics and optionally print Codex messages.

Accept directories or *.episodes.jsonl files. Uses only the Python standard
library; no model, simulator, credentials, or network connection is needed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import unicodedata

# Keep the report runnable from any cwd, using only standard-library modules.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.robolab_bench.episode_stats import EPISODE_SCHEMA_VERSION


SUMMARY_SUFFIX = ".episodes.jsonl"
MESSAGE_METHODS = {"item/started", "item/completed", "item/agentMessage/delta"}


def warn(path: Path, message: str, line: int | None = None) -> None:
    location = f"{path}:{line}" if line is not None else str(path)
    print(f"Warning: {location}: {message}", file=sys.stderr)


def discover_summary_files(paths) -> list[Path]:
    """Find summaries once, even when input directories/files overlap."""
    found = set()
    for value in paths:
        path = Path(value).expanduser()
        try:
            if path.is_dir():
                found.update(p.resolve() for p in path.rglob(f"*{SUMMARY_SUFFIX}") if p.is_file())
            elif path.is_file() and path.name.endswith(SUMMARY_SUFFIX):
                found.add(path.resolve())
            else:
                warn(path, f"expected a directory or an existing *{SUMMARY_SUFFIX} file")
        except OSError as error:
            warn(path, str(error))
    return sorted(found)


def iter_json_objects(path: Path):
    """Read one line at a time, retaining usable records around damaged lines."""
    try:
        with path.open("rb") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("expected a JSON object")
                except (ValueError, UnicodeError) as error:
                    warn(path, str(error), number)
                    continue
                yield number, row
    except OSError as error:
        warn(path, str(error))


def validate_summary_schema(row: dict) -> None:
    version = row.get("schema_version")
    if type(version) is not int or version != EPISODE_SCHEMA_VERSION:
        raise ValueError(f"unsupported episode schema {version!r}; expected {EPISODE_SCHEMA_VERSION}")


def load_episode_summaries(path: Path) -> list[dict]:
    episodes = []
    for line, row in iter_json_objects(path):
        try:
            validate_summary_schema(row)
        except ValueError as error:
            warn(path, str(error), line)
            continue
        index, episode_id = row.get("episode_index"), row.get("episode_id")
        if type(index) is not int or index < 1 or not isinstance(episode_id, str) or not episode_id:
            warn(path, "expected a positive episode_index and a nonempty episode_id", line)
            continue
        episodes.append(row)
    return sorted(episodes, key=lambda row: row["episode_index"])


def event_log_path(summary_path: Path, episode_index: int) -> Path:
    """Resolve alongside the summary, including after moving an entire run."""
    run_dir = summary_path.with_name(summary_path.name[:-len(SUMMARY_SUFFIX)])
    return run_dir / f"episode-{episode_index:04d}" / "events.jsonl"


@dataclass
class AgentMessage:
    text: str = ""
    phase: str | None = None
    completed: bool = False


def read_agent_messages(path: Path) -> list[AgentMessage]:
    """Prefer completed text over deltas, in first-appearance order per item.

    Only public agentMessage text is collected. Images, tool calls, user input,
    and reasoning items never enter the transcript.
    """
    messages = {}
    fragments = {}
    for line, row in iter_json_objects(path):
        if row.get("direction") != "receive":
            continue
        event = row.get("event")
        if not isinstance(event, dict):
            warn(path, "expected an event object", line)
            continue
        method = event.get("method")
        if not isinstance(method, str) or method not in MESSAGE_METHODS:
            continue
        try:
            params = event.get("params")
            if not isinstance(params, dict):
                raise ValueError("expected message params")
            item = params if method == "item/agentMessage/delta" else params.get("item")
            if not isinstance(item, dict):
                raise ValueError("expected a message item")
            if method != "item/agentMessage/delta" and item.get("type") != "agentMessage":
                continue
            item_id = item.get("itemId" if method == "item/agentMessage/delta" else "id")
            if not isinstance(item_id, str) or not item_id:
                raise ValueError("expected a nonempty message ID")
            identifiers = (params.get("threadId"), params.get("turnId"))
            if any(value is not None and not isinstance(value, str) for value in identifiers):
                raise ValueError("invalid message thread/turn ID")
            text = item.get("delta" if method == "item/agentMessage/delta" else "text",
                            "" if method == "item/started" else None)
            if not isinstance(text, str):
                raise ValueError("expected message text to be a string")
        except ValueError as error:
            warn(path, str(error), line)
            continue

        key = (*identifiers, item_id)
        message = messages.setdefault(key, AgentMessage())
        if message.completed:
            continue
        phase = item.get("phase")
        if isinstance(phase, str) and phase:
            message.phase = phase
        if method == "item/completed":
            message.text, message.completed = text, True
            fragments.pop(key, None)
        elif method == "item/agentMessage/delta":
            fragments.setdefault(key, []).append(text)
        elif not message.text:
            message.text = text

    for key, parts in fragments.items():
        messages[key].text += "".join(parts)
    return [message for message in messages.values() if message.text]


def display_width(text: str) -> int:
    return sum(0 if unicodedata.combining(char) else
               2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)


def brief_task(value, limit: int = 44) -> str:
    text = " ".join(value.split()) if isinstance(value, str) else "N/A"
    if display_width(text) <= limit:
        return text
    prefix = ""
    for char in text:
        if display_width(prefix + char) > limit - 1:
            break
        prefix += char
    return prefix + "…"


def format_seconds(value) -> str:
    if type(value) in (int, float) and math.isfinite(value) and value >= 0:
        return f"{value:.3f}"
    return "N/A"


def format_tokens(value) -> str:
    return f"{value:,}" if type(value) is int and value >= 0 else "N/A"


def valid_seconds(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def format_summary(path: Path, episodes: list[dict]) -> str:
    headers = ["Episode", "Task", "Wall(s)", "Harness(s)", "Input", "Output", "Total",
               "CachedInput", "Usage", "Status", "Finish(s)"]
    rows = [headers]
    models = []
    for episode in episodes:
        model = f"{episode.get('model') or 'N/A'} (effort={episode.get('reasoning_effort') or 'N/A'})"
        if model not in models:
            models.append(model)
        tokens = episode.get("tokens")
        tokens = tokens if isinstance(tokens, dict) else {}
        complete = episode.get("usage_complete")
        usage = "complete" if complete is True else "partial" if complete is False else "N/A"
        status = str(episode.get("status") or "N/A")
        finish = episode.get("codex_finish")
        if finish and finish != "completed":
            status += f"/{finish}"
        rows.append([
            f"{episode['episode_index']:04d}", brief_task(episode.get("instruction")),
            format_seconds(episode.get("wall_seconds")), format_seconds(episode.get("harness_seconds")),
            *(format_tokens(tokens.get(key)) for key in
              ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens")),
            usage, status, format_seconds(episode.get("finish_seconds")),
        ])

    widths = [max(display_width(row[i]) for row in rows) for i in range(len(headers))]
    output = [f"Run: {path}", f"Model: {', '.join(models)}",
              "Times: Raw elapsed seconds, including waiting. "
              "Wall/Harness exclude wrap-up; Finish reports wrap-up separately."]
    for index, row in enumerate(rows):
        cells = []
        for column, (cell, width) in enumerate(zip(row, widths)):
            padding = " " * (width - display_width(cell))
            cells.append(padding + cell if column in (0, 2, 3, 4, 5, 6, 7, 10) else cell + padding)
        output.append("  ".join(cells).rstrip())
        if index == 0:
            output.append("  ".join("-" * width for width in widths))
    return "\n".join(output)


def print_episode_messages(path: Path, episodes: list[dict]) -> None:
    for episode in episodes:
        print(f"\nEpisode {episode['episode_index']:04d} | ID: {episode['episode_id']}")
        print(f"Task: {episode.get('instruction') or 'N/A'}")
        messages = read_agent_messages(event_log_path(path, episode["episode_index"]))
        if not messages:
            print("(No readable agent messages.)")
        for index, message in enumerate(messages, 1):
            incomplete = " (incomplete)" if not message.completed else ""
            print(f"\n[{index} {message.phase or 'message'}]{incomplete}")
            print(message.text)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path,
                        help="Directories (searched recursively) or *.episodes.jsonl files")
    parser.add_argument("--show-messages", action="store_true",
                        help="Also print all recorded Codex commentary/final messages per episode")
    parser.add_argument("--benchmark", choices=("robolab",),
                        help="Filter by benchmark; historical records without a benchmark are RoboLab")
    args = parser.parse_args(argv)
    printed = False
    for path in discover_summary_files(args.paths):
        episodes = load_episode_summaries(path)
        if args.benchmark is not None:
            episodes = [row for row in episodes if row.get("benchmark", "robolab") == args.benchmark]
        if not episodes:
            continue
        if printed:
            print()
        print(format_summary(path, episodes))
        printed = True
        if args.show_messages:
            print_episode_messages(path, episodes)
    if not printed:
        print(f"No valid episode summaries found; expected *{SUMMARY_SUFFIX} files.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
