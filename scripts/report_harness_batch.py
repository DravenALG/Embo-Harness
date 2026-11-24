"""Report committed batch trajectories and their mean costs, without model calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
from statistics import fmean
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/RoboLab"))

from policies.harness_policy.resume import inspect_output  # noqa: E402
from policies.harness_policy.batch_resume import read_batch, validate_gpus  # noqa: E402
from scripts.read_harness_results import validate_summary_schema, valid_seconds  # noqa: E402


class BatchReporter:
    def __init__(self, run_dir: Path, gpus: list[int], runs: int, emit=print):
        if type(runs) is not int or runs < 1:
            raise ValueError("runs must be a positive integer")
        self.root = run_dir
        validate_gpus(gpus)
        self.batch = read_batch(run_dir, runs)
        self.workers = self.batch.workers
        self.record_sources = {}
        self.runs = runs
        self.emit = emit
        self.records = {}
        self.stats = {}
        self._ledger_cache = {}
        self._summary_cache = {}
        self._warnings = set()
        self._last_summary = None

    def warn(self, path: Path, error) -> None:
        message = f"[Batch] Warning: {path}: {error}"
        if message not in self._warnings:
            self._warnings.add(message)
            print(message, file=sys.stderr, flush=True)

    @staticmethod
    def stamp(path: Path):
        try:
            stat = path.stat()
            return stat.st_ino, stat.st_size, stat.st_mtime_ns
        except FileNotFoundError:
            return None

    def read_results(self) -> dict:
        found = {}
        for worker in self.workers:
            output = worker.directory / "robolab"
            try:
                stamp = tuple(self.stamp(output / name) for name in
                              ("episode_results.jsonl", "episode_results.json"))
                cached = self._ledger_cache.get(output)
                if cached is None or stamp != cached[0]:
                    progress = inspect_output(output)
                    worker.validate(progress, self.runs)
                    cached = stamp, progress.records
                    self._ledger_cache[output] = cached
            except (OSError, ValueError) as error:
                # Keep earlier committed results while a writer repairs its tail.
                self.warn(output, error)
                continue
            for key, row in cached[1].items():
                if key in found:
                    raise ValueError(f"Duplicate episode result across batches: {key[0]}/{key[1]}")
                found[key] = row
                self.record_sources[key] = worker
        return found

    def read_stats(self) -> None:
        index = {}
        ambiguous = set()
        for worker in self.workers:
            directory = worker.directory / "harness"
            for path in sorted(directory.glob("*.episodes.jsonl")):
                try:
                    stamp = self.stamp(path)
                    cached = self._summary_cache.get(path)
                    if cached is None or stamp != cached[0]:
                        rows = []
                        with path.open("rb") as stream:
                            for number, line in enumerate(stream, 1):
                                if not line.endswith(b"\n"):
                                    break  # A live writer has not committed this line yet.
                                if not line.strip():
                                    continue
                                try:
                                    row = json.loads(line)
                                    if not isinstance(row, dict):
                                        raise ValueError("expected a summary object")
                                    validate_summary_schema(row)
                                    identifier = row.get("episode_id")
                                    if not isinstance(identifier, str) or not identifier:
                                        raise ValueError("missing episode_id")
                                    rows.append(row)
                                except (ValueError, UnicodeError) as error:
                                    self.warn(path, f"line {number}: {error}")
                        cached = stamp, rows
                        self._summary_cache[path] = cached
                    for row in cached[1]:
                        key = worker.directory, row["episode_id"]
                        if key in index and index[key] != row:
                            ambiguous.add(key)
                            self.warn(path, f"conflicting summaries for {key[1]}")
                        index[key] = row
                except OSError as error:
                    self.warn(path, error)
        self.stats = {key: row for key, row in index.items() if key not in ambiguous}

    def summary(self) -> str:
        total_tasks = len(self.batch.costs)
        completed = len(self.records)
        successes = sum(row["success"] for row in self.records.values())
        tasks = sum(all((task, run) in self.records for run in range(self.runs))
                    for task in self.batch.costs)
        success_rate = f"{100 * successes / completed:.2f}% ({successes}/{completed})" if completed else "N/A"
        values = {key: [] for key in ("harness_seconds", "wall_seconds", "total_tokens")}
        for key, result in self.records.items():
            identifier = result.get("harness_episode_id")
            if not isinstance(identifier, str) or not identifier:
                continue  # Never guess associations for historical results.
            row = self.stats.get((self.record_sources[key].directory, identifier), {})
            if row.get("status") == "ended":
                for key in ("harness_seconds", "wall_seconds"):
                    if valid_seconds(row.get(key)):
                        values[key].append(row[key])
            tokens = row.get("tokens")
            token_count = tokens.get("total_tokens") if isinstance(tokens, dict) else None
            if row.get("usage_complete") is True and type(token_count) is int and token_count >= 0:
                values["total_tokens"].append(token_count)

        def mean(key):
            samples = values[key]
            formatted = (f"{fmean(samples):,.1f}" if key == "total_tokens" else
                         f"{fmean(samples):.3f}s") if samples else "N/A"
            if len(samples) != completed:
                formatted += f" ({len(samples)}/{completed} complete)"
            return formatted

        return (f"Tasks {tasks}/{total_tasks} | Episodes {completed}/{total_tasks * self.runs} | "
                f"Success {success_rate} | Mean Harness {mean('harness_seconds')} | "
                f"Mean Wall {mean('wall_seconds')} | Mean Tokens {mean('total_tokens')}")

    def report(self, event: str) -> None:
        self._last_summary = self.summary()
        self.emit(f"[Batch] {event} | {self._last_summary}")

    def poll(self, *, initial=False) -> None:
        found = self.read_results()
        # Read statistics after the ledger so a just-committed result is matched
        # against fresh server summaries, including summaries from earlier retries.
        self.read_stats()
        if initial:
            self.records.update(found)
            self.report("Initial")
            return
        new_results = False
        for key, row in found.items():
            if key in self.records:
                continue
            self.records[key] = row
            task, episode = key
            gpu = self.record_sources[key].gpu
            self.report(f"GPU {gpu} finished {task}/{episode}: {'SUCCESS' if row['success'] else 'FAILURE'}")
            new_results = True
        if not new_results and self.summary() != self._last_summary:
            self.report("Statistics updated")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpus", help="Required for legacy RoboLab batches")
    parser.add_argument("--benchmark", choices=("robolab",), default="robolab")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--watch", action="store_true", help="Refresh every second until stopped.")
    parser.add_argument("--log", action="store_true", help="Append output to RUN_DIR/progress.log.")
    parser.add_argument("--ready-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.gpus:
        parser.error("--gpus is required for RoboLab batches")
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    # Install handlers before reading files: even a short batch needs its final scan.
    if args.watch:
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, stop)
    log = None
    try:
        def emit(line):
            if log is not None:
                log.write(line + "\n")
                log.flush()
            print(line, flush=True)

        reporter = BatchReporter(args.run_dir, [int(g) for g in args.gpus.split(",")], args.runs, emit)
        if args.log:
            log = (args.run_dir / "progress.log").open("a", encoding="utf-8")
        emit("[Batch] Committed trajectories only; means exclude interrupted retries. "
             "Corrected times require complete network and rate-limit coverage; raw times include waiting. Both exclude wrap-up; tokens require complete usage.")
        reporter.poll(initial=True)
        if args.ready_file is not None:
            args.ready_file.write_text("ready\n", encoding="utf-8")
        if args.watch:
            while not stopped:
                time.sleep(1)
                if not stopped:
                    reporter.poll()
            reporter.poll()
            reporter.report("Final")
        return 0
    except (OSError, ValueError) as error:
        print(f"[Batch] Warning: progress reporting stopped: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        if log is not None:
            log.close()


if __name__ == "__main__":
    raise SystemExit(main())
