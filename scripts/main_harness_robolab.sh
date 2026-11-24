#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
SERVER_PYTHON="${SERVER_PYTHON:-python}"
CLIENT_PYTHON="${CLIENT_PYTHON:-$PWD/third_party/RoboLab/.venv/bin/python}"
BASE_PORT="${BASE_PORT:-8000}"
VLA_BASE_PORT="${VLA_BASE_PORT:-}"
run_dir_supplied="${RUN_DIR:+1}"
RUN_DIR="${RUN_DIR:-outputs/harness/robolab_$(date -u +%Y%m%dT%H%M%S)_$$}"
[[ "$RUN_DIR" = /* ]] || RUN_DIR="$PWD/$RUN_DIR"

usage() {
    cat <<'EOF'
Usage: bash scripts/main_harness_robolab.sh --gpus 0,1,2,3 [--set full|sub] [--runs 1] [--resume] [--episode-retries 10] [--episode-retry-delay 30] [--plan|--no-plan] [--multiact|--no-multiact] [--codex-source system|local] [--codex-local-path PATH] [--codex-stream-idle-timeout SECONDS] [--episode-length-scale 1] [--include-note|--no-include-note] [--note-file note.md] [--adaptive-action-chunk|--no-adaptive-action-chunk] [--direct-action-chunk|--no-direct-action-chunk] [--async-inference|--no-async-inference] [--async-tail-steps N] [--include-async-prompt|--no-include-async-prompt] [--enable-vla] [--vla-as-reference|--no-vla-as-reference] [--dry-run]

  --gpus      Comma-separated physical GPU indices (required).
  --set       full: all 120 built-in tasks (default).
              sub: the fixed 20-task exploration subset.
              Both use built-in historical mean wall times to balance tasks across GPUs; no CSV needed.
  --runs      Sequential episodes per task (default: 1).
  --resume    Continue an explicitly specified existing RUN_DIR, skipping completed episodes.
              Change --gpus to redistribute remaining tasks; keep --runs, config and evaluation options.
              Interrupted episodes restart; completed failures are preserved across all batches.
              Changed GPUs write resume_batches/NNNNNN; original files stay in place.
              Add --dry-run for a read-only preview.
              Omit --set to reuse the saved selection; an explicit --set must match it.
              Old batches without a saved set are treated as full with their original tasks.
  --episode-retries  Extra attempts per interrupted episode (default: 10; 0 disables retries).
                     Committed model/protocol failures count once and are never retried.
              Restart only the affected worker; stop the entire batch when its budget is exhausted.
              Completed benchmark failures are kept. Manual --resume starts a fresh retry budget.
  --episode-retry-delay  Initial retry wait in whole seconds (default: 30; 0 disables waiting).
              Doubles after each failure of the same episode, capped at 300 seconds.
  --plan / --no-plan  Explicit subgoal planning or implicit planning; overrides YAML (default: off).
  --multiact / --no-multiact  Short multi-action sequences in one tool call; overrides YAML (default: off).
  --action-format  eef_absolute (default), eef_delta, joint_absolute, joint_delta, or eef_destination; overrides YAML.
              YAML disable_ik: true probes joint-level bias via a prompt-only IK/FK ban; requires a joint format.
  --codex-source  Use system Codex from PATH or a local compiled binary; overrides YAML (default: system Codex from PATH).
  --codex-local-path  Compiled binary, relative to the repository root; overrides YAML, only used with source local. Compile before launching.
  --codex-stream-idle-timeout SECONDS  Positive integer seconds without model stream events before timeout.
              Overrides YAML codex_stream_idle_timeout (300 if absent; bundled config: 600).
              Action-generation timeout is set separately by YAML timeout (includes retries).
  --episode-length-scale  Multiply each task's default episode limit by K (default: 1).
  --include-note / --no-include-note  Include or exclude notice reminders; overrides YAML (default: off).
  --note-file  Bundled notice-reminder file, note.md; overrides YAML; does not enable reminders.
  --adaptive-action-chunk / --no-adaptive-action-chunk  Adaptive or fixed action chunking; overrides YAML (default: fixed).
              Fixed length uses YAML action_chunk_size (default: 15 control frames per target).
  --direct-action-chunk / --no-direct-action-chunk  Frame-wise action generation (trajectory-level bias);
              direct mode cannot be combined with adaptive action chunks and leaves multiact unchanged;
              eef_destination only supports fixed single-target act.
  --async-inference / --no-async-inference  Asynchronous execution: overlap inference and motion (default: off).
  --async-tail-steps N  Reserve N positive control steps for overlap; overrides YAML (default: 5).
  --include-async-prompt / --no-include-async-prompt  Include or exclude async_inference.md in async mode; overrides YAML (default: on).
  --enable-vla  Enable pi0.5 for VLA selection or refinement on each worker GPU; overrides YAML.
  --vla-as-reference / --no-vla-as-reference  VLA refinement via refine_act (15 frames), or VLA selection;
              effective only with VLA enabled, overrides YAML (default: off).
  --config    Harness YAML (default: config/harness/robolab.yaml).
  --dry-run   Print assignments and commands without starting processes or writing files.
  Progress    Each committed trajectory prints global progress, success rate and mean
              Harness/Wall/Tokens to the terminal and RUN_DIR/progress.log.
              Interrupted retries are excluded; missing metrics show their coverage.
  --help      Show this help.

Environment overrides: SERVER_PYTHON, CLIENT_PYTHON, BASE_PORT, VLA_BASE_PORT, RUN_DIR.
VLA_BASE_PORT defaults to YAML vla.port (9000); workers use consecutive ports.
Activate the server environment. Codex requires OPENAI_API_KEY; claude-* models
use the installed Claude Code and its user settings. Export model proxies as needed.
EOF
}

die() { echo "Error: $*" >&2; exit 1; }
gpus="" runs=1 episode_length_scale=1 dry_run=0 resume=0 config="config/harness/robolab.yaml"
task_set=""
enable_vla=""
vla_reference_flag=""
episode_retries=10 episode_retry_delay=30
note_flag=""
plan_flag=""
multiact_flag=""
note_file_args=()
codex_source_args=()
codex_path_args=()
codex_stream_idle_timeout=""
codex_stream_idle_timeout_args=()
action_chunk_flag=""
direct_action_chunk_flag=""
action_format_args=()
async_args=()
while (($#)); do
    case "$1" in
        --gpus|--runs|--config)
            (($# >= 2)) || die "$1 requires a value"
            case "$1" in --gpus) gpus="$2" ;; --runs) runs="$2" ;; --config) config="$2" ;; esac
            shift 2 ;;
        --set)
            (($# >= 2)) || die "$1 requires a value"
            [[ "$2" == full || "$2" == sub ]] || die "$1 must be full or sub"
            task_set="$2"
            shift 2 ;;
        --episode-length-scale|--episode_length_scale)
            (($# >= 2)) || die "$1 requires a value"
            episode_length_scale="$2"
            shift 2 ;;
        --episode-retries|--episode-retry-delay)
            (($# >= 2)) || die "$1 requires a value"
            case "$1" in
                --episode-retries) episode_retries="$2" ;;
                --episode-retry-delay) episode_retry_delay="$2" ;;
            esac
            shift 2 ;;
        --codex-source)
            (($# >= 2)) || die "$1 requires a value"
            [[ "$2" == system || "$2" == local ]] || die "$1 must be system or local"
            codex_source_args=(--codex-source "$2")
            shift 2 ;;
        --codex-local-path)
            (($# >= 2)) || die "$1 requires a value"
            [[ -n "$2" ]] || die "$1 requires a non-empty path"
            codex_path_args=(--codex-local-path "$2")
            shift 2 ;;
        --codex-stream-idle-timeout)
            (($# >= 2)) || die "$1 requires a value"
            [[ -n "$2" ]] || die "$1 requires positive integer seconds"
            codex_stream_idle_timeout="$2"
            codex_stream_idle_timeout_args=(--codex-stream-idle-timeout "$2")
            shift 2 ;;
        --include-note|--no-include-note) note_flag="$1"; shift ;;
        --plan|--no-plan) plan_flag="$1"; shift ;;
        --multiact|--no-multiact) multiact_flag="$1"; shift ;;
        --note-file)
            (($# >= 2)) || die "$1 requires a value"
            [[ "$2" == note.md ]] || die "$1 must be note.md"
            note_file_args=(--note-file "$2")
            shift 2 ;;
        --adaptive-action-chunk|--no-adaptive-action-chunk) action_chunk_flag="$1"; shift ;;
        --direct-action-chunk|--no-direct-action-chunk) direct_action_chunk_flag="$1"; shift ;;
        --action-format)
            (($# >= 2)) || die "$1 requires a value"
            case "$2" in
                eef_absolute|eef_delta|joint_absolute|joint_delta|eef_destination) ;;
                *) die "$1 must be eef_absolute, eef_delta, joint_absolute, joint_delta, or eef_destination" ;;
            esac
            action_format_args=(--action-format "$2")
            shift 2 ;;
        --async-inference|--no-async-inference|--include-async-prompt|--no-include-async-prompt) async_args+=("$1"); shift ;;
        --async-tail-steps)
            (($# >= 2)) || die "$1 requires a value"
            [[ "$2" =~ ^[0-9]+$ && "$2" =~ [1-9] ]] || die "$1 must be a positive integer"
            async_args+=("$1" "$2")
            shift 2 ;;
        --enable-vla) enable_vla=1; shift ;;
        --vla-as-reference|--no-vla-as-reference) vla_reference_flag="$1"; shift ;;
        --resume) resume=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done
[[ -n "$gpus" ]] || die "Specify --gpus, for example --gpus 0,1"
if ((resume)); then
    [[ -n "$run_dir_supplied" ]] || die "--resume requires an explicit RUN_DIR"
    [[ -d "$RUN_DIR" && ! -L "$RUN_DIR" ]] || die "--resume requires an existing directory: $RUN_DIR"
    [[ -f "$RUN_DIR/assignments.tsv" ]] || die "Missing $RUN_DIR/assignments.tsv"
    if (( ! dry_run )); then
        command -v flock >/dev/null || die "Executable not found: flock"
        # Lock the directory itself: no stale lock file, and no writes on a completed run.
        exec 9<"$RUN_DIR"
        flock -n 9 || die "Another launcher is using RUN_DIR: $RUN_DIR"
    fi
fi

launcher_settings=$("$SERVER_PYTHON" - "$config" "$episode_retries" "$episode_retry_delay" "$codex_stream_idle_timeout" "${action_format_args[1]-}" "$enable_vla" "$VLA_BASE_PORT" "$vla_reference_flag" <<'PY'
import sys
import yaml
from src.agent.client import is_claude_model, validate_codex_stream_idle_timeout
from src.agent.prompts import validate_disable_ik
from src.agent.vla import VLAConfig
import math

try:
    retry_values = []
    for flag, value in zip(("--episode-retries", "--episode-retry-delay"), sys.argv[2:4]):
        if not value.isascii() or not value.isdecimal() or int(value) >= sys.maxsize:
            raise ValueError(f"{flag} must be a nonnegative integer smaller than {sys.maxsize}")
        retry_values.append(int(value))
    with open(sys.argv[1]) as stream:
        document = yaml.safe_load(stream) or {}
    validate_disable_ik(document.get("disable_ik", False),
                        action_format=sys.argv[5] or document.get("action_format", "eef_absolute"))
    stream_idle_timeout = document.get("codex_stream_idle_timeout", 300)
    if sys.argv[4]:
        try:
            stream_idle_timeout = int(sys.argv[4])
        except ValueError:
            raise ValueError("--codex-stream-idle-timeout must be positive integer seconds") from None
    validate_codex_stream_idle_timeout(stream_idle_timeout)
    options = document.get("vla", {})
    if not isinstance(options, dict):
        raise ValueError("vla must be a mapping")
    options = dict(options)
    if sys.argv[6]:
        options["enabled"] = True
    if sys.argv[7]:
        options["port"] = int(sys.argv[7])
    vla = VLAConfig(**options)
    reference = (sys.argv[8] == "--vla-as-reference" if sys.argv[8]
                 else document.get("vla_as_reference", False))
    if type(reference) is not bool:
        raise ValueError("vla_as_reference must be a boolean")
    print(math.ceil(vla.startup_timeout) + 60 if vla.enabled else 60, *retry_values,
          int(is_claude_model(document.get("model", ""))), int(vla.enabled), vla.port)
except (OSError, TypeError, ValueError) as error:
    sys.exit(f"Invalid launcher configuration: {error}")
PY
)
read -r readiness_timeout episode_retries episode_retry_delay claude_backend enable_vla vla_base_port <<<"$launcher_settings"

# All scheduling, output and resume state uses evaluation GPUs. Keep the server
# GPU list in the same order, including workers with no pending tasks on resume.
gpu_settings=$("$SERVER_PYTHON" - "$gpus" <<'PY'
import sys

try:
    gpus = [int(gpu) for gpu in sys.argv[1].split(",")]
    if min(gpus) < 0 or len(set(gpus)) != len(gpus):
        raise ValueError("GPU indices must be nonnegative and unique")
    print(",".join(map(str, gpus)), ",".join(map(str, gpus)))
except ValueError as error:
    sys.exit(f"Cannot assign GPUs: {error}")
PY
)
read -r gpus server_gpus <<<"$gpu_settings"
IFS=, read -r -a server_gpu_ids <<<"$server_gpus"

# Longest tasks first, each assigned to the GPU with the least estimated work.
assignments=$("$SERVER_PYTHON" - "$gpus" "$runs" "$BASE_PORT" "$dry_run" "$episode_length_scale" "$resume" "$RUN_DIR" "$task_set" "$enable_vla" "$vla_base_port" <<'PY'
import math
import socket
import sys
from pathlib import Path
from scripts.robolab_task_sets import load_task_costs

try:
    gpus = [int(gpu) for gpu in sys.argv[1].split(",")]
    runs, base_port = int(sys.argv[2]), int(sys.argv[3])
    episode_length_scale = float(sys.argv[5])
    if not math.isfinite(episode_length_scale) or episode_length_scale <= 0:
        raise ValueError("--episode-length-scale must be finite and positive")
    if runs < 1:
        raise ValueError("--runs must be a positive integer")
    if not 1 <= base_port <= 65536 - len(gpus):
        raise ValueError("BASE_PORT and all assigned ports must be in 1..65535")
    enable_vla, vla_base_port = int(sys.argv[9]), int(sys.argv[10])
    if enable_vla:
        if not 1 <= vla_base_port <= 65536 - len(gpus):
            raise ValueError("VLA_BASE_PORT and all assigned ports must be in 1..65535")
        if set(range(base_port, base_port + len(gpus))) & set(range(vla_base_port, vla_base_port + len(gpus))):
            raise ValueError("Harness and VLA port ranges overlap")
    tasks, loads = [[] for _ in gpus], [0.0 for _ in gpus]
    resuming = sys.argv[6] == "1"
    if resuming:
        sys.path.insert(0, "third_party/RoboLab")
        from policies.harness_policy.batch_resume import plan_resume

        plan = plan_resume(Path(sys.argv[7]), gpus, runs, task_set=sys.argv[8] or None)
        print(f"Task set: {plan.batch.task_set} ({len(plan.batch.costs)} tasks)", file=sys.stderr)
        print(f"Resume: {len(plan.completed)}/{len(plan.batch.costs) * runs} episodes complete; "
              f"batch {plan.generation.relative}", file=sys.stderr)
        for worker in plan.generation.workers if plan.remaining else []:
            assignment = worker.assignment
            index = gpus.index(worker.gpu)
            pending = plan.pending(worker)
            remaining_tasks = {task for task, _ in pending}
            tasks[index] = [task for task in assignment.tasks if task in remaining_tasks]
            total = len(assignment.tasks) * runs
            loads[index] = sum(plan.batch.costs[task] for task, _ in pending)
            print(f"GPU {assignment.gpu}: {total - len(pending)}/{total} episodes complete, "
                  f"{len(pending)} remaining" + ("; skipping worker" if not pending else ""), file=sys.stderr)
            if not plan.changed:
                progress = worker.progress(runs)
                artifacts = progress.artifacts(assignment.tasks, runs)
                if artifacts or progress.tail_offset is not None:
                    print(f"  Recovery: archive {len(artifacts)} interrupted artifacts"
                          + (" and repair truncated result tail" if progress.tail_offset is not None else ""),
                          file=sys.stderr)
    else:
        task_set = sys.argv[8] or "full"
        costs = load_task_costs(task_set)
        print(f"Task set: {task_set} ({len(costs)} tasks)", file=sys.stderr)
        for task in sorted(costs, key=lambda task: (-costs[task], task)):
            index = min(range(len(gpus)), key=lambda index: loads[index])
            tasks[index].append(task)
            loads[index] += costs[task] * runs
    for index, gpu in enumerate(gpus):
        if not tasks[index]:
            continue
        port = base_port + index
        if sys.argv[4] == "0":
            for checked_port in ([port, vla_base_port + index] if enable_vla else [port]):
                with socket.socket() as probe:
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    probe.bind(("0.0.0.0", checked_port))
        if not resuming:
            print(f"GPU {gpu}: {len(tasks[index])} tasks, {runs} run(s)/task, "
                  f"estimated {loads[index] / 3600:.2f} h, port {port}", file=sys.stderr)
        print(f"{gpu}\t{port}\t{loads[index]:.3f}\t{' '.join(tasks[index])}")
except (OSError, ValueError, KeyError) as error:
    sys.exit(f"Cannot assign tasks: {error}")
PY
)
if [[ -z "$assignments" ]]; then
    echo "All requested episodes are complete. Results: $RUN_DIR"
    # Keep the completed-run and dry-run paths read-only.
    "$SERVER_PYTHON" -u scripts/report_harness_batch.py --run-dir "$RUN_DIR" --gpus "$gpus" --runs "$runs" \
        || echo "Warning: batch progress reporting unavailable" >&2
    exit 0
fi

# SIGINT lets Python run its finally blocks, including closing Codex subprocesses.
stop_group() {
    local pid="$1" attempt
    [[ -n "$pid" ]] || return 0
    kill -INT -- "-$pid" 2>/dev/null || true
    for ((attempt = 0; attempt < 100; attempt++)); do
        kill -0 -- "-$pid" 2>/dev/null || break
        sleep 0.1
    done
    kill -KILL -- "-$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

# Only committed benchmark results advance the worker. Run with both processes
# stopped, including before repairing a truncated tail or moving recordings.
worker_progress() {
    "$SERVER_PYTHON" - "$RUN_DIR" "$gpus" "$gpu" "$runs" "$1" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "third_party/RoboLab")
from policies.harness_policy.resume import prepare_output
from policies.harness_policy.batch_resume import read_batch

try:
    root, gpu, runs = Path(sys.argv[1]), int(sys.argv[3]), int(sys.argv[4])
    batch = read_batch(root, runs)
    batch.committed()
    worker = next(w for w in batch.current.workers if w.gpu == gpu)
    assignment = worker.assignment
    progress = worker.progress(runs, artifacts=True)
    pending = worker.pending(progress, runs)
    if sys.argv[5] == "recover":
        archive = prepare_output(progress.output, assignment.tasks, runs)
        if archive is not None:
            print(f"Archived interrupted output: {archive}", file=sys.stderr)
    if pending:
        print(*pending[0], sep="/")
except (OSError, ValueError, StopIteration) as error:
    sys.exit(f"Cannot recover GPU {sys.argv[3]}: {error}")
PY
}

worker_log() {
    local message="[$(date -u +%FT%TZ)] GPU $gpu: $*"
    printf '%s\n' "$message"
    printf '%s\n' "$message" >>"$gpu_dir/server.log"
    printf '%s\n' "$message" >>"$gpu_dir/client.log"
}

stop_attempt() {
    stop_group "$client_pid"
    client_pid=""
    stop_group "$server_pid"
    server_pid=""
}

stop_worker() {
    trap '' INT TERM
    if [[ -n "$retry_sleep_pid" ]]; then
        kill "$retry_sleep_pid" 2>/dev/null || true
        wait "$retry_sleep_pid" 2>/dev/null || true
    fi
    stop_attempt
}

server_exited() {
    local status=0
    wait "$server_pid" || status=$?
    failure_reason="server exited (status=$status); see $gpu_dir/server.log"
    return 1
}

# Called in an if statement: explicitly handle failures rather than relying on
# errexit, which Bash disables in this context. PIDs/reason belong to run_gpu.
run_attempt() {
    local deadline status=0
    setsid "${server_cmd[@]}" >>"$gpu_dir/server.log" 2>&1 </dev/null 9<&- &
    server_pid=$!
    deadline=$((SECONDS + readiness_timeout))
    until curl --noproxy '*' -fsS --max-time 1 "http://127.0.0.1:$port/healthz" >/dev/null 2>&1; do
        if ! kill -0 "$server_pid" 2>/dev/null; then server_exited; return 1; fi
        if ((SECONDS >= deadline)); then
            failure_reason="server readiness timed out; see $gpu_dir/server.log"
            return 1
        fi
        sleep 0.2
    done
    if ! kill -0 "$server_pid" 2>/dev/null; then server_exited; return 1; fi
    setsid "${client_cmd[@]}" >>"$gpu_dir/client.log" 2>&1 </dev/null 9<&- &
    client_pid=$!
    while kill -0 "$client_pid" 2>/dev/null; do
        if ! kill -0 "$server_pid" 2>/dev/null; then server_exited; return 1; fi
        sleep 0.5
    done
    wait "$client_pid" || status=$?
    if ((status)); then
        failure_reason="client exited (status=$status); see $gpu_dir/client.log"
        return 1
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then server_exited; return 1; fi
}

run_gpu() {
    local server_pid="" client_pid="" retry_sleep_pid="" failure_reason=""
    local next_episode retry_episode="" retries_used=0 delay=0 status
    local -a attempt_resume=()
    trap stop_worker EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    mkdir -p "$gpu_dir"
    next_episode=$(worker_progress inspect) || die "GPU $gpu progress inspection failed"
    while [[ -n "$next_episode" ]]; do
        attempt_resume=()
        if [[ -d "$gpu_dir/robolab" ]]; then attempt_resume=(--resume); fi
        client_cmd=("${client_base_cmd[@]}" "${attempt_resume[@]}" --task "${tasks[@]}")
        worker_log "Starting worker at $next_episode (resume=${attempt_resume[*]:-false})"
        failure_reason=""
        if run_attempt; then status=0; else status=$?; fi
        stop_attempt
        next_episode=$(worker_progress recover) || die "GPU $gpu recovery failed; no retry"
        if [[ -z "$next_episode" ]]; then
            if ((status)); then worker_log "$failure_reason; all episode results already committed"; fi
            break
        fi
        if ((status == 0)); then failure_reason="client exited (status=0) with incomplete results"; fi
        if [[ "$next_episode" != "$retry_episode" ]]; then
            retry_episode="$next_episode"
            retries_used=0
            delay=$((episode_retry_delay < 300 ? episode_retry_delay : 300))
        fi
        if ((retries_used >= episode_retries)); then
            worker_log "$next_episode: $failure_reason; exhausted $episode_retries retries"
            die "GPU $gpu episode $next_episode exhausted $episode_retries retries; stopping batch"
        fi
        retries_used=$((retries_used + 1))
        worker_log "$next_episode: $failure_reason; retry $retries_used/$episode_retries in ${delay}s"
        if ((delay)); then
            # Background wait lets INT/TERM interrupt even a five-minute backoff.
            sleep "$delay" 9<&- &
            retry_sleep_pid=$!
            wait "$retry_sleep_pid"
            retry_sleep_pid=""
        fi
        delay=$((delay < 150 ? delay * 2 : 300))
    done
    echo "GPU $gpu finished. Results: $gpu_dir/robolab"
}

workers=()
progress_pid=""
progress_ready=""

stop_progress() {
    local attempt
    if [[ -n "$progress_pid" ]]; then
        kill -TERM "$progress_pid" 2>/dev/null || true
        for ((attempt = 0; attempt < 50; attempt++)); do
            kill -0 "$progress_pid" 2>/dev/null || break
            sleep 0.1
        done
        if kill -0 "$progress_pid" 2>/dev/null; then
            echo "Warning: batch progress reporter did not stop; terminating it" >&2
            kill -KILL "$progress_pid" 2>/dev/null || true
        fi
        wait "$progress_pid" 2>/dev/null || echo "Warning: batch progress reporter exited with an error" >&2
        progress_pid=""
    fi
    if [[ -n "$progress_ready" ]]; then
        rm -f -- "$progress_ready" || true
        progress_ready=""
    fi
}

start_progress() {
    local attempt
    progress_ready="$RUN_DIR/.progress-ready-$$"
    "$SERVER_PYTHON" -u scripts/report_harness_batch.py --run-dir "$RUN_DIR" --gpus "$gpus" --runs "$runs" \
        --watch --log --ready-file "$progress_ready" </dev/null 9<&- &
    progress_pid=$!
    # Establish the historical baseline before any worker can commit new results.
    for ((attempt = 0; attempt < 100; attempt++)); do
        [[ ! -f "$progress_ready" ]] || break
        kill -0 "$progress_pid" 2>/dev/null || break
        sleep 0.1
    done
    if [[ ! -f "$progress_ready" ]]; then
        echo "Warning: batch progress reporter could not start; evaluation will continue" >&2
        stop_progress
    else
        rm -f -- "$progress_ready" || true
        progress_ready=""
    fi
}

check_progress() {
    if [[ -n "$progress_pid" ]] && ! kill -0 "$progress_pid" 2>/dev/null; then
        wait "$progress_pid" 2>/dev/null || true
        progress_pid=""
        echo "Warning: batch progress reporter stopped; evaluation will continue" >&2
    fi
}

cleanup() {
    local status=$?
    trap '' INT TERM
    if ((${#workers[@]})); then
        kill -TERM "${workers[@]}" 2>/dev/null || true
        for pid in "${workers[@]}"; do wait "$pid" 2>/dev/null || true; done
    fi
    # Workers have now flushed their final summaries; let the reporter drain them.
    stop_progress
    return "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if (( ! dry_run )); then
    if (( ! claude_backend )); then
        [[ -n "${OPENAI_API_KEY:-}" ]] || die "Export OPENAI_API_KEY before launching"
    fi
    for executable in "$CLIENT_PYTHON" curl setsid flock; do
        command -v "$executable" >/dev/null || die "Executable not found: $executable"
    done
    if ((resume)); then
        "$SERVER_PYTHON" - "$RUN_DIR" "$gpus" "$runs" "$BASE_PORT" "$task_set" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "third_party/RoboLab")
from policies.harness_policy.batch_resume import plan_resume, commit_resume

try:
    plan = plan_resume(Path(sys.argv[1]), [int(g) for g in sys.argv[2].split(",")], int(sys.argv[3]),
                       task_set=sys.argv[5] or None)
    commit_resume(plan, int(sys.argv[4]))
except (OSError, ValueError) as error:
    sys.exit(f"Cannot prepare resume: {error}")
PY
    else
        mkdir -p "$(dirname "$RUN_DIR")"
        mkdir "$RUN_DIR" || die "RUN_DIR must be a new directory: $RUN_DIR"
        exec 9<"$RUN_DIR"
        flock -n 9 || die "Another launcher is using RUN_DIR: $RUN_DIR"
        printf 'gpu\tport\testimated_seconds\ttasks\n%s\n' "$assignments" >"$RUN_DIR/assignments.tsv"
        "$SERVER_PYTHON" - "$RUN_DIR" "$gpus" "$runs" "$task_set" <<'PY'
import sys
from pathlib import Path
from scripts.robolab_task_sets import load_task_costs
sys.path.insert(0, "third_party/RoboLab")
from policies.harness_policy.batch_resume import initialize_batch
task_set = sys.argv[4] or "full"
initialize_batch(Path(sys.argv[1]), [int(g) for g in sys.argv[2].split(",")],
                 int(sys.argv[3]), load_task_costs(task_set), task_set=task_set)
PY
    fi
fi
batch_dir="$RUN_DIR"
if ((resume)); then
    batch_relative=$("$SERVER_PYTHON" - "$RUN_DIR" "$gpus" "$runs" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "third_party/RoboLab")
from policies.harness_policy.batch_resume import plan_resume
print(plan_resume(Path(sys.argv[1]), [int(g) for g in sys.argv[2].split(",")],
                  int(sys.argv[3])).generation.relative)
PY
)
    if [[ "$batch_relative" != "." ]]; then batch_dir="$RUN_DIR/$batch_relative"; fi
fi
echo "Output: $RUN_DIR"
echo "Active batch: $batch_dir"
echo "Episode retries: $episode_retries extra attempts; initial delay ${episode_retry_delay}s, doubling up to 300s"
if (( ! dry_run )); then start_progress; fi
while IFS=$'\t' read -r gpu port estimated_seconds task_list; do
    read -r -a tasks <<<"$task_list"
    gpu_dir="$batch_dir/gpu_$gpu"
    server_gpu="${server_gpu_ids[port - BASE_PORT]}"
    resume_args=()
    episode_plan_args=()
    if [[ "$batch_dir" != "$RUN_DIR" ]]; then
        episode_plan_args=(--episode-plan "$gpu_dir/episode_plan.json")
    fi
    if ((resume)) && [[ -d "$gpu_dir/robolab" ]]; then
        resume_args=(--resume)
    fi
    # Reset background-job signal dispositions so Python can handle SIGINT.
    server_cmd=(env --default-signal=INT,TERM CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$server_gpu"
        "$SERVER_PYTHON" -u scripts/serve_harness.py --config "$config" "${action_format_args[@]}"
        --host 127.0.0.1 --port "$port" --log-dir "$gpu_dir/harness")
    if [[ -n "$note_flag" ]]; then
        server_cmd+=("$note_flag")
    fi
    if [[ -n "$plan_flag" ]]; then
        server_cmd+=("$plan_flag")
    fi
    if [[ -n "$multiact_flag" ]]; then
        server_cmd+=("$multiact_flag")
    fi
    server_cmd+=("${note_file_args[@]}")
    server_cmd+=("${codex_source_args[@]}" "${codex_path_args[@]}"
                 "${codex_stream_idle_timeout_args[@]}" "${async_args[@]}")
    if [[ -n "$action_chunk_flag" ]]; then
        server_cmd+=("$action_chunk_flag")
    fi
    if [[ -n "$direct_action_chunk_flag" ]]; then
        server_cmd+=("$direct_action_chunk_flag")
    fi
    if [[ -n "$vla_reference_flag" ]]; then
        server_cmd+=("$vla_reference_flag")
    fi
    if ((enable_vla)); then
        server_cmd+=(--enable-vla --vla-port "$((vla_base_port + port - BASE_PORT))")
    fi
    # Isaac's Vulkan renderer uses physical indices, so keep CUDA indices unmasked.
    client_base_cmd=(env -u CUDA_VISIBLE_DEVICES --default-signal=INT,TERM CUDA_DEVICE_ORDER=PCI_BUS_ID
        "$CLIENT_PYTHON" -u third_party/RoboLab/policies/harness_policy/run.py
        --device "cuda:$gpu" --headless --num-envs 1 --num-runs "$runs"
        --episode-length-scale "$episode_length_scale" "${episode_plan_args[@]}"
        --remote-host 127.0.0.1 --remote-port "$port" --output-folder-name "$gpu_dir/robolab"
        --kit_args '--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false')
    client_cmd=("${client_base_cmd[@]}" "${resume_args[@]}" --task "${tasks[@]}")
    if ((dry_run)); then
        printf '%q ' "${server_cmd[@]}"; printf '\n'
        printf '%q ' "${client_cmd[@]}"; printf '\n'
    else
        run_gpu &
        workers+=("$!")
    fi
done <<<"$assignments"

# Workers recover independently; an unrecoverable/exhausted worker stops the batch.
while ((${#workers[@]})); do
    check_progress
    for index in "${!workers[@]}"; do
        if ! kill -0 "${workers[index]}" 2>/dev/null; then
            wait "${workers[index]}" || exit "$?"
            unset 'workers[index]'
        fi
    done
    ((${#workers[@]} == 0)) || sleep 0.2
done
