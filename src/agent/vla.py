"""Managed pi0.5 inference in openpi's own Python environment."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np
from websockets.sync.client import connect

from src.agent.executor import robot_state
from src.agent.async_context import PrefetchCancelled
from src.evaluation.robolab_bench.websocket_server import Packer, unpackb


logger = logging.getLogger(__name__)
DEFAULT_CHECKPOINT = "/data/NTU_slab/draven/.cache/openpi/openpi-assets-simeval/pi05_droid_jointpos/pi05_droid_jointpos"


@dataclass(frozen=True)
class VLAConfig:
    enabled: bool = False
    checkpoint: str = DEFAULT_CHECKPOINT
    port: int = 9000
    startup_timeout: float = 1200.0
    timeout: float = 60.0

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("vla.enabled must be a boolean")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("vla.port must be an integer in 1..65535")
        if not isinstance(self.checkpoint, str) or not self.checkpoint.strip():
            raise ValueError("vla.checkpoint must be a nonempty path")
        for name in ("startup_timeout", "timeout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"vla.{name} must be positive and finite")


@dataclass(frozen=True)
class JointActionChunk:
    """Absolute seven-joint positions and a binary gripper, one row per frame."""

    actions: np.ndarray
    execution_limit: int | None = None


def validate_joint_actions(value) -> np.ndarray:
    actions = np.asarray(value, dtype=np.float64)
    if actions.shape != (15, 8) or not np.isfinite(actions).all():
        raise ValueError("pi0.5 must return 15 x 8 finite joint actions")
    if np.any(np.abs(actions[:, :7]) > np.finfo(np.float32).max):
        raise ValueError("Joint positions must be representable as finite float32 values")
    actions = actions.copy()
    actions[:, 7] = actions[:, 7] > 0.5
    return actions.astype(np.float32)


def pack_observation(observation: dict, instruction: str) -> dict:
    state = robot_state(observation)
    images = observation["vla_images"]
    request = {}
    for source, destination in (
        ("exterior_image", "observation/exterior_image_1_left"),
        ("wrist_image", "observation/wrist_image_left"),
    ):
        image = np.asarray(images[source])
        if image.dtype != np.uint8 or image.shape != (224, 224, 3):
            raise ValueError("VLA images must be 224 x 224 uint8 RGB arrays")
        request[destination] = image
    request.update({
        "observation/joint_position": np.asarray(state["joint_positions"], dtype=np.float32),
        "observation/gripper_position": np.asarray([state["gripper"]], dtype=np.float32),
        "prompt": instruction,
    })
    return request


class VLAService:
    """Owned by the harness entrypoint, never by an individual Codex episode.

    Inference failures latch unavailable until restart. Requests are never resent.
    Startup connection polling happens before any inference or robot execution.
    """

    def __init__(self, config: VLAConfig, *, root: Path, log_path: Path):
        self.config = config
        self.root = Path(root).absolute()
        self.log_path = Path(log_path)
        self.process = self.connection = self._log_stream = None
        self._connections = ExitStack()
        self.error = None
        self._packer = Packer()
        self._cancelled_request = False

    def reconnect_cancelled(self, check_cancelled) -> None:
        """Replace only a deliberately cancelled transport; never retry a prediction."""
        if not self._cancelled_request or self.error is not None:
            return
        try:
            self._check_process()
            self.connection = self._connections.enter_context(connect(
                f"ws://127.0.0.1:{self.config.port}", proxy=None, compression=None,
                max_size=None, open_timeout=10, close_timeout=1, ping_interval=None,
            ))
            self._receive(time.monotonic() + self.config.timeout, check_cancelled)
            self._cancelled_request = False
        except BaseException as error:
            self.error = f"{type(error).__name__}: {error}"
            self._close_connection()
            raise

    def start(self) -> None:
        if self.process is not None:
            raise RuntimeError("VLA service has already been started")
        openpi = self.root / "third_party/openpi"
        # Do not resolve the Python symlink: that would lose the virtualenv.
        python = openpi / ".venv/bin/python"
        checkpoint = self.root / self.config.checkpoint
        if not python.is_file():
            raise FileNotFoundError(f"Missing openpi Python: {python}")
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Missing pi0.5 checkpoint: {checkpoint}")
        # Never attach to somebody else's already-running policy server.
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("0.0.0.0", self.config.port))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.config.startup_timeout
        try:
            self._log_stream = self.log_path.open("ab")
            env = dict(os.environ)
            env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
            command = [str(python), "-u", "scripts/serve_policy.py", "--port", str(self.config.port),
                       "policy:checkpoint", "--policy.config=pi05_droid_jointpos", f"--policy.dir={checkpoint}"]
            self.process = subprocess.Popen(
                command,
                cwd=openpi, env=env, stdin=subprocess.DEVNULL,
                stdout=self._log_stream, stderr=subprocess.STDOUT,
            )
            logger.info("Loading pi0.5 with %s; log: %s", python, self.log_path)
            while self.connection is None:
                self._check_process()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out loading pi0.5")
                try:
                    self.connection = self._connections.enter_context(connect(
                        f"ws://127.0.0.1:{self.config.port}", proxy=None, compression=None,
                        max_size=None, open_timeout=min(1, remaining), close_timeout=1, ping_interval=None,
                    ))
                except (OSError, TimeoutError):
                    time.sleep(min(0.1, max(0, deadline - time.monotonic())))
            self._receive(deadline, lambda: None)  # Server metadata.
            # Compile before readiness; this advances no simulator or episode.
            self.connection.send(self._packer.pack({
                "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
                "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.uint8),
                "observation/joint_position": np.zeros(7, dtype=np.float32),
                "observation/gripper_position": np.zeros(1, dtype=np.float32),
                "prompt": "hold position",
            }))
            validate_joint_actions(self._receive(deadline, lambda: None)["actions"])
            logger.info("pi0.5 warmed up and standing by on port %d", self.config.port)
        except BaseException:
            self.close()
            raise

    def _check_process(self) -> None:
        if self.process is None:
            raise RuntimeError(f"pi0.5 process is unavailable; see {self.log_path}")
        returncode = self.process.poll()
        if returncode is None:
            return
        message = f"pi0.5 process exited with code {returncode}; see {self.log_path}"
        try:
            with self.log_path.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                stream.seek(max(0, stream.tell() - 8192))
                tail = "\n".join(stream.read().decode("utf-8", errors="replace").splitlines()[-20:])
            if tail:
                message += f"\nLast pi0.5 log lines:\n{tail}"
        except OSError:
            pass
        raise RuntimeError(message)

    def _receive(self, deadline: float, check_cancelled) -> dict:
        while True:
            check_cancelled()
            self._check_process()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for pi0.5")
            try:
                response = self.connection.recv(timeout=min(0.2, remaining))
            except TimeoutError:
                continue
            check_cancelled()
            if isinstance(response, str):
                raise RuntimeError(f"pi0.5 inference failed: {response}")
            return unpackb(response)

    def infer(self, observation: dict, instruction: str, *, check_cancelled) -> np.ndarray:
        return self._infer_response(observation, instruction, check_cancelled=check_cancelled)["actions"]

    def _infer_response(self, observation: dict, instruction: str, *, check_cancelled) -> dict:
        check_cancelled()
        if self.error is not None:
            raise RuntimeError(f"VLA unavailable until harness restart: {self.error}")
        try:
            self._check_process()
            request = pack_observation(observation, instruction)
            self.connection.send(self._packer.pack(request))
            response = self._receive(time.monotonic() + self.config.timeout, check_cancelled)
            response["actions"] = validate_joint_actions(response["actions"])
            return response
        except PrefetchCancelled:
            self._cancelled_request = True
            self._close_connection()
            raise
        except BaseException as error:
            # A true transport/service error stays unavailable until restart.
            self.error = f"{type(error).__name__}: {error}"
            self._close_connection()
            raise

    def _close_connection(self) -> None:
        self.connection = None
        try:
            self._connections.close()
        except Exception:
            logger.warning("Failed to close pi0.5 connection", exc_info=True)

    def close(self) -> None:
        self._close_connection()
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process = None
        if self._log_stream is not None:
            self._log_stream.close()
            self._log_stream = None
