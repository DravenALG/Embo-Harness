"""Observation-time and execution-time context for speculative robot decisions."""

from copy import deepcopy
from dataclasses import dataclass


class PrefetchCancelled(Exception):
    """An unused prediction was cancelled at an episode boundary, not a service failure."""


@dataclass(frozen=True)
class ObservationSnapshot:
    observation: dict
    control_step: int
    chunk_id: int | None
    phase: str
    episode_id: str

    @classmethod
    def capture(cls, observation, control_step, chunk_id, phase, episode_id):
        return cls(deepcopy(observation), control_step, chunk_id, phase, episode_id)


@dataclass(frozen=True)
class AsyncInferenceContext:
    previous: ObservationSnapshot
    current: ObservationSnapshot
    execution_start: int
    tail: dict | None = None

    @property
    def skipped_steps(self):
        return self.execution_start - self.current.control_step
