"""Select robot-specific action math without changing model session behavior."""

from src.agent import executor
from pathlib import Path


def get_executor(benchmark="robolab"):
    if benchmark != "robolab":
        raise ValueError("Only the RoboLab harness is supported")
    return executor


def action_dim(benchmark, action_format):
    get_executor(benchmark)
    return 8


def gripper_columns(benchmark, action_format):
    action_dim(benchmark, action_format)
    return [7]


def camera_names(benchmark, views):
    get_executor(benchmark)
    return ("exterior_image",) if views == "single" else ("exterior_image", "wrist_image")


def prompt_path(benchmark, configured=None):
    get_executor(benchmark)
    directory = Path(__file__).parent / "prompts"
    return Path(configured) if configured is not None else directory / "system.md"
