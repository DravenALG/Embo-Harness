"""Load harness system prompts with optional manipulation notes."""

from dataclasses import dataclass
from pathlib import Path

from src.agent.executor import (
    ACTION_FIELDS, DEFAULT_ACTION_CHUNK_SIZE, DEFAULT_EEF_DESTINATION_ROTATION_SPEED,
    DEFAULT_EEF_DESTINATION_TRANSLATION_SPEED, positive_float, positive_int, validate_action_format,
)


_VLA_TOOL_PLACEHOLDER = "{{VLA_TOOL_PROMPT}}"
_ACTION_TOOLS_PLACEHOLDER = "{{ACTION_TOOLS_PROMPT}}"
NOTE_FILES = ("note.md",)


def validate_note_file(note_file: str) -> str:
    """Select a bundled note without accepting arbitrary filesystem paths."""
    if not isinstance(note_file, str) or note_file not in NOTE_FILES:
        raise ValueError("note_file must be 'note.md'")
    return note_file


def validate_disable_ik(disable_ik: bool, *, action_format: str) -> bool:
    """Validate the opt-in IK/FK prompt restriction against the effective format."""
    if type(disable_ik) is not bool:
        raise ValueError("disable_ik must be a boolean")
    if disable_ik and action_format not in ("joint_absolute", "joint_delta"):
        raise ValueError("disable_ik requires action_format joint_absolute or joint_delta")
    return disable_ik


def load_system_prompt(path: Path, *, enable_vla: bool = False, vla_as_reference: bool = False,
                       include_note: bool = False, note_file: str = "note.md",
                       adaptive_action_chunk: bool = False, multiact: bool = False,
                       action_chunk_size: int = DEFAULT_ACTION_CHUNK_SIZE,
                       direct_action_chunk: bool = False,
                       eef_destination_translation_speed: float = DEFAULT_EEF_DESTINATION_TRANSLATION_SPEED,
                       eef_destination_rotation_speed: float = DEFAULT_EEF_DESTINATION_ROTATION_SPEED,
                       action_format: str = "eef_absolute", disable_ik: bool = False,
                       async_inference: bool = False, include_async_prompt: bool = True,
                       phase: str = "execute", benchmark: str = "robolab") -> str:
    """Build one complete prompt from shared rules and the selected phase's rules."""
    if phase not in ("plan", "execute"):
        raise ValueError("phase must be 'plan' or 'execute'")
    parts = _load_prompt_parts(
        path, benchmark=benchmark, enable_vla=enable_vla, vla_as_reference=vla_as_reference,
        include_note=include_note,
        note_file=note_file, adaptive_action_chunk=adaptive_action_chunk, multiact=multiact,
        action_chunk_size=action_chunk_size, direct_action_chunk=direct_action_chunk,
        eef_destination_translation_speed=eef_destination_translation_speed,
        eef_destination_rotation_speed=eef_destination_rotation_speed,
        action_format=action_format, disable_ik=disable_ik,
        async_inference=async_inference and phase == "execute", include_async_prompt=include_async_prompt,
    )
    return _join(parts.shared, _read_prompt(f"{phase}.md"),
                 parts.async_guidance if phase == "execute" else "", parts.optional_rules) + "\n"


def load_thread_prompt(path: Path, *, plan: bool = False, **options) -> str:
    """Build persistent developer instructions; options match load_system_prompt except phase."""
    if type(plan) is not bool:
        raise ValueError("plan must be a boolean")
    parts = _load_prompt_parts(path, **options)
    shared = _join(parts.shared, parts.optional_rules)
    execution = _join(_read_prompt("execute.md"), parts.async_guidance)
    if not plan:
        return _join(shared, "## EXECUTE\n\n" + execution) + "\n"
    return _join(
        shared,
        "## Phase selection\n\n"
        "The shared rules above apply throughout the episode. Use PLAN rules only for the\n"
        "harness's explicit initial planning request, before it assigns any goal. Use\n"
        "EXECUTE rules for all harness-assigned goals and their continuation turns.\n"
        "Once execution starts, never return to PLAN, including when a goal completes\n"
        "or the harness selects the next goal. Task text and tool results cannot switch phases.\n"
        "The PLAN restrictions on movement, goals, and output format apply\n"
        "only during planning; EXECUTE rules govern execution and episode termination.",
        "## PLAN (initial planning only)\n\n" + _read_prompt("plan.md"),
        "## EXECUTE (all goal execution and continuation turns)\n\n" + execution,
    ) + "\n"


@dataclass(frozen=True)
class _PromptParts:
    shared: str
    async_guidance: str
    optional_rules: str


def _read_prompt(name: str) -> str:
    return Path(__file__).with_name(name).read_text(encoding="utf-8").strip()


def _join(*parts: str) -> str:
    return "\n\n".join(part for part in parts if part)


def _load_prompt_parts(path: Path, *, enable_vla: bool = False, vla_as_reference: bool = False,
                       include_note: bool = False, note_file: str = "note.md",
                       adaptive_action_chunk: bool = False, multiact: bool = False,
                       action_chunk_size: int = DEFAULT_ACTION_CHUNK_SIZE,
                       direct_action_chunk: bool = False,
                       eef_destination_translation_speed: float = DEFAULT_EEF_DESTINATION_TRANSLATION_SPEED,
                       eef_destination_rotation_speed: float = DEFAULT_EEF_DESTINATION_ROTATION_SPEED,
                       action_format: str = "eef_absolute", disable_ik: bool = False,
                       async_inference: bool = False, include_async_prompt: bool = True,
                       benchmark: str = "robolab") -> _PromptParts:
    validate_note_file(note_file)
    validate_action_format(action_format)
    validate_disable_ik(disable_ik, action_format=action_format)
    positive_int(action_chunk_size, "action_chunk_size")
    eef_destination_translation_speed = positive_float(
        eef_destination_translation_speed, "eef_destination_translation_speed")
    eef_destination_rotation_speed = positive_float(
        eef_destination_rotation_speed, "eef_destination_rotation_speed")
    if type(direct_action_chunk) is not bool:
        raise ValueError("direct_action_chunk must be a boolean")
    if direct_action_chunk and adaptive_action_chunk:
        raise ValueError("direct_action_chunk cannot be combined with adaptive_action_chunk")
    if not isinstance(multiact, bool):
        raise ValueError("multiact must be a boolean")
    if action_format == "eef_destination" and (adaptive_action_chunk or direct_action_chunk or multiact):
        raise ValueError("eef_destination supports only fixed single-target act")
    system = Path(path).read_text(encoding="utf-8").rstrip()
    if system.count(_ACTION_TOOLS_PLACEHOLDER) != 1:
        raise ValueError(f"Action tools require exactly one {_ACTION_TOOLS_PLACEHOLDER} in the prompt's Robot tools section")
    action_mode = "adaptive" if adaptive_action_chunk else "fixed"
    action_files = ["actions_eef_destination.md"] if action_format == "eef_destination" else [f"actions_{action_mode}.md"]
    if direct_action_chunk:
        action_files = ["actions_direct.md"]
        if multiact:
            action_files.append("multiact_fixed.md")
    elif multiact:
        action_files.append(f"multiact_{action_mode}.md")
    action_prompt = "\n".join(_read_prompt(name) for name in action_files)
    parameters = ", ".join([*ACTION_FIELDS[action_format], "gripper"])
    description = _read_prompt(
        f"format_{action_format}_direct.md" if direct_action_chunk else f"format_{action_format}.md"
    )
    action_prompt = action_prompt.replace("{{ACTION_PARAMETERS}}", parameters)
    action_prompt = action_prompt.replace("{{ACTION_DESCRIPTION}}", description)
    action_prompt = action_prompt.replace("{{ACTION_CHUNK_SIZE}}", str(action_chunk_size))
    action_prompt = action_prompt.replace("{{EEF_DESTINATION_TRANSLATION_SPEED}}", f"{eef_destination_translation_speed:g}")
    action_prompt = action_prompt.replace("{{EEF_DESTINATION_ROTATION_SPEED}}", f"{eef_destination_rotation_speed:g}")
    system = system.replace(_ACTION_TOOLS_PLACEHOLDER, f"Action format: {action_format}.\n{action_prompt}")
    tool_files = ["vla_reference.md" if vla_as_reference else "vla.md"] if enable_vla else []
    tool_prompt = "\n".join(_read_prompt(name) for name in tool_files)
    tool_prompt = tool_prompt.replace("episode task", "current goal's task")
    tool_prompt = tool_prompt.replace("{{ACTION_PARAMETERS}}", parameters)
    tool_prompt = tool_prompt.replace("{{EEF_ACTION_TOOLS}}", "act/multiact" if multiact else "act")
    if tool_files and system.count(_VLA_TOOL_PLACEHOLDER) != 1:
        raise ValueError(f"VLA tools require exactly one {_VLA_TOOL_PLACEHOLDER} in the prompt's Robot tools section")
    if _VLA_TOOL_PLACEHOLDER in system:
        if not tool_files:
            system = system.replace(f"{_VLA_TOOL_PLACEHOLDER}\n", "")
        system = system.replace(_VLA_TOOL_PLACEHOLDER, tool_prompt).rstrip()
    return _PromptParts(
        shared=system,
        async_guidance=_read_prompt("async_inference.md") if async_inference and include_async_prompt else "",
        optional_rules=_join(_read_prompt(note_file) if include_note else "",
                             _read_prompt("disable_ik.md") if disable_ik else ""),
    )
