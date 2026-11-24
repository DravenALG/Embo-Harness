"""Validate frontier-model actions and resolve them from measured state at execution time."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


DEFAULT_ACTION_CHUNK_SIZE = 15
DEFAULT_EEF_DESTINATION_TRANSLATION_SPEED = 1.2  # m/s, calibrated from embo_chunk8
DEFAULT_EEF_DESTINATION_ROTATION_SPEED = 6.0  # rad/s, calibrated from embo_chunk8
ACTION_FIELDS = {
    "eef_absolute": {"position": 3, "quaternion": 4},
    "eef_delta": {"delta_position": 3, "delta_rotation": 3},
    "joint_absolute": {"joint_positions": 7},
    "joint_delta": {"delta_joint_positions": 7},
    "eef_destination": {"position": 3, "quaternion": 4},
}
ACTION_FORMATS = tuple(ACTION_FIELDS)


@dataclass(frozen=True)
class ActionChunk:
    """Prepared absolute control frames returned by a direct action-chunk tool."""

    actions: np.ndarray
    mode: int
    execution_limit: int | None = None
    # Delta chunks are rebased by the async scheduler at the measured handoff
    # state. Absolute chunks leave this unset and use ``actions`` verbatim.
    relative_actions: tuple[dict, ...] | None = None
    action_format: str | None = None


def validate_action_format(value: str) -> str:
    if not isinstance(value, str) or value not in ACTION_FIELDS:
        raise ValueError(f"action_format must be one of {', '.join(ACTION_FORMATS)}")
    return value


def action_schema(action_format: str = "eef_absolute", *, adaptive_action_chunk: bool = True) -> dict:
    fields = ACTION_FIELDS[validate_action_format(action_format)]
    properties = {name: {"type": "array", "items": {"type": "number"}, "minItems": size, "maxItems": size}
                  for name, size in fields.items()}
    properties["gripper"] = {"type": "integer", "enum": [0, 1]}
    if adaptive_action_chunk:
        properties["steps"] = {"type": "integer", "minimum": 1}
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def finite_vector(value, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain {size} finite numbers")
    return vector


def unit_quaternion(value) -> np.ndarray:
    quaternion = finite_vector(value, 4, "quaternion (wxyz)")
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("quaternion must have a nonzero finite norm")
    return quaternion / norm


def positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def positive_float(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def robot_state(observation: dict) -> dict:
    """Select public proprioception; never forward arbitrary simulator fields."""
    state = {
        "position": finite_vector(observation["position"], 3, "position").tolist(),
        "quaternion": unit_quaternion(observation["quaternion"]).tolist(),
        "joint_positions": finite_vector(observation["joint_positions"], 7, "joint_positions").tolist(),
        "gripper": float(finite_vector([observation["gripper"]], 1, "gripper")[0]),
    }
    if not 0 <= state["gripper"] <= 1:
        raise ValueError("Observed gripper must be in [0, 1]")
    return state


def validate_target(target: dict, *, remaining_steps: int | None = None,
                    action_format: str = "eef_absolute") -> dict:
    fields = ACTION_FIELDS[validate_action_format(action_format)]
    if not isinstance(target, dict) or set(target) != {*fields, "gripper", "steps"}:
        raise ValueError(f"Target must contain only {', '.join(fields)}, gripper, and steps")
    normalized = {}
    for name, size in fields.items():
        vector = unit_quaternion(target[name]) if name == "quaternion" else finite_vector(target[name], size, name)
        if np.any(np.abs(vector) > np.finfo(np.float32).max):
            raise ValueError(f"Target {name} must be representable as finite float32 values")
        normalized[name] = vector.tolist()
    gripper = target["gripper"]
    if isinstance(gripper, bool) or gripper not in (0, 1):
        raise ValueError("Target gripper must be 0 (open) or 1 (closed)")
    steps = positive_int(target["steps"], "steps")
    if remaining_steps is not None and steps > remaining_steps:
        raise ValueError("steps cannot exceed the remaining episode steps")
    return {**normalized, "gripper": int(gripper), "steps": steps}


def validate_fixed_target(target: dict, *, action_format: str = "eef_absolute",
                          action_chunk_size: int = DEFAULT_ACTION_CHUNK_SIZE) -> dict:
    """Supply the fixed duration without resolving relative targets yet."""
    positive_int(action_chunk_size, "action_chunk_size")
    fields = ACTION_FIELDS[validate_action_format(action_format)]
    if action_format == "eef_destination":
        if not isinstance(target, dict) or set(target) != {*fields, "gripper"}:
            raise ValueError(f"Fixed target must contain only {', '.join(fields)}, and gripper (no steps)")
        normalized = {}
        for name, size in fields.items():
            vector = unit_quaternion(target[name]) if name == "quaternion" else finite_vector(target[name], size, name)
            if np.any(np.abs(vector) > np.finfo(np.float32).max):
                raise ValueError(f"Target {name} must be representable as finite float32 values")
            normalized[name] = vector.tolist()
        gripper = target["gripper"]
        if isinstance(gripper, bool) or gripper not in (0, 1):
            raise ValueError("Target gripper must be 0 (open) or 1 (closed)")
        return {**normalized, "gripper": int(gripper)}
    if not isinstance(target, dict) or set(target) != {*fields, "gripper"}:
        raise ValueError(f"Fixed target must contain only {', '.join(fields)}, and gripper (no steps)")
    return validate_target({**target, "steps": action_chunk_size}, action_format=action_format)


def validate_actions(actions: list, *, remaining_steps: int | None = None,
                     adaptive_action_chunk: bool = True, action_format: str = "eef_absolute",
                     action_chunk_size: int = DEFAULT_ACTION_CHUNK_SIZE) -> list[dict]:
    """Validate every target; fixed batches may outlive the remaining budget."""
    positive_int(action_chunk_size, "action_chunk_size")
    if action_format == "eef_destination":
        raise ValueError("eef_destination supports only fixed single-target act")
    if not isinstance(actions, list) or not actions:
        raise ValueError("actions must be a nonempty list of targets")
    targets = [validate_target(target, action_format=action_format) if adaptive_action_chunk
               else validate_fixed_target(target, action_format=action_format, action_chunk_size=action_chunk_size)
               for target in actions]
    if (adaptive_action_chunk and remaining_steps is not None
            and sum(target["steps"] for target in targets) > remaining_steps):
        raise ValueError("Total action steps cannot exceed the remaining episode steps")
    return targets


def validate_action_chunk(actions: list, *, action_chunk_size: int,
                          action_format: str = "eef_absolute") -> list[dict]:
    """Validate one complete direct per-frame action chunk."""
    positive_int(action_chunk_size, "action_chunk_size")
    fields = ACTION_FIELDS[validate_action_format(action_format)]
    if not isinstance(actions, list) or len(actions) != action_chunk_size:
        raise ValueError(f"actions must contain exactly {action_chunk_size} items")
    normalized = []
    for index, action in enumerate(actions):
        if not isinstance(action, dict) or set(action) != {*fields, "gripper"}:
            raise ValueError(
                f"actions[{index}] must contain only {', '.join(fields)}, and gripper"
            )
        item = {}
        for name, size in fields.items():
            vector = unit_quaternion(action[name]) if name == "quaternion" else finite_vector(action[name], size, name)
            if np.any(np.abs(vector) > np.finfo(np.float32).max):
                raise ValueError(f"actions[{index}] {name} must be representable as finite float32 values")
            item[name] = vector.tolist()
        gripper = action["gripper"]
        if isinstance(gripper, bool) or gripper not in (0, 1):
            raise ValueError(f"actions[{index}] gripper must be 0 (open) or 1 (closed)")
        item["gripper"] = int(gripper)
        normalized.append(item)
    return normalized


def prepare_action_chunk(observation: dict, actions: list, *, action_chunk_size: int,
                         action_format: str = "eef_absolute") -> tuple[list[dict], np.ndarray]:
    """Convert direct per-frame actions to absolute controller frames without interpolation."""
    normalized = validate_action_chunk(actions, action_chunk_size=action_chunk_size, action_format=action_format)
    if action_format == "eef_absolute":
        rows = [np.r_[item["position"], item["quaternion"], item["gripper"]] for item in normalized]
    elif action_format == "joint_absolute":
        rows = [np.r_[item["joint_positions"], item["gripper"]] for item in normalized]
    elif action_format == "joint_delta":
        joints = finite_vector(observation["joint_positions"], 7, "current joint_positions")
        rows = []
        for item in normalized:
            joints = joints + np.asarray(item["delta_joint_positions"], dtype=np.float64)
            rows.append(np.r_[joints, item["gripper"]])
    else:
        position = finite_vector(observation["position"], 3, "current position")
        rotation = Rotation.from_quat(unit_quaternion(observation["quaternion"])[[1, 2, 3, 0]])
        rows = []
        for item in normalized:
            position = position + np.asarray(item["delta_position"], dtype=np.float64)
            rotation = Rotation.from_rotvec(np.asarray(item["delta_rotation"], dtype=np.float64)) * rotation
            quaternion = rotation.as_quat()[[3, 0, 1, 2]]
            rows.append(np.r_[position, quaternion, item["gripper"]])
    prepared = np.asarray(rows, dtype=np.float32)
    if prepared.shape != (action_chunk_size, 8) or not np.isfinite(prepared).all():
        raise ValueError("Direct action chunk must produce finite H x 8 float32 actions")
    return normalized, prepared


def resolve_target(target: dict, observation: dict, *, action_format: str = "eef_absolute") -> dict:
    """Resolve one total delta against this segment's measured execution start."""
    target = validate_target(target, action_format=action_format)
    common = {key: target[key] for key in ("gripper", "steps")}
    if action_format == "eef_delta":
        position = finite_vector(observation["position"], 3, "current position") + target["delta_position"]
        start = Rotation.from_quat(unit_quaternion(observation["quaternion"])[[1, 2, 3, 0]])
        rotation = Rotation.from_rotvec(target["delta_rotation"]) * start
        target = {**common, "position": position.tolist(),
                  "quaternion": rotation.as_quat()[[3, 0, 1, 2]].tolist()}
    elif action_format == "joint_delta":
        joints = finite_vector(observation["joint_positions"], 7, "current joint_positions")
        target = {**common, "joint_positions": (joints + target["delta_joint_positions"]).tolist()}
    absolute_format = "joint_absolute" if action_format.startswith("joint_") else "eef_absolute"
    return validate_target(target, action_format=absolute_format)


def reference_target(target: dict, observation: dict, *, action_format: str = "eef_absolute") -> dict:
    """Express an absolute VLA endpoint in the frontier model's action format at query time."""
    validate_action_format(action_format)
    absolute_format = "joint_absolute" if action_format.startswith("joint_") else "eef_absolute"
    target = validate_target(target, action_format=absolute_format)
    common = {key: target[key] for key in ("gripper", "steps")}
    if action_format == "eef_delta":
        start = Rotation.from_quat(unit_quaternion(observation["quaternion"])[[1, 2, 3, 0]])
        end = Rotation.from_quat(np.asarray(target["quaternion"])[[1, 2, 3, 0]])
        target = {**common,
                  "delta_position": (np.asarray(target["position"]) - observation["position"]).tolist(),
                  "delta_rotation": (end * start.inv()).as_rotvec().tolist()}
    elif action_format == "joint_delta":
        target = {**common,
                  "delta_joint_positions": (np.asarray(target["joint_positions"]) - observation["joint_positions"]).tolist()}
    return validate_target(target, action_format=action_format)


def prepare_actions(observation: dict, target: dict, *, action_format: str = "eef_absolute") -> tuple[dict, np.ndarray]:
    """Interpolate a frontier-model target pose into absolute control frames."""
    resolved = resolve_target(target, observation, action_format=action_format)
    if not action_format.startswith("joint_"):
        return resolved, interpolate_actions(observation["position"], observation["quaternion"], resolved)
    start = finite_vector(observation["joint_positions"], 7, "current joint_positions")
    horizon = resolved["steps"]
    fractions = np.arange(1, horizon + 1, dtype=np.float64) / horizon
    joints = start + fractions[:, None] * (np.asarray(resolved["joint_positions"]) - start)
    actions = np.column_stack([joints, np.full(horizon, resolved["gripper"])]).astype(np.float32)
    if not np.isfinite(actions).all():
        raise ValueError("Interpolated actions must be finite float32 values")
    return resolved, actions


def prepare_eef_destination_actions(observation: dict, target: dict, *, control_dt: float,
                              action_chunk_size: int,
                              translation_speed: float = DEFAULT_EEF_DESTINATION_TRANSLATION_SPEED,
                              rotation_speed: float = DEFAULT_EEF_DESTINATION_ROTATION_SPEED,
                              remaining_steps: int | None = None) -> tuple[dict, np.ndarray]:
    """Move toward an EEF destination at configured velocity limits and return one prefix."""
    control_dt = positive_float(control_dt, "control_dt")
    positive_int(action_chunk_size, "action_chunk_size")
    translation_speed = positive_float(translation_speed, "eef_destination_translation_speed")
    rotation_speed = positive_float(rotation_speed, "eef_destination_rotation_speed")
    target = validate_fixed_target(target, action_format="eef_destination", action_chunk_size=action_chunk_size)
    current_position = finite_vector(observation["position"], 3, "current position")
    current_quaternion = unit_quaternion(observation["quaternion"])
    target_position = np.asarray(target["position"], dtype=np.float64)
    target_quaternion = np.asarray(target["quaternion"], dtype=np.float64)
    translation_steps = int(np.ceil(np.linalg.norm(target_position - current_position)
                                    / (translation_speed * control_dt)))
    start_rotation = Rotation.from_quat(current_quaternion[[1, 2, 3, 0]])
    end_rotation = Rotation.from_quat(target_quaternion[[1, 2, 3, 0]])
    rotation_angle = (end_rotation * start_rotation.inv()).magnitude()
    rotation_steps = int(np.ceil(rotation_angle / (rotation_speed * control_dt)))
    horizon = max(1, translation_steps, rotation_steps)
    resolved = {**target, "steps": horizon}
    full = interpolate_actions(current_position, current_quaternion, resolved)
    limit = min(action_chunk_size, horizon)
    if remaining_steps is not None:
        positive_int(remaining_steps, "remaining_steps")
        limit = min(limit, remaining_steps)
    if limit < 1:
        raise ValueError("eef_destination has no remaining execution steps")
    return resolved, full[:limit]


def interpolate_actions(position, quaternion, target: dict) -> np.ndarray:
    """Return H x 8 actions: xyz, quaternion wxyz, binary gripper.

    The first action advances 1/H toward the target; the final action is the
    target. Pose commands are for the same control point as the observation.
    """
    start_position = finite_vector(position, 3, "current position")
    start_quaternion = unit_quaternion(quaternion)
    target = validate_target(target)
    horizon = target["steps"]
    fractions = np.arange(1, horizon + 1, dtype=np.float64) / horizon
    positions = start_position + fractions[:, None] * (np.asarray(target["position"]) - start_position)
    # SciPy expects xyzw; RoboLab and the model use wxyz.
    rotations = Rotation.from_quat(np.array([start_quaternion, target["quaternion"]])[:, [1, 2, 3, 0]])
    quaternions = Slerp([0.0, 1.0], rotations)(fractions).as_quat()[:, [3, 0, 1, 2]]
    actions = np.column_stack([positions, quaternions, np.full(horizon, target["gripper"])])
    actions = actions.astype(np.float32)
    if not np.isfinite(actions).all():
        raise ValueError("Interpolated actions must be finite float32 values")
    return actions
