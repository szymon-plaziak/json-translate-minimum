"""Robot profile: the only source of robot knowledge.

A profile defines the action vocabulary, the parameter types and hard
limits, and the example answers shown verbatim to the LLM. It also backs
the deterministic safety layer: every proposed step is validated here
against the profile before anything reaches the robot.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Param(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    type: Literal["int", "float", "str", "bool"]
    description: str | None = None
    unit: str | None = None
    min: float | None = None
    max: float | None = None
    required: bool = False
    default: int | float | str | bool | None = None


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    description: str | None = None
    params: list[Param] = Field(default_factory=list)


class Step(BaseModel):
    """One command step as it appears in an example: an action name plus
    free-form parameter keys (validated against the actions later)."""

    model_config = ConfigDict(extra="allow")

    action: str = Field(min_length=1)

    def params(self) -> dict[str, Any]:
        return self.model_dump(exclude={"action"}, exclude_none=True)


class Example(BaseModel):
    """An exactly-correct answer in the universal envelope."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1]
    mode: Literal["action", "plan"]
    steps: list[Step] = Field(min_length=1, max_length=8)


class RobotInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str | None = None


class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_version: Literal[1]
    robot: RobotInfo
    actions: list[Action] = Field(min_length=1, max_length=32)
    examples: list[Example] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def _examples_must_validate(self) -> "Profile":
        """Every example must be an exactly-correct answer: right envelope
        shape and steps that validate against the declared actions."""
        by_name = {a.action: a for a in self.actions}
        for i, example in enumerate(self.examples):
            if example.mode == "action" and len(example.steps) != 1:
                raise ValueError(f"examples[{i}]: mode 'action' requires exactly 1 step")
            if example.mode == "plan" and not 2 <= len(example.steps) <= 8:
                raise ValueError(f"examples[{i}]: mode 'plan' requires 2-8 steps")
            for j, step in enumerate(example.steps):
                _, errors = validate_step(by_name, {"action": step.action, **step.params()})
                if errors:
                    raise ValueError(f"examples[{i}].steps[{j}]: " + "; ".join(errors))
        return self


def _type_error(param: Param, value: Any) -> str | None:
    """Type and range check of a single parameter value."""
    if param.type == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            return f"expected int, got {type(value).__name__}"
    elif param.type == "float":
        if isinstance(value, bool) or not isinstance(value, int | float):
            return f"expected float, got {type(value).__name__}"
    elif param.type == "str":
        if not isinstance(value, str):
            return f"expected str, got {type(value).__name__}"
    elif param.type == "bool":
        if not isinstance(value, bool):
            return f"expected bool, got {type(value).__name__}"
    if param.type in ("int", "float") and isinstance(value, int | float) and not isinstance(value, bool):
        if param.min is not None and value < param.min:
            return f"value {value} below min {param.min}"
        if param.max is not None and value > param.max:
            return f"value {value} above max {param.max}"
    return None


def validate_step(
    by_name: dict[str, Action], step: dict[str, Any]
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate one step dict against the action table.

    Returns (normalized_step, errors). On success errors is empty and the
    step is complete: optional params with a default are filled in. On
    failure the step is None and errors lists every problem found.
    """
    action_name = step.get("action")
    action = by_name.get(action_name) if isinstance(action_name, str) else None
    if action is None:
        return None, [f"unknown action {action_name!r}"]
    errors: list[str] = []
    out: dict[str, Any] = {"action": action_name}
    given = {k: v for k, v in step.items() if k != "action"}
    declared = {p.name: p for p in action.params}
    for key in given:
        if key not in declared:
            errors.append(f"unknown parameter {key!r}")
    for param in action.params:
        if param.name in given:
            err = _type_error(param, given[param.name])
            if err:
                errors.append(f"{param.name}: {err}")
            else:
                out[param.name] = given[param.name]
        elif param.default is not None:
            out[param.name] = param.default
        elif param.required:
            errors.append(f"{param.name}: missing required parameter")
    if errors:
        return None, errors
    return out, []


def validate_plan(profile: Profile, plan: Any) -> tuple[list[dict[str, Any]] | None, list[str]]:
    """Validate a proposed envelope against the profile.

    Returns (normalized_steps, errors) where normalized steps are ready to
    send to the robot (defaults filled in, execution order preserved).
    """
    if not isinstance(plan, dict):
        return None, ["plan must be a JSON object"]
    errors: list[str] = []
    if plan.get("version") != 1:
        errors.append('"version" must be 1')
    mode = plan.get("mode")
    steps = plan.get("steps")
    if mode not in ("action", "plan"):
        errors.append('"mode" must be "action" or "plan"')
    if not isinstance(steps, list):
        return None, errors + ['"steps" must be a list']
    if mode == "action" and len(steps) != 1:
        errors.append('"mode": "action" requires exactly 1 step')
    if mode == "plan" and not 2 <= len(steps) <= 8:
        errors.append('"mode": "plan" requires 2-8 steps')
    by_name = {a.action: a for a in profile.actions}
    normalized: list[dict[str, Any]] = []
    for i, raw in enumerate(steps):
        if not isinstance(raw, dict):
            errors.append(f"steps[{i}]: step must be a JSON object")
            continue
        step, step_errors = validate_step(by_name, raw)
        if step_errors:
            errors.extend(f"steps[{i}]: {e}" for e in step_errors)
        else:
            normalized.append(step)
    if errors:
        return None, errors
    return normalized, []
