"""LLM planner: natural language -> universal envelope JSON.

The LLM only *proposes* plans. Everything it needs to speak the robot's
language comes from the uploaded profile; every proposal is parsed
strictly and validated by deterministic code (see profile.py) before it
may reach the robot. Ambiguous commands are decided here, never asked
back — with conservative values.
"""

import json
import os
from typing import Any

import httpx

from .profile import Profile

TEMPERATURE = 0.2
TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT_S", "60"))


def _extra_body() -> dict[str, Any]:
    """Extra request-body JSON from LLM_EXTRA_BODY (e.g. disable Qwen3
    thinking: {"chat_template_kwargs": {"enable_thinking": false}})."""
    raw = os.environ.get("LLM_EXTRA_BODY", "").strip()
    if not raw:
        return {}
    try:
        extra = json.loads(raw)
    except json.JSONDecodeError as e:
        raise LLMError(f"LLM_EXTRA_BODY is not valid JSON: {e}") from e
    if not isinstance(extra, dict):
        raise LLMError("LLM_EXTRA_BODY must be a JSON object")
    return extra

ENVELOPE_RULES = """\
Universal envelope — the ONLY output format you may produce:
  {"version": 1, "mode": "action" | "plan", "steps": [<step>, ...]}

- "mode": "action" = exactly ONE step. "plan" = 2 to 8 ordered steps.
- Each step is one object: {"action": "<name>", <param>: <value>, ...},
  using ONLY the actions and parameters listed below, with correct types
  and values inside the declared min/max ranges.
- Respond with ONLY that JSON object. No prose, no markdown fences.

Rules:
- DECIDE, NEVER ASK. There is no clarification mode. Pick the most
  reasonable interpretation of a vague instruction and use conservative
  values (smaller distances, slower speeds).
- A request exceeding a single action's limits becomes a multi-step plan
  that stays within the declared limits.
- Never invent actions or parameters. Omit optional parameters you have
  no reason to set; their declared defaults are applied automatically.
- Convert spoken units to the declared units ("half a meter" -> 0.5 m).
"""


class LLMError(RuntimeError):
    """The LLM endpoint is unreachable, misconfigured or returned garbage."""


def _actions_block(profile: Profile) -> str:
    lines: list[str] = []
    for action in profile.actions:
        line = f"- {action.action}"
        if action.description:
            line += f": {action.description}"
        lines.append(line)
        for param in action.params:
            bits = [param.type]
            if param.unit:
                bits.append(f"[{param.unit}]")
            if param.min is not None or param.max is not None:
                bits.append(f"min={param.min} max={param.max}")
            if param.required:
                bits.append("required")
            elif param.default is not None:
                bits.append(f"default={param.default!r}")
            note = f" — {param.description}" if param.description else ""
            lines.append(f"    {param.name}: " + ", ".join(bits) + note)
    return "\n".join(lines)


def _examples_block(profile: Profile) -> str:
    return "\n\n".join(json.dumps(ex.model_dump(), indent=2) for ex in profile.examples)


def build_system_prompt(profile: Profile, telemetry: dict[str, Any]) -> str:
    robot = profile.robot
    robot_line = robot.name
    if robot.description:
        robot_line += f" — {robot.description}"
    return (
        "You translate natural-language instructions into JSON command plans "
        f"for the robot {robot_line}.\n\n"
        f"{ENVELOPE_RULES}\n\n"
        "## Actions (the only vocabulary)\n"
        f"{_actions_block(profile)}\n\n"
        "## Example answers (mimic this exactly)\n"
        f"{_examples_block(profile)}\n\n"
        "## Current robot telemetry (context)\n"
        f"{json.dumps(telemetry, indent=2)}"
    )


def initial_messages(
    profile: Profile, command: str, telemetry: dict[str, Any]
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_system_prompt(profile, telemetry)},
        {"role": "user", "content": command},
    ]


def corrective_messages(
    messages: list[dict[str, str]], invalid_answer: str, errors: list[str]
) -> list[dict[str, str]]:
    """Feed the validation failures back for one corrective retry."""
    return messages + [
        {"role": "assistant", "content": invalid_answer},
        {
            "role": "user",
            "content": (
                "Your answer failed validation:\n- "
                + "\n- ".join(errors)
                + "\nReturn a corrected JSON envelope that passes these "
                "checks. Respond with only the JSON object."
            ),
        },
    ]


def chat(messages: list[dict[str, str]]) -> str:
    """One OpenAI-compatible chat-completions call; returns the raw
    assistant message content. Raises LLMError on any failure."""
    base_url = os.environ.get("LLM_BASE_URL", "").rstrip("/")
    if not base_url:
        raise LLMError("LLM_BASE_URL is not set")
    if not os.environ.get("LLM_MODEL"):
        raise LLMError("LLM_MODEL is not set")
    headers = {}
    if os.environ.get("LLM_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['LLM_API_KEY']}"
    payload = {
        "model": os.environ["LLM_MODEL"],
        "messages": messages,
        "temperature": TEMPERATURE,
        "response_format": {"type": "json_object"},
    }
    payload.update(_extra_body())
    try:
        response = httpx.post(f"{base_url}/chat/completions", json=payload, headers=headers, timeout=TIMEOUT_S)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as e:
        raise LLMError(f"LLM call failed: {e}") from e
