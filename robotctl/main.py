"""robotctl REST API: natural language in, validated JSON out, executed
on the real robot over HTTP.

Minimal implementation of the concept in README.md: commands run
asynchronously (POST /commands -> 202 + id, poll GET /commands/{id}) in
a background thread; profiles live in memory only.
"""

import json
import os
import secrets
import threading
from typing import Any

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError

from . import llm
from .profile import Profile, validate_plan

load_dotenv()

API_HOST = os.environ.get("API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("API_PORT", "8081"))
ROBOT_BASE_URL = os.environ.get("ROBOT_BASE_URL", "").rstrip("/")
ROBOT_COMMAND_PATH = os.environ.get("ROBOT_COMMAND_PATH", "/command")
ROBOT_TELEMETRY_PATH = os.environ.get("ROBOT_TELEMETRY_PATH", "/telemetry")
ROBOT_STOP_PATH = os.environ.get("ROBOT_STOP_PATH", "/stop")
ROBOT_TIMEOUT_S = float(os.environ.get("ROBOT_TIMEOUT_S", "30"))
STEP_TIMEOUT_S = float(os.environ.get("STEP_TIMEOUT_S", "60"))

MOTION_STATUSES = ("idle", "moving", "turning", "stopped")

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
_client = httpx.Client(timeout=ROBOT_TIMEOUT_S)
stop_event = threading.Event()
_profile: Profile | None = None
_commands: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


class CommandBody(BaseModel):
    command: str = Field(min_length=1, max_length=500)


def _require_profile() -> Profile:
    if _profile is None:
        raise HTTPException(422, "no profile loaded; upload one via POST /profile first")
    return _profile


def _normalize_telemetry(raw: dict[str, Any]) -> dict[str, Any]:
    """Fields the robot does not provide become null; an unknown
    motion_status maps to 'error' (fail safe)."""
    status = raw.get("motion_status")
    pose = raw.get("pose")
    return {
        "timestamp": raw.get("timestamp"),
        "monotonic": raw.get("monotonic"),
        "connected": raw.get("connected"),
        "motion_status": status if status in MOTION_STATUSES else "error",
        "pose": pose if isinstance(pose, dict) else None,
        "linear_speed_mps": raw.get("linear_speed_mps"),
        "angular_speed_dps": raw.get("angular_speed_dps"),
        "battery_pct": raw.get("battery_pct"),
        "obstacle_distance_m": raw.get("obstacle_distance_m"),
        "last_action_id": raw.get("last_action_id"),
    }


def _url(path: str) -> str:
    return f"{ROBOT_BASE_URL}{path}"


def _fetch_telemetry() -> dict[str, Any]:
    if not ROBOT_BASE_URL:
        raise HTTPException(503, "robot not configured (ROBOT_BASE_URL is empty)")
    try:
        response = _client.get(_url(ROBOT_TELEMETRY_PATH))
        response.raise_for_status()
        raw = response.json()
    except (httpx.HTTPError, ValueError) as e:
        raise HTTPException(503, f"robot telemetry unavailable: {e}") from e
    if not isinstance(raw, dict):
        raise HTTPException(503, "robot telemetry is not a JSON object")
    return _normalize_telemetry(raw)


def _robot_stop() -> None:
    """Fire the robot's stop endpoint; best effort, result ignored."""
    if not ROBOT_BASE_URL or not ROBOT_STOP_PATH:
        return
    try:
        _client.post(_url(ROBOT_STOP_PATH), timeout=2.0)
    except httpx.HTTPError:
        pass


def _send_command(step: dict[str, Any]) -> tuple[bool, str | None]:
    """Forward one validated step to the robot. 2xx = success, unless the
    body carries a failure ack ('ok': false or status failed/error)."""
    try:
        response = _client.post(_url(ROBOT_COMMAND_PATH), json=step, timeout=STEP_TIMEOUT_S)
    except httpx.TimeoutException:
        return False, "timeout"
    except httpx.HTTPError as e:
        return False, f"request failed: {e}"
    if response.status_code >= 400:
        return False, f"HTTP {response.status_code}"
    try:
        body = response.json()
    except ValueError:
        return True, None
    if isinstance(body, dict):
        if body.get("ok") is False or body.get("status") in ("failed", "error"):
            return False, str(body.get("error") or body)
    return True, None


def _plan_steps(
    profile: Profile, command: str, telemetry: dict[str, Any]
) -> tuple[list[dict[str, Any]] | None, list[str]]:
    """NL -> LLM -> envelope, with one corrective retry on validation
    errors. Returns (normalized_steps, errors); steps is None when the
    proposal was rejected."""
    messages = llm.initial_messages(profile, command, telemetry)
    errors: list[str] = []
    for _ in range(2):
        raw = llm.chat(messages)
        try:
            proposal = json.loads(raw)
        except json.JSONDecodeError:
            proposal, errors = None, ["answer is not valid JSON"]
        else:
            steps, errors = validate_plan(profile, proposal)
            if steps is not None:
                return steps, []
        messages = llm.corrective_messages(messages, raw, errors)
    return None, errors


def _execute(steps: list[dict[str, Any]], plan: list[dict[str, Any]]) -> dict[str, Any]:
    """Run the plan step by step. Fresh telemetry is required before every
    step; a dead link, a failure ack, a timeout or an e-stop aborts the
    plan and tells the robot to stop."""
    results: list[dict[str, Any]] = []
    for i, step in enumerate(steps):
        if stop_event.is_set():
            _robot_stop()
            return {"status": "stopped", "plan": plan, "steps": results}
        try:
            telemetry = _fetch_telemetry()
        except HTTPException as e:
            _robot_stop()
            return {"status": "failed", "plan": plan, "steps": results, "error": e.detail}
        if telemetry.get("connected") is False:
            _robot_stop()
            return {"status": "failed", "plan": plan, "steps": results, "error": "robot disconnected"}
        ok, detail = _send_command(step)
        results.append({"index": i, **step, "ok": ok, "detail": detail})
        if not ok:
            _robot_stop()
            return {"status": "failed", "plan": plan, "steps": results}
    return {"status": "completed", "plan": plan, "steps": results}


def _set(command_id: str, **fields: Any) -> None:
    with _lock:
        _commands[command_id].update(fields)


def _run_command(command_id: str, profile: Profile, command: str) -> None:
    """Background worker: telemetry -> plan -> execute, updating the
    command record. Always ends in a terminal status."""
    try:
        try:
            telemetry = _fetch_telemetry()
        except HTTPException as e:
            _set(command_id, status="failed", error=e.detail)
            return
        steps, errors = _plan_steps(profile, command, telemetry)
        if steps is None:
            _set(command_id, status="rejected", reason="; ".join(errors))
            return
        _set(command_id, status="executing", plan=steps)
        result = _execute(steps, steps)
        _set(
            command_id,
            status=result["status"],
            plan=result["plan"],
            steps=result["steps"],
            error=result.get("error"),
        )
    except llm.LLMError as e:
        _set(command_id, status="failed", error=str(e))
    except Exception as e:  # noqa: BLE001 - a record must never stay "executing"
        _set(command_id, status="failed", error=f"internal error: {e!r}")


@app.post("/profile")
def upload_profile(body: dict[str, Any]) -> dict[str, Any]:
    global _profile
    try:
        profile = Profile.model_validate(body)
    except ValidationError as e:
        raise HTTPException(422, json.loads(e.json())) from e
    _profile = profile  # hot-apply
    return {"loaded": True, "robot": profile.robot.name, "actions": [a.action for a in profile.actions]}


@app.get("/profile")
def get_profile() -> dict[str, Any]:
    if _profile is None:
        raise HTTPException(404, "no profile loaded")
    return _profile.model_dump()


@app.post("/commands", status_code=202)
def submit_command(body: CommandBody) -> dict[str, Any]:
    profile = _require_profile()
    with _lock:
        if any(r["status"] in ("planning", "executing") for r in _commands.values()):
            raise HTTPException(409, "another command is still running")
        command_id = secrets.token_hex(6)
        _commands[command_id] = {
            "id": command_id,
            "status": "planning",
            "command": body.command,
            "plan": None,
            "steps": [],
            "error": None,
            "reason": None,
        }
    stop_event.clear()
    threading.Thread(target=_run_command, args=(command_id, profile, body.command), daemon=True).start()
    return {"id": command_id, "status": "planning"}


@app.get("/commands/{command_id}")
def command_status(command_id: str) -> dict[str, Any]:
    with _lock:
        record = _commands.get(command_id)
        if record is None:
            raise HTTPException(404, "unknown command id")
        return dict(record)


@app.get("/state")
def state() -> dict[str, Any]:
    return _fetch_telemetry()


@app.post("/stop")
def stop() -> dict[str, Any]:
    stop_event.set()
    _robot_stop()
    return {"stopped": True}


@app.get("/health")
def health() -> dict[str, Any]:
    ok = False
    if ROBOT_BASE_URL:
        try:
            _fetch_telemetry()
            ok = True
        except HTTPException:
            ok = False
    return {
        "ok": ok,
        "adapter": "http",
        "robot_configured": bool(ROBOT_BASE_URL),
        "profile_loaded": _profile is not None,
    }


def main() -> None:
    print(
        f"[robotctl] api=http://{API_HOST}:{API_PORT}"
        f" llm={os.environ.get('LLM_BASE_URL') or '(not set)'}"
        f" model={os.environ.get('LLM_MODEL') or '(not set)'}"
        f" robot={ROBOT_BASE_URL or '(not set)'}",
        flush=True,
    )
    uvicorn.run(app, host=API_HOST, port=API_PORT)


if __name__ == "__main__":
    main()
