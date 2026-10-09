"""Run one baseline (A) Stage I episode and write an audit JSONL log.

Example:
  set -a; source .env; set +a
  python -m experiments.stage1.run_episode --seed 1234 --log /tmp/construct-stage1-1234.jsonl

The log is an experimenter-side record and includes hidden world data. Never
pass the full log to the agent or treat hidden-data fields as observations.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import uuid
from typing import Any

from experiments.stage1.agent import DeepSeekAgent
from experiments.stage1.environment import Environment, EpisodeConfig


ENVIRONMENT_VERSION = "stage1-grid-v1"


def jsonable(value: Any) -> Any:
    """Convert dataclasses, tuples, and nested containers to JSON-safe values."""
    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_episode(
    seed: int,
    log_path: str | Path,
    *,
    swapped_law: bool = False,
    max_steps: int = 100,
    model: str = "deepseek-flash",
    api_key: str | None = None,
) -> dict[str, Any]:
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    resolved_key = api_key if api_key is not None else os.environ.get("DEEPSEEK_API_KEY", "")
    if not resolved_key.strip():
        raise ValueError("DEEPSEEK_API_KEY is not set; export it before running the episode")

    output_path = Path(log_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects previous experimental evidence from overwrite.
    log_handle = output_path.open("x", encoding="utf-8")

    def write_event(event: dict[str, Any]) -> None:
        log_handle.write(json.dumps(jsonable(event), ensure_ascii=False, separators=(",", ":")) + "\n")
        log_handle.flush()

    episode_id = str(uuid.uuid4())
    config = EpisodeConfig()
    env = Environment(config)
    try:
        episode = env.generate(seed, swapped_law=swapped_law, require_resource_feasible=True)
    except Exception:
        log_handle.close()
        output_path.unlink(missing_ok=True)
        raise
    observation = env.reset(episode)
    agent = DeepSeekAgent(api_key=resolved_key, model=model)
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_reported_tokens = 0
    terminal: str | None = None
    status = "STEP_LIMIT"
    step_count = 0

    write_event({
        "event": "episode_start",
        "timestamp": utc_now(),
        "episode_id": episode_id,
        "mode": "A",
        "environment_version": ENVIRONMENT_VERSION,
        "model": model,
        "seed": seed,
        "configuration": config,
        "swapped_law": swapped_law,
        "initial_observation": observation,
        # Research-only data is logged for replay but never sent to the LLM.
        "research_only": {
            "full_map": episode.grid,
            "start": episode.start,
            "goal": episode.goal,
            "hidden_cost_law": episode.cost_law,
        },
        "retrieved_memories": [],
    })

    try:
        for step_index in range(1, max_steps + 1):
            available_actions = [move["action"] for move in observation["moves"]]
            available_actions.append("SEARCH")
            try:
                decision = agent.decide(
                    observation=jsonable(observation),
                    available_actions=available_actions,
                )
            except Exception as exc:
                status = "ABORTED_AGENT_ERROR"
                write_event({
                    "event": "agent_error",
                    "timestamp": utc_now(),
                    "episode_id": episode_id,
                    "step": step_index,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "observation": observation,
                    "available_actions": available_actions,
                    "api_calls": agent.calls,
                })
                break

            result, next_observation = env.step(decision.action)
            step_count += 1
            usage = decision.usage
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            reported_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_reported_tokens += reported_tokens

            public_result = {
                key: value for key, value in result.items()
                if key not in ("before", "after")
            }
            write_event({
                "event": "step",
                "timestamp": utc_now(),
                "episode_id": episode_id,
                "mode": "A",
                "step": step_index,
                "observation": observation,
                "available_actions": available_actions,
                "selected_action": decision.action,
                "expectation": decision.expectation,
                "llm_input": decision.messages,
                "llm_output": decision.raw_content,
                "model": decision.model,
                "finish_reason": decision.finish_reason,
                "usage": usage,
                "immediate_memory": None,
                "retrieved_memories": [],
                "objective_result": public_result,
                "new_observation": next_observation,
                "resource_state": {
                    "time": next_observation["time"],
                    "energy": next_observation["energy"],
                },
                "terminal_state": result["terminal"],
                "research_only": {
                    "position": env.position,
                    "hidden_tile_under_agent": episode.grid[env.position[1]][env.position[0]],
                },
            })

            observation = next_observation
            if result["terminal"] is not None:
                terminal = result["terminal"]
                status = terminal
                break
    finally:
        write_event({
            "event": "episode_end",
            "timestamp": utc_now(),
            "episode_id": episode_id,
            "mode": "A",
            "status": status,
            "terminal_state": terminal,
            "steps": step_count,
            "api_calls": agent.calls,
            "remaining_time": observation["time"],
            "remaining_energy": observation["energy"],
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "reported_tokens": total_reported_tokens,
            "log_path": str(output_path.resolve()),
        })
        log_handle.close()

    return {
        "episode_id": episode_id,
        "status": status,
        "terminal_state": terminal,
        "steps": step_count,
        "api_calls": agent.calls,
        "remaining_time": observation["time"],
        "remaining_energy": observation["energy"],
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "reported_tokens": total_reported_tokens,
        "log_path": str(output_path.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one Construct Stage I baseline episode (mode A).")
    parser.add_argument("--seed", type=int, required=True, help="Deterministic world seed")
    parser.add_argument("--log", required=True, help="New JSONL path; existing files are never overwritten")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--swapped-law", action="store_true", help="Run the same seed with A/B costs swapped")
    args = parser.parse_args()

    try:
        summary = run_episode(
            seed=args.seed,
            log_path=args.log,
            max_steps=args.max_steps,
            model=args.model,
            swapped_law=args.swapped_law,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] in ("WIN", "LOSS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
