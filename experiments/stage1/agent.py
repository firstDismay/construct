"""Minimal stateless DeepSeek client for Construct Stage I.

Each decision uses only the current public observation. No hidden environment
state, cross-episode history, or research journal is sent to the model.
External memory/RAG is intentionally not implemented in this baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from experiments.stage1.environment import Environment


API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-flash"

SYSTEM_PROMPT = """You are the decision-making component in a controlled grid-world experiment.
Your imperative is: win the current expedition.

The environment is deterministic, but you do not see the full map or the hidden
movement-cost law. You receive only the current public observation.
- MOVE_N, MOVE_S, MOVE_E, MOVE_W attempt to enter the adjacent cell in that direction.
- Only directions listed in available_actions are legal moves from the current cell.
- A and B cells have different time/energy movement costs. The mapping is hidden.
- SEARCH spends 2 time units and reveals the costs for A/B tile types adjacent to you.
- Entering S or G costs 1 time and 1 energy. Reaching G wins; exhausting time or
  energy before reaching G loses.
- You observe remaining time and energy after each action.

Choose one action from available_actions. Do not invent unseen map data and do
not assume a hidden cost mapping without evidence. Respond with exactly one JSON
object, no Markdown, in this shape:
{"action":"MOVE_E","expectation":"One short sentence about the expected immediate result."}
The expectation is a brief prediction, not a step-by-step explanation.
"""


class AgentResponseError(ValueError):
    """The model response did not contain a valid action object."""


@dataclass(frozen=True)
class AgentDecision:
    action: str
    expectation: str
    raw_content: str
    usage: dict[str, Any]
    messages: list[dict[str, str]]
    model: str
    finish_reason: str | None


def _parse_decision(content: str) -> tuple[str, str]:
    """Parse the required JSON object, tolerating a surrounding JSON fence."""
    text = content.strip()
    fenced = re.fullmatch(r"\x60\x60\x60(?:json)?\s*(.*?)\s*\x60\x60\x60", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AgentResponseError("Model response was not valid JSON") from exc

    if not isinstance(payload, dict):
        raise AgentResponseError("Model response must be a JSON object")
    action = payload.get("action")
    expectation = payload.get("expectation", "")
    if not isinstance(action, str) or action not in Environment.ACTIONS:
        raise AgentResponseError("Model response has an unknown or missing action")
    if not isinstance(expectation, str):
        expectation = str(expectation)
    return action, expectation.strip()


class DeepSeekAgent:
    """Small standard-library client; the API key is never logged."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        timeout: float = 60.0,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("DEEPSEEK_API_KEY is empty")
        self.api_key = api_key.strip()
        self.model = model
        self.timeout = timeout
        self.calls = 0

    def decide(self, observation: dict[str, Any], available_actions: list[str]) -> AgentDecision:
        # The model receives no cross-step transcript in baseline mode.
        public_input = {
            "objective": "win the current expedition",
            "available_actions": available_actions,
            "observation": observation,
        }
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(public_input, ensure_ascii=False, separators=(",", ":")),
            },
        ]
        request_body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "temperature": 0.2,
                "max_tokens": 256,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = Request(
            API_URL,
            data=request_body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        self.calls += 1
        try:
            with urlopen(request, timeout=self.timeout) as response:
                api_result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DeepSeek API returned HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach DeepSeek API: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError("DeepSeek API request timed out") from exc

        try:
            choice = api_result["choices"][0]
            raw_content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("DeepSeek response did not contain choices[0].message.content") from exc

        if not isinstance(raw_content, str):
            raise RuntimeError("DeepSeek returned a non-text message content")
        action, expectation = _parse_decision(raw_content)
        usage = api_result.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {}
        return AgentDecision(
            action=action,
            expectation=expectation,
            raw_content=raw_content,
            usage=usage,
            messages=messages,
            model=str(api_result.get("model", self.model)),
            finish_reason=choice.get("finish_reason"),
        )
