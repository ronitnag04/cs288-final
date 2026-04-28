"""LLM judge for whether a model response follows the prompt's instructions."""
from __future__ import annotations

import json
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_JUDGE_MODEL = "gpt-4o-mini"

INSTRUCTION_FOLLOWING_PROMPT = """
You are a careful instruction-following judge.

Given only the task prompt and the model's response, evaluate whether the
response follows the prompt's explicit instructions.

Only evaluate explicit, checkable constraints that are directly stated in the
prompt. Do not judge factual quality, helpfulness, creativity, style quality,
redundancy, or domain correctness unless those are explicitly requested as
constraints.

Return:
- follows_instructions: whether the response satisfies the prompt's explicit
  instructions.
- instruction_following_score: 1.0 if the response clearly follows the
  instructions, 0.0 if it clearly violates them, and 0.5 if unrelated,
  ambiguous, or impossible to judge.
- violated_instruction: the exact instruction violated, or an empty string if
  none.
- reason: one concise sentence explaining the judgment.

[PROMPT]: {prompt}
[RESPONSE]: {response}

=======================
Output format:
```json

{{
  "follows_instructions": boolean,
  "instruction_following_score": float,
  "violated_instruction": string,
  "reason": string
}}

```
"""


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse first JSON object from model text, with light cleanup fallback."""
    text = text.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return {}


def evaluate(prompt: str, response: str) -> float | None:
    """Judge whether ``response`` follows ``prompt``'s explicit instructions.

    Returns only ``instruction_following_score`` (0.0–1.0). The model's reason is printed.
    """
    client = OpenAI()
    judge_prompt = INSTRUCTION_FOLLOWING_PROMPT.format(
        prompt=prompt, response=response
    )
    completion = client.chat.completions.create(
        model=_JUDGE_MODEL,
        messages=[{"role": "user", "content": judge_prompt}],
        max_tokens=256,
        temperature=0.0,
    )
    text = (completion.choices[0].message.content or "").strip()
    result = _extract_json_object(text)
    if not result:
        return None

    reason = str(result.get("reason", "") or "").strip()
    if reason:
        print(reason)

    raw = result.get("instruction_following_score")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
