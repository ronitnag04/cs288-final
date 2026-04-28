"""LLM judge for factual soundness of a model response given the task prompt."""
from __future__ import annotations

import json
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_JUDGE_MODEL = "gpt-4o-mini"

TRUTHFULNESS_PROMPT = """
You are a careful factuality judge.

Given only the task prompt and the model's response, evaluate whether the
response is factually sound as an answer to the prompt (correct claims, no clear
falsehoods, and it addresses what was asked).

Do not require any reference answer or fixed wording. Penalize clear false
claims, wrong entities or dates, unsupported factual assertions where the
prompt calls for accuracy, and answers that fail to address the question.

Return:
- is_factual: whether the response is factually correct for the prompt.
- truthfulness_score: 1.0 if the response is fully factual and on-topic, 0.0 if
  it is clearly false or misleading where a factual answer was expected, and
  0.5 if unrelated, ambiguous, subjective-only, or impossible to judge.
- reason: one concise sentence explaining the judgment.

[PROMPT]: {prompt}
[RESPONSE]: {response}

=======================
Output format:
```json

{{
  "is_factual": boolean,
  "truthfulness_score": float,
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
    """Judge factual soundness of ``response`` for ``prompt``.

    Returns only ``truthfulness_score`` (0.0–1.0). The model's reason is printed.
    """
    client = OpenAI()
    judge_prompt = TRUTHFULNESS_PROMPT.format(prompt=prompt, response=response)
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

    raw = result.get("truthfulness_score")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    prompt = "What is the capital of France?"
    response = "London"
    print(evaluate(prompt, response))
