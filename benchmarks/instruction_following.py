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
prompt, such as required format, structure, length, presence or absence of specific words, or other clear instructions about how to answer.
Ignore whether the response is factually correct, truthful, factually plausible, helpful, or domain-appropriate unless those aspects are directly specified as instructions in the prompt. 
For example, if the prompt says "Respond in exactly three words" or "Answer only in JSON", only judge whether these instructions were followed, regardless of the factual truth of the content.

Do not judge truthfulness, factual quality, helpfulness, creativity, style, redundancy, or domain correctness unless they are explicitly requested as constraints.

Return:
- instruction_following_score: A continuous score anywhere in [0.0, 1.0], where:
  1.0 = clearly follows the instructions
  0.0 = clearly violates the instructions
  0.5 = unrelated, ambiguous, or impossible to judge
- reason: one concise sentence explaining the judgment.
- violated_instruction: the exact instruction violated, or an empty string if
  none.
- follows_instructions: whether the response satisfies the prompt's explicit
  instructions.

[PROMPT]: {prompt}
[RESPONSE]: {response}

=======================
Output format:
```json

{{
  "instruction_following_score": float,
  "reason": string,
  "violated_instruction": string,
  "follows_instructions": boolean
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

    # reason = str(result.get("reason", "") or "").strip()
    # if reason:
    #     print(reason)

    raw = result.get("instruction_following_score")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
