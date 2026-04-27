"""LLM judge for whether a counterfactual response still follows instructions."""
from __future__ import annotations

import json
from typing import Any

INSTRUCTION_FOLLOWING_PROMPT = """
You are a careful instruction-following judge.

The reference answer is assumed to follow the prompt's instructions. Evaluate
whether the counterfactual answer still follows the prompt's explicit
instructions.

Only evaluate explicit, checkable constraints that are directly stated in the
prompt. Do not judge factual quality, helpfulness, creativity, style quality,
redundancy, or domain correctness unless those are explicitly requested as
constraints. Do not require the counterfactual to match the reference wording.

Return:
- counterfactual_follows_instructions: whether the counterfactual follows the
  prompt's explicit instructions.
- preserves_reference_behavior: whether the counterfactual preserves the same
  instruction-following behavior as the reference.
- instruction_following_score: 1.0 if the counterfactual clearly follows the
  instructions, 0.0 if it clearly violates them, and 0.5 if unrelated,
  ambiguous, or impossible to judge.
- violated_instruction: the exact instruction violated, or an empty string if
  none.
- reason: one concise sentence explaining the judgment.

[PROMPT]: {prompt}
[REFERENCE]: {reference}
[COUNTERFACTUAL]: {counterfactual}

=======================
Output format:
```json

{{
  "counterfactual_follows_instructions": boolean,
  "preserves_reference_behavior": boolean,
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


def check_instruction_following(
    prompt: str, reference: str, counterfactual: str
) -> float | None:
    """Judge whether a counterfactual still follows instructions, assuming reference does.

    Returns only ``instruction_following_score`` (0.0–1.0). The model's reason is printed.
    """
    from anthropic import Anthropic

    client = Anthropic()
    judge_prompt = INSTRUCTION_FOLLOWING_PROMPT.format(
        prompt=prompt, reference=reference, counterfactual=counterfactual
    )
    response = client.messages.create(
        model="claude-haiku-4-5",
        messages=[{"role": "user", "content": judge_prompt}],
        max_tokens=256,
        temperature=0.0,
    )
    text = response.content[0].text.strip()
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
