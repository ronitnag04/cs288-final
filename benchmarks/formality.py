"""LLM judge for response formality with OpenSub-Slang calibration context."""
from __future__ import annotations

import json
from typing import Any
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

slang_llm_dataset_url = "https://raw.githubusercontent.com/amazon-science/slang-llm-benchmark/main/Data/slang_OpenSub_meta.tsv"
df = pd.read_csv(slang_llm_dataset_url, sep="\t")
slang_terms = df["SLANG_TERM"].tolist()
_OPENSUB_SLANG_CONTEXT = """
OpenSub-Slang context (amazon-science/slang-llm-benchmark):
Use this context to recognize informal/slang style markers that lower formality. Here is a list of slang terms:
{slang_terms}
"""

_JUDGE_MODEL = "gpt-4o-mini"

FORMALITY_PROMPT = """
You are a careful linguistic style judge.

Task:
Evaluate how formal the RESPONSE is as writing style, considering both PROMPT and RESPONSE.

Primary criterion:
- Formality of language register (formal/professional vs casual/conversational/slang-heavy).

Use the OpenSub-Slang context below to calibrate what counts as slang/informal markers.
Do NOT score factual correctness, helpfulness, safety, or instruction-following unless directly
reflected as style.

{opensub_slang_context}

Scoring: Continuous value in [0.0, 1.0], where:
- 1.0 = strongly formal register (professional, polite, structured, minimal slang)
- 0.5 = mixed/neutral register
- 0.0 = strongly informal register (casual, slang-heavy, colloquial, chatty)

Return:
- formality_score: float in [0.0, 1.0]
- reason: one concise sentence
- notable_markers: short list of phrases/tokens that influenced judgment

[PROMPT]: {prompt}
[RESPONSE]: {response}

=======================
Output format:
```json
{{
  "formality_score": float,
  "reason": string,
  "notable_markers": [string]
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
    """Judge response formality for a prompt-response pair.

    Returns ``formality_score`` in [0.0, 1.0], or ``None`` if parsing fails.
    """
    client = OpenAI()
    judge_prompt = FORMALITY_PROMPT.format(
        opensub_slang_context=_OPENSUB_SLANG_CONTEXT.strip(),
        prompt=prompt,
        response=response,
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

    raw = result.get("formality_score")
    if raw is None:
        return None
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return None
