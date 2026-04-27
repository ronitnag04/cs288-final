import json
from anthropic import Anthropic
from dotenv import load_dotenv
import regex as re
load_dotenv()


TRUTHFULNESS_PROMPT = """
You are a careful factuality judge.

The reference answer is assumed to be factually correct. Evaluate whether the
counterfactual answer remains factually correct for the question.

Do not require identical wording. Penalize contradictions, incorrect entities,
unsupported factual claims, or answers that no longer address the question.

Return:
- counterfactual_factual: whether the counterfactual is a factually correct
  answer to the question.
- preserves_reference_claim: whether the counterfactual preserves the same core
  factual claim as the reference.
- truthfulness_score: 1.0 if the counterfactual is fully factual, 0.0 if it is
  factually false, and 0.5 if it is unrelated, ambiguous, or cannot be judged.
- reason: one concise sentence explaining the judgment.

[QUESTION]: {question}
[REFERENCE]: {reference}
[COUNTERFACTUAL]: {counterfactual}

=======================
Output format: 
```json

{{
  "counterfactual_factual": boolean,
  "preserves_reference_claim": boolean,
  "truthfulness_score": float,
  "reason": string
}}

```
"""




def check_truthfulness(
    question: str, reference: str, counterfactual: str
) -> float | None:
    """Return ``truthfulness_score`` (0.0–1.0). The model's reason is printed."""
    client = Anthropic()
    prompt = TRUTHFULNESS_PROMPT.format(
        question=question, reference=reference, counterfactual=counterfactual
    )
    response = client.messages.create(
        model="claude-haiku-4-5",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.0,
    )
    text = response.content[0].text.strip()
    match = re.search(r"\{(?:[^{}]|(?R))*\}", text)
    if not match:
        return None

    result = json.loads(match.group(0))
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
    question = "What is the capital of France?"
    reference = "Paris"
    counterfactual = "London"
    print(check_truthfulness(question, reference, counterfactual))

