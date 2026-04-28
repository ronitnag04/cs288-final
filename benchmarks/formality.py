"""
Formality axis benchmark via zero-shot NLI over a response string.

Model:
https://huggingface.co/cross-encoder/nli-deberta-v3-large

`evaluate(prompt, response)` returns a scalar in [0.0, 1.0]:
- 0.0 -> strongly informal
- 0.5 -> neutral/uncertain
- 1.0 -> strongly formal
"""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_MODEL_ID = "cross-encoder/nli-deberta-v3-large"
_tokenizer: AutoTokenizer | None = None
_model: AutoModelForSequenceClassification | None = None
_entail_idx: int | None = None

_LABELS = ("formal", "informal")
_HYPOTHESIS = {
    "formal": "The response is written in a professional or polite tone, avoiding contractions, everyday words, or slang, and writing in a formal or academic style.",
    "informal": "The response is written in a casual or informal tone, using contractions, everyday words, or slang, and writing in a conversational or everyday style.",
}


def _load_model() -> tuple[AutoTokenizer, AutoModelForSequenceClassification, int]:
    global _tokenizer, _model, _entail_idx
    if _tokenizer is None or _model is None or _entail_idx is None:
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
        _model = AutoModelForSequenceClassification.from_pretrained(_MODEL_ID)
        _model.eval()
        label2id = {str(k).lower(): int(v) for k, v in (_model.config.label2id or {}).items()}
        if "entailment" in label2id:
            _entail_idx = label2id["entailment"]
        else:
            # Known mapping for this model family: [contradiction, entailment, neutral].
            _entail_idx = 1
    return _tokenizer, _model, _entail_idx


def evaluate(prompt: str, response: str) -> float:
    """
    Return formality score for ``response`` in ``[0.0, 1.0]``.

    Parameters
    ----------
    prompt:
        Kept for interface consistency with other evaluators; unused.
    response:
        Arbitrary model text to score.

    Returns
    -------
    float
        Score in ``[0.0, 1.0]`` (inclusive), derived from NLI Deberta-v3-large.
    """
    del prompt  # not used for this axis; formality is judged on response text
    text = (response or "").strip()
    if not text:
        raise ValueError(f"Invalid response: {text}")

    tokenizer, model, entail_idx = _load_model()
    pairs = [(text, _HYPOTHESIS[label]) for label in _LABELS]
    encoded = tokenizer(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        logits = model(**encoded).logits
        entail_logits = logits[:, entail_idx]
        probs = torch.softmax(entail_logits, dim=0)
    # Probability assigned to "formal" over {"formal","informal"}.
    return float(probs[0].item())
