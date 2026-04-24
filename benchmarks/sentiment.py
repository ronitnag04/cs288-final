"""
Sentiment axis benchmark: map text polarity to a scalar in [0.0, 1.0].

This mirrors the *binary sentiment* setting used in GLUE SST-2 (negative vs.
positive), as implemented in baseline stacks such as
https://github.com/nyu-mll/GLUE-baselines — we do not re-run those training
pipelines here; instead we use a lightweight, rule-augmented lexicon model that
targets the same axis.

Scoring uses VADER (Valence Aware Dictionary and sEntiment Reasoner):
https://github.com/cjhutto/vadersentiment

VADER's *compound* score lies in [-1.0, 1.0]. We linearly map it to [0.0, 1.0]
so that 0.0 is maximally negative, 0.5 is neutral, and 1.0 is maximally
positive, consistent with ``evaluate(prompt, response)`` in the project architecture.
"""

from __future__ import annotations

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def evaluate(prompt: str, response: str) -> float:
    """
    Return sentiment polarity for ``response`` on a scale from 0.0 (negative)
    through 0.5 (neutral) to 1.0 (positive).

    Parameters
    ----------
    prompt:
        Kept for interface consistency with other evaluators; unused.
    response:
        Arbitrary model text to score.

    Returns
    -------
    float
        Score in ``[0.0, 1.0]`` (inclusive), derived from VADER's compound metric.
    """
    del prompt
    text = response.strip()
    if not text:
        raise ValueError(f"Invalid response: {text}")
        
    compound = _analyzer.polarity_scores(text)["compound"]
    # Map [-1, 1] -> [0, 1]; clamp for numerical safety.
    return max(0.0, min(1.0, (compound + 1.0) * 0.5))
