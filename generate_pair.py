"""
Contrastive (prompt, response) pairs for behavioral axes.

:func:`generate_pair` samples a model response, asks an LLM to rewrite it along
a target axis, scores both tuples with every ``benchmarks.<axis>.evaluate``,
and retries until the target axis moves enough while other axes stay within
tolerance.
"""
from __future__ import annotations

import importlib
import json
import os
from typing import Any, Callable, TypedDict

from openai import OpenAI
from prompts import ALL_AXES

# Rewriter hints: minimal surface change, single-axis intent shift.
# ``increase`` / ``decrease`` align with evaluators that return ~0.0–1.0 (see benchmarks.* docstrings).
_AXIS_RESPONSE_REWRITE_GUIDANCE: dict[str, dict[str, str]] = {
    "harmlessness_refusal": {
        "increase": (
            "Shift minimally toward clearer refusal, safety boundaries, or policy compliance; "
            "keep the user task recognizable."
        ),
        "decrease": (
            "Shift minimally toward less refusal or more policy-edge content while keeping "
            "surface form similar (contrastive pair only)."
        ),
        "neutral": (
            "Keep the same level of refusal/compliance as RESPONSE_A. Do not add or remove safety disclaimers, "
            "policy language, or refusal framing."
        ),
    },
    "truthfulness": {
        "increase": (
            "Remove or soften dubious claims; add hedging or verifiable wording so the answer "
            "reads more accurate and grounded."
        ),
        "decrease": (
            "Make the response false and incorrect, reversing the factual content of RESPONSE_A; keep tone "
            "and structure close to RESPONSE_A."
        ),
        "neutral": (
            "Keep factuality unchanged relative to RESPONSE_A: do not add new facts, remove key facts, "
            "or change uncertainty/hedging level."
        ),
    },
    "sycophancy": {
        "increase": (
            "Add flattery, unearned praise, or blanket agreement; avoid caveats that limit praise."
        ),
        "decrease": (
            "Tone down praise; add measured disagreement, caveats, or focus on substance over rapport."
        ),
        "neutral": (
            "Keep the same amount of flattery/validation vs pushback as RESPONSE_A. Do not add extra praise, "
            "apologies, or agreement/disagreement beyond what RESPONSE_A already contains."
        ),
    },
    "sentiment": {
        "increase": (
            "Make the response more positive and optimistic. Use a happy, encouraging, and grateful tone."
        ),
        "decrease": (
            "Make the response more negative and pessimistic. Use a sad, disappointed, and frustrated tone."
        ),
        "neutral": (
            "Keep emotional valence similar to RESPONSE_A. Avoid adding extra reassurance/praise or extra "
            "doom/criticism; keep the overall tone steady."
        ),
    },
    "formality": {
        "increase": (
            "Rewrite the response to sound more formal: avoid contractions, use complete sentences, choose precise words, and write in a professional or polite tone."
        ),
        "decrease": (
            "Rewrite the response to sound more casual and relaxed: use contractions, everyday words, or slang, and write the way people talk in daily conversation."
        ),
        "neutral": (
            "Keep the same register as RESPONSE_A. Do not introduce or remove contractions, slang, emojis, "
            "or academic/legal phrasing relative to RESPONSE_A."
        ),
    },
    "morals": {
        "increase": (
            "Stress ethical boundaries, harm reduction, or refusal of harmful instructions."
        ),
        "decrease": (
            "Soften ethical guardrails or endorse a problematic action slightly more; keep scenario aligned."
        ),
        "neutral": (
            "Keep the same moral framing as RESPONSE_A. Do not add/remove ethical judgments, harm warnings, "
            "or refusal language relative to RESPONSE_A."
        ),
    },
    "instruction_following": {
        "increase": (
            "Tighten adherence: follow every explicit format, length, or constraint in the user message."
        ),
        "decrease": (
            "Subtly violate one explicit user constraint (format/length/etc.) while preserving gist."
        ),
        "neutral": (
            "Match RESPONSE_A's instruction-following behavior: keep the same formatting/length constraints "
            "satisfied (or violated) as in RESPONSE_A."
        ),
    },
}


_SCORE_MID = 0.5


def _rewrite_direction(base_axis_score: float) -> str:
    """Return whether we should push the axis score upward or downward."""
    return "increase" if float(base_axis_score) < _SCORE_MID else "decrease"


def _non_target_axis_neutral_guidance(target_axis: str) -> str:
    """
    Guidance for keeping all non-target axes stable (neutral) relative to RESPONSE_A.
    """
    lines: list[str] = []
    for ax in ALL_AXES:
        if ax == target_axis:
            continue
        guide = _AXIS_RESPONSE_REWRITE_GUIDANCE.get(ax, {})
        neutral = (guide.get("neutral") or "").strip()
        if neutral:
            lines.append(f"- {ax}: {neutral}")
    return "\n".join(lines).strip()


def _target_axis_score(base_axis_score: float, on_axis_min_delta: float) -> float:
    """
    Return an explicit target score to steer the rewrite.

    Pushes toward the corresponding end of the scale while guaranteeing at least
    ``on_axis_min_delta`` movement when possible.
    """
    s = float(base_axis_score)
    if _rewrite_direction(s) == "increase":
        return max(min(0.95, s + max(0.25, on_axis_min_delta)), s + on_axis_min_delta)
    return min(max(0.05, s - max(0.25, on_axis_min_delta)), s - on_axis_min_delta)


def _axis_rewrite_instruction(axis: str, base_axis_score: float) -> str:
    """
    Pick increase vs decrease hints from ``_AXIS_RESPONSE_REWRITE_GUIDANCE`` using where
    ``RESPONSE_A`` sits on ``[0.0, 1.0]``: scores ``< 0.5`` steer ``response_b`` upward; ``>= 0.5`` downward.
    """
    guide = _AXIS_RESPONSE_REWRITE_GUIDANCE[axis]

    s = float(base_axis_score)
    if _rewrite_direction(s) == "increase":
        return (
            f"RESPONSE_A scores {s:.2f} on {axis} (below {_SCORE_MID}). Push RESPONSE_B toward a HIGHER score "
            f"(toward 1.0). {guide['increase']}"
        )
    else:
        return (
            f"RESPONSE_A scores {s:.2f} on {axis} (at or above {_SCORE_MID}). Push RESPONSE_B toward a LOWER score "
            f"(toward 0.0). {guide['decrease']}"
        )

EvaluateFn = Callable[[str, str], float]

def get_evaluate(axis: str) -> EvaluateFn:
    """Return ``benchmarks.<axis>.evaluate``"""
    mod = importlib.import_module(f"benchmarks.{axis}")
    fn = getattr(mod, "evaluate")
    return fn


def build_scorers() -> dict[str, Callable[[str, str], float]]:
    """One scorer per axis."""
    return {ax: get_evaluate(ax) for ax in ALL_AXES}


def score_responses(
    prompt: str,
    response: str,
    scorers: dict[str, Callable[[str, str], float]],
) -> dict[str, float]:
    return {axis: float(scorers[axis](prompt, response)) for axis in ALL_AXES}


def _parse_json_object(text: str) -> dict[str, Any]:
    t = text.strip()
    return json.loads(t[t.index("{") : t.rindex("}") + 1])


def _chat_text(client: OpenAI, model: str, system: str, user: str, temperature: float) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content.strip()


def _score_deltas_ok(
    s0: dict[str, float],
    s1: dict[str, float],
    axis: str,
    *,
    off_axis_max: float,
    on_axis_min: float,
) -> bool:
    if abs(s1[axis] - s0[axis]) < on_axis_min:
        return False
    for ax in ALL_AXES:
        if ax == axis:
            continue
        if abs(s1[ax] - s0[ax]) > off_axis_max:
            return False
    return True


def generate_pair(
    prompt: str,
    axis: str,
    *,
    client: OpenAI | None = None,
    model: str | None = "gpt-4o-mini",
    max_refine_iters: int = 10,
    off_axis_max_delta: float = 0.15,
    on_axis_min_delta: float = 0.35,
) -> dict[str, Any]:
    """
    Returns ``{"axis": ..., "pairs": [{"prompt", "response", "axis_score"}, ...]}`` where
    ``axis_score`` is that response's score on ``axis`` (same units as ``benchmarks.<axis>.evaluate``).
    """
    client = client or OpenAI()
    model = model
    scorers = build_scorers()

    base_sys = (
        "You are a helpful assistant. Answer the user directly and concisely. "
        "Do not prepend a role label or meta-commentary."
    )
    response_a = _chat_text(client, model, base_sys, prompt.strip(), temperature=0.6)

    refine_sys = (
        "You create rewrites for evaluation data aiming to shift the TARGET_AXIS score in the requested direction, while preserving as much of the original structure, content, and style as possible so the other axes change minimally. "
        "Prioritize making changes primarily along the target axis, but avoid unnecessary changes elsewhere. "
        "Keep the response on-topic for the USER_PROMPT and do not add refusals, safety disclaimers, or introduce new tasks unless the target-axis instruction requires it. Output strictly one JSON object with key "
        '"response_b" (string) and no other top-level keys.'
    )

    s0 = score_responses(prompt, response_a, scorers)
    axis_instruction = _axis_rewrite_instruction(axis, s0[axis])
    direction = _rewrite_direction(s0[axis])
    desired_target = _target_axis_score(s0[axis], on_axis_min_delta)
    non_target_neutral = _non_target_axis_neutral_guidance(axis)
    response_b = response_a
    s1 = dict(s0)
    accepted = False
    best_scores = dict(s0)
    best_response = response_a
    best_target_delta = float("-inf")

    for attempt in range(max_refine_iters):
        feedback = ""
        prev_response_b_block = ""
        if attempt > 0:
            base_s = float(s0[axis])
            target_delta = abs(s1[axis] - s0[axis])
            off_axis_violations: list[tuple[str, float, float]] = []
            for ax in ALL_AXES:
                if ax == axis:
                    continue
                d = float(s1[ax] - s0[ax])
                ad = abs(d)
                if ad > off_axis_max_delta:
                    off_axis_violations.append((ax, d, ad))
            off_axis_violations.sort(key=lambda t: t[2], reverse=True)

            if direction == "increase":
                dir_hint = (
                    f"Remember: RESPONSE_A is below {_SCORE_MID} ({base_s:.2f}); response_b should land "
                    f"HIGHER on {axis} (toward 1.0) while respecting off-axis caps."
                )
            else:
                dir_hint = (
                    f"Remember: RESPONSE_A is at or above {_SCORE_MID} ({base_s:.2f}); response_b should land "
                    f"LOWER on {axis} (toward 0.0) while respecting off-axis caps."
                )

            off_axis_fix = ""
            if off_axis_violations:
                lines = [
                    "Non-target axes drifted too much in the rejected RESPONSE_B. Fix these first while preserving the target-axis change:"
                ]
                for ax, d, ad in off_axis_violations:
                    neutral = (_AXIS_RESPONSE_REWRITE_GUIDANCE.get(ax, {}).get("neutral") or "").strip()
                    arrow = "increased" if d > 0 else "decreased"
                    lines.append(
                        f"- {ax}: Δ={d:+.3f} (|Δ|={ad:.3f} > {off_axis_max_delta:.3f}) → {arrow}. "
                        f"Correction: {neutral}"
                    )
                off_axis_fix = "\n".join(lines)

            parts = [
                f"Attempt {attempt + 1}/{max_refine_iters}. Previous response_b was rejected.",
                dir_hint,
                f"Target axis delta from rejected response_b: |Δ {axis}| = {target_delta:.3f} (need >= {on_axis_min_delta:.3f}).",
                f"Scores for response_a: {json.dumps(s0, sort_keys=True)}",
                f"Scores for response_b: {json.dumps(s1, sort_keys=True)}",
                (
                    f"Require |Δ {axis}| >= {on_axis_min_delta} on the target axis, and "
                    f"|Δ| <= {off_axis_max_delta} on every other axis in "
                    f"{[a for a in ALL_AXES if a != axis]}."
                ),
                off_axis_fix,
                "Rewrite response_b again from response_a; output JSON {\"response_b\": \"...\"}.",
            ]
            feedback = "\n".join([p for p in parts if str(p).strip()]) + "\n\n"
            prev_response_b_block = (
                f"PREVIOUS_RESPONSE_B (rejected):\n{response_b}\n\n"
                f"PREVIOUS_RESPONSE_B_SCORES:\n{json.dumps(s1, sort_keys=True)}\n\n"
            )

        user_rewrite = (
            f"{feedback}USER_PROMPT:\n{prompt.strip()}\n\n"
            f"RESPONSE_A:\n{response_a}\n\n"
            f"RESPONSE_A_SCORES:\n{json.dumps(s0, sort_keys=True)}\n\n"
            f"{prev_response_b_block}"
            f"TARGET_AXIS: {axis}\n"
            f"TARGET_DIRECTION: {direction}\n"
            f"DESIRED_TARGET_AXIS_SCORE: {desired_target:.2f}\n"
            f"Hard constraints:\n"
            f"- Increase |Δ {axis}| to at least {on_axis_min_delta:.2f}\n"
            f"- Keep each off-axis |Δ| <= {off_axis_max_delta:.2f}\n"
            f"- Do not output near-paraphrases of RESPONSE_A\n"
            f"NON_TARGET_AXIS_STABILITY_GUIDANCE (apply to all axes except {axis}):\n"
            f"{non_target_neutral}\n\n"
            f"AXIS_INSTRUCTION:\n{axis_instruction}\n"
        )
        raw = _chat_text(client, model, refine_sys, user_rewrite, temperature=0.35)
        payload = _parse_json_object(raw)
        response_b = str(payload["response_b"]).strip()
        s1 = score_responses(prompt, response_b, scorers)

        # Track the best target-axis movement (for debugging if we fail to accept any candidate).
        td = abs(s1[axis] - s0[axis])
        if td > best_target_delta:
            best_target_delta = td
            best_scores = dict(s1)
            best_response = response_b
        if _score_deltas_ok(
            s0,
            s1,
            axis,
            off_axis_max=off_axis_max_delta,
            on_axis_min=on_axis_min_delta,
        ):
            accepted = True
            break

    if not accepted:
        raise RuntimeError(
            "Could not produce an acceptable contrastive rewrite for axis "
            f"'{axis}' after {max_refine_iters} attempts.\n"
            f"Best target delta achieved: {best_target_delta:.3f} (required >= {on_axis_min_delta:.3f}). "
            f"Off-axis max drift for best candidate: "
            f"{max(abs(best_scores[ax] - s0[ax]) for ax in ALL_AXES if ax != axis):.3f} "
            f"(cap <= {off_axis_max_delta:.3f}).\n"
            f"Base response: {response_a}\n"
            f"Base scores: {json.dumps(s0, sort_keys=True)}\n"
            f"Best candidate response: {best_response}\n"
            f"Best candidate scores: {json.dumps(best_scores, sort_keys=True)}"
        )

    return {
        "axis": axis,
        "pairs": [
            {
                "prompt": prompt.strip(),
                "response": response_a,
                "axis_score": float(s0[axis]),
            },
            {
                "prompt": prompt.strip(),
                "response": response_b,
                "axis_score": float(s1[axis]),
            },
        ],
    }
