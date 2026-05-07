"""
Evaluate saved steering inference results across all benchmark axes.

Reads JSONL rows produced by ``steering_inference.py``/``batch_steering_inference.py``
and writes one output row per (input row, evaluation axis). Each output row includes
three scores for the same prompt:
  - baseline_score
  - baseline_with_prompt_score
  - steered_score
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Callable

ALLOWED_AXES = ("formality", "instruction_following", "truthfulness")


def infer_axis_from_checkpoint(path_text: str) -> str | None:
    p = path_text.lower()
    for axis in ALLOWED_AXES:
        if axis in p:
            return axis
    return None


def get_evaluator(axis: str) -> Callable[[str, str], float]:
    if axis not in ALLOWED_AXES:
        raise ValueError(f"Unsupported axis {axis!r}. Expected one of {ALLOWED_AXES}.")
    mod = importlib.import_module(f"benchmarks.{axis}")
    fn = getattr(mod, "evaluate")
    return fn


def main() -> None:
    p = argparse.ArgumentParser(
        description="Score baseline, baseline+prompt, and steered responses on all benchmark axes."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "results.jsonl",
        help="Input JSONL from steering_inference.py.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "evaluation.jsonl",
        help="Output JSONL with per-row scores.",
    )
    args = p.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    with args.input.open(encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            prompt = str(row.get("prompt", "")).strip()
            baseline = str(row.get("baseline_response", "")).strip()
            baseline_with_prompt = str(row.get("baseline_with_prompt_response", "")).strip()
            steered = str(row.get("steered_response", "")).strip()
            checkpoint = str(row.get("steering_checkpoint", ""))
            direction = row.get("direction")
            steering_axis = infer_axis_from_checkpoint(checkpoint)

            if not prompt or not baseline or not steered:
                skipped += 1
                continue
            for eval_axis in ALLOWED_AXES:
                eval_fn = get_evaluator(eval_axis)
                baseline_score = float(eval_fn(prompt, baseline))
                steered_score = float(eval_fn(prompt, steered))
                baseline_with_prompt_score = (
                    float(eval_fn(prompt, baseline_with_prompt))
                    if baseline_with_prompt
                    else None
                )

                out_row = {
                    "axis": eval_axis,
                    "steering_axis": steering_axis,
                    "prompt": prompt,
                    "steering_checkpoint": checkpoint,
                    "direction": direction,
                    "baseline_response": baseline,
                    "baseline_with_prompt_response": baseline_with_prompt,
                    "steered_response": steered,
                    "baseline_score": baseline_score,
                    "baseline_with_prompt_score": baseline_with_prompt_score,
                    "steered_score": steered_score,
                }
                fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                written += 1

    print(f"Wrote {written} scored rows to {args.output}")
    if skipped:
        print(f"Skipped {skipped} row(s) due to parse issues, missing fields, or unknown axis.")


if __name__ == "__main__":
    main()
