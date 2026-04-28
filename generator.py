"""
Generate contrastive (prompt, response) rows via :func:`generate_pair.generate_pair`.

Writes ``counterfactuals.jsonl`` (or ``--output``): one JSON object per line, each the
return value of ``generate_pair`` for that run (``axis`` + ``pairs``). For each axis,
there is one row per prompt in ``prompts.AXIS_PROMPTS``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from generate_pair import generate_pair
from prompts import AXIS_PROMPTS, ALL_AXES


def main() -> None:
    p = argparse.ArgumentParser(description="Build counterfactuals.jsonl from generate_pair.")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("counterfactuals.jsonl"),
        help="JSONL path (default: counterfactuals.jsonl)",
    )
    args = p.parse_args()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY (e.g. export or a .env next to this file).")

    client = OpenAI()

    with args.output.open("w", encoding="utf-8") as out:
        for axis in ALL_AXES:
            prompts_for_axis = AXIS_PROMPTS.get(axis, [])
            if not prompts_for_axis:
                print(f"[warn] No prompts configured for axis '{axis}'; skipping.")
                continue
            for prompt in prompts_for_axis:
                row = generate_pair(
                    prompt,
                    axis,
                    client=client
                )
                out.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
