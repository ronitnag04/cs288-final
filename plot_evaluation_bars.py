"""
Plot mean baseline vs baseline+prompt vs steered scores grouped by direction.

Reads evaluation JSONL (e.g. from ``evaluate_all.py``) with:
  - direction
  - baseline_score
  - baseline_with_prompt_score
  - steered_score

Error bars use the standard error of the mean (SEM) within each direction group.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import DefaultDict

import pandas as pd

BENCHMARK_AXES = ("formality", "instruction_following", "truthfulness")


def _sem(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    return stdev(values) / math.sqrt(n)


def load_means_by_direction(
    path: Path,
    axis_filter: str,
) -> tuple[list[str], list[float], list[float], list[float], list[float], list[float], list[float]]:
    """
    Group rows by ``direction``; return ordered labels and mean scores + SEM per group.
    """
    by_dir: DefaultDict[str, list[tuple[float, float, float]]] = defaultdict(list)

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            axis = str(row.get("axis", "")).strip().lower()
            if axis != axis_filter:
                continue
            d = row.get("direction")
            if d is None or d == "":
                continue
            direction = str(d).strip().lower()
            if direction not in ("add", "subtract"):
                continue
            try:
                baseline = float(row["baseline_score"])
                baseline_with_prompt = float(row["baseline_with_prompt_score"])
                steered = float(row["steered_score"])
            except (KeyError, TypeError, ValueError):
                continue
            by_dir[direction].append((baseline, baseline_with_prompt, steered))

    # Stable order: add then subtract; only show groups that have data
    order = ("add", "subtract")
    labels: list[str] = []
    baseline_means: list[float] = []
    baseline_with_prompt_means: list[float] = []
    steered_means: list[float] = []
    baseline_sems: list[float] = []
    baseline_with_prompt_sems: list[float] = []
    steered_sems: list[float] = []

    for d in order:
        pairs = by_dir.get(d)
        if not pairs:
            continue
        bs = [p[0] for p in pairs]
        bps = [p[1] for p in pairs]
        ss = [p[2] for p in pairs]
        suffix = " (more formal)" if d == "add" else " (less formal)"
        labels.append(d + suffix)
        baseline_means.append(mean(bs))
        baseline_with_prompt_means.append(mean(bps))
        steered_means.append(mean(ss))
        baseline_sems.append(_sem(bs))
        baseline_with_prompt_sems.append(_sem(bps))
        steered_sems.append(_sem(ss))

    if not labels:
        raise ValueError(
            f"No valid rows for axis={axis_filter!r} with direction in ('add','subtract') and scores in {path}"
        )

    return (
        labels,
        baseline_means,
        baseline_with_prompt_means,
        steered_means,
        baseline_sems,
        baseline_with_prompt_sems,
        steered_sems,
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot mean baseline vs steered scores by steering direction."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "evaluation.jsonl",
        help="Input JSONL (evaluate_all.py output) or benchmark CSV.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "evaluation_barplot.png",
        help="Output image path (e.g., .png).",
    )
    p.add_argument(
        "--csv-metric",
        type=str,
        default="prompt_level_loose_acc",
        help="CSV mode: metric column to plot.",
    )
    p.add_argument(
        "--csv-stderr",
        type=str,
        default="prompt_level_loose_stderr",
        help="CSV mode: stderr column for error bars.",
    )
    p.add_argument(
        "--stderr",
        choices=("auto", "on", "off"),
        default="auto",
        help="Control error bars/labels: auto (default), on, or off.",
    )
    p.add_argument("--title", type=str, default=None, help="Optional chart title override.")
    args = p.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install with: python3 -m pip install matplotlib"
        ) from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.input.suffix.lower() == ".csv":
        df = pd.read_csv(args.input)
        required = {"run_label", args.csv_metric}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise SystemExit(f"CSV missing required columns: {missing}")

        run_order = ("baseline", "baseline_with_prompting", "steering_1p0")
        label_map = {
            "baseline": "Baseline",
            "baseline_with_prompting": "Baseline + prompt",
            "steering_1p0": "Steered",
        }
        labels: list[str] = []
        values: list[float] = []
        errs: list[float] = []
        has_stderr_data = args.csv_stderr in df.columns and df[args.csv_stderr].notna().any()
        for run in run_order:
            match = df[df["run_label"] == run]
            if match.empty:
                continue
            labels.append(label_map.get(run, run))
            values.append(float(match.iloc[0][args.csv_metric]))
            if args.csv_stderr in match.columns:
                e = match.iloc[0][args.csv_stderr]
                errs.append(float(e) if pd.notna(e) else 0.0)
            else:
                errs.append(0.0)

        if not labels:
            raise SystemExit("CSV mode found no recognized run_label rows.")

        # Match JSONL style: one grouped x-category with three side-by-side bars.
        width = 0.22
        x = [0]
        x_positions = [-width, 0.0, width]
        fig_w = 7
        fig, ax = plt.subplots(figsize=(fig_w, 5))
        show_err = args.stderr == "on" or (args.stderr == "auto" and has_stderr_data)
        for xi, y, e, label in zip(x_positions, values, errs, labels):
            if show_err:
                ax.bar(
                    [xi],
                    [y],
                    width=width,
                    yerr=[e],
                    capsize=4,
                    alpha=0.55,
                    label=label,
                )
                y_pos = min(1.02, y + e + 0.015)
                ax.text(xi, y_pos, f"{e:.2f}", ha="center", va="bottom", fontsize=8)
            else:
                ax.bar(
                    [xi],
                    [y],
                    width=width,
                    alpha=0.55,
                    label=label,
                )

        direction_value = str(df.get("steering_direction", pd.Series([""])).iloc[0]).strip().lower()
        group_label = direction_value if direction_value in ("add", "subtract") else args.input.stem
        ax.set_xlabel("Direction")
        ax.set_ylabel(args.csv_metric)
        ax.set_title(args.title or f"{args.input.stem}: {args.csv_metric}")
        ax.set_xticks(x)
        ax.set_xticklabels([group_label], rotation=0, ha="center")
        ax.set_ylim(0.0, 1.06)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output, dpi=150)
        plt.close(fig)
        print(f"Saved plot to {args.output}")
        return

    written = 0
    for axis_name in BENCHMARK_AXES:
        try:
            (
                labels,
                baseline_means,
                baseline_with_prompt_means,
                steered_means,
                baseline_sems,
                baseline_with_prompt_sems,
                steered_sems,
            ) = load_means_by_direction(args.input, axis_filter=axis_name)
        except ValueError:
            print(f"[warn] Skipping {axis_name}: no valid rows found.")
            continue

        x = list(range(len(labels)))
        width = 0.22

        fig_w = max(7, len(labels) * 2.5)
        fig, ax = plt.subplots(figsize=(fig_w, 5))
        baseline_x = [i - width for i in x]
        baseline_with_prompt_x = x
        steered_x = [i + width for i in x]

        ax.bar(
            baseline_x,
            baseline_means,
            width=width,
            label="Baseline",
            yerr=baseline_sems if args.stderr != "off" else None,
            capsize=4 if args.stderr != "off" else 0,
            alpha=0.55,
        )
        ax.bar(
            baseline_with_prompt_x,
            baseline_with_prompt_means,
            width=width,
            label="Baseline + prompt",
            yerr=baseline_with_prompt_sems if args.stderr != "off" else None,
            capsize=4 if args.stderr != "off" else 0,
            alpha=0.55,
        )
        ax.bar(
            steered_x,
            steered_means,
            width=width,
            label="Steered",
            yerr=steered_sems if args.stderr != "off" else None,
            capsize=4 if args.stderr != "off" else 0,
            alpha=0.55,
        )

        if args.stderr != "off":
            for xi, y, e in zip(baseline_x, baseline_means, baseline_sems):
                y_pos = min(1.02, y + e + 0.015)
                ax.text(xi, y_pos, f"{e:.2f}", ha="center", va="bottom", fontsize=8)
            for xi, y, e in zip(
                baseline_with_prompt_x,
                baseline_with_prompt_means,
                baseline_with_prompt_sems,
            ):
                y_pos = min(1.02, y + e + 0.015)
                ax.text(xi, y_pos, f"{e:.2f}", ha="center", va="bottom", fontsize=8)
            for xi, y, e in zip(steered_x, steered_means, steered_sems):
                y_pos = min(1.02, y + e + 0.015)
                ax.text(xi, y_pos, f"{e:.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_xlabel("Direction")
        ax.set_ylabel("Score")
        ax.set_title(args.title or f"{axis_name}: mean baseline vs steered by direction")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center")
        ax.set_ylim(0.0, 1.06)
        ax.legend()
        fig.tight_layout()

        out_path = args.output.with_name(f"{args.output.stem}_{axis_name}{args.output.suffix}")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        written += 1
        print(f"Saved plot to {out_path}")

    if written == 0:
        raise SystemExit("No plots were generated (no valid rows for any benchmark axis).")


if __name__ == "__main__":
    main()
