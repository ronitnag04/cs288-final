"""
Run steering inference over many prompts with one model load.

This is much faster than invoking ``steering_inference.py`` once per prompt because
tokenizer/model initialization happens only once.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from steering_inference import (
    append_result_jsonl,
    apply_steering,
    format_prompt,
    load_steering_checkpoint,
)


def _read_prompts(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    prompts: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("{"):
                try:
                    d = json.loads(s)
                    p = str(d.get("prompt", "")).strip()
                    if p:
                        prompts.append(p)
                        continue
                except json.JSONDecodeError:
                    pass
            prompts.append(s)
    return prompts


def _pick_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    p = argparse.ArgumentParser(description="Batch Qwen steering inference with persistent JSONL logging.")
    p.add_argument(
        "--steering",
        type=Path,
        default=Path(__file__).resolve().parent / "steering_vectors.pt",
        help="Checkpoint from steering_train.py (steering_vectors + meta).",
    )
    p.add_argument(
        "--prompts-file",
        type=Path,
        required=True,
        help="Text or JSONL file of prompts. For JSONL, each line may have a 'prompt' field.",
    )
    p.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Override base model id.",
    )
    p.add_argument("--coeff", type=float, default=1.0, help="Steering strength.")
    p.add_argument("--direction", choices=("subtract", "add"), default="subtract")
    p.add_argument("--normalize", action="store_true", help="L2-normalize each layer vector.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    p.add_argument("--device", type=str, default=None, help="e.g. cuda, mps, cpu")
    p.add_argument(
        "--results-jsonl",
        type=Path,
        default=Path(__file__).resolve().parent / "results.jsonl",
        help="Append run records (prompt, baseline, steered) to this JSONL file.",
    )
    p.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Print progress every N prompts.",
    )
    args = p.parse_args()

    prompts = _read_prompts(args.prompts_file)
    if not prompts:
        raise SystemExit(f"No prompts found in {args.prompts_file}")

    device = _pick_device(args.device)
    dtype = getattr(torch, args.dtype)
    subtract = args.direction == "subtract"

    steering_vectors, meta = load_steering_checkpoint(args.steering)
    model_name = args.model or meta.get("model_name")
    if not model_name:
        raise SystemExit("Pass --model or ensure steering checkpoint meta contains model_name.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    total = len(prompts)
    for i, prompt_text in enumerate(prompts, start=1):
        batch_with_prompt = {k: v.to(device) for k, v in format_prompt(tokenizer, prompt_text + " Be formal.").items()}
        batch = {k: v.to(device) for k, v in format_prompt(tokenizer, prompt_text).items()}

        # Baseline with prompt
        with torch.inference_mode():
            out_base_with_prompt = model.generate(**batch_with_prompt, **gen_kwargs)
        text_base_with_prompt = tokenizer.decode(out_base_with_prompt[0], skip_special_tokens=True)

        # Baseline without prompt
        with torch.inference_mode():
            out_base = model.generate(**batch, **gen_kwargs)
        text_base = tokenizer.decode(out_base[0], skip_special_tokens=True)

        # Steered, use baseline without prompt
        with apply_steering(
            model,
            steering_vectors,
            coeff=args.coeff,
            subtract=subtract,
            normalize=args.normalize,
        ):
            with torch.inference_mode():
                out = model.generate(**batch, **gen_kwargs)
        text_steered = tokenizer.decode(out[0], skip_special_tokens=True)

        append_result_jsonl(
            args.results_jsonl,
            {
                "prompt": prompt_text,
                "baseline_response": text_base,
                "baseline_with_prompt_response": text_base_with_prompt,
                "steered_response": text_steered,
                "steering_checkpoint": str(args.steering),
                "direction": args.direction,
            },
        )

        if args.print_every > 0 and (i % args.print_every == 0 or i == total):
            print(f"[{i}/{total}] wrote result for prompt {i}", flush=True)

    print(f"Done. Appended {total} rows to {args.results_jsonl}")


if __name__ == "__main__":
    main()
