"""
Apply steering vectors from ``steering_train.py`` during dense Qwen3 inference.

Vectors are ``E[h_counterfactual - h_original]`` per layer (unsafe-minus-benign direction).
Default intervention: ``h <- h - coeff * v`` (``--direction subtract``), which shifts
activations opposite to that contrastive direction (typical for refusal / safety steering).

Uses forward hooks on decoder layer outputs so steering runs on every forward during generation.
"""
from __future__ import annotations

import argparse
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Generator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_steering_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if "steering_vectors" not in payload:
        raise KeyError(f"Expected 'steering_vectors' in {path}")
    return payload["steering_vectors"], payload.get("meta", {})


def parse_layer_key(key: str) -> int:
    m = re.match(r"layer_(\d+)$", key)
    if not m:
        raise ValueError(f"Unexpected steering key (expected layer_<int>): {key}")
    return int(m.group(1))


def _steering_hook(
    vector: torch.Tensor,
    coeff: float,
    subtract: bool,
) -> Callable:
    """Hook modifies decoder layer output in-place-safe way by returning new tensor."""

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = None

        sign = -1.0 if subtract else 1.0
        v = vector.to(device=h.device, dtype=h.dtype)
        delta = (sign * coeff) * v.view(1, 1, -1)
        h_new = h + delta

        if isinstance(output, tuple):
            return (h_new,) + rest
        return h_new

    return hook


@contextmanager
def apply_steering(
    model: AutoModelForCausalLM,
    steering_vectors: dict[str, torch.Tensor],
    coeff: float,
    subtract: bool = True,
    normalize: bool = False,
) -> Generator[None, None, None]:
    """
    Register forward hooks on ``model.model.layers[i]`` for each ``layer_i`` key in
    ``steering_vectors``. Restores the model on exit.
    """
    inner = model.model
    handles: list = []

    for key, vec in steering_vectors.items():
        layer_idx = parse_layer_key(key)
        v = vec.clone().float()
        if normalize:
            v = v / (v.norm() + 1e-8)
        h = inner.layers[layer_idx].register_forward_hook(
            _steering_hook(v, coeff=coeff, subtract=subtract)
        )
        handles.append(h)

    try:
        yield
    finally:
        for h in handles:
            h.remove()


def format_prompt(tokenizer: AutoTokenizer, user_text: str) -> dict[str, torch.Tensor]:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": user_text}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = user_text
    return tokenizer(text, return_tensors="pt", padding=False, truncation=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Qwen3 dense LM inference with activation steering.")
    p.add_argument(
        "--steering",
        type=Path,
        default=Path(__file__).resolve().parent / "steering_vectors.pt",
        help="Checkpoint from steering_train.py (steering_vectors + meta).",
    )
    p.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Override base model id (default: Qwen/Qwen3-4B-Instruct-2507).",
    )
    p.add_argument("--prompt", type=str, default=None, help="User message; if omitted, read stdin.")
    p.add_argument(
        "--coeff",
        type=float,
        default=1.0,
        help="Strength of steering (scaled steering vector).",
    )
    p.add_argument(
        "--direction",
        choices=("subtract", "add"),
        default="subtract",
        help="subtract => h + (-coeff)*v (default; away from contrastive unsafe direction).",
    )
    p.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize each layer vector before applying.",
    )
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also print an unsteered generation for comparison.",
    )
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = getattr(torch, args.dtype)

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

    prompt_text = args.prompt
    if prompt_text is None:
        raise SystemExit("Empty prompt.")

    batch = {k: v.to(device) for k, v in format_prompt(tokenizer, prompt_text).items()}
    subtract = args.direction == "subtract"

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    if args.compare_baseline:
        with torch.inference_mode():
            out_base = model.generate(**batch, **gen_kwargs)
        text_base = tokenizer.decode(out_base[0], skip_special_tokens=True)
        print("--- baseline (no steering) ---")
        print(text_base)
        print()

    with apply_steering(
        model,
        steering_vectors,
        coeff=args.coeff,
        subtract=subtract,
        normalize=args.normalize,
    ):
        with torch.inference_mode():
            out = model.generate(**batch, **gen_kwargs)

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print("--- steered ---")
    print(text)


if __name__ == "__main__":
    main()
