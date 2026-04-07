"""
Train steering vectors from counterfactual prompt pairs using contrastive activation differences.

Designed for **dense** Qwen3 causal LMs (e.g. Qwen/Qwen3-8B, Qwen/Qwen3-4B-Instruct-2507).
MoE variants (e.g. Qwen3-30B-A3B) are not targeted here.

For each JSONL row with `original_prompt` (benign) and `counterfactual_prompt` (unsafe variant),
we run a single forward pass per prompt, collect residual-stream activations at selected decoder
layers, pool over sequence positions, and average (counterfactual − original) across pairs.

Default direction: ``v_l ≈ E[ h(counterfactual) - h(original) ]`` at layer ``l``.
Apply steering as ``h' = h - coeff * v`` to push away from the unsafe direction (sign depends on task).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class CounterfactualRow:
    original_prompt: str
    counterfactual_prompt: str
    safety_category: str | None = None
    changed_aspect: str | None = None
    why_model_should_refuse: str | None = None


def load_counterfactuals(path: Path) -> list[CounterfactualRow]:
    rows: list[CounterfactualRow] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append(
                CounterfactualRow(
                    original_prompt=d["original_prompt"],
                    counterfactual_prompt=d["counterfactual_prompt"],
                    safety_category=d.get("safety_category"),
                    changed_aspect=d.get("changed_aspect"),
                    why_model_should_refuse=d.get("why_model_should_refuse"),
                )
            )
    return rows


def format_for_model(tokenizer, user_text: str) -> dict[str, torch.Tensor]:
    """Apply Qwen3 chat template when available; otherwise raw encode."""
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": user_text}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = user_text
    enc = tokenizer(
        text,
        return_tensors="pt",
        padding=False,
        truncation=True,
    )
    return {k: v for k, v in enc.items()}


def pool_hidden(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor | None,
    how: str,
) -> torch.Tensor:
    """
    hidden: (1, seq, dim)
    Returns (dim,) vector.
    """
    if how == "last":
        if attention_mask is not None:
            idx = attention_mask[0].sum().item() - 1
            idx = max(0, int(idx))
        else:
            idx = hidden.shape[1] - 1
        return hidden[0, idx].float()
    if how == "mean":
        if attention_mask is not None:
            mask = attention_mask[0].float().unsqueeze(-1)
            summed = (hidden[0].float() * mask).sum(0)
            denom = mask.sum().clamp(min=1.0)
            return summed / denom
        return hidden[0].float().mean(0)
    raise ValueError(f"Unknown pooling: {how}")


def parse_layers(spec: str, num_layers: int) -> list[int]:
    if spec.strip().lower() in ("all", "*"):
        return list(range(num_layers))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    for i in out:
        if i < 0 or i >= num_layers:
            raise ValueError(f"Layer index {i} out of range [0, {num_layers})")
    return sorted(set(out))


def collect_steering_vectors(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    rows: list[CounterfactualRow],
    layer_indices: list[int],
    pooling: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Returns dict:
      - per-layer tensors ``steer_L`` of shape (hidden_size,)
      - ``meta`` is stored separately by caller
    """
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    # Accumulate sum of (h_cf - h_orig) per layer
    sums = {i: torch.zeros(hidden_size, device=device) for i in layer_indices}
    count = 0

    model.eval()
    inner = model.model  # Qwen3: stacked decoder layers

    for row in rows:
        acts_orig: dict[int, torch.Tensor] = {}
        acts_cf: dict[int, torch.Tensor] = {}

        def make_hook(storage: dict[int, torch.Tensor], idx: int):
            def hook(_module, _inp, out):
                # Decoder layer returns residual stream (batch, seq, dim)
                if isinstance(out, tuple):
                    h = out[0]
                else:
                    h = out
                storage[idx] = h.detach()

            return hook

        batch_orig = {k: v.to(device) for k, v in format_for_model(tokenizer, row.original_prompt).items()}
        batch_cf = {k: v.to(device) for k, v in format_for_model(tokenizer, row.counterfactual_prompt).items()}

        hooks: list = []
        try:
            for li in layer_indices:
                hooks.append(
                    inner.layers[li].register_forward_hook(
                        make_hook(acts_orig, li)
                    )
                )

            with torch.inference_mode():
                model(**batch_orig)
        finally:
            for h in hooks:
                h.remove()

        hooks = []
        try:
            for li in layer_indices:
                hooks.append(
                    inner.layers[li].register_forward_hook(
                        make_hook(acts_cf, li)
                    )
                )

            with torch.inference_mode():
                model(**batch_cf)
        finally:
            for h in hooks:
                h.remove()

        mask_o = batch_orig.get("attention_mask")
        mask_c = batch_cf.get("attention_mask")
        for li in layer_indices:
            vo = pool_hidden(acts_orig[li], mask_o, pooling)
            vc = pool_hidden(acts_cf[li], mask_c, pooling)
            sums[li] = sums[li] + (vc - vo)
        count += 1

    if count == 0:
        raise ValueError("No counterfactual rows to process.")

    steer = {f"layer_{i}": sums[i] / float(count) for i in layer_indices}
    return steer


def main() -> None:
    p = argparse.ArgumentParser(
        description="Contrastive steering vectors for dense Qwen3 models from counterfactual JSONL."
    )
    p.add_argument(
        "--jsonl",
        type=Path,
        default=Path(__file__).resolve().parent / "counterfactuals.jsonl",
        help="Path to counterfactuals.jsonl (original_prompt / counterfactual_prompt).",
    )
    p.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Hugging Face id for a **dense** Qwen3 Causal LM (not MoE).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "steering_vectors.pt",
        help="Where to save a dict of tensors + metadata.",
    )
    p.add_argument(
        "--layers",
        type=str,
        default="all",
        help='Comma-separated layer indices (0..L-1) or "all".',
    )
    p.add_argument(
        "--pooling",
        choices=("last", "mean"),
        default="last",
        help="How to pool token activations into one vector per prompt.",
    )
    p.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = getattr(torch, args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
    )
    model.to(device)

    rows = load_counterfactuals(args.jsonl)
    layer_indices = parse_layers(args.layers, model.config.num_hidden_layers)

    steer = collect_steering_vectors(
        model=model,
        tokenizer=tokenizer,
        rows=rows,
        layer_indices=layer_indices,
        pooling=args.pooling,
        device=device,
    )

    payload = {
        "steering_vectors": steer,
        "meta": {
            "model_name": args.model,
            "jsonl": str(args.jsonl),
            "num_pairs": len(rows),
            "layers": layer_indices,
            "pooling": args.pooling,
            "description": "steer[layer_k] = mean over pairs of (pool(h_counterfactual) - pool(h_original))",
            "architecture": "dense_qwen3",
        },
    }
    torch.save(payload, args.output)
    print(f"Saved steering vectors for layers {layer_indices} to {args.output}")


if __name__ == "__main__":
    main()
