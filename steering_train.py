"""
Train steering vectors from counterfactual prompt pairs using contrastive activation differences.

Designed for **dense** Qwen3 causal LMs (e.g. Qwen/Qwen3-8B, Qwen/Qwen3-4B-Instruct-2507).
MoE variants (e.g. Qwen3-30B-A3B) are not targeted here.

For each JSONL row with ``axis`` and ``pairs`` (two dicts sharing the same ``prompt``, differing in
``response`` / ``axis_score``), we treat the higher-scoring response as ``original`` and the lower
as ``counterfactual``, run one forward per side on the full user+assistant chat, pool activations,
and average (counterfactual − original) across rows.

Default direction: ``v_l ≈ E[ h(counterfactual) - h(original) ]`` at layer ``l``.
Apply steering as ``h' = h - coeff * v`` to push away from the unsafe direction (sign depends on task).

By default, **LayerNavigator**-style scores (``S = D + C``) rank layers; when ``--layers`` is ``all``,
only the top ``--top-k`` layers receive extracted steering vectors. Pass ``--no-layer-navigator`` to
skip ranking and use every layer, or set an explicit ``--layers`` list (ranking is still logged to meta).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class AxisPairItem:
    prompt: str
    response: str
    axis_score: float


@dataclass
class CounterfactualRow:
    axis: str
    high: AxisPairItem
    low: AxisPairItem


def load_counterfactuals(path: Path) -> list[CounterfactualRow]:
    rows: list[CounterfactualRow] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            axis = d["axis"]
            raw_pairs = d["pairs"]
            items = [
                AxisPairItem(
                    prompt=p["prompt"],
                    response=p["response"],
                    axis_score=float(p["axis_score"]),
                )
                for p in raw_pairs
            ]
            items_sorted = sorted(items, key=lambda x: x.axis_score, reverse=True)
            high, low = items_sorted[0], items_sorted[-1]
            rows.append(CounterfactualRow(axis=axis, high=high, low=low))
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


def format_user_assistant_for_model(tokenizer, prompt: str, response: str) -> dict[str, torch.Tensor]:
    """Encode a single user turn and assistant reply (no extra generation prompt)."""
    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        text = f"{prompt}\n\n{response}"
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


def layers_spec_is_all(spec: str) -> bool:
    return spec.strip().lower() in ("all", "*")


# --- LayerNavigator-style ranking (Sun et al., NeurIPS 2025; see layer_navigator.py) ---


def layer_scores_from_pos_neg(
    pos: torch.Tensor,
    neg: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[float, float, float]:
    """
    pos, neg: (N, d) pooled activations for + / - sides at one layer.

    Returns (D_l, C_l, S_l) as Python floats.
    """
    n = pos.shape[0]
    if n < 1:
        return 0.0, 0.0, 0.0

    both = torch.cat([pos, neg], dim=0)  # (2N, d)
    mu = both.mean(dim=0)
    sig = both.std(dim=0).clamp_min(eps)
    z_pos = (pos - mu) / sig
    z_neg = (neg - mu) / sig

    mu_pos = z_pos.mean(dim=0)
    mu_neg = z_neg.mean(dim=0)
    v = (z_pos - z_neg).mean(dim=0)

    vb = n * ((v @ mu_pos) ** 2 + (v @ mu_neg) ** 2)
    centered_pos = z_pos - mu_pos
    centered_neg = z_neg - mu_neg
    vw = (v * centered_pos).sum(dim=1).pow(2).sum() + (v * centered_neg).sum(dim=1).pow(2).sum()
    denom = vb + vw
    d_l = (vb / denom.clamp_min(eps)).item() if denom.item() > eps else 0.0

    diff = z_pos - z_neg
    v_norm = v.norm().clamp_min(eps)
    diff_norm = diff.norm(dim=1).clamp_min(eps)
    cosines = (diff * v).sum(dim=1) / (diff_norm * v_norm)
    c_l = cosines.mean().item()

    return d_l, c_l, d_l + c_l


def collect_pos_neg_activations_all_layers(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    rows: list[CounterfactualRow],
    pooling: str,
    device: torch.device,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Two forwards per row; pooled residual at every decoder layer (+ = high score, − = low)."""
    num_layers = model.config.num_hidden_layers
    inner = model.model

    pos_lists: dict[int, list[torch.Tensor]] = {l: [] for l in range(num_layers)}
    neg_lists: dict[int, list[torch.Tensor]] = {l: [] for l in range(num_layers)}

    model.eval()

    def run_one(pair: AxisPairItem) -> dict[int, torch.Tensor]:
        acts: dict[int, torch.Tensor] = {}

        def make_hook(li: int):
            def hook(_module, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                acts[li] = h.detach()

            return hook

        batch = {
            k: v.to(device)
            for k, v in format_user_assistant_for_model(
                tokenizer, pair.prompt, pair.response
            ).items()
        }
        
        hooks: list = []
        try:
            for li in range(num_layers):
                hooks.append(inner.layers[li].register_forward_hook(make_hook(li)))
            with torch.inference_mode():
                model(**batch)
        finally:
            for h in hooks:
                h.remove()

        mask = batch.get("attention_mask")
        return {li: pool_hidden(acts[li], mask, pooling).float() for li in range(num_layers)}

    for row in rows:
        high_vecs = run_one(row.high)
        low_vecs = run_one(row.low)
        for li in range(num_layers):
            pos_lists[li].append(high_vecs[li])
            neg_lists[li].append(low_vecs[li])

    pos_stacked = {l: torch.stack(pos_lists[l], dim=0) for l in range(num_layers)}
    neg_stacked = {l: torch.stack(neg_lists[l], dim=0) for l in range(num_layers)}
    return pos_stacked, neg_stacked


def rank_layers_by_steerability(
    pos_by_layer: dict[int, torch.Tensor],
    neg_by_layer: dict[int, torch.Tensor],
) -> list[dict]:
    """Per-layer D, C, S; sorted by S descending."""
    rows_out: list[dict] = []
    for l in sorted(pos_by_layer.keys()):
        d_l, c_l, s_l = layer_scores_from_pos_neg(pos_by_layer[l], neg_by_layer[l])
        rows_out.append({"layer": l, "D": d_l, "C": c_l, "S": s_l})
    rows_out.sort(key=lambda r: r["S"], reverse=True)
    return rows_out


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

        batch_orig = {
            k: v.to(device)
            for k, v in format_user_assistant_for_model(
                tokenizer, row.high.prompt, row.high.response
            ).items()
        }
        batch_cf = {
            k: v.to(device)
            for k, v in format_user_assistant_for_model(
                tokenizer, row.low.prompt, row.low.response
            ).items()
        }

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
        required=True,
        help="Path to counterfactuals JSON file (axis + pairs with prompt/response/axis_score).",
   
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
        help='Comma-separated layer indices (0..L-1) or "all". When --layer-navigator is on and this is "all", only the top --top-k layers by steerability S=D+C are used.',
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="With --layer-navigator and --layers all: number of highest-S layers to extract steering vectors for.",
    )
    p.add_argument(
        "--layer-navigator",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rank layers via simplified LayerNavigator (S=D+C). Default on; use --no-layer-navigator to disable.",
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
    num_layers = model.config.num_hidden_layers
    layer_indices = parse_layers(args.layers, num_layers)

    layer_nav_ranking: list[dict] | None = None
    if args.layer_navigator:
        pos_by_l, neg_by_l = collect_pos_neg_activations_all_layers(
            model, tokenizer, rows, args.pooling, device
        )
        layer_nav_ranking = rank_layers_by_steerability(pos_by_l, neg_by_l)
        if layers_spec_is_all(args.layers):
            layer_indices = sorted(
                r["layer"] for r in layer_nav_ranking[: args.top_k]
            )
            print(
                f"LayerNavigator (--layer-navigator): using top-{args.top_k} layers by S=D+C: {layer_indices}"
            )
        else:
            print(
                "LayerNavigator (--layer-navigator): ranking saved to meta; "
                f"--layers is explicit, so extracting {layer_indices}"
            )

    steer = collect_steering_vectors(
        model=model,
        tokenizer=tokenizer,
        rows=rows,
        layer_indices=layer_indices,
        pooling=args.pooling,
        device=device,
    )

    meta: dict = {
        "model_name": args.model,
        "jsonl": str(args.jsonl),
        "num_pairs": len(rows),
        "layers": layer_indices,
        "pooling": args.pooling,
        "description": "steer[layer_k] = mean over rows of (pool(h_low_score) - pool(h_high_score)) on user+assistant chats",
        "architecture": "dense_qwen3",
        "layer_navigator_enabled": args.layer_navigator,
        "top_k": args.top_k,
    }
    if layer_nav_ranking is not None:
        meta["layer_navigator"] = {
            "paper": "LayerNavigator (NeurIPS 2025), simplified S=D+C",
            "openreview": "https://openreview.net/forum?id=wj4lM45xQR",
            "layers_ranked": layer_nav_ranking,
            "layers_used": layer_indices,
        }

    payload = {"steering_vectors": steer, "meta": meta}
    torch.save(payload, args.output)
    print(f"Saved steering vectors for layers {layer_indices} to {args.output}")


if __name__ == "__main__":
    main()
