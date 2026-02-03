#!/usr/bin/env python3
"""Validation harness for sequence-metric vs sampled-inference alignment."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = next(
    p for p in _THIS_FILE.parents if (p / "mech_interp" / "__init__.py").is_file()
)
sys.path.insert(0, str(PROJECT_ROOT))

from mech_interp.decision_metrics import compute_action_sequence_preference, prepare_prompt
from mech_interp.model_loader import LoRAModelLoader
from mech_interp.prompt_generator import load_prompt_dataset
from mech_interp.utils import MODEL_LABELS


KEY_SEPARATION_PAIRS = [
    ("PT2_COREDe", "PT3_COREDe"),
    ("PT2_COREDe", "PT3_COREUt"),
]


def permutation_p_value(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_perm: int = 20_000,
    seed: int = 0,
) -> float:
    """Two-sided permutation p-value for difference in means."""
    rng = np.random.default_rng(seed)
    observed = float(np.mean(x) - np.mean(y))
    pooled = np.concatenate([x, y])
    n_x = len(x)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        stat = float(np.mean(perm[:n_x]) - np.mean(perm[n_x:]))
        if abs(stat) >= abs(observed):
            count += 1
    return (count + 1) / (n_perm + 1)


def bootstrap_ci_diff(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int = 20_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Bootstrap CI for difference in means (x - y)."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        diffs[i] = float(np.mean(xb) - np.mean(yb))
    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1 - alpha / 2))
    return lo, hi


def pairwise_tests(
    df: pd.DataFrame,
    *,
    group_cols: Iterable[str],
    metric_col: str,
    pairs: Iterable[Tuple[str, str]],
    n_perm: int,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Run pairwise model tests globally or within groups."""
    rows = []
    grouped = df.groupby(list(group_cols)) if group_cols else [((), df)]
    for group_key, g in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        for m1, m2 in pairs:
            x = g[g["model_id"] == m1][metric_col].to_numpy(dtype=np.float64)
            y = g[g["model_id"] == m2][metric_col].to_numpy(dtype=np.float64)
            if len(x) == 0 or len(y) == 0:
                continue
            diff = float(np.mean(x) - np.mean(y))
            p = permutation_p_value(x, y, n_perm=n_perm, seed=seed)
            ci_lo, ci_hi = bootstrap_ci_diff(x, y, n_boot=n_boot, seed=seed)
            row = {
                "metric": metric_col,
                "model_a": m1,
                "model_b": m2,
                "mean_diff_a_minus_b": diff,
                "perm_p_value_two_sided": p,
                "bootstrap_ci_low": ci_lo,
                "bootstrap_ci_high": ci_hi,
                "n_a": len(x),
                "n_b": len(y),
            }
            for idx, col in enumerate(group_cols):
                row[col] = group_key[idx]
            rows.append(row)
    return pd.DataFrame(rows)


def majority_label(values: List[str]) -> str:
    """Return majority label; ties and empty map to 'unknown'."""
    if not values:
        return "unknown"
    counts = pd.Series(values).value_counts()
    if counts.empty:
        return "unknown"
    if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
        return "unknown"
    return str(counts.index[0])


def parse_action_from_text(text: str) -> str:
    """Parse generated text into action1/action2/unknown."""
    t = text.lower()

    match = re.search(r"\baction\s*([12])\b", t)
    if match:
        return "action1" if match.group(1) == "1" else "action2"

    if re.search(r"\bcooperat", t):
        return "action1"
    if re.search(r"\bdefect", t):
        return "action2"

    if re.search(r"\b[12]\b", t):
        return "action1" if re.search(r"\b1\b", t) else "action2"

    return "unknown"


def sample_actions_for_prompt(
    *,
    model,
    tokenizer,
    prompt: str,
    samples_per_prompt: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    use_chat_template: bool,
) -> List[str]:
    """Sample model responses and parse actions."""
    prepared = prepare_prompt(tokenizer, prompt, use_chat_template=use_chat_template)
    encoded = tokenizer(prepared, return_tensors="pt")
    input_ids = encoded.input_ids.to(model.device)
    attention_mask = encoded.attention_mask.to(model.device)

    with torch.no_grad():
        generated = model.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_return_sequences=samples_per_prompt,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = input_ids.shape[1]
    actions: List[str] = []
    for i in range(generated.shape[0]):
        continuation = generated[i, prompt_len:]
        text = tokenizer.decode(continuation, skip_special_tokens=True)
        actions.append(parse_action_from_text(text))
    return actions


def run_validation(args: argparse.Namespace) -> None:
    """Run alignment validation and export CSV summaries."""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompt_dataset(args.prompt_dataset)
    if args.scenarios:
        prompts = [p for p in prompts if p["scenario"] in args.scenarios]

    print(f"Loaded prompts: {len(prompts)}")
    print(f"Models: {args.models}")

    per_prompt_rows = []

    for model_id in args.models:
        print(f"\n=== {MODEL_LABELS.get(model_id, model_id)} ===")
        hooked_model = LoRAModelLoader.load_hooked_model(
            model_id,
            device=args.device,
            merge_lora=False,
            use_4bit=True,
        )
        hooked_model.name = model_id
        tokenizer = hooked_model.tokenizer

        for idx, prompt_data in enumerate(prompts, start=1):
            scenario = prompt_data["scenario"]
            variant = int(prompt_data["variant"])
            prompt = prompt_data["prompt"]

            prepared = prepare_prompt(tokenizer, prompt, use_chat_template=not args.no_chat_template)
            input_ids = tokenizer(prepared, return_tensors="pt").input_ids.to(hooked_model.device)

            seq_pref = compute_action_sequence_preference(
                forward_logits_fn=hooked_model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                action1_label=args.action1_label,
                action2_label=args.action2_label,
            )

            sampled_actions = sample_actions_for_prompt(
                model=hooked_model,
                tokenizer=tokenizer,
                prompt=prompt,
                samples_per_prompt=args.samples_per_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                use_chat_template=not args.no_chat_template,
            )

            known = [a for a in sampled_actions if a in {"action1", "action2"}]
            majority = majority_label(known)
            n_action2 = sum(a == "action2" for a in known)
            n_action1 = sum(a == "action1" for a in known)
            n_unknown = len(sampled_actions) - len(known)
            rate_action2 = float(n_action2 / len(known)) if known else np.nan
            seq_action = seq_pref.preferred_action
            agree = int(seq_action == majority) if majority != "unknown" else 0

            per_prompt_rows.append(
                {
                    "model_id": model_id,
                    "scenario": scenario,
                    "variant": variant,
                    "prompt_id": prompt_data.get("id"),
                    "seq_preferred_action": seq_action,
                    "seq_p_action2": float(seq_pref.p_action2),
                    "seq_delta_logp_action2_minus_action1": float(seq_pref.delta_logp_action2_minus_action1),
                    "sampled_majority_action": majority,
                    "sampled_action2_rate": rate_action2,
                    "n_sampled": len(sampled_actions),
                    "n_known": len(known),
                    "n_unknown": n_unknown,
                    "n_action1": n_action1,
                    "n_action2": n_action2,
                    "agreement_seq_vs_sampled_majority": agree,
                }
            )

            print(
                f"[{idx:02d}/{len(prompts)}] {scenario} v{variant} | "
                f"seq={seq_action} p2={seq_pref.p_action2:.3f} | sampled={majority} "
                f"(a2_rate={0.0 if np.isnan(rate_action2) else rate_action2:.3f}, unknown={n_unknown})"
            )

        del hooked_model
        torch.cuda.empty_cache()

    per_prompt_df = pd.DataFrame(per_prompt_rows).sort_values(["model_id", "scenario", "variant"])
    per_prompt_path = out_dir / "alignment_per_prompt.csv"
    per_prompt_df.to_csv(per_prompt_path, index=False)

    agg_rows = []
    confusion_rows = []
    grouped = per_prompt_df.groupby(["scenario", "model_id"])
    for (scenario, model_id), g in grouped:
        seq_majority = majority_label(g["seq_preferred_action"].tolist())
        sampled_majority = majority_label(g["sampled_majority_action"].tolist())
        valid = g[g["sampled_majority_action"] != "unknown"]
        agreement_rate = (
            float(valid["agreement_seq_vs_sampled_majority"].mean()) if len(valid) > 0 else np.nan
        )

        agg_rows.append(
            {
                "scenario": scenario,
                "model_id": model_id,
                "n_prompts": len(g),
                "seq_majority_action": seq_majority,
                "sampled_majority_action": sampled_majority,
                "agreement_rate": agreement_rate,
                "seq_p_action2_mean": float(g["seq_p_action2"].mean()),
                "sampled_action2_rate_mean": float(g["sampled_action2_rate"].mean()),
                "sampled_unknown_rate_mean": float((g["n_unknown"] / g["n_sampled"]).mean()),
            }
        )

        conf = (
            g.groupby(["seq_preferred_action", "sampled_majority_action"])
            .size()
            .reset_index(name="count")
        )
        for _, r in conf.iterrows():
            confusion_rows.append(
                {
                    "scenario": scenario,
                    "model_id": model_id,
                    "seq_preferred_action": r["seq_preferred_action"],
                    "sampled_majority_action": r["sampled_majority_action"],
                    "count": int(r["count"]),
                }
            )

    agg_df = pd.DataFrame(agg_rows).sort_values(["scenario", "model_id"])
    agg_path = out_dir / "alignment_by_scenario_model.csv"
    agg_df.to_csv(agg_path, index=False)

    confusion_df = pd.DataFrame(confusion_rows).sort_values(
        ["scenario", "model_id", "seq_preferred_action", "sampled_majority_action"]
    )
    confusion_path = out_dir / "alignment_confusion_table.csv"
    confusion_df.to_csv(confusion_path, index=False)

    metrics = [
        "seq_p_action2",
        "seq_delta_logp_action2_minus_action1",
        "sampled_action2_rate",
    ]
    clean_df = per_prompt_df.dropna(subset=["sampled_action2_rate"]).copy()
    global_tests = []
    scenario_tests = []
    for metric in metrics:
        global_tests.append(
            pairwise_tests(
                clean_df,
                group_cols=[],
                metric_col=metric,
                pairs=KEY_SEPARATION_PAIRS,
                n_perm=args.n_perm,
                n_boot=args.n_boot,
                seed=args.seed,
            )
        )
        scenario_tests.append(
            pairwise_tests(
                clean_df,
                group_cols=["scenario"],
                metric_col=metric,
                pairs=KEY_SEPARATION_PAIRS,
                n_perm=args.n_perm,
                n_boot=args.n_boot,
                seed=args.seed,
            )
        )

    global_tests_df = pd.concat(global_tests, ignore_index=True)
    scenario_tests_df = pd.concat(scenario_tests, ignore_index=True)
    global_tests_df.to_csv(out_dir / "significance_global_strategic_vs_de_ut.csv", index=False)
    scenario_tests_df.to_csv(out_dir / "significance_by_scenario_strategic_vs_de_ut.csv", index=False)

    summary_lines = [
        "# Mech-Interp Alignment Validation",
        "",
        f"- Prompt dataset: `{args.prompt_dataset}`",
        f"- Models: `{', '.join(args.models)}`",
        f"- Prompts evaluated: {len(per_prompt_df)} (model x prompt rows)",
        f"- Samples per prompt: {args.samples_per_prompt}",
        "",
        "## Agreement Snapshot",
    ]
    for _, row in agg_df.iterrows():
        summary_lines.append(
            f"- {row['scenario']} | {row['model_id']}: "
            f"seq_majority={row['seq_majority_action']}, "
            f"sampled_majority={row['sampled_majority_action']}, "
            f"agreement_rate={row['agreement_rate']:.3f}" if not np.isnan(row["agreement_rate"])
            else f"- {row['scenario']} | {row['model_id']}: no valid sampled-majority actions"
        )

    (out_dir / "SUMMARY.md").write_text("\n".join(summary_lines))

    print(f"\nSaved: {per_prompt_path}")
    print(f"Saved: {agg_path}")
    print(f"Saved: {confusion_path}")
    print(f"Saved: {out_dir / 'significance_global_strategic_vs_de_ut.csv'}")
    print(f"Saved: {out_dir / 'significance_by_scenario_strategic_vs_de_ut.csv'}")
    print(f"Saved: {out_dir / 'SUMMARY.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate sequence-metric alignment with sampled inference.")
    parser.add_argument(
        "--prompt-dataset",
        type=str,
        default=str(PROJECT_ROOT / "mech_interp_outputs" / "prompt_datasets" / "ipd_eval_prompts.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "mech_interp_outputs" / "validation"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["PT2_COREDe", "PT3_COREDe", "PT3_COREUt", "PT4_COREDe"],
    )
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--samples-per-prompt", type=int, default=9)
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-perm", type=int, default=20_000)
    parser.add_argument("--n-boot", type=int, default=20_000)
    parser.add_argument("--action1-label", type=str, default="action1")
    parser.add_argument("--action2-label", type=str, default="action2")
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable chat-template formatting (debug only).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_validation(args)
