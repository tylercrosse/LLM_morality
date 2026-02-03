#!/usr/bin/env python3
"""Analyze prompt sensitivity and model-separation significance from logit-lens outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


KEY_MODEL_PAIRS = [
    ("PT2_COREDe", "PT3_COREDe"),
    ("PT2_COREDe", "PT3_COREUt"),
    ("PT2_COREDe", "PT4_COREDe"),
    ("PT3_COREDe", "PT3_COREUt"),
    ("PT3_COREUt", "PT4_COREDe"),
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
            x = g[g["model_id"] == m1][metric_col].to_numpy()
            y = g[g["model_id"] == m2][metric_col].to_numpy()
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        default="mech_interp_outputs/logit_lens/decision_statistics_by_variant.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="mech_interp_outputs/logit_lens/prompt_sensitivity",
    )
    parser.add_argument("--n-perm", type=int, default=20000)
    parser.add_argument("--n-boot", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    in_path = Path(args.input_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # 1) Prompt sensitivity summaries
    by_scenario_model = (
        df.groupby(["scenario", "model_id"])
        .agg(
            n_variants=("variant", "count"),
            seq_p_action2_mean=("seq_p_action2", "mean"),
            seq_p_action2_std=("seq_p_action2", "std"),
            seq_p_action2_min=("seq_p_action2", "min"),
            seq_p_action2_max=("seq_p_action2", "max"),
            seq_delta_mean=("seq_delta_logp_action2_minus_action1", "mean"),
            seq_delta_std=("seq_delta_logp_action2_minus_action1", "std"),
            final_delta_mean=("final_delta", "mean"),
            final_delta_std=("final_delta", "std"),
            n_unique_actions=("seq_preferred_action", "nunique"),
        )
        .reset_index()
    )
    by_scenario_model["variant_flip"] = by_scenario_model["n_unique_actions"] > 1
    by_scenario_model.to_csv(out_dir / "prompt_sensitivity_by_scenario_model.csv", index=False)

    by_model = (
        df.groupby("model_id")
        .agg(
            n_points=("seq_p_action2", "count"),
            seq_p_action2_mean=("seq_p_action2", "mean"),
            seq_p_action2_std=("seq_p_action2", "std"),
            seq_delta_mean=("seq_delta_logp_action2_minus_action1", "mean"),
            seq_delta_std=("seq_delta_logp_action2_minus_action1", "std"),
            final_delta_mean=("final_delta", "mean"),
            final_delta_std=("final_delta", "std"),
        )
        .reset_index()
    )
    by_model.to_csv(out_dir / "prompt_sensitivity_by_model.csv", index=False)

    # 2) Significance tests: global + per-scenario
    metrics = ["seq_p_action2", "seq_delta_logp_action2_minus_action1", "final_delta"]
    global_rows = []
    scenario_rows = []
    for metric in metrics:
        global_rows.append(
            pairwise_tests(
                df,
                group_cols=[],
                metric_col=metric,
                pairs=KEY_MODEL_PAIRS,
                n_perm=args.n_perm,
                n_boot=args.n_boot,
                seed=args.seed,
            )
        )
        scenario_rows.append(
            pairwise_tests(
                df,
                group_cols=["scenario"],
                metric_col=metric,
                pairs=KEY_MODEL_PAIRS,
                n_perm=args.n_perm,
                n_boot=args.n_boot,
                seed=args.seed,
            )
        )
    global_tests = pd.concat(global_rows, ignore_index=True)
    scenario_tests = pd.concat(scenario_rows, ignore_index=True)
    global_tests.to_csv(out_dir / "significance_global.csv", index=False)
    scenario_tests.to_csv(out_dir / "significance_by_scenario.csv", index=False)

    # 3) Lightweight markdown summary for writeups
    lines = []
    lines.append("# Prompt Sensitivity Validation")
    lines.append("")
    lines.append(f"- Input: `{in_path}`")
    lines.append(f"- Rows: {len(df)}")
    lines.append("")
    lines.append("## Prompt-Variant Stability")
    flips = int(by_scenario_model["variant_flip"].sum())
    total_cells = len(by_scenario_model)
    lines.append(f"- Scenario-model cells with action flips across variants: **{flips}/{total_cells}**")
    lines.append("")
    lines.append("## Model-Level Sensitivity (seq_p_action2 std)")
    for _, r in by_model.sort_values("model_id").iterrows():
        lines.append(f"- {r['model_id']}: {r['seq_p_action2_std']:.4f}")
    lines.append("")
    lines.append("## Key Global Pairwise Tests (seq_p_action2)")
    seq_global = global_tests[global_tests["metric"] == "seq_p_action2"].sort_values(
        "perm_p_value_two_sided"
    )
    for _, r in seq_global.iterrows():
        lines.append(
            f"- {r['model_a']} - {r['model_b']}: "
            f"diff={r['mean_diff_a_minus_b']:.4f}, p={r['perm_p_value_two_sided']:.6f}, "
            f"CI=[{r['bootstrap_ci_low']:.4f}, {r['bootstrap_ci_high']:.4f}]"
        )

    (out_dir / "SUMMARY.md").write_text("\n".join(lines))
    print(f"Saved prompt sensitivity analysis to: {out_dir}")


if __name__ == "__main__":
    main()

