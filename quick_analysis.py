#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
import textwrap
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ACTION_C = "action1"
ACTION_D = "action2"
GAME_FILES = ["IPD", "ISH", "ICN", "ICD", "BOS"]


def parse_action(text):
    if not isinstance(text, str):
        return "illegal"
    s = text.lower()
    matches = [(m.start(), m.group(0)) for m in re.finditer(r"action1|action2", s)]
    if not matches:
        return "illegal"
    matches.sort(key=lambda x: x[0])
    token = matches[0][1]
    return "C" if token == ACTION_C else "D"


def load_game_csvs(results_dir, model_dir):
    run_dir = Path(results_dir) / model_dir / "run1"
    data = {}
    for game in GAME_FILES:
        pattern = f"*independent eval {game}.csv"
        files = list(run_dir.glob(pattern))
        if not files:
            continue
        data[game] = pd.read_csv(files[0])
    return data


def load_unrelated_csv(results_dir, model_dir):
    run_dir = Path(results_dir) / model_dir / "run1"
    files = list(run_dir.glob("*independent eval 4 unrelated queries.csv"))
    if not files:
        return None
    return pd.read_csv(files[0])


def load_unstructured_csv(results_dir, model_dir):
    run_dir = Path(results_dir) / model_dir / "run1"
    files = list(run_dir.glob("*independent eval 2 unstructured IPD queries.csv"))
    if not files:
        return None
    return pd.read_csv(files[0])


def compute_metrics_for_game(df):
    before = df["response (before)"].apply(parse_action)
    after = df["response (after)"].apply(parse_action)

    metrics = {
        "coop_rate_before": (before == "C").mean(),
        "coop_rate_after": (after == "C").mean(),
        "illegal_rate_before": (before == "illegal").mean(),
        "illegal_rate_after": (after == "illegal").mean(),
    }

    for reward in ["Game", "De", "Ut", "GameDe"]:
        before_col = f"rewards_{reward} (before)"
        after_col = f"rewards_{reward} (after)"
        if before_col in df.columns:
            metrics[f"reward_{reward}_before"] = df[before_col].mean()
        if after_col in df.columns:
            metrics[f"reward_{reward}_after"] = df[after_col].mean()

    if "opponent_move" in df.columns:
        opp = df["opponent_move"].apply(parse_action)
        pairs = pd.DataFrame({"model": after, "opp": opp})
        pairs = pairs[pairs["model"].isin(["C", "D"]) & pairs["opp"].isin(["C", "D"])]
        if not pairs.empty:
            counts = pairs.value_counts(normalize=True).rename("freq").reset_index()
            metrics["reciprocity"] = counts
    return metrics


def summarize(results_dir, models):
    rows = []
    reciprocity = []
    for model_dir in models:
        game_data = load_game_csvs(results_dir, model_dir)
        for game, df in game_data.items():
            metrics = compute_metrics_for_game(df)
            row = {
                "model": model_dir,
                "game": game,
                **{k: v for k, v in metrics.items() if k != "reciprocity"},
            }
            rows.append(row)
            if "reciprocity" in metrics:
                rec = metrics["reciprocity"].copy()
                rec["model"] = model_dir
                rec["game"] = game
                reciprocity.append(rec)
    summary_df = pd.DataFrame(rows)
    reciprocity_df = pd.concat(reciprocity, ignore_index=True) if reciprocity else pd.DataFrame()
    return summary_df, reciprocity_df


def plot_cooperation(summary_df, output_dir):
    plt.figure(figsize=(10, 5))
    sns.barplot(data=summary_df, x="game", y="coop_rate_after", hue="model")
    plt.title("Cooperation Rate After Fine-Tuning")
    plt.ylabel("Cooperation rate")
    plt.xlabel("Game")
    plt.tight_layout()
    path = Path(output_dir) / "cooperation_by_game.png"
    plt.savefig(path, dpi=200)
    plt.close()


def plot_rewards(summary_df, output_dir):
    reward_cols = [c for c in summary_df.columns if c.endswith("_after") and c.startswith("reward_")]
    if not reward_cols:
        return
    melted = summary_df.melt(id_vars=["model", "game"], value_vars=reward_cols, var_name="reward", value_name="value")
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=melted, x="reward", y="value", hue="model")
    plt.title("Reward Distributions After Fine-Tuning")
    plt.xticks(rotation=20)
    plt.tight_layout()
    path = Path(output_dir) / "rewards_comparison.png"
    plt.savefig(path, dpi=200)
    plt.close()


def plot_model_comparison(summary_df, output_dir):
    agg = summary_df.groupby("model").agg({
        "coop_rate_after": "mean",
        "illegal_rate_after": "mean",
        "reward_Game_after": "mean" if "reward_Game_after" in summary_df.columns else "mean",
    }).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=agg, x="model", y="coop_rate_after", ax=ax)
    ax.set_title("Average Cooperation Rate (After)")
    ax.set_ylabel("Cooperation rate")
    ax.set_xlabel("Model")
    plt.tight_layout()
    path = Path(output_dir) / "model_comparison.png"
    plt.savefig(path, dpi=200)
    plt.close()


def plot_reciprocity(reciprocity_df, output_dir):
    if reciprocity_df.empty:
        return
    reciprocity_df = reciprocity_df.copy()
    reciprocity_df["pair"] = reciprocity_df["model"] + "|" + reciprocity_df["opp"]
    for (model, game), group in reciprocity_df.groupby(["model", "game"]):
        pivot = group.pivot_table(index="model", columns="opp", values="freq", aggfunc="sum")
        if pivot.empty:
            continue
    # Aggregate across games for a compact heatmap
    agg = reciprocity_df.groupby(["model", "pair"]).freq.mean().reset_index()
    pivot = agg.pivot(index="model", columns="pair", values="freq")
    plt.figure(figsize=(10, 4))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Reciprocity Patterns (Model vs Opponent Move)")
    plt.ylabel("Model")
    plt.xlabel("Model action | Opponent action")
    plt.tight_layout()
    path = Path(output_dir) / "reciprocity_heatmap.png"
    plt.savefig(path, dpi=200)
    plt.close()


def write_report(summary_df, output_dir, results_dir, models):
    lines = []
    lines.append("# Summary Report")
    lines.append("")
    lines.append(f"Results dir: `{results_dir}`")
    lines.append(f"Models: {', '.join(models)}")
    lines.append("")
    if summary_df.empty:
        lines.append("No game CSVs found.")
        Path(output_dir, "summary_report.md").write_text("\n".join(lines))
        return

    overall = summary_df.groupby("model").agg({
        "coop_rate_after": "mean",
        "illegal_rate_after": "mean",
        "reward_Game_after": "mean" if "reward_Game_after" in summary_df.columns else "mean",
    }).reset_index()
    lines.append("## Overall (averaged across games)")
    lines.append("")
    lines.append(overall.to_markdown(index=False))
    lines.append("")

    lines.append("## By Game")
    lines.append("")
    lines.append(summary_df[["model", "game", "coop_rate_after", "illegal_rate_after"]].to_markdown(index=False))
    lines.append("")

    Path(output_dir, "summary_report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--output-dir", default="./analysis_output")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [p.name for p in results_dir.iterdir() if p.is_dir()]
    models.sort()

    summary_df, reciprocity_df = summarize(results_dir, models)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)

    plot_cooperation(summary_df, output_dir)
    plot_rewards(summary_df, output_dir)
    plot_model_comparison(summary_df, output_dir)
    plot_reciprocity(reciprocity_df, output_dir)

    write_report(summary_df, output_dir, results_dir, models)

    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
