#!/usr/bin/env python3
import argparse
from pathlib import Path
import re

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


def build_summary(results_dir):
    models = [p.name for p in Path(results_dir).iterdir() if p.is_dir()]
    models.sort()
    rows = []
    for model in models:
        game_data = load_game_csvs(results_dir, model)
        for game, df in game_data.items():
            before = df["response (before)"].apply(parse_action)
            after = df["response (after)"].apply(parse_action)
            row = {
                "model": model,
                "game": game,
                "coop_rate_before": (before == "C").mean(),
                "coop_rate_after": (after == "C").mean(),
                "illegal_rate_before": (before == "illegal").mean(),
                "illegal_rate_after": (after == "illegal").mean(),
            }
            for reward in ["Game", "De", "Ut", "GameDe"]:
                bc = f"rewards_{reward} (before)"
                ac = f"rewards_{reward} (after)"
                if bc in df.columns:
                    row[f"reward_{reward}_before"] = df[bc].mean()
                if ac in df.columns:
                    row[f"reward_{reward}_after"] = df[ac].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def plot_cooperation(summary_df, output_dir):
    plt.figure(figsize=(10, 5))
    sns.barplot(data=summary_df, x="game", y="coop_rate_after", hue="model")
    plt.title("Cooperation Rate After Fine-Tuning")
    plt.ylabel("Cooperation rate")
    plt.xlabel("Game")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "cooperation_by_model_game.png", dpi=200)
    plt.close()


def plot_reward_box(summary_df, output_dir):
    reward_cols = [c for c in summary_df.columns if c.startswith("reward_") and c.endswith("_after")]
    if not reward_cols:
        return
    melted = summary_df.melt(id_vars=["model", "game"], value_vars=reward_cols, var_name="reward", value_name="value")
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=melted, x="reward", y="value", hue="model")
    plt.title("Reward Distributions After Fine-Tuning")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "reward_distributions.png", dpi=200)
    plt.close()


def plot_before_after(summary_df, output_dir):
    if "reward_Game_before" not in summary_df.columns or "reward_Game_after" not in summary_df.columns:
        return
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=summary_df, x="reward_Game_before", y="reward_Game_after", hue="model", style="game")
    plt.plot([summary_df["reward_Game_before"].min(), summary_df["reward_Game_before"].max()],
             [summary_df["reward_Game_before"].min(), summary_df["reward_Game_before"].max()],
             color="gray", linestyle="--")
    plt.title("Before vs After Game Reward")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "before_after_comparison.png", dpi=200)
    plt.close()


def plot_generalization(summary_df, output_dir):
    ipd = summary_df[summary_df["game"] == "IPD"]
    other = summary_df[summary_df["game"] != "IPD"]
    agg_other = other.groupby("model").coop_rate_after.mean().reset_index()
    agg_ipd = ipd[["model", "coop_rate_after"]].rename(columns={"coop_rate_after": "ipd"})
    merged = agg_other.merge(agg_ipd, on="model", how="left")
    plt.figure(figsize=(8, 4))
    x = np.arange(len(merged))
    width = 0.35
    plt.bar(x - width/2, merged["ipd"], width, label="IPD")
    plt.bar(x + width/2, merged["coop_rate_after"], width, label="Other games")
    plt.xticks(x, merged["model"], rotation=20)
    plt.ylabel("Cooperation rate")
    plt.title("Generalization: IPD vs Other Games")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "generalization_plot.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--output-dir", default="./figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = build_summary(args.results_dir)
    if summary_df.empty:
        print("No game CSVs found.")
        return

    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)

    plot_cooperation(summary_df, output_dir)
    plot_reward_box(summary_df, output_dir)
    plot_before_after(summary_df, output_dir)
    plot_generalization(summary_df, output_dir)

    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
