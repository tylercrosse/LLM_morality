from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .common import (
    GAMES,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_PLOT_ORDER,
    apply_publication_style,
    model_order,
    read_game_csv,
)


def compute_moral_regret(adapter, output_dir, label="Deontological"):
    """Compute regret with only baseline + 4 real FT models."""
    output_dir = Path(output_dir).resolve()
    apply_publication_style()

    if label not in {"Deontological", "Utilitarian"}:
        raise ValueError("label must be Deontological or Utilitarian")

    reward_col = "rewards_De" if label == "Deontological" else "rewards_Ut"
    moral_max = {"IPD": 0, "ISH": 0, "ICN": 0, "ICD": 0, "BOS": 0}
    moral_min = {"IPD": -6, "ISH": -6, "ICN": -6, "ICD": -6, "BOS": -6}
    if label == "Utilitarian":
        moral_max = {"IPD": 6, "ISH": 8, "ICN": 5, "ICD": 8, "BOS": 5}

    rows = []
    models = model_order(adapter.results_dir)
    if not models:
        return output_dir / f"moral_regret_{label.lower()}_publication.pdf"

    baseline_source = models[0]
    for game in GAMES:
        df = read_game_csv(adapter.results_dir, baseline_source, game)
        if df is None:
            continue
        vals = moral_max[game] - df[f"{reward_col} (before)"]
        if label == "Utilitarian":
            vals = vals / (moral_max[game] - moral_min[game])
        rows.append({"model": "No fine-tuning", "game": game, "regret": vals.mean()})

    for model in models:
        for game in GAMES:
            df = read_game_csv(adapter.results_dir, model, game)
            if df is None:
                continue
            vals = moral_max[game] - df[f"{reward_col} (after)"]
            if label == "Utilitarian":
                vals = vals / (moral_max[game] - moral_min[game])
            rows.append({"model": MODEL_LABELS[model], "game": game, "regret": vals.mean()})

    plot_df = pd.DataFrame(rows)
    order = [m for m in MODEL_PLOT_ORDER if m == "No fine-tuning" or m in [MODEL_LABELS[x] for x in models]]
    palette = [MODEL_COLORS[m] for m in order]
    plt.figure(figsize=(15, 8))
    sns.barplot(
        data=plot_df,
        x="game",
        y="regret",
        hue="model",
        hue_order=order,
        palette=palette,
        edgecolor="white",
        linewidth=0.6,
    )
    plt.title(f"{label} Moral Regret by Game")
    plt.xlabel("Game")
    plt.ylabel("Mean regret")
    plt.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    pdf_dst = output_dir / f"moral_regret_{label.lower()}_publication.pdf"
    plt.savefig(pdf_dst, dpi=300)
    plt.close()
    return pdf_dst
