from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import (
    ACTION_COLORS,
    MODEL_LABELS,
    MODEL_PLOT_ORDER,
    apply_publication_style,
    model_order,
    parse_action,
    read_unrelated_csv,
)


def create_reciprocity_comparison(adapter, output_dir):
    """Create 5-condition reciprocity plot (baseline + 4 FT models)."""
    output_dir = Path(output_dir).resolve()
    apply_publication_style()

    rows = []
    models = model_order(adapter.results_dir)
    if not models:
        return output_dir / "reciprocity_comparison_publication.pdf"

    # Baseline from PT2 "before" columns.
    base_df = read_unrelated_csv(adapter.results_dir, models[0])
    if base_df is not None:
        opp = base_df["opp_prev_move (for moral eval only)"].apply(parse_action)
        act = base_df["response (before) - moral"].apply(parse_action)
        for state in ["C", "D"]:
            denom = (opp == state).sum() or 1
            for action in ["C", "D", "illegal"]:
                rows.append(
                    {
                        "model": "No fine-tuning",
                        "pair": f"{action} | {state}",
                        "pct": 100 * ((opp == state) & (act == action)).sum() / denom,
                    }
                )

    for key in models:
        df = read_unrelated_csv(adapter.results_dir, key)
        if df is None:
            continue
        opp = df["opp_prev_move (for moral eval only)"].apply(parse_action)
        act = df["response (after) - moral"].apply(parse_action)
        for state in ["C", "D"]:
            denom = (opp == state).sum() or 1
            for action in ["C", "D", "illegal"]:
                rows.append(
                    {
                        "model": MODEL_LABELS[key],
                        "pair": f"{action} | {state}",
                        "pct": 100 * ((opp == state) & (act == action)).sum() / denom,
                    }
                )

    plot_df = pd.DataFrame(rows)
    order = [m for m in MODEL_PLOT_ORDER if m == "No fine-tuning" or m in [MODEL_LABELS[x] for x in models]]
    pair_order = ["C | C", "D | C", "illegal | C", "C | D", "D | D", "illegal | D"]
    pivot = plot_df.pivot_table(index="model", columns="pair", values="pct", aggfunc="mean").reindex(order)
    pivot = pivot.reindex(columns=pair_order).fillna(0)

    ax = pivot.plot(
        kind="bar",
        stacked=True,
        figsize=(16, 8),
        width=0.75,
        color=[ACTION_COLORS[p] for p in pair_order],
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_ylabel("Percent of responses (conditioned on opponent state)")
    ax.set_xlabel("Model")
    ax.set_title("Action choices on an unrelated prompt containing Action+Game+State")
    ax.legend(title="Action | Opponent", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
    ax.margins(x=0.02)
    plt.tight_layout()
    dst = output_dir / "reciprocity_comparison_publication.pdf"
    plt.savefig(dst, dpi=300)
    plt.close()
    return dst
