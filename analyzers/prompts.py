from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .common import (
    MODEL_LABELS,
    MODEL_PLOT_ORDER,
    apply_publication_style,
    model_order,
    parse_action,
    read_unstructured_csv,
)


def analyze_prompt_robustness(adapter, output_dir):
    """Prompt robustness for baseline + 4 FT models only."""
    output_dir = Path(output_dir).resolve()
    apply_publication_style()

    rows = []
    models = model_order(adapter.results_dir)
    if not models:
        return output_dir / "prompt_robustness_publication.pdf"

    prompts = ["structured_IPD", "unstructured_IPD", "poetic_IPD", "explicit_IPD"]

    # Baseline from before columns in first model's file.
    base_df = read_unstructured_csv(adapter.results_dir, models[0])
    if base_df is not None:
        for prompt_name in prompts:
            col = f"response (before) - {prompt_name}"
            if col not in base_df.columns:
                continue
            vals = base_df[col].apply(parse_action)
            for action in ["C", "D", "illegal"]:
                rows.append(
                    {
                        "model": "No fine-tuning",
                        "prompt": prompt_name,
                        "action": action,
                        "pct": 100 * (vals == action).mean(),
                    }
                )

    for model in models:
        df = read_unstructured_csv(adapter.results_dir, model)
        if df is None:
            continue
        for prompt_name in prompts:
            col = f"response (after) - {prompt_name}"
            if col not in df.columns:
                continue
            vals = df[col].apply(parse_action)
            for action in ["C", "D", "illegal"]:
                rows.append(
                    {
                        "model": MODEL_LABELS[model],
                        "prompt": prompt_name,
                        "action": action,
                        "pct": 100 * (vals == action).mean(),
                    }
                )

    plot_df = pd.DataFrame(rows)
    order = [m for m in MODEL_PLOT_ORDER if m == "No fine-tuning" or m in [MODEL_LABELS[x] for x in models]]
    if plot_df.empty:
        return output_dir / "prompt_robustness_publication.pdf"

    prompt_titles = {
        "structured_IPD": "Structured IPD",
        "unstructured_IPD": "Unstructured IPD",
        "poetic_IPD": "Poetic IPD",
        "explicit_IPD": "Explicit IPD",
    }
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), sharey=True)
    for idx, prompt in enumerate(prompt_titles):
        ax = axes[idx // 2][idx % 2]
        sdf = plot_df[plot_df["prompt"] == prompt]
        pivot = sdf.pivot_table(index="model", columns="action", values="pct", aggfunc="mean").reindex(order).fillna(0)
        pivot = pivot.reindex(columns=["C", "D", "illegal"]).fillna(0)
        pivot.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            legend=False,
            color=["#28641E", "#8E0B52", "#9A9A9A"],
            edgecolor="white",
            linewidth=0.5,
            width=0.78,
        )
        ax.set_title(prompt_titles[prompt])
        ax.set_xlabel("")
        ax.set_ylabel("Percent")
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", rotation=22)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", title="Action", frameon=True)
    fig.suptitle("Action choices on four types of IPD prompt", y=0.99, fontsize=22)
    plt.tight_layout(rect=[0, 0, 0.96, 0.97])
    dst = output_dir / "prompt_robustness_publication.pdf"
    plt.savefig(dst, dpi=300)
    plt.close()
    return dst
