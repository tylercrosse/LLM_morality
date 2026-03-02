# Advanced Analysis Guide

This guide covers the advanced analyses for your real checkpoints in `./results/`.

## What this adds

- State-conditioned reciprocity (C|C, C|D, D|C, D|D)
- Moral regret (Deontological and Utilitarian)
- Cross-game generalization plots
- Prompt robustness plots (structured/unstructured/poetic/explicit IPD)
- Publication-style figure sizing and font scaling
- Uses only 5 real conditions: Baseline + PT2 + PT3-De + PT3-Ut + PT4

## Run the advanced CLI

```
python advanced_analysis.py \
  --results-dir ./results \
  --output-dir ./publication_figures \
  --analyses reciprocity regret generalization prompts
```

Outputs (in `./publication_figures/`):
- `reciprocity_comparison_publication.pdf`
- `moral_regret_deontological_publication.pdf`
- `moral_regret_utilitarian_publication.pdf`
- `cross_game_generalization_publication.pdf`
- `prompt_robustness_publication.pdf`

## Model conditions shown in plots

- `Baseline` (from `response (before)` columns)
- `PT2 (Game)` (`PT2_COREDe`)
- `PT3-De` (`PT3_COREDe`)
- `PT3-Ut` (`PT3_COREUt`)
- `PT4 (Game+De)` (`PT4_COREDe`)

## Notes and caveats

- Plots are generated directly from your current `./results` files; no synthetic extra model slots are included.
- If you only have `run1`, results reflect that single run (no across-run CI).
- All plots are generated headlessly using the Agg backend, so no GUI is required.

## Troubleshooting

- If you see missing-file errors, verify your `./results/<MODEL>/run1` folders are intact.
- If a plot looks empty, inspect the corresponding CSV to ensure responses contain `action1` or `action2`.
- To change the opponent name or episode count, update the adapter parameters in `advanced_analysis.py`.
