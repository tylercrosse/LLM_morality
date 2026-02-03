# Analysis Guide

This repo includes three analysis entry points for the evaluation results in `./results/`:

- `analyze_results.ipynb` — interactive exploration notebook
- `quick_analysis.py` — fast CLI summary + core plots
- `visualize_results.py` — plot suite for common figures

## Expected results layout

```
./results/
├── PT2_COREDe/
│   └── run1/
│       ├── EVAL After FT _PT2 - independent eval IPD.csv
│       ├── EVAL After FT _PT2 - independent eval ISH.csv
│       ├── EVAL After FT _PT2 - independent eval ICN.csv
│       ├── EVAL After FT _PT2 - independent eval ICD.csv
│       ├── EVAL After FT _PT2 - independent eval BOS.csv
│       ├── EVAL After FT _PT2 - independent eval 2 unstructured IPD queries.csv
│       └── EVAL After FT _PT2 - independent eval 4 unrelated queries.csv
├── PT3_COREDe/run1/...
├── PT3_COREUt/run1/...
└── PT4_COREDe/run1/...
```

## Quick analysis (CLI)

```
python quick_analysis.py --results-dir ./results --output-dir ./analysis_output
```

Outputs:
- `analysis_output/summary_report.md`
- `analysis_output/summary_metrics.csv`
- `analysis_output/cooperation_by_game.png`
- `analysis_output/rewards_comparison.png`
- `analysis_output/model_comparison.png`
- `analysis_output/reciprocity_heatmap.png`

## Visualization suite

```
python visualize_results.py --results-dir ./results --output-dir ./figures
```

Outputs:
- `figures/summary_metrics.csv`
- `figures/cooperation_by_model_game.png`
- `figures/reward_distributions.png`
- `figures/before_after_comparison.png`
- `figures/generalization_plot.png`

## Notebook (interactive)

Open `analyze_results.ipynb` in Jupyter and run all cells. It:
- Loads all game CSVs
- Computes cooperation/illegal rates
- Compares rewards across models
- Visualizes reciprocity patterns

## Metrics interpretation

- **Cooperation rate**: fraction of responses containing `action1` (mapped to C).
- **Defection rate**: fraction containing `action2` (mapped to D).
- **Illegal rate**: responses without `action1` or `action2`.
- **Reward metrics**: mean of `rewards_Game`, `rewards_De`, `rewards_Ut`, and `rewards_GameDe` (before/after).

## Tips

- If plots look empty, confirm the CSV file names match the expected pattern.
- If you want to change the action mapping, update `parse_action()` in the scripts/notebook.
- For heavier statistical tests (t-tests, ANOVA), add `scipy` and extend the scripts.

## About `plotting.py`

`plotting.py` contains many plotting utilities but executes a lot of code on import and expects a different directory layout. If you want to use it, consider refactoring it into import-safe functions or create a dedicated symlink tree matching its expectations.
