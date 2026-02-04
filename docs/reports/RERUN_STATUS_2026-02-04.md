# Re-run Status Summary (Feb 4, 2026)

This note summarizes the queued post-fix re-run attempts and outcomes for
the updated mech-interp pipeline.

## Scope queued in tmux

The following scripts were queued to run sequentially:

1. `docs/reports/scripts/mech_interp/run_dla.py`
2. `docs/reports/scripts/mech_interp/run_attention_analysis.py`
3. `docs/reports/scripts/mech_interp/run_component_interactions.py`
4. `docs/reports/scripts/mech_interp/validate_mech_interp_alignment.py`

(`run_probes.py` was intentionally left optional for this metric-fix cycle.)

## Outcome by script

### 1) `run_dla.py` — Failed at import stage

- Log: `mech_interp_outputs/logs/run_dla.log`
- Error:
  - `ModuleNotFoundError: No module named 'mech_interp'`

### 2) `run_attention_analysis.py` — Failed at import stage

- Log: `mech_interp_outputs/logs/run_attention.log`
- Error:
  - `ModuleNotFoundError: No module named 'mech_interp'`

### 3) `run_component_interactions.py` — Failed at import stage

- Log: `mech_interp_outputs/logs/run_component_interactions.log`
- Error:
  - `ModuleNotFoundError: No module named 'mech_interp'`

### 4) `validate_mech_interp_alignment.py` — Completed successfully

- Log: `mech_interp_outputs/logs/validate_alignment.log`
- Outputs written to `mech_interp_outputs/validation/`:
  - `alignment_per_prompt.csv`
  - `alignment_by_scenario_model.csv`
  - `alignment_confusion_table.csv`
  - `significance_global_strategic_vs_de_ut.csv`
  - `significance_by_scenario_strategic_vs_de_ut.csv`
  - `SUMMARY.md`

## High-level results from successful validation run

- Sequence-vs-sampled majority agreement:
  - Mean agreement rate = **1.0** for all four evaluated models
    (`PT2_COREDe`, `PT3_COREDe`, `PT3_COREUt`, `PT4_COREDe`)
- Separation signal (`seq_p_action2_mean` by model):
  - Strategic (`PT2_COREDe`): ~**0.9997**
  - Deontological (`PT3_COREDe`): ~**0.0003**
  - Utilitarian (`PT3_COREUt`): ~**0.0703**
  - Hybrid (`PT4_COREDe`): ~**0.4116**
- Global permutation tests (Strategic vs De/Ut pairs):
  - Significant for `seq_p_action2`,
    `seq_delta_logp_action2_minus_action1`, and sampled action rate
  - `p ≈ 0.00005` across reported global tests

## Current state of generated artifacts

- `mech_interp_outputs/patching/` is fresh (updated Feb 4, 2026) and
  sequence-metric aware.
- `mech_interp_outputs/dla/`, `mech_interp_outputs/attention_analysis/`,
  and `mech_interp_outputs/component_interactions/` were **not refreshed**
  in this run due to the import failure above.

## Immediate follow-up needed

Re-run the three failed scripts after resolving Python module path setup
for script execution (so `mech_interp` imports resolve correctly). Then
reconcile DLA/attention/component outputs against the successful validation
results.
