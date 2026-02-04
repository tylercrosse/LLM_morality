# Advanced Patching Views: Methodology and Figure Interpretation

This guide explains how the advanced patching figures were generated and how
to interpret the current outputs in `mech_interp_outputs/patching/overview_plus/`.

Source script:

- `docs/reports/scripts/mech_interp/create_patching_advanced_views.py`

Input data:

- `mech_interp_outputs/patching/patch_results_*.csv` (all four experiments)
- `mech_interp_outputs/patching/component_recurrence_summary.csv` (indirectly,
  for context from prior overview)

---

## 1) `advanced_cross_experiment_component_stability.png`

### Methodology

For each component and experiment, we aggregate:

- `mean_abs_delta_change = mean(abs(delta_change))`

Then we take the top components by cross-experiment average effect magnitude
and plot each component's value across experiments as a line chart.

### What is represented

- X-axis: patching experiments (source -> target model)
- Y-axis: mean absolute sequence-preference shift caused by patching that
  component
- One line per component (top-N components)

### What to look for

- **Flat high lines**: components that are robustly influential across all
  model-pair settings.
- **High variance lines**: context-dependent components that may be tied to
  specific source/target dynamics.

### Current run highlights

Most stable high-effect components are concentrated around:

- `L25_MLP`
- `L16H*` (same-layer head family; attention-output-level patching caveat)
- `L16_MLP`
- `L18H*`

This is consistent with the earlier recurrence summary.

---

## 2) `advanced_direction_consistency_heatmap.png`

### Methodology

For each `(component, experiment, scenario)` cell, we compute:

- `sign_consistency = abs(mean(sign(delta_change)))`

This ranges from:

- `1.0`: always same direction across prompt variants
- `0.0`: direction cancels across variants (unstable sign)

### What is represented

- Rows: top components by effect magnitude
- Columns: experiment | scenario
- Color: directional consistency strength (0 to 1)

### What to look for

- **Bright horizontal bands**: components with directionally stable influence
  regardless of scenario.
- **Patchy rows**: components whose effect direction is prompt-sensitive.

### Current run highlights

Many components are highly direction-consistent, including several with `1.0`
average consistency. That indicates variant-level directional robustness for
those components, even when effect sizes differ.

---

## 3) `advanced_variant_robustness_boxplot.png`

### Methodology

Take top-N components globally by mean absolute effect, then plot distribution
of raw `delta_change` values across prompt variants and experiments.

### What is represented

- X-axis: component
- Y-axis: raw `delta_change` (signed)
- Hue: experiment
- Box spread: variability across variants/scenarios

### What to look for

- **Narrow boxes + clear offset from 0**: stable, repeatable effects.
- **Wide boxes crossing 0**: variable components; interpret with caution.
- **Between-experiment separation**: same component behaves differently by
  source->target patch setup.

### Current run highlights

Top components generally show persistent non-zero shifts, but spread differs
by experiment. This supports "stable influence with context-dependent
magnitude" rather than brittle one-off behavior.

---

## 4) `advanced_cumulative_topk_curves.png`

### Methodology

Within each `(experiment, scenario)` group:

1. Rank components by `mean(abs(delta_change))`
2. Compute cumulative fraction of total effect mass explained by top-k

### What is represented

- X-axis: top-k components
- Y-axis: cumulative fraction of total absolute patch effect explained
- Separate lines by scenario and experiment

### What to look for

- **Steep early rise**: sparse mechanism (few components dominate).
- **Gradual rise**: distributed mechanism (many components needed).

### Current run highlights

Average cumulative effect captured:

- Top 5: ~0.118
- Top 10: ~0.210
- Top 20: ~0.331
- Top 50: ~0.567

Interpretation: effects are distributed. Even top-10 components explain only
~21% of total perturbation mass on average. `DC_exploited` and `DD_trapped`
show steeper early curves than `CC_continue`.

---

## 5) `advanced_pair_non_additivity_heatmap.png` and
##    `advanced_pair_non_additivity_matrix.csv`

### Methodology

For a chosen experiment + scenario (current run:
`PT2_COREDe_to_PT3_COREDe`, `DC_exploited`):

1. Select top-N components by single-patch effect (deduplicated by effective
   patch module key to avoid same-layer head aliases).
2. For each component pair `(A, B)` and each prompt variant:
   - Compute joint patch change: `joint_change`
   - Compute additive baseline: `single(A) + single(B)`
   - Non-additivity: `joint_change - additive`
3. Average non-additivity across variants.

### What is represented

- Heatmap/csv cell `(A, B)` = mean non-additivity
- Positive: synergistic (joint > additive expectation)
- Negative: antagonistic/redundant (joint < additive expectation)

### What to look for

- **Large |value| pairs**: strong interaction candidates.
- **Structured blocks**: subsystem interactions rather than isolated pairs.
- **Near-zero region**: approximately additive behavior.

### Current run highlights

Selected components:

- `L16H2`, `L25_MLP`, `L16_MLP`, `L18H0`, `L24_MLP`, `L19H0`, `L15_MLP`,
  `L15H7`

Largest absolute non-additivity examples:

- `L18H0` x `L19H0`: `+2.2578`
- `L18H0` x `L24_MLP`: `+1.9115`
- `L25_MLP` x `L18H0`: `+1.4870`
- `L16_MLP` x `L15_MLP`: `-0.8516`

Mean absolute pair non-additivity is ~`0.516`, indicating notable
non-additive interactions in this setting.

---

## Interpretation Summary

Across these advanced views, the patching evidence remains consistent with a
distributed mechanism:

- no single-component flips in the main patching runs,
- but recurring high-influence regions (late MLP + specific attention layers),
- and measurable pairwise non-additivity that suggests interaction effects
  beyond simple additive component contributions.

Use these plots as triage tools to decide where to run targeted multi-
component or interaction-focused follow-ups.
