# Patching Overview Figures: Interpretation Guide

This document explains how each overview figure was produced, what data it
shows, and what to look for when interpreting the results.

---

## `overview_experiment_scenario_heatmap.png`

This heatmap is built from the four `patch_results_*.csv` files in
`mech_interp_outputs/patching/`, one per source->target patching experiment.
Each CSV contains one row per patched component and prompt variant
(5 scenarios x 3 variants x all components), including sequence-metric patch
effects via `delta_change` (patched minus baseline sequence preference). To
make this comparable across experiments, values are aggregated to **mean
absolute `delta_change`** per experiment/scenario cell.

The plot represents **effect magnitude**, not direction. Higher values mean
patching components in that experiment/scenario tends to move sequence
preference more strongly, regardless of whether the movement is toward
`action1` or `action2`. Because magnitudes are averaged over all components
and prompt variants, this is a top-level sensitivity map.

What to look for: rows/scenarios with warmer colors indicate contexts where
interventions have stronger influence on sequence preference, even if no
behavior flips occur. Use these cells to prioritize deeper layer/component
inspection.

---

## `overview_flip_rates.png`

This bar chart uses the same patching CSVs, but only `action_flipped`. For
each experiment, flip rate is computed as:

`sum(action_flipped) / count(rows)`

It captures how often a **single-component patch** changed the target model's
final preferred action under the sequence metric.

The represented data is intentionally simple: one scalar per experiment that
summarizes causal "hard switch" behavior. It complements magnitude-based
figures by asking "did the patch cross the decision boundary?" rather than
"how much did it move preference?"

What to look for: non-zero bars indicate components that can individually flip
behavior. Near-zero bars indicate distributed robustness, where patches move
preferences but usually do not flip final action.

---

## `overview_layer_type_heatmap.png`

This heatmap parses each `component` into `(layer, type)` where type is `head`
or `mlp`, then aggregates `mean(abs(delta_change))` over all experiments,
scenarios, and variants within each layer/type bucket.

The plot represents **average perturbation strength by architectural region**,
not per-head identity. It provides a compact profile of where in the network
patches matter most, and whether attention outputs or MLP outputs dominate at
each layer.

What to look for: layer bands with high values point to likely decision-
relevant regions. Differences between `head` and `mlp` columns suggest which
pathway class is more influential at a given depth.

---

## `overview_component_recurrence_scatter.png`

This scatter plot is based on
`mech_interp_outputs/patching/component_recurrence_summary.csv`, computed from
all four experiments. For each component:

- `experiments_present` measures cross-experiment recurrence.
- `mean_of_mean_abs_delta` measures average effect magnitude across
  experiments.

Top components by effect are labeled.

The represented data is a **stability vs strength** view: components in the
upper-right are both recurrent and strong, making them better candidates for
shared mechanism hypotheses than components that are strong only once.

What to look for: components with high recurrence and high effect should be
prioritized for multi-component patch tests or targeted follow-up analysis.
Treat them as influence candidates, not guaranteed single-component causal
switches.

---

# Lower-Level Patching Figure Guide

The figures below are per-experiment/per-scenario diagnostics. They are more
granular than the overview plots and are useful for detailed follow-up once
you know which experiment/scenario to inspect.

---

## `patch_heatmap_<experiment>_<scenario>.png`

Example:
`patch_heatmap_PT2_COREDe_to_PT3_COREDe_DD_trapped.png`

### How data is collected

For one source->target experiment and one scenario, the patching run applies
single-component patches across all heads and MLPs for each prompt variant.
Each patch row records sequence-level outcomes:

- `delta_change` = `patched_delta - baseline_delta`
- `effect_size` (normalized effect)
- `action_flipped` (boolean)

The heatmap reshapes these component-level values onto layer/head coordinates
(and layer-MLP entries), usually averaging over variants in that scenario.

### What the plot represents

Each cell shows how strongly patching that component changes sequence
preference for `action2` vs `action1` in the target model.

- Large absolute values: influential component
- Positive/negative sign: direction of movement (toward `action2` or
  `action1`)

### What to look for

1. **Hot bands across a layer**: suggests layer-level influence concentration.
2. **Isolated strong cells**: candidate high-leverage components.
3. **Sign structure**: whether components push consistently in one direction.

Practical caveat: in the current implementation, head patching is at the
layer attention-output level, so multiple heads in one layer can show very
similar values. Treat layer-level patterns as more reliable than fine head
differences.

---

## `patch_top_components_<experiment>_<scenario>.png`

Example:
`patch_top_components_PT2_COREDe_to_PT3_COREUt_DC_exploited.png`

### How data is collected

From the same per-scenario patch rows, components are ranked by effect
magnitude (typically `|delta_change|` or `effect_size`), and the top-k are
plotted as a sorted bar chart.

### What the plot represents

This is a ranking view: it answers "which components moved the decision metric
the most for this scenario?"

- Bar length: effect magnitude
- Bar sign/color (if present): direction of movement

### What to look for

1. **Recurring components** (e.g., same layer/MLP showing up across
   scenarios/experiments): stronger mechanism candidates.
2. **Tail drop-off**: if top few components dominate, targeted follow-up is
   easier.
3. **Consistency with heatmap**: top-ranked components should align with hot
   heatmap cells.

Interpretation caveat: top influence does **not** imply causal flipping. A
component can be highly ranked but still never produce `action_flipped = 1`.

---

## `circuit_discovery_<experiment>_<scenario>_v<variant>.png`

Example:
`circuit_discovery_PT2_COREDe_to_PT3_COREDe_DD_trapped_v0.png`

### How data is collected

Circuit discovery starts from ranked single-component effects for one specific
prompt variant. It then adds components incrementally (greedy/minimal-circuit
style) and tracks cumulative behavioral movement to see whether a small set
can reproduce a behavior change.

### What the plot represents

A trajectory of cumulative patch influence as components are added, plus the
reported "minimal circuit" candidate (if a threshold/flip criterion is met).

### What to look for

1. **Rapid early gain**: suggests a compact influential subset.
2. **Plateau behavior**: indicates diminishing returns from extra components.
3. **Whether criterion is actually met**:
   - If no action flip occurs, treat the discovered set as an influence set,
     not a true minimal causal circuit.

In your current runs (zero single-component flips), these plots are best read
as ranked cumulative influence diagnostics rather than definitive
"this is the causal circuit" evidence.
