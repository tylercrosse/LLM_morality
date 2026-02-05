# Causal Routing Experiments

This directory contains outputs from causal routing experiments designed to test the network rewiring hypothesis with direct interventions.

## Experiments

### 1. Frankenstein (LoRA Weight Transplant)

**Hypothesis**: If L2_MLP acts as a routing switch, transplanting its LoRA weights from one model to another should shift behavior.

**Method**:
- Extract L2_MLP LoRA weights (gate_proj, up_proj, down_proj) from source model
- Replace target model's L2_MLP LoRA weights with source weights
- Evaluate behavior change on all 15 IPD scenarios

**Key Tests**:
1. Strategic + Deontological_L2 → Expect cooperation increase
2. Deontological + Strategic_L2 → Expect cooperation decrease
3. Utilitarian + Deontological_L2 → Expect cooperation increase
4. Deontological + Utilitarian_L2 → Expect slight cooperation decrease

**Success Criteria**: >5% cooperation rate change in expected direction

**Outputs**:
- `frankenstein_{source}_to_{target}_L{layer}.csv` - Per-scenario results
- `frankenstein_{source}_to_{target}_L{layer}_summary.json` - Summary statistics
- `frankenstein_summary_all.csv` - All experiments combined
- `frankenstein_comparison.png` - Baseline vs transplanted cooperation rates
- `frankenstein_per_scenario_heatmap.png` - Per-scenario effect sizes

**Interpretation**:
- If hypotheses supported (≥3/4): Strong evidence for L2_MLP routing role
- If moderate (2/4): Some evidence, but may need additional context
- If weak (≤1/4): L2_MLP weights alone insufficient, investigate pathways

### 2. Path Patching (Planned)

**Hypothesis**: Information flows causally through L2_MLP → residual stream → L9_MLP pathway.

**Method**:
- Cache activations from source model (e.g., Deontological)
- Replace target model's (e.g., Strategic) residual stream activations layer-by-layer
- Measure progressive effect as path extends from L2 → L3 → ... → L9

**Key Tests**:
- Progressive patching to find critical layer range
- MLP-only vs Attention-only path contributions
- Compare effect size to single-component patching (which showed zero flips)

**Success Criteria**: >50% behavior change when patching L2→L9 path

### 3. Activation Steering (Planned)

**Hypothesis**: L2_MLP activations can be steered to control downstream routing.

**Method**:
- Compute steering vector: `mean(moral_acts) - mean(strategic_acts)` at L2_MLP
- Add scaled steering vector to L2_MLP activations during forward pass
- Test strengths: [-2.0, -1.5, ..., 0.0, ..., 1.5, 2.0]

**Key Tests**:
- Steering sweep: p_action2 vs steering strength (expect monotonic)
- Downstream effects: How does L9_MLP activation change?
- Isolation: What dimensions of L2_MLP affect which downstream components?

**Success Criteria**: Linear relationship between steering strength and cooperation rate

### 4. Causal Mediation Analysis (Planned)

**Hypothesis**: Significant fraction of behavioral difference flows through L2→L9 pathway.

**Method**:
- Total Effect (TE): p_action2(strategic) - p_action2(deontological)
- Natural Indirect Effect (NIE): Effect mediated through L2→L9 pathway
- Natural Direct Effect (NDE): Effect through other pathways
- Mediation proportion: NIE / TE

**Success Criteria**: >30% of total effect mediated through L2→L9

**Interpretation**:
- High mediation (>50%): L2→L9 is dominant pathway
- Moderate (30-50%): Important but not sole pathway
- Low (<30%): Effect is highly distributed

## Running Experiments

```bash
# Frankenstein (weight transplant)
python scripts/mech_interp/run_frankenstein.py

# Path patching (when implemented)
python scripts/mech_interp/run_path_patching.py

# Activation steering (when implemented)
python scripts/mech_interp/run_activation_steering.py

# Causal mediation (when implemented)
python scripts/mech_interp/run_causal_mediation.py
```

## Dependencies

All experiments reuse existing infrastructure:
- `mech_interp/weight_analysis.py` - LoRA weight loading
- `mech_interp/model_loader.py` - HookedGemmaModel with activation caching
- `mech_interp/decision_metrics.py` - Sequence-level behavior evaluation
- `mech_interp/activation_patching.py` - Hook registration patterns
- `mech_interp/direct_logit_attribution.py` - Component contribution analysis

## Expected Runtime

- Frankenstein: ~30-45 minutes (4 experiments × 15 scenarios × 2 model loads)
- Path Patching: ~60-90 minutes (progressive patching with multiple path lengths)
- Activation Steering: ~30-45 minutes (9 steering strengths × 15 scenarios)
- Causal Mediation: ~45-60 minutes (counterfactual computation)

**Total**: ~3-4 hours for all experiments (GPU recommended)

## Validation

All experiments use:
- Sequence-level probability metrics (validated Feb 4, 2026)
- Same 15 IPD scenarios as original analyses
- Reproducible random seeds
- Automatic sanity checks (e.g., steering strength=0 should equal baseline)

## Integration with Other Findings

These causal experiments complement correlational findings:
- **DLA**: Shows L8/L9 encode cooperation/defection (what components do)
- **Attention**: 99.99% identical (rules out attention mechanism)
- **Linear Probes**: Identical representations (rules out representation differences)
- **Component Interactions**: 29 pathways differ (correlational connectivity evidence)
- **Weight Analysis**: L2_MLP not heavily modified (rules out massive retraining)
- **→ Causal Routing**: Tests if L2→L9 pathway causally mediates behavior (causal evidence)

## Limitations

- Only tested on Gemma-2-2b-it with LoRA fine-tuning
- Only tested on IPD scenarios (one task)
- Path patching assumes information flows through residual stream
- Steering assumes linear separability of moral vs strategic directions
- Mediation analysis makes counterfactual assumptions

## File Manifest

After running all experiments, expect:
- 16+ CSV files (4 Frankenstein + 3 path patching + 2 steering + others)
- 8+ PNG visualizations
- 4+ JSON summaries
- This README

## Citation

If using these experiments, cite:
- Original paper: [ICLR 2025 submission]
- Validation methodology: docs/reports/LOGIT_DECISION_METRIC_LESSONS.md
- Research log: docs/reports/MECH_INTERP_RESEARCH_LOG.md
