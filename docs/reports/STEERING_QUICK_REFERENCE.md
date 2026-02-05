# Activation Steering Quick Reference

**TL;DR**: L16-17 show 25-30% cooperation changes under steering. Early/mid layers (L2-11) show <3% or paradoxical effects. Moral decisions crystallize in final layers.

## Key Findings at a Glance

### Effectiveness Ranking

| Layer | Δ Cooperation | Status | Interpretation |
|-------|---------------|--------|----------------|
| **L17_MLP** | **+29.6%** | ✓ SUCCESS | Best performer - final decision layer |
| **L16_MLP** | **+25.7%** | ✓ SUCCESS | Second-best - pre-decision layer |
| L19_ATTN | +4.3% | ~ MODERATE | Weak attention routing effect |
| L9_MLP | -2.8% | ✗ FAILED | Signal washes out |
| L8_MLP | -0.9% | ✗ FAILED | Signal washes out |
| L2_MLP | -0.8% | ✗ FAILED | Signal washes out |
| **L11_MLP** | **-3.9%** | ⚠ PARADOX | Makes cooperation worse (!) |

### What This Means

✅ **Late layers work**: L16-17 are 10-30× more effective than other layers
✅ **Universal patterns**: Same layers work for both Strategic and Deontological models
✅ **Room for improvement**: Strategic model shows larger dynamic range (8% → 45%)
❌ **Early steering fails**: Signal washes out through 14-18 downstream layers
⚠️ **L11 paradox**: Steering in the "wrong" direction - needs investigation

---

## Visualizations Guide

### 1. Steering Sweep Plots

**File**: `steering_sweep_PT2_COREDe_L16_mlp.png`

**What to look for**:
- **Monotonic increase**: Good steering shows smooth upward curve
- **Large Δ**: Distance between baseline (strength=0) and max (strength=+2.0)
- **Flat curves**: Failed steering (e.g., L8, L9)

**Interpretation**:
```
High Δ + Monotonic = Effective steering (L16, L17)
Low Δ + Noisy = Failed steering (L8, L9, L11)
```

### 2. Logit Lens Trajectories

**File**: `steering_logit_lens_PT2_COREDe_L17_MLP_str+2.0_CC_temptation.png`

**What to look for**:
- **Red dashed line**: Where steering was applied
- **Orange vs Blue separation**: Magnitude of steering effect
- **Persistence**: Does separation maintain or converge back?

**Key comparisons**:
- **L17 steering**: Immediate divergence at L17, persists to output
- **L8 steering**: Initial divergence at L8, converges back by L15

### 3. Overlay Comparisons

**File**: `overlay_all_layers_PT2_COREDe_CC_temptation_str+2.0.png`

**What to look for**:
- **Separation from baseline**: Distance of colored lines from gray baseline
- **L16-17 dominance**: Should clearly separate from other layers
- **L8-11 flatness**: Should track close to baseline

**Best files**:
- `KEY_l17_vs_l16_comparison.png` - Head-to-head of top performers
- `KEY_early_vs_late_washout.png` - Shows why early steering fails
- `KEY_model_universality_l17.png` - Tests cross-model patterns

---

## Quick Interpretation Rules

### Reading Logit Difference Values

```
Positive = Prefers Defect (selfish)
Negative = Prefers Cooperate (moral)
```

**Example**:
- Baseline: +0.52 (prefers Defect)
- Steered: -0.23 (prefers Cooperate)
- **Effect**: Δ = -0.75 (shifted 0.75 logits toward cooperation)

### Reading Cooperation Rates

**Strategic model** (baseline 15.8%):
- Good steering: 30-50% cooperation
- Moderate: 20-30%
- Failed: <20%

**Deontological model** (baseline 95.2%):
- Already near ceiling
- Negative steering more visible (can drop to 87%)
- Positive steering limited (ceiling at ~97%)

### Monotonicity Check

**Good steering** (L16, L17):
```
Strength:  -2.0  -1.0   0.0  +1.0  +2.0
Coop:       8%    12%   16%   29%   45%  ← Monotonic increase
```

**Failed steering** (L8, L9):
```
Strength:  -2.0  -1.0   0.0  +1.0  +2.0
Coop:      16%   15%   16%   15%   15%  ← Flat/noisy
```

---

## Common Questions

### Why do early layers fail?

**Signal washout**: 14-18 downstream layers overwrite the steering signal
- L8 intervention → 17 layers of processing → washed out
- L17 intervention → 8 layers of processing → persists

### Why is L11 paradoxical?

**Three hypotheses**:
1. L11 encodes "strategic thinking" not cooperation
2. L11 is task-general (IPD rules) not preference-specific
3. Steering disrupts balanced computation

**Next step**: Path patching from L11 → L17 to trace information flow

### Can steering transfer to other models?

**Likely yes** for same architecture:
- Strategic and Deontological show r=0.94 correlation in layer rankings
- Same base model (Gemma-2-2b-it) suggests architectural invariants

**Unknown** for different architectures (Llama, Mistral, etc.)

### What strength should I use?

**Recommendation**: ±1.0 to ±2.0 for clear effects
- Strength < 1.0: Subtle changes, good for fine control
- Strength 1.0-2.0: Clear behavioral shifts
- Strength > 2.0: May destabilize (untested)

### Why test negative steering?

**Tests symmetry**:
- Can we steer *away* from cooperation as easily as toward it?
- Deontological model shows **asymmetry** (-7.9% vs +1.6%)
- Ceiling effect: harder to increase already-high cooperation

---

## File Locations

### Data
```
mech_interp_outputs/causal_routing/
├── steering_vector_*.pt              # Normalized steering vectors
├── steering_sweep_*.csv              # Strength sweep data
├── steering_effectiveness_ranking.csv # Summary table
└── downstream_effects_*.csv          # Activation changes
```

### Visualizations
```
mech_interp_outputs/causal_routing/logit_lens_steering/
├── steering_logit_lens_*.png         # Individual trajectories (120 files)
└── overlays/
    ├── KEY_*.png                      # Publication-ready comparisons (3 files)
    ├── overlay_all_layers_*.png       # Layer comparisons (20 files)
    ├── overlay_both_models_*.png      # Model comparisons (30 files)
    └── overlay_all_scenarios_*.png    # Scenario comparisons (12 files)
```

### Code
```
mech_interp/activation_steering.py                    # Core steering logic
scripts/mech_interp/run_activation_steering_comprehensive.py  # Main pipeline
scripts/mech_interp/generate_steering_logit_lens.py   # Trajectory plots
scripts/mech_interp/plot_logit_lens_steering_overlays.py # Overlay comparisons
```

---

## Next Steps

### Immediate Actions

1. **Examine KEY plots** in `overlays/`:
   - `KEY_l17_vs_l16_comparison.png`
   - `KEY_early_vs_late_washout.png`
   - `KEY_model_universality_l17.png`

2. **Check effectiveness ranking**:
   ```bash
   cat mech_interp_outputs/causal_routing/steering_effectiveness_ranking.csv
   ```

3. **Investigate L11 paradox**: Look at L11 logit lens plots

### Research Directions

- **Multi-layer steering**: Test L8+L16 combined
- **Path patching**: Trace L11 → L17 information flow
- **Cross-architecture**: Test on Llama/Mistral
- **Minimal circuits**: Find which neurons/heads are causal

---

## Citation

```bibtex
@techreport{steering2026,
  title={Activation Steering in Moral LLMs: Late-Layer Localization},
  author={Research Team},
  institution={ICLR 2025 LLM Morality Project},
  year={2026},
  month={February}
}
```

**Full documentation**: `docs/reports/ACTIVATION_STEERING_EXPERIMENTS.md`
