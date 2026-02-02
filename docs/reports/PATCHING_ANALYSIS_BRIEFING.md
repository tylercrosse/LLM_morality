# Activation Patching Analysis Briefing

**Date**: February 2, 2026
**Status**: Patching experiments running (~1 hour elapsed, 2-4 experiments likely complete)
**Task**: Analyze and interpret patching results, particularly for RQ2

---

## Background Context

### Project Overview
Mechanistic interpretability study of LoRA-finetuned LLMs trained on Iterated Prisoner's Dilemma (IPD) with different moral reward structures. We're investigating how moral fine-tuning changes internal model computations.

### Models Being Analyzed
- **Base**: Gemma-2-2b-it (no fine-tuning)
- **PT2_COREDe** (Strategic): PPO-trained with standard IPD payoffs only
- **PT3_COREDe** (Deontological): PPO-trained with game payoffs + betrayal penalty (-3)
- **PT3_COREUt** (Utilitarian): PPO-trained to maximize collective welfare (sum of both players)
- **PT4_COREDe** (Hybrid): Game payoffs + deontological penalty

All models trained for 1,000 episodes with Tit-for-Tat opponent, LoRA rank 64.

### Research Questions
1. **RQ1**: How are "selfish" attention heads suppressed during moral fine-tuning?
2. **RQ2**: Do Deontological vs. Utilitarian agents develop distinct circuit structures? ⚠️ **PRIMARY FOCUS**
3. **RQ3**: Can we identify which parts to fine-tune specifically for targeted training?

---

## What We've Learned So Far

### From Direct Logit Attribution (DLA)

**Key Finding**: Component-level contributions are remarkably similar across all models.

**Universal Components**:
- **L8_MLP**: Strongly pro-Defect (+6.8 to +7.7) across ALL models
- **L9_MLP**: Strongly pro-Cooperate (-8.2 to -9.3) across ALL models
- These adjacent layers dominate decision-making (7-9x stronger than other components)

**Model Similarities**:
| Model | Mean Contribution | Pro-Defect Components | Pro-Cooperate Components |
|-------|-------------------|----------------------|-------------------------|
| All models | ~0.135 | ~2380 (68%) | ~1130 (32%) |

**Moral Fine-Tuning Effects** (PT3 vs PT2):
- Maximum component change: 0.047 (L13_MLP toward cooperation)
- L8_MLP (most pro-defect) actually INCREASED +0.013 in moral models
- Changes concentrated in mid-late MLPs (L11-L23)
- **Interpretation**: Moral reasoning emerges from subtle rebalancing, not suppression

**Implication for RQ1**: ✅ **ANSWERED** - Selfish components aren't suppressed; behavior emerges from distributed adjustments.

### From Activation Patching (Experiments 1-2)

**Experiment 1**: PT2 → PT3_De (Strategic → Deontological)
- **0 behavioral flips** in 3,510 patches
- Mean effect: **-0.012** (made model MORE cooperative, not less!)
- Top components: L0_MLP, all L0 heads in temptation scenarios (+0.094)
- Minimal circuits failed even at 10 components

**Experiment 2**: PT2 → PT3_Ut (Strategic → Utilitarian)
- **0 behavioral flips** in 3,510 patches
- Mean effect: **+0.0005** (nearly neutral)
- More balanced: 39% increase defection, 40% increase cooperation
- Top components: L2 heads, L12 heads in temptation (+0.109)

**Key Differences Found**:
1. **Response patterns**: De pushes cooperation (-0.012), Ut is neutral (+0.0005)
2. **Key layers**: De uses L0, Ut uses L2/L12
3. **Scenario sensitivity**: Ut shows context-dependent effects (±0.052 range)

**Robustness**: Both moral models highly resistant to strategic influence (no flips!)

---

## The Challenge with RQ2

### Problem Statement
**Deontological and Utilitarian models show very similar component-level patterns**, making it difficult to identify distinct circuit structures. This is concerning because:

1. Behavioral evaluation shows they DO behave differently (different regret, reciprocity patterns)
2. Original paper reported "similar but slightly different policies"
3. Need to distinguish circuits to fully answer RQ2

### Possible Explanations
1. **Similar circuits, subtle differences**: Differences in weighting/interaction, not structure
2. **Context-dependent differences**: Circuits differ in specific scenarios, not overall
3. **Interaction vs individual components**: Differences in how components combine, not individual magnitudes
4. **Training consistency**: Models converged to similar solutions (not a fluke)
5. **Measurement granularity**: Our methods aren't fine-grained enough

---

## Current Patching Run Details

### Experiments Running
**Experiment 3**: PT3_De → PT3_Ut (Deontological → Utilitarian)
**Experiment 4**: PT3_Ut → PT3_De (Utilitarian → Deontological)

### What These Experiments Test
- **Direct comparison**: Patching between the two moral models (not via strategic)
- **Bidirectional**: Both directions to find asymmetric effects
- **RQ2 focus**: Most relevant for identifying distinguishing circuits

### Technical Details
- **234 components per model**: 26 layers × 8 heads + 26 MLPs
- **15 scenarios**: 5 game states × 3 variants each
- **Total patches per experiment**: 3,510 (234 × 15)
- **Measurements**: Δ logit change, action flips, effect size
- **Circuit discovery**: Greedy search for minimal circuits (max 10 components)

### Evaluation Scenarios
1. **CC_continue**: Mutual cooperation maintenance
2. **CC_temptation**: Cooperation with defection incentive
3. **CD_punished**: Cooperated but got defected on
4. **DC_exploited**: Defected on cooperator
5. **DD_trapped**: Mutual defection cycle

---

## Analysis Tasks

### Primary Questions to Answer

#### 1. Are there ANY behavioral flips?
```bash
# Check if experiments 3-4 caused action flips
grep "action_flipped,True" mech_interp_outputs/patching/patch_results_PT3_*.csv
```

If **YES**: Celebrate! These components distinguish moral frameworks.
If **NO**: Moral frameworks are even more similar than De/Ut vs Strategic.

#### 2. What are the effect sizes?
```python
# Compare effect distributions
df_de_to_ut = pd.read_csv('patch_results_PT3_COREDe_to_PT3_COREUt.csv')
df_ut_to_de = pd.read_csv('patch_results_PT3_COREUt_to_PT3_COREDe.csv')

print("De → Ut effects:")
print(f"  Mean: {df_de_to_ut['delta_change'].mean():.4f}")
print(f"  Std:  {df_de_to_ut['delta_change'].std():.4f}")
print(f"  Max:  {df_de_to_ut['delta_change'].max():.4f}")
print(f"  Min:  {df_de_to_ut['delta_change'].min():.4f}")

# Compare to PT2 → PT3 experiments
# Expected: Smaller effects (models more similar)
```

**Hypothesis**: Effects should be smaller than PT2→PT3 (±0.012, ±0.0005) since moral models are more similar.

#### 3. Are there asymmetric effects?
```python
# Compare bidirectional patching
# Does De → Ut differ from Ut → De?

for component in components:
    de_to_ut = df_de_to_ut[df_de_to_ut.component == component].delta_change.mean()
    ut_to_de = df_ut_to_de[df_ut_to_de.component == component].delta_change.mean()

    asymmetry = abs(de_to_ut - ut_to_de)
    if asymmetry > 0.01:  # Threshold for "interesting"
        print(f"{component}: asymmetry = {asymmetry:.4f}")
```

**Interpretation**: Asymmetric effects reveal directional causal relationships.

#### 4. Which components distinguish moral frameworks?
```python
# Find components with largest effects in cross-patching
# but small effects in PT2 → PT3

# Load all results
pt2_to_de = pd.read_csv('patch_results_PT2_COREDe_to_PT3_COREDe.csv')
pt2_to_ut = pd.read_csv('patch_results_PT2_COREDe_to_PT3_COREUt.csv')

# For each component, compare:
# - Effect in PT2 → PT3_De (strategic influence on De)
# - Effect in PT2 → PT3_Ut (strategic influence on Ut)
# - Effect in PT3_De → PT3_Ut (De → Ut distinction)

# Components that matter for De/Ut distinction should have:
# - Small effect in PT2 → both (not strategic-related)
# - Large effect in De ↔ Ut (moral framework-specific)
```

#### 5. Are there scenario-specific differences?
```python
# Do De and Ut differ more in certain scenarios?

for scenario in scenarios:
    scenario_df = df_de_to_ut[df_de_to_ut.scenario == scenario]
    mean_effect = scenario_df.delta_change.mean()
    print(f"{scenario}: {mean_effect:.4f}")

# Hypothesis: Might differ more in:
# - CC_temptation (De: duty, Ut: collective good)
# - CD_punished (De: forgiveness, Ut: pragmatic)
# - DC_exploited (De: guilt, Ut: consequence)
```

#### 6. Did circuit discovery find minimal circuits?
```bash
# Check circuits CSV files
cat mech_interp_outputs/patching/circuits_PT3_*.csv

# Questions:
# - Did any circuits flip behavior?
# - What's the circuit size? (< 10 = success)
# - Are circuit components consistent across scenarios?
# - Do they overlap with PT2 → PT3 circuits?
```

### Secondary Analyses

#### Statistical Significance
```python
# Test if De and Ut contributions differ significantly
from scipy import stats

for component in all_components:
    # Get DLA contributions
    de_contrib = dla_df[(dla_df.model == 'PT3_COREDe') &
                        (dla_df.component == component)].contribution
    ut_contrib = dla_df[(dla_df.model == 'PT3_COREUt') &
                        (dla_df.component == component)].contribution

    t_stat, p_value = stats.ttest_rel(de_contrib, ut_contrib)

    if p_value < 0.05:
        print(f"{component}: p={p_value:.4f}, Δ={de_contrib.mean() - ut_contrib.mean():.4f}")
```

#### Component Interaction Patterns
```python
# Hypothesis: Differences might be in interactions, not individual components

# For each model, compute correlation matrix of component activations
# Compare correlation patterns between De and Ut

# Example: Does De have stronger L0→L13 connection?
#          Does Ut have stronger L2→L17 connection?
```

#### Pairwise Model Distance
```python
# Compute all pairwise distances to find most different models
from scipy.spatial.distance import cosine

models = ['base', 'PT2_COREDe', 'PT3_COREDe', 'PT3_COREUt', 'PT4_COREDe']

for m1 in models:
    for m2 in models:
        if m1 < m2:
            vec1 = dla_df[dla_df.model == m1].contribution.values
            vec2 = dla_df[dla_df.model == m2].contribution.values
            dist = cosine(vec1, vec2)
            print(f"{m1} vs {m2}: {dist:.6f}")

# Expected ranking (most to least similar):
# - PT3_De vs PT3_Ut: smallest (moral models)
# - Base vs PT2: small (both pre-moral)
# - PT2 vs PT3: medium (strategic vs moral)
# - Base vs PT3: largest (no training vs moral)
```

---

## Interpretation Guidelines

### Possible Outcomes & Interpretations

#### Outcome 1: Weak effects, no flips (like Experiments 1-2)
**Interpretation**:
- Deontological and Utilitarian are circuit-level indistinguishable
- Behavioral differences emerge from subtle weighting, not structure
- **RQ2 Answer**: "Moral frameworks share circuits; differences in weighting"

**Follow-up**:
- Statistical tests for subtle differences
- Interaction pattern analysis
- Scenario-specific analysis
- Reframe as "convergent moral computation"

#### Outcome 2: Moderate effects, some flips
**Interpretation**:
- Distinguishing components exist but are distributed
- Certain scenarios show clearer differences
- **RQ2 Answer**: "Partial circuit distinction; context-dependent differences"

**Follow-up**:
- Characterize flipping components
- Map to behavioral differences
- Build minimal distinguishing circuit

#### Outcome 3: Strong effects, many flips
**Interpretation**:
- Clear circuit-level distinction
- De and Ut use different computational strategies
- **RQ2 Answer**: "Distinct moral circuits identified"

**Follow-up**:
- Characterize distinguishing circuits
- Map to moral reasoning differences
- Publication-quality finding!

#### Outcome 4: Asymmetric effects
**Interpretation**:
- Directional causal relationships
- One framework more robust than the other
- **RQ2 Answer**: "Asymmetric moral circuits; one dominates"

**Follow-up**:
- Understand why asymmetry exists
- Relate to training objectives
- Test if other models show asymmetry

### Connection to RQ2

**Original question**: "Do Deontological vs. Utilitarian agents develop distinct circuit structures?"

**Possible reformulations based on findings**:

1. **If similar**: "How do similar circuits produce different moral behaviors?"
   - Focus on interaction patterns
   - Emphasize distributed representation
   - Highlight robustness of moral encoding

2. **If scenario-dependent**: "When and where do moral frameworks diverge computationally?"
   - Map scenarios to circuit differences
   - Connect to ethical theory (duty vs consequence)
   - Identify critical decision points

3. **If asymmetric**: "Which moral framework is computationally dominant?"
   - Characterize the asymmetry
   - Relate to training stability
   - Consider evolutionary/learning dynamics

---

## File Locations

### Expected Output Files
```
mech_interp_outputs/patching/
├── patch_results_PT3_COREDe_to_PT3_COREUt.csv
├── patch_results_PT3_COREUt_to_PT3_COREDe.csv
├── patch_summary_PT3_COREDe_to_PT3_COREUt.csv
├── patch_summary_PT3_COREUt_to_PT3_COREDe.csv
├── circuits_PT3_COREDe_to_PT3_COREUt.csv
├── circuits_PT3_COREUt_to_PT3_COREDe.csv
├── top_components_PT3_COREDe_to_PT3_COREUt.csv
├── top_components_PT3_COREUt_to_PT3_COREDe.csv
├── patch_heatmap_*_PT3_*.png
├── patch_top_components_*_PT3_*.png
├── circuit_discovery_*_PT3_*.png
├── patch_consistency_PT3_*.png
└── cross_experiment_summary.csv
```

### Reference Data
```
mech_interp_outputs/dla/
├── dla_full_results.csv              # DLA component attributions
├── dla_summary_stats.csv             # Statistical summaries
└── dla_top_components.csv            # Ranked components

mech_interp_outputs/patching/
├── patch_results_PT2_COREDe_to_PT3_COREDe.csv   # Experiment 1
├── patch_results_PT2_COREDe_to_PT3_COREUt.csv   # Experiment 2
└── ...                                           # Related files
```

---

## Success Criteria

### Minimum Requirements
- [ ] Quantify effect sizes for cross-patching (De ↔ Ut)
- [ ] Compare to baseline (PT2 → PT3 effects)
- [ ] Identify top 10 distinguishing components
- [ ] Provide clear answer to RQ2 (with caveats)

### Bonus Analyses
- [ ] Statistical significance tests
- [ ] Scenario-specific patterns
- [ ] Asymmetry analysis
- [ ] Component interaction patterns
- [ ] Visualization of distinguishing circuits

### Deliverables
1. **Summary statistics**: Effect distributions, flip counts, top components
2. **Interpretation**: Clear answer to RQ2 with supporting evidence
3. **Visualizations**: Key plots showing De/Ut differences (or lack thereof)
4. **Recommendations**: Next steps if findings are ambiguous

---

## Context on Expectations

### Why This Matters
The paper author (Elizaveta Tennant) is interested in this analysis. If we find clear distinctions, it's a strong result. If we find similarities, it's still interesting (convergent moral computation) but needs careful framing.

### Time Pressure
The patching run has been ongoing for ~1 hour. Results should be available soon (if not already). This is the final major analysis before writing up findings.

### Research Log
All findings should be documented in `/root/LLM_morality/MECH_INTERP_RESEARCH_LOG.md` for future reference.

---

## Quick Start Commands

### 1. Check if patching is complete
```bash
ps aux | grep run_patching
ls -lh mech_interp_outputs/patching/*PT3*.csv
```

### 2. Quick summary
```bash
python3 << 'EOF'
import pandas as pd
import glob

# Find PT3 cross-patching files
files = glob.glob('mech_interp_outputs/patching/patch_results_PT3_*.csv')

for f in files:
    df = pd.read_csv(f)
    print(f"\n{f.split('/')[-1]}")
    print(f"  Total patches: {len(df)}")
    print(f"  Action flips: {df['action_flipped'].sum()}")
    print(f"  Mean Δ: {df['delta_change'].mean():.4f}")
    print(f"  Std Δ:  {df['delta_change'].std():.4f}")
    print(f"  Max |Δ|: {df['delta_change'].abs().max():.4f}")
EOF
```

### 3. Detailed analysis
See analysis tasks section above for specific Python snippets.

---

## Questions for the Analyst

1. Are the PT3 ↔ PT3 effects smaller than PT2 → PT3 effects? (Expected: yes)
2. Do any components cause behavioral flips? (Hope: yes, but prepared for no)
3. Are effects symmetric or asymmetric? (Either is interesting)
4. Which scenarios show largest De/Ut differences? (Connect to ethics)
5. Can we identify a minimal distinguishing circuit? (For RQ2)
6. Should we recommend retraining with stronger moral signals? (If effects are too weak)
7. How should we frame RQ2 for the paper? (Depends on findings)

---

**Good luck with the analysis! Remember: Even if De and Ut are very similar, that's a valid and interesting finding. The key is to interpret it correctly and provide clear evidence.**
