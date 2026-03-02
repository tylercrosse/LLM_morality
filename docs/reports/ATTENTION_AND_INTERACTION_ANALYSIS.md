# Attention Pattern and Component Interaction Analysis

## Overview

This document describes two additional mechanistic interpretability analyses designed to investigate **how** Deontological and Utilitarian models produce different behaviors despite having nearly identical component-level representations.

### Motivation from RQ2 Analysis

Our comprehensive RQ2 analysis revealed that:
- PT3_COREDe and PT3_COREUt have 99.9999% similar component strengths (cosine distance = 0.00000588)
- Cross-patching produced 0 behavioral flips in 7,020 experiments
- However, 78% of components show directional asymmetry (different effects when patched in each direction)

This suggests: **Similar circuits with different information routing or processing strategies.**

These two new analyses test specific hypotheses about the mechanisms:

1. **Attention Pattern Analysis**: Do the models attend to different information?
2. **Component Interaction Analysis**: Do the models wire components together differently?

---

## Analysis 1: Attention Pattern Analysis

### Hypothesis

**Deontological models attend more to opponent's previous actions** (testing for betrayal, reciprocity norms)

**Utilitarian models attend more to joint payoff information** (computing collective welfare)

### Methodology

For each model and each scenario:

1. **Extract attention weights** from all 26 layers × 8 heads
2. **Aggregate attention to the final token** (where decision is made)
3. **Classify tokens** into three categories:
   - **Action keywords**: "action1", "action2"
   - **Opponent action context**: "they played", "opponent chose", "other player"
   - **Payoff information**: "points", "reward", "receive", "outcome"
4. **Compute attention sums** for each category
5. **Compare** between Deontological and Utilitarian models

### Key Files

**Implementation:**
- [mech_interp/attention_analysis.py](mech_interp/attention_analysis.py) - Core analysis module (430 lines)
  - `AttentionAnalyzer`: Extracts attention patterns
  - `AttentionComparator`: Compares models
  - `AttentionPatternResult`: Data structure for results

**Execution:**
- [scripts/mech_interp/run_attention_analysis.py](scripts/mech_interp/run_attention_analysis.py) - Run the analysis

### Usage

```bash
# Run attention analysis
python scripts/mech_interp/run_attention_analysis.py
```

### Expected Outputs

**Data files:**
- `attention_results_PT3_COREDe.json` - All attention patterns for Deontological model
- `attention_results_PT3_COREUt.json` - All attention patterns for Utilitarian model
- `attention_comparison_De_vs_Ut.csv` - Comparison statistics

**Visualizations:**
- `attention_comparison_Deontological_vs_Utilitarian.png` - Bar charts showing attention to each token type
- `attention_heatmap_*.png` - Individual attention heatmaps (optional)

### Interpretation Guide

**If Deontological attends more to opponent actions:**
- Evidence for "reciprocity-based moral reasoning"
- Model checks opponent's previous move to decide on cooperation
- Supports norm-following interpretation (tit-for-tat, reciprocity)

**If Utilitarian attends more to payoff information:**
- Evidence for "consequentialist reasoning"
- Model computes expected collective welfare
- Supports optimization interpretation

**If attention patterns are similar:**
- Both frameworks use same information
- Differences arise in how information is processed (see Component Interaction Analysis)

---

## Analysis 2: Component Interaction Analysis

### Hypothesis

**Deontological and Utilitarian models wire similar components together with different strengths**

Example predictions:
- **De model**: Strong L8_MLP → L25H0 connection (opponent state → decision)
- **Ut model**: Strong L9_MLP → L25H3 connection (collective payoff → decision)

### Methodology

For each model and each scenario:

1. **Extract component activations** (L2 norm of final token hidden state for each head and MLP)
2. **Compute correlation matrix** across scenarios (which components co-activate?)
3. **Compare correlation matrices** between De and Ut models
4. **Identify pathways** with large correlation differences (>0.3)
5. **Categorize pathway types**:
   - **Intra-layer**: Same layer connections
   - **Adjacent-layer**: Sequential processing
   - **Early-to-late**: L0-L9 → L16-L25 (long-range information flow)
   - **Late-stage**: L16-L25 → L16-L25 (decision refinement)

### Key Files

**Implementation:**
- [mech_interp/component_interactions.py](mech_interp/component_interactions.py) - Core analysis module (550 lines)
  - `ComponentInteractionAnalyzer`: Extracts component activations and computes correlations
  - `InteractionComparator`: Compares correlation matrices
  - `ComponentInteractionResult`: Data structure for results

**Execution:**
- [scripts/mech_interp/run_component_interactions.py](scripts/mech_interp/run_component_interactions.py) - Run the analysis

### Usage

```bash
# Run component interaction analysis
python scripts/mech_interp/run_component_interactions.py
```

### Expected Outputs

**Data files:**
- `component_activations_PT3_COREDe.json` - Activation magnitudes for Deontological model
- `component_activations_PT3_COREUt.json` - Activation magnitudes for Utilitarian model
- `correlation_matrix_PT3_COREDe.npz` - 234×234 correlation matrix for De
- `correlation_matrix_PT3_COREUt.npz` - 234×234 correlation matrix for Ut
- `interaction_comparison_De_vs_Ut.csv` - All pairwise correlation differences (27,261 rows)
- `significant_pathways_De_vs_Ut.csv` - Pathways with |diff| > 0.3
- `key_component_pathways_De_vs_Ut.csv` - Connections involving L8_MLP, L9_MLP, L25 heads

**Visualizations:**
- `correlation_matrix_PT3_COREDe.png` - Heatmap of De model correlations (top 50 components)
- `correlation_matrix_PT3_COREUt.png` - Heatmap of Ut model correlations (top 50 components)
- `interaction_diff_Deontological_vs_Utilitarian.png` - Difference heatmap

### Interpretation Guide

**If many pathways differ significantly:**
- Models use different information routing strategies
- Same components, different "wiring diagram"
- Explains how similar components produce different behaviors

**If early-to-late pathways differ:**
- Different information integration strategies
- De: Direct opponent state → decision
- Ut: Multi-stage payoff computation → decision

**If late-stage pathways differ:**
- Different decision refinement processes
- Final layers implement different "override" or "correction" mechanisms

**If key component pathways differ (L8_MLP, L9_MLP → L25 heads):**
- Direct evidence for different functional roles
- L8_MLP may connect differently in De vs Ut
- Supports "similar components, different connections" hypothesis

---

## How These Analyses Address RQ2

### Original RQ2: Are Deontological and Utilitarian models distinct?

**Previous answer (from DLA + Patching):**
"Partially distinct - 99.9999% similar in component strengths but 78% show directional asymmetry"

### Refined RQ2: How do similar circuits produce different moral behaviors?

**Attention Analysis answers:**
- Do they attend to different aspects of the problem?
- Which information does each framework prioritize?

**Interaction Analysis answers:**
- Do they wire components together differently?
- Are information flow pathways distinct?

### Combined Interpretation Framework

| Finding | Interpretation |
|---------|---------------|
| **Attention patterns differ + Interaction patterns differ** | Strong evidence for distinct mechanisms: different information selection AND different processing |
| **Attention patterns differ + Interaction patterns similar** | Different input focus, same processing architecture |
| **Attention patterns similar + Interaction patterns differ** | Same information, different computational pathways (most interesting!) |
| **Both similar** | Differences arise at an even finer grain (individual neuron directions, residual stream geometry) |

---

## Technical Details

### Attention Weight Extraction

Gemma-2 stores attention weights in cache:
```python
cache["model.layers.{layer}.self_attn.attn_weights"]  # Shape: (batch, num_heads, seq_len, seq_len)
```

We focus on the **final token's attention** (row -1 of the attention matrix), which determines what information is used for the decision.

### Component Activation Extraction

For attention heads:
```python
head_out = cache["model.layers.{layer}.self_attn.head_out"][0, -1, head, :]  # (head_dim,)
magnitude = torch.norm(head_out, p=2)  # L2 norm
```

For MLPs:
```python
mlp_out = cache["model.layers.{layer}.mlp.output"][0, -1, :]  # (hidden_dim,)
magnitude = torch.norm(mlp_out, p=2)  # L2 norm
```

### Correlation Computation

Given activation magnitudes across N=15 scenarios:
```
Activation matrix: (15 scenarios, 234 components)

Correlation[i, j] = pearson_correlation(activations[:, i], activations[:, j])
```

This produces a 234×234 symmetric matrix showing which components co-activate.

### Why Use Correlation?

- **Component strength alone** (from DLA) shows individual contributions
- **Correlation** reveals which components work together
- **High correlation** → components activate in sync → likely part of the same functional pathway
- **Correlation differences** between models → different functional organization

---

## Computational Requirements

### Attention Analysis
- **Memory**: ~6GB GPU (loading 2 models)
- **Time**: ~2 minutes per model (15 scenarios)
- **Total**: ~4 minutes

### Component Interaction Analysis
- **Memory**: ~6GB GPU (loading 2 models)
- **Time**: ~2 minutes per model (extracting activations) + ~10 seconds (computing correlations)
- **Total**: ~5 minutes

---

## Integration with Existing Results

### Complements DLA Analysis

**DLA shows:** L8_MLP and L9_MLP have similar strengths in both models

**Interaction Analysis can show:** L8_MLP connects to different downstream heads in De vs Ut

### Complements Patching Analysis

**Patching shows:** Directional asymmetry (component has different effect depending on patch direction)

**Attention Analysis can show:** Models attend to different contexts, so same component processes different information

---

## Example Hypotheses to Test

### Hypothesis 1: Opponent Action Attention
**Test:** Do De models show higher attention to "they played action2" tokens in CD_punished scenario?

**Prediction:** Yes - Deontological models encode betrayal norms, so they focus on opponent's defection

**Null result implication:** Deontological reasoning doesn't rely on explicit opponent tracking

### Hypothesis 2: Early-to-Late Pathway Differences
**Test:** Do De and Ut models show different correlation strengths for L8_MLP → L25H0 pathways?

**Prediction:** Yes - Different moral frameworks wire information flow differently

**Null result implication:** Information routing is universal, differences are in component-level tuning only

### Hypothesis 3: Late-Stage Decision Refinement
**Test:** Do L20-L25 components show different correlation patterns between De and Ut?

**Prediction:** Yes - Final decision layers implement different moral "override" mechanisms

**Null result implication:** Final decision is computed similarly, differences are in earlier layers

---

## Publication-Ready Framing

### For Paper

**Title:** "Similar Circuits, Different Connections: Attention and Interaction Analysis Reveals How Moral Fine-Tuning Rewires Information Flow"

**Key Claims:**
1. Deontological and Utilitarian models have nearly identical component-level representations (99.9999% similar)
2. Despite this similarity, they differ in:
   - **What information they attend to** (Attention Analysis)
   - **How components coordinate with each other** (Interaction Analysis)
3. This reveals fine-tuning operates by rewiring information flow, not by creating distinct feature detectors

**Figure Suggestions:**
- **Figure 1:** Attention comparison bar charts (3 panels: action keywords, opponent context, payoffs)
- **Figure 2:** Correlation matrix heatmaps (De and Ut side-by-side)
- **Figure 3:** Difference heatmap with annotated key pathways
- **Figure 4:** Key pathway analysis showing L8_MLP and L9_MLP connections

---

## Next Steps After Running Analyses

1. **Run both analyses:**
   ```bash
   python scripts/mech_interp/run_attention_analysis.py
   python scripts/mech_interp/run_component_interactions.py
   ```

2. **Examine results:**
   - Check `attention_comparison_De_vs_Ut.csv` for attention differences
   - Check `significant_pathways_De_vs_Ut.csv` for correlation differences
   - Look at visualizations in output directories

3. **Statistical testing:**
   - Run t-tests on attention differences (paired samples)
   - Compute effect sizes (Cohen's d)
   - Bonferroni correction for multiple comparisons

4. **Integrate with RQ2 analysis:**
   - Add findings to [RQ2_ANALYSIS_RESULTS.md](RQ2_ANALYSIS_RESULTS.md)
   - Update "How do similar circuits produce different behaviors?" section
   - Create final answer to RQ2

5. **Create publication figures:**
   - High-quality attention comparison plots
   - Annotated correlation difference heatmaps
   - Pathway diagram showing key connections

---

## References to Previous Work

- **DLA Analysis**: [mech_interp_outputs/dla_results/](mech_interp_outputs/dla_results/)
- **Patching Analysis**: [mech_interp_outputs/patching/](mech_interp_outputs/patching/)
- **RQ2 Comprehensive Analysis**: [RQ2_ANALYSIS_RESULTS.md](RQ2_ANALYSIS_RESULTS.md)
- **Project Summary**: [PROJECT_SUMMARY_FOR_PAPER_AUTHORS.md](PROJECT_SUMMARY_FOR_PAPER_AUTHORS.md)

---

## Questions These Analyses Answer

### Attention Analysis
1. Do Deontological models focus more on opponent's previous actions?
2. Do Utilitarian models focus more on joint payoff information?
3. Are attention patterns scenario-dependent (e.g., more opponent focus in CD_punished)?
4. Which layers show the largest attention differences?

### Component Interaction Analysis
1. Do L8_MLP and L9_MLP connect to different downstream components?
2. Are early-to-late pathways (L0-L9 → L20-L25) different between models?
3. Do late-stage components (L20-L25) show different coordination patterns?
4. Which component pairs show the largest correlation differences?

### Combined
5. Can we identify a "Deontological circuit" (specific attention patterns + specific pathways)?
6. Can we identify a "Utilitarian circuit" (different attention patterns + different pathways)?
7. Does fine-tuning operate by rewiring connections or by changing component strengths?

---

## Expected Outcomes

### Best Case (Strong Signal)
- Attention patterns differ significantly (p < 0.001)
- 50+ pathways have |correlation_diff| > 0.3
- Clear narrative: "De attends to opponent, wires L8_MLP → L25; Ut attends to payoffs, wires L9_MLP → L25"

### Moderate Signal
- Attention patterns show weak differences (p < 0.05)
- 10-20 pathways have |correlation_diff| > 0.3
- Narrative: "Subtle differences in information processing"

### Null Result
- No significant attention differences
- <5 pathways have |correlation_diff| > 0.3
- Conclusion: Differences operate at finer grain (neuron-level directions, residual stream geometry)
- Suggests need for even more detailed analysis (e.g., Distributed Alignment Search, Subspace Analysis)

---

## Fallback Analyses (If Null Results)

If both analyses show minimal differences, consider:

1. **Subspace Analysis**: Do De and Ut use different subspaces of the residual stream?
2. **Distributed Alignment Search (DAS)**: Are there distributed linear combinations that distinguish behaviors?
3. **Activation Clustering**: Do De and Ut cluster differently in activation space?
4. **Token-level Analysis**: Do specific tokens (e.g., "cooperate" vs "defect") activate different patterns?

---

## Code Organization

```
mech_interp/
├── attention_analysis.py           # NEW: Attention pattern analysis
├── component_interactions.py       # NEW: Component correlation analysis
├── direct_logit_attribution.py    # Existing: Component contribution decomposition
├── activation_patching.py          # Existing: Causal intervention
├── logit_lens.py                   # Existing: Layer-wise decision trajectory
├── model_loader.py                 # Model loading infrastructure
└── utils.py                        # Shared utilities

scripts/
├── run_attention_analysis.py       # NEW: Execute attention analysis
├── run_component_interactions.py   # NEW: Execute interaction analysis
├── run_dla.py                      # Existing: Execute DLA
├── run_patching.py                 # Existing: Execute patching
└── run_logit_lens.py              # Existing: Execute logit lens

mech_interp_outputs/
├── attention_analysis/             # NEW: Attention results
├── component_interactions/         # NEW: Interaction results
├── dla_results/                    # Existing: DLA results
├── patching/                       # Existing: Patching results
└── logit_lens/                     # Existing: Logit lens results
```

---

## Implementation Status

✅ **Completed:**
- Attention analysis module implementation
- Component interaction module implementation
- Execution scripts
- Documentation
- Module exports updated

⏳ **Ready to run:**
- `python scripts/mech_interp/run_attention_analysis.py`
- `python scripts/mech_interp/run_component_interactions.py`

📊 **Pending:**
- Execute analyses and collect results
- Statistical significance testing
- Integration with RQ2 analysis document
- Publication figure creation

---

## Contact

For questions about these analyses, see:
- Implementation: [mech_interp/README.md](mech_interp/README.md)
- Research context: [PROJECT_SUMMARY_FOR_PAPER_AUTHORS.md](PROJECT_SUMMARY_FOR_PAPER_AUTHORS.md)
- RQ2 findings: [RQ2_ANALYSIS_RESULTS.md](RQ2_ANALYSIS_RESULTS.md)
