# Causal Routing Experiments: Quick Reference

**Date**: February 5, 2026
**Status**: Complete

## Hypothesis Support Table

| Experiment | Hypothesis | Result | Support |
|------------|-----------|--------|---------|
| **Frankenstein** | L2_MLP weights control routing | 1/4 tests supported (De→Ut: +71.31% coop) | ⚠️ Partial |
| **Activation Steering** | L2_MLP is routing switch | L2: +0.56%, L17: +29.58% (50x larger) | ❌ Rejected |
| **Activation Steering** | L16/L17 MLPs are routing hubs | L16: +26.17%, L17: +29.58% | ✅ Strong |
| **Path Patching** | L2→L9 pathway causally mediates behavior | De→St: +61.73% cooperation, 61.7x single-component | ✅ Very Strong |
| **Path Patching** | Attention > MLP pathways | Attention: 34.4%, MLP: 11.2% | ✅ Strong |

## Key Effect Sizes

- **Single-component patching**: 0% flips (21,060 patches)
- **L2_MLP steering**: +0.56% cooperation
- **L16_MLP steering**: +26.17% cooperation (46x baseline)
- **L17_MLP steering**: +29.58% cooperation (52x baseline)
- **Full path patching (L2→L9)**: +61.73% cooperation (61.7x single-component)

## Revised Mechanism

**Original**: L2_MLP is an early-layer routing switch
**Revised**: Distributed routing through late-layer hubs (L16/L17) operating via attention-mediated pathways

## Confidence Levels

- ✅ **Very High**: Pathway-level causality exists (path patching: +61.73%)
- ✅ **High**: L16/L17 are dominant routing hubs (steering: 50x L2_MLP)
- ⚠️ **Medium**: L2_MLP has some routing role (Frankenstein: 1/4 supported)
- ❌ **Low**: L2_MLP is THE routing switch (steering shows minimal effect)

## Experimental Details

### Experiment 1: Frankenstein (LoRA Weight Transplant)

**Duration**: ~45 minutes
**Method**: Transplanted L2_MLP LoRA weights (gate_proj, up_proj, down_proj) between models
**Test Cases**: 4 transplant combinations
**Success Rate**: 1/4 hypotheses supported

**Key Finding**: L2_MLP weights alone are insufficient for consistent behavioral control. One strong effect (De→Ut: +71.31%) suggests component has some role, but inconsistent results indicate need for broader context or different layer.

### Experiment 2: Activation Steering

**Duration**: ~45 minutes
**Method**: Computed steering vectors (moral - strategic), tested at multiple layers with strengths [-2.0 to +2.0]
**Layers Tested**: L2_MLP (original hypothesis), L16_MLP, L17_MLP, others
**Success Rate**: Strong evidence for L16/L17 hubs

**Key Finding**: Real routing switches are in deep layers (L16/L17), not early (L2). L17_MLP steering 52x more effective than L2_MLP steering. Suggests final-third layers where decisions finalize.

### Experiment 3: Path Patching

**Duration**: ~90 minutes
**Method**: Progressive replacement of residual stream activations (L2→L2, L2→L3, ..., L2→L9)
**Pathway Types**: Full residual, MLP-only, attention-only
**Success Rate**: Very strong support for pathway hypothesis

**Key Finding**:
- Saturation at L5 (critical window L2→L5)
- Attention pathways 3x more effective than MLP pathways
- Path effects 61.7x larger than single-component effects
- Confirms information flows through multi-layer attention-mediated pathways

## Files Generated

**Logs**: `mech_interp_outputs/causal_routing/logs/*.log` (3 files)
**CSVs**: `mech_interp_outputs/causal_routing/*.csv` (12 files)
**Figures**: `mech_interp_outputs/causal_routing/*.png` (8 files)
**Documentation**: `mech_interp_outputs/causal_routing/README.md`

## Integration with Prior Findings

### Correlational Evidence (Phase 1-2)
- Component interactions: 541 pathways differ
- Weight analysis: 99%+ weight similarity
- Linear probes: Identical representations
- Attention: 99.99% identical

### Causal Evidence (Phase 3)
- **Validates**: Network rewiring hypothesis (pathway-level causality proven)
- **Refines**: Location of routing switches (L16/L17, not L2)
- **Extends**: Mechanism understanding (attention-mediated, not MLP-only)

## Implications for Future Work

**For Targeted Fine-Tuning**:
- Focus interventions on deep layers (L16/L17)
- Prioritize attention mechanisms over MLPs
- Consider pathway-level modifications rather than single-component

**For Interpretability**:
- Causal interventions necessary to distinguish correlation from causation
- Steering experiments can rapidly identify functional hubs
- Path patching reveals information flow patterns

**For This Model/Task**:
- Moral behavior emerges from deep-layer routing (L16/L17)
- Attention pathways dominate information flow
- Distributed mechanism (no single switch)
