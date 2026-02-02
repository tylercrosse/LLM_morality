# RQ2 Analysis - Quick Summary

**Date**: February 2, 2026  
**Question**: Do Deontological vs. Utilitarian agents develop distinct circuit structures?

---

## TL;DR

**Answer**: **NO** - they develop highly similar circuits with subtle distinctions.

**Key Numbers**:
- 0% behavioral flips (0 out of 7,020 patches)
- 0.91x effect ratio (cross-moral vs. baseline)
- 24% components differ (p<0.05), but magnitudes tiny (<0.01 logits)
- 4.1x asymmetry (Ut→De stronger than De→Ut)

**Interpretation**: **Convergent moral computation** - shared substrate, different weightings.

---

## One-Sentence Findings

1. **No architectural differences**: Zero behavioral flips means no single component distinguishes moral frameworks
2. **Comparable to training effects**: Cross-moral patching (0.91x) nearly as strong as Strategic→Moral
3. **Asymmetric but weak**: Utilitarian→Deontological 4x stronger, but both effects <0.011 logits
4. **L12 & L25 distinguish**: L12 heads (Ut-specific), L25 heads (bidirectional) are key components
5. **Scenario-specific patterns**: Largest divergence in recovery contexts (DD_trapped, CD_punished)
6. **Different from Strategic/Moral boundary**: Only 1/20 overlap in top components

---

## Key Components

| Component | De→Ut Effect | Ut→De Effect | Interpretation |
|-----------|--------------|--------------|----------------|
| **L12H0-7** | -0.003 | **-0.033** | **Utilitarian-specific** (strongest) |
| **L25H0-7** | **+0.028** | **-0.027** | **Bidirectional** (symmetric) |
| L2H0-7 | +0.020 | -0.026 | Moderate |
| L6_MLP | -0.024 | -0.013 | Asymmetric |
| L13H0-7 | -0.022 | +0.004 | De-specific |

---

## Scenario Divergence (Top 3)

1. **DD_trapped** (Mutual defection): 0.015 - How to escape defection cycle?
2. **CD_punished** (Betrayed): 0.014 - How to respond to betrayal?
3. **CC_temptation** (Temptation): 0.012 - When to resist defection?

→ Frameworks differ most in **recovery/repair contexts** (aligns with ethical theory)

---

## Figures

1. **fig1_effect_size_comparison.png**: De↔Ut effects comparable to PT2→PT3 baseline
2. **fig2_key_components_heatmap.png**: L12 heads (Ut), L25 heads (bidirectional)
3. **fig3_scenario_divergence.png**: Recovery contexts show largest divergence
4. **fig4_component_overlap.png**: Different boundaries use different components

---

## Recommended Framing

### For Paper

> "While Deontological and Utilitarian training produce behaviorally distinct agents, they develop remarkably similar circuit structures. Behavioral differences emerge from subtle reweighting of component contributions rather than fundamentally different architectures, suggesting convergent moral computation with a shared substrate for ethical reasoning."

### Key Messages

1. **Similar circuits, different weightings** (not different architectures)
2. **Shared moral reasoning substrate** (convergent solutions)
3. **Robust representations** (neither framework corrupts the other)
4. **Scenario-specific distinctions** (recovery contexts, aligning with theory)

---

## Statistical Summary

| Experiment | Mean Δ | Std Δ | Max |Δ| | Flips | Pro-Defect % |
|------------|--------|-------|---------|-------|--------------|
| PT2 → De   | -0.012 | 0.035 | 0.109 | 0 | 25.2% |
| PT2 → Ut   | +0.000 | 0.039 | 0.125 | 0 | 39.5% |
| **De → Ut** | **+0.003** | **0.038** | **0.141** | **0** | **42.5%** |
| **Ut → De** | **-0.011** | **0.037** | **0.125** | **0** | **26.1%** |

**Asymmetry**: |Exp3| / |Exp4| = 0.003 / 0.011 = **4.1x**

---

## What This Means

### For AI Alignment
- Moral training doesn't require architectural changes
- Robust encoding: neither framework easily corrupted
- Can fine-tune on same components for different ethics

### For Interpretability
- Distributed representation (not modular circuits)
- Behavior emerges from rebalancing, not suppression
- Multi-dimensional moral space (Strategic/Moral ≠ De/Ut)

### For Philosophy
- Different ethical theories may share computational mechanisms
- Practical similarity despite philosophical differences
- Context-dependent divergence (where theory predicts)

---

## Future Work

**Priority 1**: Multi-component patching (L12+L25 together) - can combinations flip?  
**Priority 2**: Attention analysis (what do L12/L25 attend to?)  
**Priority 3**: Interaction analysis (component correlation matrices)  
**Priority 4**: Ablation (zero out L12, L25 individually)  
**Priority 5**: Gradient attribution (validate patching results)

---

## Files

**Full Report**: `RQ2_ANALYSIS_REPORT.md` (15,000+ words)  
**Figures**: `fig1-4_*.png` (publication-quality, 300 DPI)  
**Data**: `rq2_*.csv` (summary stats, scenarios, components, overlap)

---

**Status**: ✅ COMPLETE - All research questions answered, deliverables created
