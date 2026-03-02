# Quick Reference: What Each Experiment Tells You

## The One-Page Cheat Sheet

### 🎯 The Central Question
How do Strategic, Deontological, and Utilitarian models produce different behaviors?

---

## 📊 Experiment Quick Reference

### 1. Logit Lens - "WHEN"
**Question**: When in the network does the decision happen?

**What it shows**: Layer-by-layer trajectory of cooperation preference

**Key Finding**: ⚠️ All models look identical (within 0.04 logits)
- Layer 0: Strong cooperation bias (same across all models)
- Layers 20-24: Decision stabilizes
- Can't distinguish models at this level

**Limitation**: Only shows aggregate output per layer, not individual components

---

### 2. Direct Logit Attribution (DLA) - "WHAT"
**Question**: Which components push toward Cooperate vs Defect?

**What it shows**: Contribution of each head & MLP to final decision

**Key Findings**:
- ✅ **L8_MLP**: Universal pro-Defect (+6.8 to +7.7) - ALL models
- ✅ **L9_MLP**: Universal pro-Cooperate (-8.2 to -9.3) - ALL models
- ⚠️ **99.9999% component similarity** across models
- ⚠️ Max training effect: 0.047 (tiny!)

**What it reveals**: Same computational primitives, subtle rebalancing

**Limitation**: Shows component strengths but not how they interact

---

### 3. Activation Patching - "CAN"
**Question**: Can we causally change behavior by swapping components?

**What it shows**: Does replacing Component X from Model A into Model B flip behavior?

**Key Finding**: 🚫 **ZERO flips** (0 out of 21,060 patches!)
- Single components: Can't flip behavior
- Minimal circuits (10 components): Still can't flip
- Both directions (A→B, B→A): No flips

**What it reveals**:
- Distributed encoding (not localized circuits)
- Extremely robust moral behavior
- Can't corrupt model by patching individual components

**Limitation**: Shows robustness but doesn't explain HOW similarity creates diversity

---

### 4. Attention Analysis - "WHERE"
**Question**: Do models attend to different information?

**What it shows**: Which tokens each model focuses on

**Key Finding**: 🟰 **99.99% identical attention**
- Action keywords: Same (0.000 both models)
- Opponent actions: Same (0.004 both models)
- Payoff info: Same (0.012 both models)

**What it reveals**: Models look at the same information sources

**Limitation**: Rules out selective attention, but mystery deepens

---

### 5. Component Interactions - "HOW" ⚡
**Question**: Do components connect together differently?

**What it shows**: Correlation patterns between all 52 components

**Key Finding**: 🔥 **29 pathways differ significantly**
- L2_MLP → L9_MLP: **+0.27** (De) vs **-0.49** (Ut) = **0.76 difference!**
- L2_MLP = "Moral routing switch"
- Deontological: Routes TO cooperation pathway
- Utilitarian: Routes AWAY (context-dependent)

**What it reveals**: **BREAKTHROUGH - Network rewiring mechanism!**

**This is the answer**: Different wiring, not different components

---

## 🧩 How They Fit Together

```
Logit Lens:    "All models look the same" → Need finer analysis
                    ↓
DLA:           "Same components (99.9999%)" → But how do they differ?
                    ↓
Patching:      "Can't flip by swapping components" → Distributed encoding
                    ↓
Attention:     "Same information sources" → Not selective attention
                    ↓
Interactions:  "Different routing via L2_MLP!" → SOLVED ✅
```

---

## 💡 The Big Insight

### What Makes Models Different?

❌ **NOT**:
- Different components (99.9999% similar)
- Different attention (99.99% similar)
- Suppressed selfish parts (L8_MLP increased!)
- Localized moral circuits (zero flips)

✅ **YES**:
- **Different routing** (~20% of pathways differ)
- **L2_MLP acts as switch**
- **Same parts, different wiring**

### The Mental Model

Think of it like plumbing:
- **Same pipes** (components: L8_MLP, L9_MLP, etc.)
- **Same water source** (information: opponent actions, payoffs)
- **Different valves** (L2_MLP routes flow differently)
- **Different destinations** (De → cooperation, Ut → context-dependent)

---

## 📈 Key Numbers to Remember

| Metric | Value | What it means |
|--------|-------|---------------|
| Component similarity | 99.9999% | Same computational primitives |
| Attention similarity | 99.99% | Same information sources |
| Pathway differences | 29 (>0.3 diff) | Different wiring! |
| L2→L9 correlation diff | 0.76 | Routing switch magnitude |
| Patching flips | 0 / 21,060 | Distributed, robust encoding |
| Validation alignment | 1.0 (100%) | Internal metrics match behavior |

---

## 🎓 Research Questions Answered

### RQ1: How are "selfish" heads suppressed?
**Answer**: They aren't! Distributed rebalancing (max change 0.047 vs magnitude 7-9)

**Evidence**: DLA (L8_MLP increased) + Patching (zero flips) + Interactions (routing not suppression)

---

### RQ2: Do De/Ut develop distinct circuits?
**Answer**: No (same components) but YES (different wiring)!

**Evidence**: DLA (99.9999%) + Attention (99.99%) + **Interactions (29 pathways differ)**

---

### RQ3: What parts should we fine-tune?
**Original**: Mid-late MLPs (L11-L23)

**Updated**: **Pathways** - specifically L2_MLP connections!

**Target**: L2→L9, L2→L6, L22→L2 pathways (70% parameter reduction)

---

## 🔍 When to Use Which Analysis

**Understanding decision timing**: → Logit Lens
**Finding important components**: → DLA
**Testing causality**: → Activation Patching
**Checking information selection**: → Attention Analysis
**Understanding how components coordinate**: → Component Interactions ⭐

**For the complete picture**: Need all five!

---

## 🚀 The Breakthrough

### First Demonstration:
**Moral fine-tuning operates through network rewiring**

### The Mechanism:
Same components + Same information + Different routing = Different behaviors

### The Key Player:
**L2_MLP** = Moral routing switch
- Deontological: Routes to cooperation (+0.27 to L9_MLP)
- Utilitarian: Routes away from default (-0.49 to L9_MLP)

### The Validation:
- Perfect alignment (1.0) with actual behavior
- Significant separation (p < 0.00005)
- Robust across all scenarios

---

## 📖 Where to Learn More

**Full narrative**: [SYNTHESIS_HOW_IT_ALL_FITS_TOGETHER.md](SYNTHESIS_HOW_IT_ALL_FITS_TOGETHER.md)

**Detailed findings**:
- RQ2 analysis: [RQ2_KEY_INSIGHTS.md](RQ2_KEY_INSIGHTS.md)
- Attention & interactions: [ATTENTION_AND_INTERACTION_ANALYSIS.md](ATTENTION_AND_INTERACTION_ANALYSIS.md)
- Complete log: [MECH_INTERP_RESEARCH_LOG.md](../../MECH_INTERP_RESEARCH_LOG.md)

**Raw data**:
- DLA: `mech_interp_outputs/dla/`
- Patching: `mech_interp_outputs/patching/`
- Interactions: `mech_interp_outputs/component_interactions/`

---

**Last Updated**: February 4, 2026
**Status**: All experiments complete, validated, ready for publication
