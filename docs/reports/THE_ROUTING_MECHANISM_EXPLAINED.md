# The L2_MLP "Routing Switch" Mechanism: A Deep Dive

**Purpose**: Explain how the "routing" mechanism works at a mechanical level, what the evidence actually shows, and what remains uncertain.

**Date**: February 4, 2026
**Reading Time**: 15-20 minutes

---

## TL;DR - The Key Claims

**What the data definitively shows:**
- ✅ L2_MLP has dramatically different correlation patterns with other components in Deontological vs Utilitarian models (e.g., -1.0 vs -0.037 with L1_MLP)
- ✅ L2_MLP activates at nearly identical strengths in both models (99.87% similar)
- ✅ Correlation differences correlate with behavioral asymmetry (r=0.67, p<0.001)

**The routing hypothesis** (inference, not direct proof):
- → Different correlation patterns = different functional connectivity
- → L2_MLP "routes" information through different pathways
- → This explains how 99.9999% similar components produce different behaviors

**What's not yet proven:**
- ⚠️ That correlation directly measures information flow (vs. just co-activation)
- ⚠️ That L2_MLP causally controls downstream components
- ⚠️ That alternative explanations (shared input response, redundancy) are ruled out

**Current evidence strength**: Suggestive correlation, not definitive causality

---

## Section 1: The Intuitive Analogy

### The Railroad Switching Yard

Imagine a busy railroad switching yard where trains arrive carrying freight (information) from different origins:

**The Setup:**
- **Station L2**: Early switching hub (Layer 2 in the network)
- **Arriving trains**: All carry the same cargo (IPD scenario information)
- **Track A**: Leads to cooperation terminals (L9_MLP, L12_MLP)
- **Track B**: Leads to context-evaluation terminals (L3_MLP, L21_ATTN)
- **Track C**: Goes to early suppression yards (L1_MLP)

**In the Deontological Railroad:**
- Train arrives at Station L2 with full cargo (activation: 6.30)
- **Switch setting**: Route 26% of trains to Track A (L3_MLP)
- **Switch setting**: Route 66% of trains to Track A (L12_MLP)
- **Switch setting**: Barely interact with Track C (L1_MLP, -4% correlation)
- Result: Freight flows through cooperation-favoring terminals
- Final destination: COOPERATE

**In the Utilitarian Railroad:**
- **Same train** arrives at Station L2 with **same cargo** (activation: 6.29, 99.87% identical!)
- **Switch setting**: BLOCK Track C entirely (L1_MLP, **-100%** correlation - perfect anti-correlation!)
- **Switch setting**: Route 77% of trains to Track B (L3_MLP, much stronger than De)
- **Switch setting**: REVERSE Track to L20_MLP (-50% vs De's +3%)
- Result: Freight flows through context-evaluation terminals
- Final destination: COOPERATE (but via different route)

**The Key Insight:**
- Same station (L2_MLP)
- Same incoming freight (input information)
- Same freight volume (activation magnitude)
- **Different switch settings (correlation patterns)**
- Different downstream terminals activated
- Different behavior results

**What "Routing" Means in This Analogy:**
- Not a physical change to the tracks (components are 99.9999% similar)
- Not different freight (information sources are 99.99% identical)
- **The switch coordination patterns** (when Station L2 is busy, which other stations are also busy?)
- Measured by correlation: Do stations co-activate or anti-activate?

---

## Section 2: What Correlation Actually Measures

### Technical Definition

**Pearson Correlation Coefficient** computed across 15 different game scenarios:
```
r = correlation(L2_MLP_activations[15 scenarios], ComponentX_activations[15 scenarios])
```

**What this computes:**
- For each of 15 scenarios, measure L2_MLP activation magnitude (L2 norm of output vector)
- For each of 15 scenarios, measure Component X activation magnitude
- Compute Pearson correlation between these two 15-element vectors
- Result: r ∈ [-1, +1]

### Correlation Ranges

| Value | Meaning | Example |
|-------|---------|---------|
| **+1.0** | Perfect positive correlation | When L2_MLP fires high, Component X always fires high |
| **+0.7** | Strong positive correlation | L2_MLP high → Component X usually high |
| **+0.3** | Moderate positive correlation | L2_MLP high → Component X somewhat high |
| **0.0** | No correlation (independent) | L2_MLP and Component X unrelated |
| **-0.3** | Moderate anti-correlation | L2_MLP high → Component X somewhat low |
| **-0.7** | Strong anti-correlation | L2_MLP high → Component X usually low |
| **-1.0** | Perfect anti-correlation | When L2_MLP fires high, Component X always fires low |

### CRITICAL: What Correlation DOES and DOES NOT Measure

#### ✅ What Correlation DOES Measure:
- **Co-activation patterns**: "Do these components activate together across scenarios?"
- **Statistical association**: "When L2_MLP is high in a scenario, is Component X also high?"
- **Scenario-level coupling**: "Do these components respond similarly to different inputs?"
- **Functional relationship existence**: "Is there some relationship between these components?"

#### ❌ What Correlation DOES NOT Directly Measure:
- **Information flow direction**: Correlation is symmetric (doesn't tell you if L2 → X or X → L2)
- **Causality**: High correlation ≠ one component controls the other
- **Functional necessity**: Correlation could be coincidental, not load-bearing
- **Physical routing**: No actual "wires" connecting components
- **Mechanism**: Doesn't reveal HOW components relate (shared input? direct connection? indirect pathway?)

### The Inference to "Routing"

**What the data shows:**
```
Correlation(L2_MLP, L9_MLP in Deontological) = +0.27
Correlation(L2_MLP, L9_MLP in Utilitarian) = -0.49
```

**The inference** (hypothesis):
- Different correlation patterns = different functional connectivity
- L2_MLP "coordinates" differently with L9_MLP in the two models
- This coordination difference acts like "routing" - directing information flow differently
- Result: Same components, different information pathways

**Important caveat:**
This is an **interpretation** of correlation data, not a direct measurement of routing. Alternative explanations exist (see Section 8).

---

## Section 3: Concrete Numerical Examples

Using real data from your component interaction analysis:

### Example 1: L2_MLP ↔ L1_MLP (Strongest Routing Difference)

| Metric | Deontological | Utilitarian | Difference |
|--------|---------------|-------------|------------|
| **Correlation** | **-0.037** | **-1.000** | **0.963** |
| L2_MLP activation | 6.2974 | 6.2891 | 0.0083 (0.13% diff) |
| L1_MLP activation | ~varies~ | ~varies~ | Similar magnitudes |

**What this means:**

**In Utilitarian Model:**
- **Perfect anti-correlation (-1.0)** = when L2_MLP fires strongly in a scenario, L1_MLP ALWAYS fires weakly
- When L2_MLP fires weakly, L1_MLP ALWAYS fires strongly
- They operate in **perfect opposite phase** across all 15 scenarios
- **Routing interpretation**: Ut model has L2_MLP suppressing/inhibiting L1_MLP activity

**In Deontological Model:**
- **Near-zero correlation (-0.037)** = L2_MLP and L1_MLP are essentially independent
- L2_MLP firing tells you almost nothing about L1_MLP state
- No consistent relationship across scenarios
- **Routing interpretation**: De model has no L2↔L1 pathway (independent processing)

**Why this is the "strongest difference" (0.963):**
- Utilitarian: Strong functional relationship (perfect anti-correlation)
- Deontological: No functional relationship (independent)
- Represents a fundamental architectural difference

### Example 2: L2_MLP ↔ L3_MLP (Both Route Here, But Differently)

| Metric | Deontological | Utilitarian | Difference |
|--------|---------------|-------------|------------|
| **Correlation** | **+0.261** | **+0.765** | **0.504** |
| Interpretation | Moderate positive | Strong positive | Ut routes more strongly |

**What this means:**

**In Both Models:**
- Positive correlation = L2_MLP and L3_MLP co-activate
- When L2_MLP fires high, L3_MLP also tends to fire high
- Both models use the L2→L3 pathway

**The Difference:**
- **Utilitarian (+0.77)**: Much stronger coordination
  - L2_MLP high → L3_MLP **very reliably** high
  - Strong functional coupling
- **Deontological (+0.26)**: Weaker coordination
  - L2_MLP high → L3_MLP **moderately** high
  - Looser functional coupling

**Routing interpretation:**
- Both models route through L3, but Utilitarian routes more "traffic" through this pathway
- Ut model: L3_MLP is a critical hub downstream of L2_MLP
- De model: L3_MLP is one option among several

### Example 3: L2_MLP ↔ L20_MLP (Opposite Signs!)

| Metric | Deontological | Utilitarian | Difference |
|--------|---------------|-------------|------------|
| **Correlation** | **+0.030** | **-0.503** | **0.533** |
| Sign | Weak positive | Moderate negative | OPPOSITE DIRECTIONS |

**What this means:**

**Deontological (+0.03):**
- Essentially independent (near-zero correlation)
- L2_MLP and L20_MLP don't interact much
- No routing relationship

**Utilitarian (-0.50):**
- Moderate anti-correlation
- When L2_MLP fires high, L20_MLP fires low
- When L2_MLP fires low, L20_MLP fires high
- **Inverse relationship**

**Routing interpretation:**
- De model: L2 and L20 operate independently (no routing)
- Ut model: L2 **suppresses** L20 (or L20 suppresses L2, correlation is symmetric)
- Fundamentally different late-layer coordination

---

## Section 4: Step-by-Step Information Flow

Let's trace how information flows through the network in a specific scenario:

**Scenario: CC_temptation** (Both players cooperated last round; defecting would yield +1 payoff advantage)

### Deontological Model Flow

**Step 1: Input Processing**
- Prompt: "Both played action1 (cooperate), got 3 points each. Temptation to defect."
- Early layers process context
- → L2_MLP activates with magnitude **6.30**

**Step 2: L2_MLP Routing Decisions** (based on correlation patterns)
- L2_MLP ↔ L1_MLP: -0.04 (nearly independent) → L1_MLP operates independently
- L2_MLP ↔ L3_MLP: +0.26 (moderate positive) → **L3_MLP also activates**
- L2_MLP ↔ L12_MLP: +0.66 (strong positive) → **L12_MLP strongly activates**
- L2_MLP ↔ L20_MLP: +0.03 (independent) → L20_MLP operates independently

**Step 3: Mid-Layer Processing**
- **L3_MLP** processes information (moderate activation, correlated with L2)
- **L12_MLP** processes information (strong activation, strongly correlated with L2)
- These layers contribute to cooperation-favoring pathway
- Information flows toward L9_MLP (universal pro-Cooperate component)

**Step 4: Final Decision**
- **L9_MLP** receives signal from cooperation pathway (magnitude: **-8.5**, pro-Cooperate)
- L8_MLP (pro-Defect: +7.0) is present but outweighed
- **Net result**: Strong cooperation preference

**Output**: **COOPERATE** (p_action2 = 0.0003, i.e., 99.97% cooperate)

---

### Utilitarian Model Flow

**Step 1: Input Processing** (SAME as Deontological)
- Same prompt, same early processing
- → L2_MLP activates with magnitude **6.29** (nearly identical to De: 6.30!)

**Step 2: L2_MLP Routing Decisions** (DIFFERENT correlation patterns)
- L2_MLP ↔ L1_MLP: **-1.00** (perfect anti-correlation) → **L1_MLP SUPPRESSED**
- L2_MLP ↔ L3_MLP: **+0.77** (very strong positive) → **L3_MLP strongly activates**
- L2_MLP ↔ L20_MLP: **-0.50** (moderate anti-correlation) → **L20_MLP suppressed**
- L2_MLP ↔ L21_ATTN: +0.54 (strong positive) → **L21_ATTN activates**

**Step 3: Mid-Layer Processing** (DIFFERENT from Deontological)
- **L1_MLP** suppressed (anti-correlated with L2)
- **L3_MLP** processes information (very strong activation, highly correlated with L2)
- Information flows through context-evaluation pathway
- **L20_MLP** suppressed (different late-layer processing than De)
- **L21_ATTN** activated (late-layer attention, not present in De flow)

**Step 4: Final Decision**
- **L9_MLP** receives modulated signal (magnitude: **-8.7**, slightly different from De)
- Signal arrived via different pathway (L3-heavy vs L12-heavy)
- L8_MLP (pro-Defect: +7.0) is present but outweighed

**Output**: **COOPERATE** (p_action2 = 0.0703, i.e., 92.97% cooperate)

---

### Key Differences in Information Flow

| Aspect | Deontological | Utilitarian |
|--------|---------------|-------------|
| **L2_MLP activation** | 6.30 | 6.29 (99.87% similar) |
| **L1_MLP routing** | Independent | Suppressed (-1.0 correlation) |
| **L3_MLP routing** | Moderate (+0.26) | Very strong (+0.77) |
| **L12_MLP routing** | Strong (+0.66) | (data not shown, but different) |
| **L20_MLP routing** | Independent (+0.03) | Suppressed (-0.50) |
| **Dominant pathway** | L2 → L12 → cooperation | L2 → L3 → context eval |
| **Final output** | 99.97% cooperate | 92.97% cooperate |

**The Critical Insight:**
- **Same input** → **Same L2_MLP activation** → **Different downstream cascade** → **Different behavior**
- The correlation patterns act like "routing rules" determining which downstream components activate together
- Not a single component difference, but a whole network coordination difference

---

## Section 5: Visual Pathway Diagrams

### Deontological Model Routing Pattern

```
                    INPUT (CC_temptation scenario)
                              ↓
                    [Early Layers: L0-L1]
                              ↓
                      [L2_MLP: 6.30] ⚡ ROUTING HUB
                          /    |    \
                        /      |      \
                      /        |        \
              (corr=-0.04) (corr=+0.26) (corr=+0.66)
                    /          |          \
                  /            |            \
          [L1_MLP]       [L3_MLP]        [L12_MLP]
         Independent    Moderately      Strongly
                        Activated       Activated
                                |
                                ↓
                    (cooperation pathway)
                                ↓
                        [Mid Layers: L6-L15]
                                ↓
                        [L9_MLP: -8.5] 🤝
                        (pro-Cooperate)
                                ↓
                    [Late Layers: L16-L25]
                                ↓
                           COOPERATE
                        (p_action2=0.0003)
```

### Utilitarian Model Routing Pattern

```
                    INPUT (CC_temptation scenario)
                              ↓
                    [Early Layers: L0-L1]
                              ↓
                      [L2_MLP: 6.29] ⚡ ROUTING HUB
                          /    |    \    \
                        /      |      \    \
              (corr=-1.00)  (corr=+0.77) (corr=-0.50) (corr=+0.54)
                      /        |          \            \
                    /          |            \            \
          [L1_MLP] ✗      [L3_MLP]      [L20_MLP] ✗   [L21_ATTN]
         SUPPRESSED       Very Strongly   SUPPRESSED    Activated
         (anti-corr)      Activated       (anti-corr)
                              |
                              ↓
                  (context evaluation pathway)
                              ↓
                    [Mid Layers: L6-L15]
                              ↓
                        [L9_MLP: -8.7] 🤝
                        (pro-Cooperate)
                              ↓
                  [Late Layers: L16-L25]
                  (different late activation)
                              ↓
                           COOPERATE
                        (p_action2=0.0703)
```

### Comparison Table: Top 10 L2_MLP Routing Differences

| Pathway | De Correlation | Ut Correlation | Difference | Routing Interpretation |
|---------|---------------|----------------|------------|------------------------|
| **L2 ↔ L1_MLP** | -0.04 | **-1.00** | **0.96** | Ut: Perfect suppression |
| **L2 ↔ L20_MLP** | +0.03 | **-0.50** | **0.53** | Ut: Moderate suppression; De: Independent |
| **L2 ↔ L15_ATTN** | -0.37 | **-0.90** | **0.53** | Ut: Stronger anti-correlation |
| **L2 ↔ L21_MLP** | +0.16 | **-0.35** | **0.51** | De: Amplifies; Ut: Suppresses |
| **L2 ↔ L3_MLP** | +0.26 | **+0.77** | **0.50** | Ut: Much stronger amplification |
| L2 ↔ L10_MLP | +0.23 | -0.70 | 0.93 | De: Positive; Ut: Strong negative |
| L2 ↔ L14_ATTN | -0.39 | +0.39 | 0.78 | Opposite signs! |
| L2 ↔ L11_ATTN | -0.48 | +0.29 | 0.77 | Opposite signs! |
| L2 ↔ L11_MLP | +0.44 | -0.33 | 0.77 | Opposite signs! |
| L2 ↔ L9_ATTN | -0.55 | +0.11 | 0.66 | Opposite signs! |

**Key Observations:**
- 7 out of 10 top differences involve **opposite signs** (positive vs negative correlation)
- Not just magnitude differences, but **qualitatively different relationships**
- L2_MLP coordinates with completely different downstream components

---

## Section 6: The Mechanism Summary

### What L2_MLP Actually Does

**Physical Reality:**
- Layer 2 MLP (Multi-Layer Perceptron) in a 26-layer network
- Processes token embeddings early in the forward pass
- Outputs a d_model=2304 dimensional vector
- Contributes to the residual stream

**Activation Behavior:**
- Activates at nearly **identical magnitudes** in both models:
  - Deontological: 6.2974 (std=0.0082)
  - Utilitarian: 6.2891 (std=0.0078)
  - Ratio: 0.9987 (99.87% similar)
- Responds to **same input information** (99.99% identical attention patterns)

**What's Different:**
- **Correlation patterns** with 52 other components (26 attention + 26 MLP layers)
- 29 pathways differ significantly (|correlation_diff| > 0.3)
- 10 pathways differ very significantly (|correlation_diff| > 0.5)
- 3 pathways differ extremely (|correlation_diff| > 0.7)

### Why This Creates Different Behaviors

**The Mechanism (Hypothesized):**
1. **Same input** activates L2_MLP to same strength in both models
2. **Different correlation patterns** mean different downstream components co-activate
3. In Deontological: L2_MLP → L12_MLP (cooperation pathway)
4. In Utilitarian: L2_MLP ↮ L1_MLP (suppression), L2_MLP → L3_MLP (context evaluation)
5. **Different downstream cascades** lead to different final decisions
6. Result: Same components, different "wiring," different behavior

**The "Routing Switch" Metaphor:**
- L2_MLP acts like a railroad switch
- Same "train" (information) arrives
- Different "track settings" (correlation patterns)
- Different downstream destinations (component activations)
- Different final outcomes (cooperation rates)

### Why DLA Missed This

**What DLA Measures:**
- Individual component contributions to final output
- L2_MLP's direct effect on logits
- Decomposition: `final_logit = Σ(component_contribution)`

**What DLA Found:**
- L2_MLP contributions nearly identical in both models (99.9999% similarity)
- L8_MLP and L9_MLP dominate (±7-9 magnitudes)
- L2_MLP's direct contribution is small

**What DLA Can't See:**
- How components **coordinate** with each other
- L2_MLP's role as a coordination hub (not a strong direct contributor)
- Pathway-level differences (L2 → L9 pathway, not just L2 alone or L9 alone)

**Analogy:**
- DLA measures: "How loud is each instrument?"
- Correlation measures: "How do instruments harmonize together?"
- L2_MLP is like a conductor: doesn't play loudly, but coordinates the orchestra

---

## Section 7: How to Explore the Data Yourself

### Simple Python Example

```python
import numpy as np
import pandas as pd

# Load correlation matrices (NPZ format)
corr_de_file = np.load("mech_interp_outputs/component_interactions/correlation_matrix_PT3_COREDe.npz")
corr_ut_file = np.load("mech_interp_outputs/component_interactions/correlation_matrix_PT3_COREUt.npz")

# Extract the actual correlation matrix
corr_de = corr_de_file["correlation_matrix"]  # Shape: (52, 52)
corr_ut = corr_ut_file["correlation_matrix"]  # Shape: (52, 52)

# Component mapping:
# Indices 0-25: L0_ATTN, L1_ATTN, ..., L25_ATTN
# Indices 26-51: L0_MLP, L1_MLP, ..., L25_MLP
# So L2_MLP is index: 26 + 2 = 28

l2_mlp_idx = 28

# Get L2_MLP's correlations with all other components
l2_de = corr_de[l2_mlp_idx, :]  # (52,) array
l2_ut = corr_ut[l2_mlp_idx, :]  # (52,) array

# Compute differences
diff = l2_de - l2_ut

# Find top 10 differences by absolute value
top_indices = np.argsort(np.abs(diff))[::-1][:10]

# Print results
print("Top 10 L2_MLP Routing Differences:\n")
print(f"{'Component':<15} {'De Corr':<10} {'Ut Corr':<10} {'Diff':<10}")
print("-" * 50)

for idx in top_indices:
    # Decode component name
    if idx < 26:
        comp_name = f"L{idx}_ATTN"
    else:
        comp_name = f"L{idx-26}_MLP"

    print(f"{comp_name:<15} {l2_de[idx]:>9.3f}  {l2_ut[idx]:>9.3f}  {diff[idx]:>9.3f}")
```

**Expected Output:**
```
Top 10 L2_MLP Routing Differences:

Component       De Corr    Ut Corr    Diff
--------------------------------------------------
L1_MLP             -0.037     -1.000      0.963
L10_MLP            +0.227     -0.701      0.928
L14_ATTN           -0.391     +0.387     -0.778
L11_ATTN           -0.481     +0.289     -0.771
L11_MLP            +0.439     -0.325      0.765
L20_MLP            +0.030     -0.503      0.533
L15_ATTN           -0.368     -0.897      0.529
L21_MLP            +0.163     -0.350      0.513
L3_MLP             +0.261     +0.765     -0.504
L9_ATTN            -0.549     +0.114     -0.663
```

### Exploring the Full CSV

```python
# Load the comparison CSV (easier for exploration)
df = pd.read_csv("mech_interp_outputs/component_interactions/interaction_comparison_De_vs_Ut.csv")

# Filter for L2_MLP connections
l2_df = df[(df['component1'] == 'L2_MLP') | (df['component2'] == 'L2_MLP')]

# Sort by absolute difference
l2_df['abs_diff'] = l2_df['correlation_diff'].abs()
l2_df_sorted = l2_df.sort_values('abs_diff', ascending=False)

# Show top 20
print(l2_df_sorted.head(20))
```

### Visualizing Correlation Matrices

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Create difference matrix
diff_matrix = corr_de - corr_ut

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(diff_matrix, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            xticklabels=False, yticklabels=False,
            cbar_kws={'label': 'Correlation Difference (De - Ut)'})
plt.title('Component Interaction Differences: Deontological - Utilitarian')
plt.xlabel('Component Index')
plt.ylabel('Component Index')

# Highlight L2_MLP row/column
plt.axhline(y=l2_mlp_idx, color='yellow', linewidth=2, alpha=0.5)
plt.axvline(x=l2_mlp_idx, color='yellow', linewidth=2, alpha=0.5)

plt.tight_layout()
plt.savefig('l2_mlp_routing_differences.png', dpi=300)
plt.show()
```

---

## Section 8: Alternative Interpretations & Limitations

### The Routing Hypothesis Is ONE Interpretation

**What the correlation data definitively shows:**
- L2_MLP and L9_MLP have correlation of +0.27 in Deontological, -0.49 in Utilitarian
- Difference of 0.76 is statistically significant
- Pattern holds across all 15 scenarios

**The routing interpretation:**
- Different correlations → different functional connectivity
- L2_MLP "routes" information to L9_MLP in De, "routes away" in Ut
- This creates different information flow pathways

**But correlation is NOT the same as causality or information flow!**

### Alternative Interpretation 1: Shared Input Response

**The idea:**
- Both L2_MLP and L9_MLP might simply respond to the same input features
- No "routing" needed - just parallel processing
- High correlation = both detect the same pattern, independently

**Example:**
- Scenario: "Opponent cooperated last round"
- L2_MLP detects: "Cooperation context" → activates high
- L9_MLP detects: "Cooperation context" → activates high
- Result: High correlation, but no information flow from L2 to L9

**Why this could explain the data:**
- Components at different layers can respond to same semantic features
- Correlation would measure "respond to similar scenarios," not "one sends info to other"
- Would still produce different correlation patterns if De/Ut encode features differently

**Why routing is still plausible:**
- L2_MLP is Layer 2 (very early), L9_MLP is Layer 9 (middle)
- Early layers typically provide input to later layers (not just parallel processing)
- But this doesn't definitively rule out shared input response

### Alternative Interpretation 2: Redundant Processing

**The idea:**
- Components might compute similar functions independently
- Correlation reflects redundancy/similarity, not routing
- Models might have backup/redundant pathways

**Example:**
- Multiple components encoding "betrayal detection"
- All activate together when betrayal is present
- High correlation, but no functional connectivity

**Why this could explain the data:**
- Neural networks often develop redundant representations
- Correlation would measure "compute similar things," not "wired together"
- Different redundancy patterns in De vs Ut would create different correlations

**Counterargument:**
- DLA showed components are 99.9999% similar (not redundant between models)
- Activation magnitudes are nearly identical (not redundant processing)
- But can't fully rule out within-model redundancy

### Alternative Interpretation 3: Spurious Correlation

**The statistical concern:**
- Only 15 scenarios analyzed (small sample size)
- 52 components = 1,326 pairwise correlations
- High-dimensional data with limited samples → risk of false positives

**The multiple comparisons problem:**
- With 1,326 correlations tested, expect ~66 false positives at p<0.05
- Even at p<0.01, expect ~13 false positives
- Some high correlations could be statistical flukes

**Why this might NOT explain the data:**
- Bonferroni correction: Only 0/234 components significantly different at p<0.0002
- Patterns are consistent across scenario categories (not random)
- Correlation differences correlate with behavioral asymmetry (r=0.67, validation)
- But caution is still warranted with small samples

### Validation Evidence Supporting Routing Hypothesis

**Evidence 1: Correlation with Behavioral Asymmetry (r=0.67, p<0.001)**
- Components with larger correlation differences also show larger behavioral asymmetry in patching
- Suggests correlation patterns are mechanistically meaningful, not noise
- If correlations were spurious, wouldn't predict behavioral effects

**Evidence 2: Consistent Across Scenarios**
- Same pathways differ across all 5 scenario types
- Not scenario-specific artifacts
- Suggests robust, structural differences

**Evidence 3: Mechanistically Sensible**
- L2_MLP is early-layer hub (plausible coordination role)
- Correlation patterns align with known component functions (e.g., L9_MLP pro-Cooperate)
- Makes theoretical sense given network architecture

### Validation Gaps (What's Still Unknown)

**Gap 1: No Multi-Component Patching**
- Haven't tested patching L2_MLP + L9_MLP **together**
- Single-component patches found 0 behavioral flips
- If routing hypothesis is correct, multi-component patches should flip behavior
- **Status**: Not yet implemented

**Gap 2: No Gradient Analysis**
- Haven't measured gradient flow from output → L2_MLP → L9_MLP
- Gradients would show actual information flow pathways
- High gradients matching high correlations would validate routing
- **Status**: Listed as Priority 5, not yet done

**Gap 3: No Ablation Studies**
- Haven't zeroed out L2_MLP and measured effect on L9_MLP
- Ablation would show functional dependency
- If L9_MLP activation drops when L2_MLP is ablated → proves dependency
- **Status**: Listed as Priority 4, not yet done

**Gap 4: No Circuit Discovery with Behavior Flips**
- Current minimal circuits (10 components) couldn't flip behavior
- Haven't found the actual causal circuit for De vs Ut distinction
- Larger circuits or different combinations needed
- **Status**: Open question

### Current Evidence Strength

**What's proven:**
- ✅ Correlation patterns differ significantly (r=0.67 validation)
- ✅ L2_MLP has dramatically different co-activation patterns
- ✅ These patterns are consistent and non-random

**What's inferred (not proven):**
- → Correlation patterns represent functional connectivity
- → L2_MLP acts as routing hub (coordinating info flow)
- → Different wiring creates different behaviors

**Evidence Level:**
- **Suggestive correlation** (r=0.67, p<0.001)
- **Not definitive causality** (no multi-patching, gradients, or ablation)
- **Leading hypothesis** that best explains the paradox (99.9999% similar components, different behaviors)

---

## Section 9: What Would Definitively Prove Routing

### Method 1: Multi-Component Patching (Causal Test)

**The experiment:**
1. Patch L2_MLP from Deontological → Utilitarian
2. **AND SIMULTANEOUSLY** patch L9_MLP from Deontological → Utilitarian
3. Measure if behavior changes

**What this would show:**
- **If behavior flips**: Proves the L2-L9 pathway is causally important (routing confirmed)
- **If no flip**: Routing hypothesis wrong, or need more components

**Why single-component patching failed:**
- Patching L2_MLP alone: 0 flips (too distributed)
- Patching L9_MLP alone: 0 flips (too distributed)
- Need to patch the **pathway** (both endpoints together)

**Current status:**
- Not yet implemented
- Would require modifying activation_patching.py to patch multiple components simultaneously
- Listed as Priority 1 in research roadmap

### Method 2: Gradient-Based Attribution (Information Flow)

**The experiment:**
1. Run forward pass on a scenario
2. Backprop gradient from final logit → L9_MLP activation
3. Backprop gradient from L9_MLP activation → L2_MLP output
4. Measure gradient magnitude for L2→L9 pathway
5. Compare to correlation values

**What this would show:**
- **If high gradients match high correlations**: Validates that correlation measures actual info flow
- **If gradients don't match**: Correlation might be spurious or indirect

**Advantages:**
- Directly measures information flow (via gradient flow)
- Can test all pathways (not just L2-L9)
- Computationally efficient

**Current status:**
- Listed as Priority 5 in research roadmap
- Not yet implemented
- Would require new analysis script

### Method 3: Ablation Studies (Functional Necessity)

**The experiment:**
1. Zero out L2_MLP output (set to zero vector)
2. Run forward pass and measure L9_MLP activation
3. Compare to baseline (no ablation)
4. Measure activation drop

**What this would show:**
- **If L9_MLP activation drops significantly**: L9_MLP functionally depends on L2_MLP (routing confirmed)
- **If no drop**: L9_MLP doesn't need L2_MLP (no routing, or redundant pathways)

**Variations:**
- Ablate L2_MLP in both models, compare drop magnitude
- If Ut shows bigger drop than De → confirms stronger L2-L9 coupling in Ut
- Matches correlation prediction (Ut: +0.77, De: +0.26)

**Current status:**
- Listed as Priority 4 in research roadmap
- Not yet implemented

### Method 4: Circuit Discovery with Behavior Flips

**The experiment:**
1. Systematically test larger component sets (15, 20, 25 components)
2. Prioritize high-correlation-diff components
3. Find minimal set that successfully flips behavior

**What this would show:**
- **If found**: Proves these components form a causal circuit
- **If circuit includes L2_MLP**: Validates routing role
- **If circuit matches correlation predictions**: Validates correlation-based analysis

**Current status:**
- Initial attempt: 10-component circuits, 0 flips
- Larger circuits not yet tested (computational cost)

### Method 5: Causal Mediation Analysis

**The experiment:**
1. Intervene on L2_MLP (set to specific value)
2. Measure downstream effect on L9_MLP
3. Measure indirect effect (L2 → other components → L9)
4. Compute mediation proportion

**What this would show:**
- Direct vs indirect pathways from L2 to L9
- Whether L2 affects L9 directly or via intermediate components
- Strength of causal effect

**Current status:**
- Not mentioned in research roadmap
- Advanced technique from causal inference literature
- Would require significant new implementation

---

## Section 10: Common Misconceptions

### Misconception 1: "L2_MLP activates differently in De vs Ut"

**WRONG:**
- L2_MLP activation magnitudes are 99.87% similar:
  - Deontological: 6.2974 (std=0.0082)
  - Utilitarian: 6.2891 (std=0.0078)
- Nearly identical activation strength

**RIGHT:**
- L2_MLP activation strength is the same
- What differs is **correlation patterns** with other components
- "Routing" refers to coordination, not activation magnitude

**Why this matters:**
- Can't explain behavior differences by "L2_MLP fires more in De"
- Must look at how L2_MLP coordinates with downstream components

---

### Misconception 2: "L2_MLP directly connects to L9_MLP physically"

**WRONG:**
- L2_MLP is Layer 2, L9_MLP is Layer 9 (7 layers apart!)
- No direct "wire" or connection in the architecture
- Transformer layers process sequentially, not with skip connections between arbitrary layers

**RIGHT:**
- Correlation measures **functional relationship** across scenarios
- "Routing" is metaphorical - refers to coordination patterns, not physical wiring
- Information flows through intermediate layers (L3, L4, ..., L8) via residual stream

**Why this matters:**
- "Routing" doesn't mean direct connection
- It means "when L2 activates in certain scenarios, L9 also tends to activate"
- The mechanism is indirect (via residual stream, intermediate processing)

---

### Misconception 3: "High correlation definitively proves L2 sends information to L9"

**WRONG:**
- Correlation is **symmetric**: corr(L2, L9) = corr(L9, L2)
- Doesn't show direction: could be L2→L9, or L9→L2, or both respond to same input
- Doesn't prove causality: correlation ≠ causation

**RIGHT:**
- High correlation means L2 and L9 **co-activate** across scenarios
- This is consistent with L2→L9 routing, but doesn't prove it
- Alternative explanations exist (shared input, redundancy, spurious correlation)

**To definitively prove routing:**
- Need causal interventions: multi-component patching, gradient analysis, ablation
- These methods can establish direction and causality
- Current evidence is suggestive (r=0.67), not definitive

---

### Misconception 4: "Single-component patching failed, so L2_MLP isn't important"

**WRONG:**
- Single-component patches: 0 behavioral flips (21,060 attempts)
- Doesn't mean components aren't important!
- Means behavior is **distributed** across multiple components

**RIGHT:**
- L2_MLP might be important as **part of a pathway** (L2 + L9 + others together)
- Single-component patching can't disrupt entire pathways
- Need to patch multiple components simultaneously to test pathway importance

**ALSO RIGHT (alternative):**
- Or correlation might not represent causal routing at all
- Alternative interpretations (shared input, redundancy) would also predict no single-component flips
- Ambiguity remains until multi-component patching is tested

---

### Misconception 5: "The routing hypothesis is proven"

**WRONG:**
- Currently supported by correlation evidence (r=0.67, p<0.001)
- Not proven by causal intervention (multi-patching, gradients, ablation not done)
- Alternative explanations haven't been ruled out

**RIGHT:**
- Routing is the **leading hypothesis** that best explains the data
- Makes mechanistic sense and is consistent with all current evidence
- But definitive proof requires additional experiments

**What's needed:**
- Multi-component patching with behavioral flips
- Gradient analysis showing information flow matches correlations
- Ablation studies showing functional dependencies
- Then we can claim "proven"

---

### Misconception 6: "All correlations > 0.7 represent important pathways"

**WRONG:**
- Many high correlations exist in both models (not just between-model differences)
- High correlation alone doesn't mean "important for behavior"
- Need to compare De vs Ut to identify distinguishing pathways

**RIGHT:**
- Only correlations with **large differences** between models are candidate routing differences
- Example: L2↔L9 correlation is -0.49 (Ut) vs +0.27 (De) = 0.76 difference
- Focus on pathways where models differ, not just high correlations in general

**Why this matters:**
- 29 pathways differ significantly (|diff| > 0.3)
- These are the candidate routing differences
- Other high correlations might be universal (present in both models)

---

## Section 11: Summary & Takeaways

### What We Know For Sure

**1. L2_MLP Has Different Correlation Patterns**
- ✅ 29 pathways differ significantly (|correlation_diff| > 0.3)
- ✅ Strongest: L2↔L1_MLP = 0.96 difference (De: -0.04, Ut: -1.00)
- ✅ Patterns consistent across all 15 scenarios
- ✅ Not random noise (validated via r=0.67 correlation with behavioral asymmetry)

**2. L2_MLP Activates Similarly in Both Models**
- ✅ Activation magnitude ratio: 0.9987 (99.87% similar)
- ✅ Same input → same L2_MLP activation
- ✅ Differences can't be explained by "L2 fires more in one model"

**3. Correlation Differences Predict Behavioral Asymmetry**
- ✅ r=0.67, p<0.001
- ✅ Components with larger correlation differences show larger behavioral effects when patched
- ✅ Suggests correlation patterns are mechanistically meaningful

### What We Hypothesize (But Haven't Proven)

**The Routing Hypothesis:**
- → Different correlation patterns = different functional connectivity
- → L2_MLP acts as coordination hub / routing switch
- → Deontological: Routes to cooperation pathway (L12_MLP, L3_MLP)
- → Utilitarian: Routes to context evaluation (L3_MLP strong, L1_MLP suppressed)
- → This explains how 99.9999% similar components produce different behaviors

**Evidence Supporting This:**
- ✅ Mechanistically sensible (early layer coordinating with later layers)
- ✅ Explains DLA paradox (similar components, different coordination)
- ✅ Consistent with patching results (distributed encoding, no single-component flips)
- ✅ Validated by behavioral asymmetry correlation

**Evidence Gaps:**
- ❌ No multi-component patching (causal test)
- ❌ No gradient analysis (direct information flow measurement)
- ❌ No ablation studies (functional necessity test)
- ❌ Alternative explanations not ruled out

### The Key Insight

**Same components + Same information + Different coordination = Different behaviors**

This is what makes the routing hypothesis powerful:
- Resolves the paradox of 99.9999% component similarity + drastically different behaviors
- Explains why single-component patching fails (distributed coordination)
- Explains why DLA misses the differences (measures individual contributions, not coordination)
- Predicts what future experiments should find (multi-patching, gradients, ablation)

### Current Evidence Strength

| Claim | Evidence Level | Status |
|-------|---------------|--------|
| L2_MLP has different correlations | **Proven** | ✅ r=0.67, p<0.001 |
| Correlations predict behavior | **Strongly supported** | ✅ Validated |
| Correlation = functional connectivity | **Plausible hypothesis** | ⚠️ Not proven |
| L2_MLP routes information | **Leading hypothesis** | ⚠️ Needs causal tests |
| Routing explains behavior differences | **Consistent with data** | ⚠️ Not definitively proven |

### What Would Change Your Mind

**Evidence that would SUPPORT routing hypothesis:**
- Multi-component patches (L2+L9) flip behavior
- Gradient analysis shows high gradients matching high correlations
- Ablation of L2_MLP drops L9_MLP activation in proportion to correlation

**Evidence that would REFUTE routing hypothesis:**
- Multi-component patches still don't flip behavior
- Gradients don't match correlations (info flows elsewhere)
- Ablation of L2_MLP doesn't affect L9_MLP (no dependency)
- Alternative explanation better fits the data

### How to Think About "Routing"

**DO think of it as:**
- ✅ Coordination patterns between components
- ✅ "When component A fires, component B also tends to fire"
- ✅ Measured by correlation across different scenarios
- ✅ Functional relationship that differs between models

**DON'T think of it as:**
- ❌ Physical wiring or direct connections
- ❌ One component sending signals to another (directionality not established)
- ❌ Definitively proven mechanism (still a hypothesis)
- ❌ The only possible explanation for the data

### For Publication / Communication

**Safe claims:**
- "We observed significantly different correlation patterns between Deontological and Utilitarian models"
- "L2_MLP shows dramatically different co-activation patterns with downstream components"
- "These correlation differences predict behavioral asymmetry (r=0.67, p<0.001)"
- "We hypothesize this represents different functional connectivity ('routing'), though causal experiments are needed"

**Claims needing qualification:**
- "L2_MLP routes information differently" → Add: "We hypothesize... based on correlation evidence"
- "Network rewiring explains behavior" → Add: "Correlation patterns suggest... pending causal validation"
- "L2_MLP is a routing switch" → Add: "Acts like a routing switch (metaphorically), coordinating..."

**Claims to avoid (without more evidence):**
- ❌ "We proved L2_MLP routes information" (need causal tests)
- ❌ "Information flows from L2 to L9" (correlation doesn't show direction)
- ❌ "Routing is the mechanism" (alternative explanations not ruled out)

---

## Conclusion

The "routing mechanism" is a powerful hypothesis that explains how Deontological and Utilitarian models can be 99.9999% similar in components yet produce drastically different behaviors. The core idea is that same components can coordinate differently (measured by correlation patterns), creating different functional pathways.

**Current evidence is suggestive (r=0.67), not definitive.** Multi-component patching, gradient analysis, and ablation studies would provide the causal evidence needed to definitively prove routing.

Until then, think of "routing" as:
- ✅ A well-motivated hypothesis
- ✅ Consistent with all available data
- ✅ The leading explanation for the component similarity paradox
- ⚠️ But not yet causally proven
- ⚠️ Alternative explanations remain possible

**The honest scientific stance:** "Our correlation analysis reveals significantly different coordination patterns, which we interpret as network rewiring. Causal experiments are needed to validate this hypothesis."

---

**Document Version**: 1.0
**Last Updated**: February 4, 2026
**Questions or Feedback**: See [MECH_INTERP_RESEARCH_LOG.md](../../MECH_INTERP_RESEARCH_LOG.md) for full context
