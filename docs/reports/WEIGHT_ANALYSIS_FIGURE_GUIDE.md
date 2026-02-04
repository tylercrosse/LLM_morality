# Weight Analysis Figure Interpretation Guide

This guide explains the visualizations generated from LoRA adapter weight analysis, including how data was collected, what each plot represents, and what patterns to look for.

---

## Data Collection Overview

All figures are based on analyzing the LoRA (Low-Rank Adaptation) adapter weights from four fine-tuned Gemma-2-2b models:
- **PT2_COREDe** (Strategic): Trained on strategic IPD play
- **PT3_COREDe** (Deontological): Trained on deontological moral reasoning
- **PT3_COREUt** (Utilitarian): Trained on utilitarian moral reasoning
- **PT4_COREDe** (Hybrid): Combined training

### How LoRA Adapters Work
LoRA fine-tuning modifies a base model by inserting small "adapter" weight matrices into each layer. For each module (like attention query projection or MLP gate), LoRA adds two low-rank matrices:
- **A matrix**: Shape (rank=64, input_dim)
- **B matrix**: Shape (output_dim, rank=64)
- **Effective update**: ΔW = (lora_alpha / rank) × B @ A

The adapter is applied as: `output = base_weights(x) + (α/r) × B @ A @ x`

### Measurement: Frobenius Norm
The **Frobenius norm** `||B @ A||_F` quantifies the magnitude of weight change:
- Formula: `sqrt(sum of all squared elements in B @ A)`
- Interpretation: How much this component's behavior was modified during fine-tuning
- High norm → Component was heavily retrained
- Low norm → Component changed minimally (may be passive responder)

### Architecture Details
- **26 layers** (L0 to L25)
- **7 modules per layer**:
  - **MLP modules**: gate_proj, up_proj, down_proj (transform information)
  - **Attention modules**: q_proj, k_proj, v_proj, o_proj (route information)
- **Total**: 182 modules per model (26 × 7)

---

## Per-Model Visualizations

### 1. `norm_heatmap_{model_id}_normalized.png`

**Data Collected:**
- Frobenius norm `||B @ A||_F` for each of 182 modules
- Normalized by `sqrt(output_dim)` to enable fair comparison across module types
- Organized into 26×7 matrix (layers × module types)

**What the Plot Shows:**
- **Rows**: 26 layers (L0 at top, L25 at bottom)
- **Columns**: 7 module types (Gate, Up, Down, Q, K, V, O)
- **Color intensity**: Magnitude of weight change (yellow = low, red = high)
- **Heatmap values**: Normalized Frobenius norms

**What to Look For:**

1. **Hot spots (red regions)**: Components that were heavily modified
   - These are the "retrained" components driving behavioral change
   - Example: If L2 MLP columns are bright red, L2_MLP was explicitly modified

2. **Vertical patterns**:
   - If an entire column is hot → That module type was heavily targeted across all layers
   - If a column is cool → That module type remained largely unchanged

3. **Horizontal patterns**:
   - If an entire row is hot → That layer was comprehensively retrained
   - If a row is cool → That layer minimally changed

4. **MLP vs Attention**:
   - First 3 columns (Gate, Up, Down) = MLP
   - Last 4 columns (Q, K, V, O) = Attention
   - Compare intensities: Which component type was modified more?

5. **Layer depth distribution**:
   - Are early layers (L0-L8) hotter → Task-specific features learned early
   - Are middle layers (L9-L17) hotter → Abstract reasoning modified
   - Are late layers (L18-L25) hotter → Output formatting changed

**Interpretation Examples:**
- **Uniform red**: Comprehensive retraining across all components
- **Sparse red patches**: Targeted modification of specific circuits
- **Cool heatmap overall**: Minimal adaptation (base model largely preserved)

---

### 2. `norm_heatmap_{model_id}_raw.png`

**Data Collected:**
- Same as normalized version, but using raw (unnormalized) Frobenius norms
- No division by `sqrt(output_dim)`

**What the Plot Shows:**
- Same 26×7 layout as normalized version
- Raw magnitude of `||B @ A||_F` without normalization

**What to Look For:**

1. **Absolute magnitude differences**:
   - MLP columns (especially Up/Gate with larger output_dim=9216) will appear hotter than attention columns
   - This reflects both the importance of the module AND its dimensionality

2. **Cross-column comparison is less fair**:
   - Use this plot to see absolute scales
   - Use normalized version for fair cross-module comparison

3. **Practical impact**:
   - Larger raw norms → More parameters changed → Potentially greater computational impact
   - Compare to normalized to distinguish "big because important" vs "big because high-dimensional"

**When to Use This vs Normalized:**
- **Normalized**: Fair comparison across module types (e.g., "Is MLP more modified than Attention?")
- **Raw**: Understanding absolute parameter change magnitude (e.g., "Which has more total parameter drift?")

---

### 3. `top_components_{model_id}.png`

**Data Collected:**
- Frobenius norm for all 182 components
- Sorted in descending order
- Top 30 components selected

**What the Plot Shows:**
- **Horizontal bar chart**
- **Y-axis**: Component names (e.g., L15_MLP_UP, L8_ATTN_Q)
- **X-axis**: Frobenius norm magnitude
- **Colors**:
  - **Red bars**: L2_MLP modules (gate, up, down) if present in top 30
  - **Model color**: All other components

**What to Look For:**

1. **Is L2_MLP in the top 30?**
   - Red bars present → L2_MLP was heavily modified (supports "switching" hypothesis)
   - No red bars → L2_MLP not in top 30 (refutes "switching" hypothesis)
   - Check position: Top 10? Top 20? This indicates strength of modification

2. **Which layers dominate?**
   - Count occurrences: Are top components concentrated in specific layers?
   - Example: Many L10-L15 components → Middle layers were main target
   - Dispersed across layers → Broad architectural changes

3. **MLP vs Attention ratio**:
   - Count MLP components (names containing MLP, GATE, UP, DOWN)
   - Count Attention components (names containing ATTN, Q, K, V, O)
   - Ratio indicates which subsystem was modified more

4. **Magnitude distribution**:
   - Steep drop-off → Few dominant components (sparse circuit modification)
   - Gradual decline → Many components modified similarly (distributed change)

5. **Specific layer patterns**:
   - If early layers dominate → Input processing changed
   - If late layers dominate → Output generation changed
   - If middle layers dominate → Core reasoning modified

**Example Interpretation:**
- "Top 30 includes L2_MLP_GATE (#18) but other L2 MLPs (#35, #42) → L2_MLP moderately modified, not dominant driver"
- "No L2_MLP in top 30 → L2_MLP changes likely passive responses to upstream modifications"

---

## Cross-Model Comparisons

### 4. `l2_mlp_comparison.png`

**Data Collected:**
- For each model: Sum of Frobenius norms for L2's three MLP modules (gate_proj + up_proj + down_proj)
- Total L2_MLP modification magnitude per model

**What the Plot Shows:**
- **Bar chart** with 4 bars (one per model)
- **Y-axis**: Total Frobenius norm for L2_MLP
- **X-axis**: Model names (Strategic, Deontological, Utilitarian, Hybrid)
- **Colors**: Model-specific colors

**What to Look For:**

1. **Relative magnitudes**:
   - Which model modified L2_MLP the most?
   - Which modified it the least?
   - How large are the differences? (2× difference? 10× difference?)

2. **Hypothesis testing**:
   - If Deontological/Utilitarian have high L2_MLP norms → Moral reasoning required L2 modification
   - If Strategic has high L2_MLP norm → L2 is domain-general, not morality-specific
   - If all norms are similar → L2_MLP modification is consistent across training regimes

3. **Absolute vs relative context**:
   - Compare to other layers' norms (from per-model heatmaps)
   - Is L2_MLP exceptionally high for ANY model?
   - Or is it low for all models?

**Example Interpretation:**
- "Utilitarian L2_MLP norm is 2.5× higher than Strategic → Utilitarian training specifically targeted L2"
- "All models have similar L2_MLP norms → L2 modification is general consequence of IPD fine-tuning, not moral-specific"

---

### 5. `layer_profile_mlp.png`

**Data Collected:**
- For each model and each layer: Sum of Frobenius norms for all MLP modules (gate + up + down)
- Creates layer-wise profile (26 data points per model)

**What the Plot Shows:**
- **Line plot** with 4 overlaid lines (one per model)
- **X-axis**: Layer index (0-25)
- **Y-axis**: Total MLP Frobenius norm for that layer
- **Lines**: Each model's layer-wise MLP modification profile

**What to Look For:**

1. **Peak layers**:
   - Where does each model have its maximum MLP norm?
   - Example: Deontological peaks at L12 → Core moral reasoning happens mid-network
   - Example: Strategic peaks at L5 → Early strategic planning dominant

2. **Profile shapes**:
   - **Front-loaded** (early peak) → Input processing and feature extraction modified
   - **Middle peak** → Abstract reasoning and concept manipulation modified
   - **Back-loaded** (late peak) → Output formatting and decision-making modified
   - **Flat profile** → Uniform modification across depth (rare)

3. **Cross-model divergence**:
   - Where do lines diverge the most?
   - Example: All models similar L0-L8, diverge L9-L15 → Moral differences emerge mid-network
   - Consistent divergence across depth → Fundamentally different architectures

4. **Overall magnitude**:
   - Which model's line is highest overall?
   - This model underwent the most total MLP modification
   - Relates to training intensity or task difficulty

5. **Layer-specific differences**:
   - Point-by-point comparison: At L10, is Utilitarian >> Deontological?
   - These specific layers may encode distinctive moral representations

**Example Interpretation:**
- "All models peak at L11-L13 with Utilitarian 40% higher → Mid-layers critical for IPD, utilitarian reasoning requires stronger modification"
- "Strategic shows early peak (L3-L5), moral models show middle peak (L10-L15) → Different processing strategies"

---

### 6. `layer_profile_self_attn.png`

**Data Collected:**
- For each model and each layer: Sum of Frobenius norms for all attention modules (q + k + v + o)
- Creates layer-wise attention modification profile

**What the Plot Shows:**
- Same format as MLP profile
- **Lines**: Each model's layer-wise attention modification profile

**What to Look For:**

1. **Attention vs MLP comparison**:
   - Compare magnitudes to MLP profile (previous figure)
   - If attention norms much lower → MLP was main target (information transformation)
   - If attention norms similar/higher → Information routing also modified

2. **Where is attention modified?**
   - Early layers: Token-level attention patterns changed
   - Middle layers: Abstract attention patterns changed
   - Late layers: Decision-relevant attention modified

3. **Model-specific attention patterns**:
   - Do different models modify attention differently?
   - Example: Deontological modifies late attention (attending to moral cues)
   - Example: Utilitarian modifies early attention (attending to payoff information)

4. **Convergence vs divergence**:
   - If lines are close together → All models changed attention similarly
   - If lines diverge → Different attentional strategies learned

**Example Interpretation:**
- "Attention norms 60% lower than MLP → MLPs were primary target, attention routing largely preserved"
- "Deontological shows peak attention modification at L18-L20 → Late attention crucial for deontological judgment"

---

## Adapter Comparison Visualizations

### 7. `adapter_similarity_heatmap.png`

**Data Collected:**
- For each model: 182-dimensional vector of component norms (one value per component)
- **Cosine similarity** computed between all model pairs
- Formula: `cosine(A, B) = (A · B) / (||A|| × ||B||)`
- Range: 0 (opposite directions) to 1 (identical directions)

**What the Plot Shows:**
- **4×4 heatmap** (symmetric matrix)
- **Both axes**: The 4 models (Strategic, Deontological, Utilitarian, Hybrid)
- **Diagonal**: Always 1.0 (model is identical to itself)
- **Off-diagonal**: Cosine similarity between model pairs
- **Color**: Yellow (low similarity ~0.99) to brown (high similarity ~1.0)

**What to Look For:**

1. **Overall similarity range**:
   - Are all values very high (>0.99)? → Models are structurally very similar (same base model, similar fine-tuning)
   - Are some values low (<0.95)? → Models diverged substantially

2. **Most similar pair**:
   - Which two models have highest cosine (darkest off-diagonal cell)?
   - These models modified components in the most similar pattern
   - Example: Strategic-Utilitarian = 0.9969 → Very similar modification patterns

3. **Most divergent pair**:
   - Which two models have lowest cosine (lightest off-diagonal cell)?
   - These models took the most different approaches
   - Example: Utilitarian-Hybrid = 0.9933 → Most distinct modification strategies

4. **Clustering patterns**:
   - Do moral models (Deontological, Utilitarian) cluster separately from Strategic?
   - Does Hybrid resemble a specific moral model more?

5. **Interpretation of small differences**:
   - Even 0.9933 vs 0.9969 (0.36% difference) can be meaningful
   - In high-dimensional spaces (182 components), small cosine differences indicate significant divergence
   - Focus on *relative* differences, not absolute values

**What Cosine Similarity Measures:**
- **Directional alignment**: Are components modified in the same proportions?
- **Does NOT measure**: Absolute magnitude (a model can be 10× stronger but still have cosine=1.0 if proportions match)
- **Use case**: Understanding if models use similar or different modification strategies

**Example Interpretation:**
- "Strategic-Utilitarian highest similarity (0.9969) → Utilitarian training didn't fundamentally restructure from strategic baseline"
- "Utilitarian-Hybrid lowest similarity (0.9933) → Hybrid's mixed training created unique modification pattern"

---

### 8. `adapter_distance_heatmap.png`

**Data Collected:**
- Same 182-dimensional component norm vectors as similarity heatmap
- **L2 distance** (Euclidean distance) computed between all model pairs
- Formula: `L2(A, B) = ||A - B|| = sqrt(sum((A_i - B_i)²))`
- Range: 0 (identical) to large positive values (very different)

**What the Plot Shows:**
- **4×4 heatmap** (symmetric matrix)
- **Both axes**: The 4 models
- **Diagonal**: Always 0 (model has zero distance to itself)
- **Off-diagonal**: L2 distance between model pairs
- **Color**: White (small distance ~0.1) to dark blue (large distance ~0.5)

**What to Look For:**

1. **Absolute magnitude scale**:
   - What is the range? (0.1 to 0.5? 0.01 to 0.1?)
   - Larger values → More total parameter divergence
   - Compare to typical weight magnitudes to assess if differences are large

2. **Most similar pair (smallest L2)**:
   - Which models are closest in Euclidean space?
   - Example: Strategic-Deontological = 0.1152 → Minimal total parameter difference
   - These models have similar overall modification magnitudes

3. **Most distant pair (largest L2)**:
   - Which models are farthest apart?
   - Example: Strategic-Hybrid = 0.5356 → Largest total parameter difference
   - These models underwent the most different amounts of modification

4. **Comparison to cosine similarity**:
   - **Cosine measures direction** (modification pattern)
   - **L2 measures magnitude** (how much was modified)
   - Example: Two models can have high cosine (similar pattern) but large L2 (one much stronger)

5. **Clustering by magnitude**:
   - Do certain models cluster as "heavily modified" vs "lightly modified"?
   - Can reveal training intensity differences

**What L2 Distance Measures:**
- **Total divergence**: How different are the absolute magnitudes?
- **Includes scale**: A model that modified everything 2× more will have large L2
- **Use case**: Understanding which models underwent similar amounts of total change

**Example Interpretation:**
- "Strategic-Deontological smallest L2 (0.12) → Similar total modification despite different objectives"
- "Strategic-Hybrid largest L2 (0.54) → Hybrid training induced much stronger overall modifications"
- "High cosine + high L2 → Same modification pattern but different intensity"
- "Low cosine + low L2 → Different patterns but similar total change"

**Cosine vs L2 Joint Interpretation:**
| Cosine | L2 | Meaning |
|--------|-----|---------|
| High | Low | Very similar: same pattern, same magnitude |
| High | High | Same pattern, different intensity (one much stronger) |
| Low | Low | Different patterns, similar total change |
| Low | High | Different patterns, very different magnitudes |

---

### 9. `adapter_top_variable_components_heatmap.png`

**Data Collected:**
- For each of 182 components: Standard deviation across 4 models
- Identifies components with highest variability (most different across models)
- Top 30 most variable components selected
- Each component Z-scored: (value - mean) / std across models

**What the Plot Shows:**
- **Heatmap**: 30 rows (components) × 4 columns (models)
- **Rows**: Component names (e.g., L12_MLP_GATE, L15_ATTN_Q)
- **Columns**: The 4 models
- **Colors**: Red-white-blue diverging scale
  - **Red**: This model's norm is *below average* for this component (negative Z-score)
  - **White**: At average (Z ≈ 0)
  - **Blue**: This model's norm is *above average* for this component (positive Z-score)

**What to Look For:**

1. **Model-specific hotspots (blue columns for specific model)**:
   - If Utilitarian shows many blue cells → These components are Utilitarian-specific
   - If Strategic shows many red cells → These components were under-modified in Strategic
   - Identifies distinctive architectural changes per model

2. **Component-specific patterns (row patterns)**:
   - **All blue except one red**: Three models agree, one is outlier
   - **Two blue, two red**: Models split into two camps (e.g., moral vs strategic)
   - **Gradient pattern**: Continuous variation (e.g., Strategic < Deon < Util < Hybrid)

3. **Layer clustering**:
   - Are top variable components concentrated in specific layers?
   - Example: Many L10-L15 components → Mid-layer modifications are model-distinctive
   - Example: Mixed across all layers → Divergence is architectural, not layer-specific

4. **MLP vs Attention divergence**:
   - Count MLP vs Attention in top 30
   - If mostly MLP → Models differ in information transformation
   - If mostly Attention → Models differ in information routing

5. **Hybrid model position**:
   - Does Hybrid show intermediate values (white/pink)?
   - Or does it show extreme values (dark blue/red)?
   - Intermediate → Hybrid averages other models
   - Extreme → Hybrid developed unique features

**What Z-Score Normalization Does:**
- Removes absolute magnitude differences
- Highlights *relative* differences in modification patterns
- A component can have small absolute norm but still be "variable" if models differ proportionally

**Example Interpretation:**
- "L12_MLP_DOWN: Utilitarian deep blue (+2.3σ), others red → Utilitarian uniquely modifies L12 MLP"
- "Top 30 includes 8 components from L15-L18 → Late-middle layers are model-distinctive"
- "Attention components absent from top 30 → All models modified attention similarly, divergence is MLP-specific"

---

### 10. `adapter_layerwise_mlp_delta_vs_strategic.png`

**Data Collected:**
- For each model and layer: Total MLP norm (gate + up + down)
- **Delta computed**: MLP_norm(model) - MLP_norm(Strategic)
- Strategic used as baseline (delta = 0 by definition)

**What the Plot Shows:**
- **Line plot**: 3 lines (Deontological, Utilitarian, Hybrid)
- **X-axis**: Layer index (0-25)
- **Y-axis**: Delta MLP norm (positive = more modified than Strategic, negative = less modified)
- **Black horizontal line**: y=0 (Strategic baseline)

**What to Look For:**

1. **Above vs below baseline**:
   - **Positive delta**: This model modified MLP more than Strategic at this layer
   - **Negative delta**: This model modified MLP less than Strategic (rare, usually positive)
   - Most lines should be positive (moral training adds to strategic training)

2. **Peak divergence layers**:
   - Where is delta maximized?
   - Example: Utilitarian peaks at L14 (+0.08) → L14 MLP is where utilitarian reasoning diverges most from strategic
   - These layers encode model-distinctive features

3. **Convergence points**:
   - Where do lines cross y=0? (Delta ≈ 0)
   - These layers were modified similarly across all training regimes
   - Example: All models cross at L0-L2 → Early layers uniformly adapted

4. **Relative ordering**:
   - Which model is consistently highest/lowest?
   - Example: Hybrid always highest → Hybrid training induced strongest modifications
   - Example: Deontological lowest → Deontological training was most conservative

5. **Layer-specific effects**:
   - **Early layers (L0-L8)**:
     - Small deltas → Strategic training already optimized input processing
     - Large deltas → Moral training changed feature extraction
   - **Middle layers (L9-L17)**:
     - Large deltas expected → Core reasoning differs
     - Peak divergence layers reveal where moral concepts emerge
   - **Late layers (L18-L25)**:
     - Small deltas → Decision output format similar across models
     - Large deltas → Different decision-making mechanisms

6. **Profile shapes**:
   - **Single peak**: Divergence concentrated in specific layers
   - **Multiple peaks**: Multiple sites of moral specialization
   - **Plateau**: Broad, distributed divergence

**Example Interpretation:**
- "Utilitarian shows peak delta at L12-L15 (~+0.06), Deontological peaks earlier L9-L11 → Different moral reasoning depths"
- "All models converge near zero at L0-L3 and L22-L25 → Divergence is mid-network only"
- "Hybrid delta 2× larger than moral models → Combined training amplified modifications"

---

### 11. `adapter_mlp_vs_attn_summary.png`

**Data Collected:**
- For each model:
  - Mean MLP norm (average across all MLP modules: 26 layers × 3 modules = 78 values)
  - Mean Attention norm (average across all Attention modules: 26 layers × 4 modules = 104 values)

**What the Plot Shows:**
- **Grouped bar chart**: 4 pairs of bars (one pair per model)
- **Each pair**:
  - **Blue bar**: Mean MLP norm
  - **Pink bar**: Mean Attention norm
- **Y-axis**: Mean Frobenius norm
- **X-axis**: Model names

**What to Look For:**

1. **MLP vs Attention magnitude (within each model)**:
   - **Blue > Pink**: MLP was modified more than Attention
     - Interpretation: Training primarily changed information transformation, not routing
     - Suggests task requires new computations, not new attention patterns
   - **Pink > Blue**: Attention was modified more than MLP (rare)
     - Interpretation: Training changed what information to attend to
     - Suggests task requires new information routing strategies
   - **Blue ≈ Pink**: Balanced modification
     - Both systems equally important

2. **Across-model MLP comparison**:
   - Which model has the highest blue bar?
   - This model underwent strongest MLP modification
   - Compare to training objectives: Does Utilitarian have higher MLP norm because payoff calculation requires stronger transformations?

3. **Across-model Attention comparison**:
   - Which model has the highest pink bar?
   - This model changed attention patterns most
   - Compare to expected mechanisms: Does Deontological have higher attention because rule-following requires attending to specific constraint tokens?

4. **Ratio consistency**:
   - Is MLP/Attention ratio similar across models?
   - **Consistent ratio**: All training regimes target same component types
   - **Variable ratio**: Different models use different mechanisms (e.g., Strategic modifies MLP more, Deontological modifies Attention more)

5. **Absolute vs relative**:
   - Look at both bar heights (absolute) and bar ratios (relative)
   - Example: Hybrid has highest absolute for both MLP and Attention (most modified overall)
   - Example: Utilitarian has highest MLP/Attention ratio (most MLP-focused)

**Mechanistic Interpretation:**

**High MLP, Low Attention:**
- Task solved by new computations using existing attention patterns
- Example: Calculating expected values (Utilitarian) requires MLP transformations but standard attention to payoff numbers

**Low MLP, High Attention:**
- Task solved by re-routing information to existing computations
- Example: Deontological rule-checking might use existing logical operations (MLP) but attend to different token positions (Attention)

**Both High:**
- Comprehensive architectural change
- Both what to attend to AND what to compute changed
- Suggests complex, multi-faceted task adaptation

**Both Low:**
- Minimal modification overall
- Base model already capable, fine-tuning was light

**Example Interpretation:**
- "All models show MLP norm 2-3× higher than Attention → IPD fine-tuning primarily about new computations (value calculation, outcome prediction), not new attention patterns"
- "Utilitarian MLP norm highest (0.071) → Utilitarian reasoning requires strongest computational modifications"
- "Attention norms nearly uniform (0.024-0.027) → All training regimes use similar attention patterns, divergence is computational"

---

## Summary: How to Use These Figures Together

### Step 1: Start with Per-Model Heatmaps
- Understand each model's modification pattern individually
- Identify hot spots and overall modification intensity

### Step 2: Check Top Components
- Verify whether L2_MLP (or other hypothesized components) are actually heavily modified
- Get ranked ordering of modification magnitudes

### Step 3: Compare Layer Profiles
- Understand where (which layers) different models diverge
- Identify peak modification layers (likely sites of distinctive processing)

### Step 4: Analyze Cross-Model Similarity
- **Cosine similarity**: Do models use similar modification strategies?
- **L2 distance**: Do models differ in total modification magnitude?
- **Top variable components**: Which specific components drive model differences?

### Step 5: Examine Delta Plots
- See layer-by-layer divergence from Strategic baseline
- Identify where moral training adds strongest modifications

### Step 6: MLP vs Attention Summary
- Understand which subsystem (MLP or Attention) was primary target
- Interpret mechanistic implications (computation vs routing)

### Integrated Example Analysis

**Scenario**: Investigating whether L2_MLP is the "switching" node for moral decisions.

1. **Per-model heatmaps**: L2 row (row 2) shows moderate warmth (yellow-orange), not red
2. **Top components**: L2_MLP not in top 30 for any model
3. **L2 MLP comparison**: All models have similar low L2_MLP norms (~0.02-0.04)
4. **Layer profiles**: MLP norm peaks at L11-L15, not L2
5. **Delta plot**: L2 shows minimal delta (~0.005), much smaller than L12-L15 deltas (~0.06)
6. **Top variable components**: L2_MLP absent, but L12-L15 MLPs present

**Conclusion**: L2_MLP is NOT heavily modified. The "switching" behavior likely emerges from:
- Upstream changes (attention routing to L2)
- Downstream interpretation (L12-L15 processing L2 outputs differently)
- L2_MLP itself is a passive component responding to altered inputs

---

## Common Patterns and Their Meanings

| Pattern | Interpretation |
|---------|----------------|
| **Early layer hotspots (L0-L8)** | Input feature extraction modified; task requires different low-level representations |
| **Middle layer hotspots (L9-L17)** | Core reasoning modified; abstract concept processing changed |
| **Late layer hotspots (L18-L25)** | Output formatting modified; decision mechanisms changed |
| **MLP >> Attention** | Computation-heavy adaptation; task requires new transformations |
| **Attention >> MLP** | Routing-heavy adaptation; task requires different information flow |
| **Uniform heatmap** | Comprehensive retraining; base model poorly suited to task |
| **Sparse hotspots** | Targeted circuit modification; minimal intervention strategy |
| **High cosine + low L2** | Models use very similar strategies with similar intensity |
| **High cosine + high L2** | Models use similar strategies but different training intensity |
| **Low cosine** | Models use fundamentally different architectural modifications |
| **Top components clustered in few layers** | Localized functional specialization |
| **Top components distributed** | Distributed processing changes |

---

## Technical Notes

### Normalization
- **Normalized norms** (`/sqrt(output_dim)`): Use for fair cross-module comparison
- **Raw norms**: Use for understanding absolute parameter change
- **Z-scores** (variable components): Use for understanding relative divergence patterns

### Statistical Significance
- These are deterministic measurements (not statistical tests)
- "Significance" is interpretive: Is a 10% difference meaningful?
- Context matters: Compare to:
  - Baseline model norms (if available)
  - Norms from unrelated fine-tuning tasks
  - Expected effect sizes from prior work

### Limitations
- **Norm ≠ Importance**: A component can have low norm but high causal impact (or vice versa)
- **Static analysis**: Norms don't show *when* components are active (need activation patching)
- **Linear assumption**: LoRA is linear, but models are nonlinear (norms don't capture interaction effects)
- **Magnitude bias**: Larger modules (higher dim) tend to have larger raw norms

---

## Quick Reference: Which Plot for Which Question?

| Question | Plot to Use |
|----------|-------------|
| "Was L2_MLP heavily modified?" | Top components bar chart |
| "Which layers were modified most?" | Per-model heatmap, layer profiles |
| "Do models use similar modification strategies?" | Similarity heatmap (cosine) |
| "Which model was modified most overall?" | Distance heatmap (L2), MLP vs Attn summary |
| "Where do moral models diverge from strategic?" | Layerwise delta plot |
| "Which components are model-distinctive?" | Top variable components |
| "Were MLPs or Attention modified more?" | MLP vs Attn summary |
| "What's the spatial pattern of modification?" | Per-model heatmaps |

---

*This guide assumes familiarity with basic linear algebra (norms, matrix multiplication) and transformer architecture (attention, MLP, layers). For deeper mechanistic interpretability concepts, see TransformerLens documentation and Anthropic's interpretability papers.*
