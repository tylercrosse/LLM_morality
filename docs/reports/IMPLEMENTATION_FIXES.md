# Implementation Fixes for Attention and Component Interaction Analyses

## Issues Fixed

### 1. Action Token ID Extraction (Component Interactions)

**Error:**
```
ValueError: too many values to unpack (expected 2)
```

**Cause:**
`get_action_token_ids()` returns a dictionary, not a tuple.

**Fix:**
```python
# OLD (incorrect)
self.c_token, self.d_token = get_action_token_ids(self.tokenizer)

# NEW (correct)
action_tokens = get_action_token_ids(self.tokenizer)
self.c_token = action_tokens["C"]
self.d_token = action_tokens["D"]
```

---

### 2. Cache Key Mismatches

**Issue:**
Both analyses used cache keys that don't exist in `HookedGemmaModel`.

**HookedGemmaModel actually provides:**
- `blocks.{idx}.hook_attn_out` - Full attention output (d_model)
- `blocks.{idx}.hook_mlp_out` - MLP output (d_model)
- `blocks.{idx}.hook_resid_post` - Residual stream after layer

**Does NOT provide:**
- Per-head attention activations
- `model.layers.{layer}.self_attn.head_out`
- `model.layers.{layer}.mlp.output`

---

### 3. Component Interaction Analysis - Simplified to Layer-Level

**Original design:** Analyze correlations between individual attention heads (234 components: 26 layers × 8 heads + 26 MLPs)

**Problem:** HookedGemmaModel doesn't decompose attention into per-head outputs

**Solution:** Simplified to layer-level granularity (52 components: 26 attention layers + 26 MLPs)

**Updated code:**
```python
for layer in range(26):
    # Attention output (full layer)
    attn_key = f"blocks.{layer}.hook_attn_out"
    if attn_key in cache:
        attn_out = cache[attn_key][0, final_pos]
        magnitude = float(torch.norm(attn_out, p=2).cpu())
        component_activations[f"L{layer}_ATTN"] = magnitude

    # MLP output
    mlp_key = f"blocks.{layer}.hook_mlp_out"
    if mlp_key in cache:
        mlp_out = cache[mlp_key][0, final_pos]
        magnitude = float(torch.norm(mlp_out, p=2).cpu())
        component_activations[f"L{layer}_MLP"] = magnitude
```

**Impact:**
- Correlation matrix: 52×52 instead of 234×234
- Still captures layer-level interaction patterns
- More interpretable (attention vs MLP roles per layer)
- Component pairs: 1,326 instead of 27,261

---

### 4. Attention Weights Not Cached

**Issue:**
Attention analysis needs attention weights, but `HookedGemmaModel.run_with_cache()` didn't capture them.

**Solution:**
Updated `run_with_cache()` to support `output_attentions=True`:

```python
def run_with_cache(self, input_ids, output_attentions=False):
    self.register_cache_hooks()

    with torch.no_grad():
        outputs = self.model(input_ids, output_attentions=output_attentions)
        logits = outputs.logits

        # Cache attention weights if requested
        if output_attentions and hasattr(outputs, 'attentions'):
            for layer_idx, attn_weights in enumerate(outputs.attentions):
                # attn_weights shape: (batch, num_heads, seq_len, seq_len)
                self.cache[f"model.layers.{layer_idx}.self_attn.attn_weights"] = attn_weights.detach()

    self.remove_hooks()
    return logits, self.cache.copy()
```

**Updated attention analysis call:**
```python
_, cache = self.model.run_with_cache(input_ids, output_attentions=True)
```

---

## Summary of Changes

### Files Modified

1. **[mech_interp/component_interactions.py](mech_interp/component_interactions.py)**
   - Fixed action token ID extraction
   - Simplified to layer-level granularity (52 components instead of 234)
   - Updated cache keys to match HookedGemmaModel

2. **[mech_interp/model_loader.py](mech_interp/model_loader.py)**
   - Added `output_attentions` parameter to `run_with_cache()`
   - Added attention weight caching when requested

3. **[mech_interp/attention_analysis.py](mech_interp/attention_analysis.py)**
   - Updated to call `run_with_cache(input_ids, output_attentions=True)`

---

## Updated Analysis Capabilities

### Attention Analysis ✅
- **Works:** Extracts attention weights per head
- **Granularity:** 26 layers × 8 heads
- **What it shows:** Which tokens each head attends to

### Component Interaction Analysis ✅
- **Works:** Computes correlation matrices between components
- **Granularity:** 26 layers × 2 component types (ATTN, MLP) = 52 components
- **What it shows:** Which layer components co-activate (layer-level pathways)

---

## Interpretation Adjustments

### Component Interactions

**Original framing:**
"Compare L8_MLP → L25H0 pathway vs L9_MLP → L25H3 pathway"

**Updated framing:**
"Compare L8_MLP → L25_ATTN pathway vs L9_MLP → L25_ATTN pathway"

### Key Questions (Updated)

1. **Does L8_MLP correlate more strongly with late-layer attention in Deontological models?**
   - Hypothesis: L8_MLP → L20-L25 ATTN pathways differ between De and Ut

2. **Do early-to-late pathways differ?**
   - Compare L0-L9 ATTN/MLP → L20-L25 ATTN/MLP correlations

3. **Are MLP-to-ATTN connections different from ATTN-to-MLP connections?**
   - Asymmetric pathways: MLP(t) → ATTN(t+1) vs ATTN(t) → MLP(t)

---

## Why Layer-Level Granularity is Appropriate

### Advantages

1. **More stable correlations**
   - Averaging across heads reduces noise
   - Captures overall layer function

2. **More interpretable**
   - Clearer narrative: "L8 MLP processes X, sends to L25 attention for Y"
   - Matches common interpretability framing (e.g., "early layers extract features, late layers make decisions")

3. **Consistent with existing DLA results**
   - DLA already reports L8_MLP and L9_MLP as key components
   - Interaction analysis complements this with "who talks to who"

4. **Computationally tractable**
   - 52×52 correlation matrix is easier to visualize
   - 1,326 component pairs instead of 27,261

### What We Lose

- Cannot distinguish if different heads within a layer connect differently
- Example: Cannot see if L8_MLP → L25H0 differs from L8_MLP → L25H3

### Potential Future Extension

If layer-level analysis shows promising differences, we could implement per-head decomposition by:
1. Adding head-level hooks to HookedGemmaModel
2. Using attention output reshaping: `attn_out.reshape(num_heads, head_dim)`
3. This would require deeper model access (not available through standard forward hooks)

---

## Expected Outputs (Updated)

### Component Interaction Analysis

**Data files:**
- `component_activations_PT3_COREDe.json` - 52 components × 15 scenarios
- `component_activations_PT3_COREUt.json` - 52 components × 15 scenarios
- `correlation_matrix_PT3_COREDe.npz` - 52×52 matrix
- `correlation_matrix_PT3_COREUt.npz` - 52×52 matrix
- `interaction_comparison_De_vs_Ut.csv` - 1,326 component pairs
- `significant_pathways_De_vs_Ut.csv` - Pathways with |diff| > 0.3
- `key_component_pathways_De_vs_Ut.csv` - Connections involving L8_MLP, L9_MLP

**Visualizations:**
- Correlation heatmaps (52×52) - easier to read than 234×234!
- Difference heatmaps
- Key pathway diagrams

---

## Ready to Run

All fixes implemented and tested. You can now run:

```bash
python scripts/mech_interp/run_full_rq2_analysis.py
```

Expected runtime: ~10 minutes

---

## Technical Notes

### Why HookedGemmaModel Doesn't Provide Per-Head Outputs

The attention module computes:
```
Q = input @ W_Q  # (seq_len, num_heads, head_dim)
K = input @ W_K
V = input @ W_V
attn_weights = softmax(Q @ K.T / sqrt(head_dim))
head_outputs = attn_weights @ V  # (seq_len, num_heads, head_dim)
attn_output = head_outputs.reshape(seq_len, d_model) @ W_O
```

Standard forward hooks only capture `attn_output` (after W_O projection), not `head_outputs`.

To get per-head outputs, we would need to:
1. Modify the attention module itself
2. Add hooks INSIDE the attention forward pass
3. Or use a library like TransformerLens that's designed for this

Since HookedGemmaModel is a lightweight wrapper, we keep it simple and work with full layer outputs.

---

## Comparison with DLA

**DLA (Direct Logit Attribution):**
- Projects component outputs through unembedding matrix
- Measures contribution to final decision
- Reports per-head contributions (but approximates by dividing evenly)

**Component Interactions:**
- Computes correlations between component activations
- Measures which components co-activate
- Layer-level granularity (ATTN + MLP)

Both analyses are complementary:
- **DLA:** "Which components push toward Cooperate vs Defect?"
- **Interactions:** "Which components work together?"
