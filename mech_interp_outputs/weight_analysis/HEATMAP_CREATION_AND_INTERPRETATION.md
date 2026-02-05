# Normalized Norm Heatmaps: Creation and Interpretation

This note explains, in practical detail, how the normalized heatmaps were produced and how to read the three figures:

- `norm_heatmap_PT2_COREDe_normalized.png`
- `norm_heatmap_PT3_COREDe_normalized.png`
- `norm_heatmap_PT3_COREUt_normalized.png`

It complements `FIGURE_GUIDE.md` with a methods-focused and model-specific readout.

---

## 1) How these figures are created

### Data source

Each model has a per-component CSV:

- `weight_norms_PT2_COREDe.csv`
- `weight_norms_PT3_COREDe.csv`
- `weight_norms_PT3_COREUt.csv`

and all models are combined in:

- `weight_norms_all_models.csv`

Each row corresponds to one LoRA-instrumented module in one layer.

### Architecture indexing used by the heatmap

- Layers: `L0` to `L25` (26 rows)
- Module columns (7):
  1. `mlp.gate_proj` (Gate)
  2. `mlp.up_proj` (Up)
  3. `mlp.down_proj` (Down)
  4. `self_attn.q_proj` (Q)
  5. `self_attn.k_proj` (K)
  6. `self_attn.v_proj` (V)
  7. `self_attn.o_proj` (O)

Total cells per model: `26 x 7 = 182`.

### LoRA update magnitude

For each component, LoRA defines:

- `A` with shape `(rank, input_dim)`
- `B` with shape `(output_dim, rank)`
- effective update `ΔW = (alpha / rank) * (B @ A)`

The scalar magnitude is the Frobenius norm:

`||ΔW||_F = sqrt(sum_ij (ΔW_ij^2))`

### Normalization used in these plots

The plotted metric is:

`frobenius_norm_normalized = ||ΔW||_F / sqrt(output_dim)`

This makes cross-module comparison fairer (especially MLP vs attention columns with different dimensions).

### Rendering

The 182 normalized values are reshaped into a `26 x 7` matrix and shown as a heatmap:

- rows: layers
- columns: module type
- warm/red: higher update magnitude
- light/yellow: lower update magnitude

---

## 2) What signal the normalized heatmaps show

These figures answer: **where did LoRA spend adaptation budget?**

- Hot cells = stronger parameter movement in that layer/module
- Vertical trends = module-family targeting (e.g., Q/K vs Gate/Up/Down)
- Horizontal trends = depth targeting (early vs middle vs late network)

Important: this is a **magnitude map**, not a direct causal-importance proof. It indicates where changes are concentrated, then causal interventions should verify function.

---

## 3) Model-specific readout for the three requested figures

## PT2_COREDe (`norm_heatmap_PT2_COREDe_normalized.png`)

- Overall mean normalized norm: `0.000652` (lowest of the three)
- MLP vs attention means: `0.000675` vs `0.000636` (slight MLP tilt)
- Top layers by mean intensity: `L24`, `L22`, `L20`, `L23`, `L19`, `L21`
- Max single cell: `L24_ATTN_K = 0.001028`

Interpretation: adaptation is present but relatively mild and smooth; strongest activity is late-layer, with mixed MLP Gate/Down and attention Q/K hotspots.

## PT3_COREDe (`norm_heatmap_PT3_COREDe_normalized.png`)

- Overall mean normalized norm: `0.000786` (~`1.20x` PT2)
- MLP vs attention means: `0.000796` vs `0.000779` (nearly balanced)
- Top layers by mean intensity: `L20`, `L22`, `L18`, `L19`, `L24`, `L21`
- Max single cell: `L25_ATTN_Q = 0.001162`

Interpretation: stronger and more attention-prominent than PT2, with late-layer concentration and clear Q/K/O involvement.

## PT3_COREUt (`norm_heatmap_PT3_COREUt_normalized.png`)

- Overall mean normalized norm: `0.001244` (~`1.91x` PT2; ~`1.58x` PT3_COREDe)
- MLP vs attention means: `0.001240` vs `0.001247` (effectively balanced)
- Top layers by mean intensity: `L24`, `L23`, `L20`, `L22`, `L25`, `L21`
- Max single cell: `L25_ATTN_K = 0.002537` (largest spike among all three)

Interpretation: highest adaptation budget and sharpest hotspots; strongest peaks in late-layer attention (especially Q/K) plus high Down/Up activity in nearby late layers.

---

## 4) Shared pattern across all three heatmaps

All three models are depth-skewed to late layers:

- Early (`L0-L8`) mean activity is lowest
- Middle (`L9-L17`) is higher
- Late (`L18-L25`) is highest

This suggests fine-tuning primarily reshapes higher-level reasoning/output-stage computations rather than early feature extraction.

---

## 5) L2 note (for switching-hypothesis context)

`L2` component magnitudes increase from PT2 -> PT3_COREDe -> PT3_COREUt, but they are still low-ranked globally within each model (roughly bottom quartile by normalized norm).

So from these heatmaps alone: **L2 changes exist, but L2 is not a dominant adaptation hotspot** relative to late-layer components.

---

## 6) Minimal reproducibility sketch (pseudocode)

```python
for model in [PT2_COREDe, PT3_COREDe, PT3_COREUt]:
    rows = []
    for layer in range(26):
        for module in [gate, up, down, q, k, v, o]:
            A, B = load_lora_matrices(model, layer, module)
            delta = (alpha / rank) * (B @ A)
            frob = frobenius_norm(delta)
            normed = frob / sqrt(output_dim(module))
            rows.append((layer, module, normed))
    heatmap = pivot_to_26x7(rows)  # layers x module types
    plot_heatmap(heatmap, cmap="YlOrRd")
```

