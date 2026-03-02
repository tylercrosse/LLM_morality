# Mechanistic Interpretability for IPD Models

This module implements white-box mechanistic interpretability methods for analyzing LoRA-finetuned LLMs on the Iterated Prisoner's Dilemma (IPD) task.

## Overview

The module provides three core analysis methods:

1. **Logit Lens** - Layer-by-layer decision trajectory visualization
2. **Direct Logit Attribution (DLA)** - Component-level contribution decomposition
3. **Activation Patching** - Causal circuit discovery through intervention

## Quick Start

### Running Complete Analyses

```bash
# Task 2: Logit Lens Analysis
python scripts/mech_interp/run_logit_lens.py

# Task 3: Direct Logit Attribution
python scripts/mech_interp/run_dla.py

# Task 4: Activation Patching
python scripts/mech_interp/run_patching.py
```

### Custom Analysis

```python
from mech_interp import (
    LoRAModelLoader,
    get_action_token_ids,
    LogitLensAnalyzer,
    DirectLogitAttributor,
    ActivationPatcher
)

# Load model
loader = LoRAModelLoader(device="cuda")
model = loader.load_hooked_model("PT3_COREDe")
action_tokens = get_action_token_ids(model.tokenizer)

# Logit lens
lens = LogitLensAnalyzer(model, action_tokens)
trajectory = lens.compute_action_trajectory("Your prompt here")

# Direct logit attribution
dla = DirectLogitAttributor(model, action_tokens)
result = dla.decompose_logits("Your prompt here")
top_components = dla.identify_top_components(result, top_k=20)

# Activation patching
source = loader.load_hooked_model("PT2_COREDe")
target = loader.load_hooked_model("PT3_COREDe")
patcher = ActivationPatcher(source, target, action_tokens)
patch_results = patcher.systematic_patch("Your prompt here", "scenario_name")
```

## Module Structure

```
mech_interp/
├── __init__.py                      # Module exports
├── utils.py                         # Action tokens, labels, colors
├── model_loader.py                  # HookedGemmaModel wrapper
├── prompt_generator.py              # IPD evaluation dataset
├── logit_lens.py                    # Task 2: Logit lens
├── direct_logit_attribution.py      # Task 3: DLA
└── activation_patching.py           # Task 4: Patching
```

## Core Components

### 1. Model Loading

**HookedGemmaModel** - Custom wrapper for Gemma-2 with activation caching:

```python
class HookedGemmaModel:
    def run_with_cache(self, input_ids):
        """Run forward pass and cache all activations."""
        # Returns: (logits, cache)
        # Cache contains:
        #   - blocks.{i}.hook_resid_post  (residual stream)
        #   - blocks.{i}.hook_attn_out    (attention output)
        #   - blocks.{i}.hook_mlp_out     (MLP output)

    def unembed(self, hidden_state):
        """Project hidden state to vocabulary logits."""
        # Applies RMSNorm before unembedding (critical for accuracy)
```

**LoRAModelLoader** - Loads models with in-memory LoRA merging:

```python
loader = LoRAModelLoader(device="cuda")
model = loader.load_hooked_model("PT3_COREDe")  # Returns HookedGemmaModel
```

Available models:
- `"base"` - Gemma-2-2b-it base model
- `"PT2_COREDe"` - Strategic (game payoffs only)
- `"PT3_COREDe"` - Deontological (betrayal penalty)
- `"PT3_COREUt"` - Utilitarian (collective sum reward)
- `"PT4_COREDe"` - Hybrid (game + deontological)

### 2. Action Tokens

IPD actions are represented as multi-token sequences:
- **action1** (Cooperate): `[2314, 235274]` → Compare on token `235274`
- **action2** (Defect): `[2314, 235284]` → Compare on token `235284`

```python
action_tokens = get_action_token_ids(tokenizer)
# Returns:
# {
#     'action1_tokens': [2314, 235274],
#     'action2_tokens': [2314, 235284],
#     'C': 235274,  # Cooperate comparison token
#     'D': 235284,  # Defect comparison token
# }
```

### 3. Evaluation Prompts

15 IPD scenarios across 5 game states:

```python
from mech_interp import load_prompt_dataset

prompts = load_prompt_dataset()
# Returns list of dicts with:
# {
#     'scenario': 'CC_continue',
#     'variant': 1,
#     'prompt': '...',
#     'description': '...'
# }
```

Scenarios:
- **CC_continue** - Mutual cooperation, maintain cooperation
- **CC_temptation** - Mutual cooperation, defection tempting
- **CD_punished** - Cooperated but got defected on
- **DC_exploited** - Defected on cooperator (guilt test)
- **DD_trapped** - Mutual defection, break cycle

## Analysis Methods

### Task 2: Logit Lens

**Purpose**: Visualize how action preferences evolve layer-by-layer.

**Usage**:
```python
analyzer = LogitLensAnalyzer(model, action_tokens)

# Compute trajectory for a single prompt
trajectory = analyzer.compute_action_trajectory(prompt)
# Returns: np.array of shape [n_layers] with Δ(D-C) logits per layer

# Get statistics
stats = analyzer.get_layer_statistics(trajectory)
# Returns: {
#     'mean_delta': float,
#     'std_delta': float,
#     'final_delta': float,
#     'final_action': str,
#     'cooperate_layers': int,
#     'defect_layers': int,
#     'flip_layers': List[int]
# }
```

**Visualization**:
```python
visualizer = LogitLensVisualizer(action_tokens)

# Compare models on one scenario
visualizer.plot_model_comparison(
    all_models,
    [result1, result2, result3],
    "CC_temptation"
)

# Grid of all scenarios
visualizer.plot_scenario_grid(results_dict, model_labels)

# Heatmap of final preferences
visualizer.plot_final_comparison_heatmap(results_dict, model_labels)
```

**Execution**:
```bash
python scripts/mech_interp/run_logit_lens.py
# Analyzes 5 models × 5 scenarios
# Generates: comparison plots, grid, heatmap, CSV, JSON
# Output: mech_interp_outputs/logit_lens/
```

### Task 3: Direct Logit Attribution

**Purpose**: Decompose final action logits into per-component contributions.

**Usage**:
```python
attributor = DirectLogitAttributor(model, action_tokens)

# Decompose logits
result = attributor.decompose_logits(prompt)
# Returns: DLAResult with:
#   - head_contributions: [n_layers, n_heads]
#   - mlp_contributions: [n_layers]
#   - final_delta: float

# Identify top components
top_comps = attributor.identify_top_components(result, top_k=20)
# Returns: {
#     'pro_defect': [(component, contribution), ...],
#     'pro_cooperate': [(component, contribution), ...]
# }

# Export to DataFrame
df = attributor.export_to_dataframe(result)
```

**Visualization**:
```python
visualizer = DLAVisualizer(action_tokens)

# Head contribution heatmap
visualizer.plot_head_heatmap(result, cmap='RdBu_r')

# MLP contributions
visualizer.plot_mlp_contributions(result)

# Top components
visualizer.plot_top_components(result, top_k=20)

# Model comparison
visualizer.plot_model_comparison(results, model_labels, scenario)
```

**Execution**:
```bash
python scripts/mech_interp/run_dla.py
# Analyzes 5 models × 5 scenarios
# Generates: heatmaps, MLP plots, top component plots
# Exports: full results CSV, summary stats, top components
# Output: mech_interp_outputs/dla/
```

**Key Metrics**:
- **contribution**: Δ logit contribution (positive = pro-Defect, negative = pro-Cooperate)
- **component**: e.g., "L12H3" (layer 12, head 3) or "L15_MLP"

### Task 4: Activation Patching

**Purpose**: Identify causal circuits by patching activations between models.

**Usage**:
```python
# Load source and target models
source = loader.load_hooked_model("PT2_COREDe")  # Strategic
target = loader.load_hooked_model("PT3_COREDe")  # Deontological

patcher = ActivationPatcher(source, target, action_tokens)

# Get baseline behavior
baseline_delta, baseline_action = patcher.get_baseline_behavior(prompt)

# Patch single component
patched_delta, patched_action = patcher.patch_component(prompt, "L5H3")

# Systematic patching (all components)
results = patcher.systematic_patch(prompt, scenario)
# Returns: List[PatchResult] with:
#   - baseline_delta, baseline_action
#   - patched_delta, patched_action
#   - delta_change (patched - baseline)
#   - action_flipped (bool)
#   - effect_size (normalized change)

# Discover minimal circuit
discovery = patcher.discover_minimal_circuit(prompt, scenario, max_components=10)
# Returns: CircuitDiscovery with:
#   - ranked_components: List[(component, effect)]
#   - minimal_circuit: List[component]
#   - all_patches: List[PatchResult]
```

**Visualization**:
```python
visualizer = PatchingVisualizer()

# Heatmap of patching effects
visualizer.plot_patch_heatmap(results, metric="delta_change")

# Top components by effect
visualizer.plot_top_components(results, top_k=30)

# Circuit discovery
visualizer.plot_circuit_discovery(discovery)
```

**Execution**:
```bash
python scripts/mech_interp/run_patching.py

# Skip slow circuit discovery:
python scripts/mech_interp/run_patching.py --no-circuits

# Analyze specific scenarios only:
python scripts/mech_interp/run_patching.py --scenarios CC_temptation CD_punished
```

**Default Experiments**:
1. **PT2 → PT3_De**: Strategic → Deontological (find "selfishness circuit")
2. **PT2 → PT3_Ut**: Strategic → Utilitarian
3. **PT3_De → PT3_Ut**: Deontological → Utilitarian (find moral differences)
4. **PT3_Ut → PT3_De**: Utilitarian → Deontological

**Output**: `mech_interp_outputs/patching/`
- Heatmaps per scenario
- Top component plots
- Circuit discovery visualizations
- Consistency analysis (components that flip across scenarios)
- Full results CSV
- Summary statistics
- Discovered circuits
- Cross-experiment comparison

**Key Metrics**:
- **delta_change**: Patched Δ - Baseline Δ (sign indicates direction of influence)
- **action_flipped**: Did patching change the action choice?
- **effect_size**: |delta_change| / |baseline_delta| (normalized effect)

## Research Questions

This infrastructure enables investigation of:

### RQ1: Suppression of "selfish" heads during moral fine-tuning
- **Method**: DLA + Patching (PT2 → PT3)
- **Analysis**: Compare head contributions in strategic vs moral models
- **Question**: Which heads are pro-Defect in PT2 but suppressed in PT3?

### RQ2: Distinct circuits for Deontological vs Utilitarian reasoning
- **Method**: Patching (PT3_De ↔ PT3_Ut)
- **Analysis**: Cross-patching to identify distinguishing components
- **Question**: Do different moral frameworks rely on distinct circuits?

### RQ3: Targeted fine-tuning guidance
- **Method**: DLA across all models
- **Analysis**: Identify components most affected by moral training
- **Question**: Can we fine-tune only specific heads/layers for efficiency?

## Technical Details

### Architecture
- **Model**: Gemma-2-2b-it (2304 hidden dim, 26 layers, 8 heads)
- **LoRA**: rank=64, alpha=32, targets=q/k/v/o_proj + gate/up/down_proj
- **Training**: PPO with 1000 episodes, Tit-for-Tat opponent

### Normalization
- **Gemma-2** uses RMSNorm (not LayerNorm)
- **Critical**: Apply `model.model.norm` before unembedding for accurate logit lens

### Memory Optimization
- In-memory LoRA merge (no disk overhead)
- Activation caching via forward hooks
- Automatic GPU cache clearing between models

### Cached Activations
Per forward pass: **78 activations** (26 layers × 3 components):
- `blocks.{i}.hook_resid_post` - Residual stream after layer i
- `blocks.{i}.hook_attn_out` - Attention output
- `blocks.{i}.hook_mlp_out` - MLP output

## Performance Notes

### Execution Times (approximate)
- **Logit Lens**: ~15-20 min (5 models × 15 prompts)
- **DLA**: ~20-30 min (5 models × 15 prompts)
- **Patching**: ~2-4 hours (4 experiments × 15 prompts × 234 components)
- **Circuit Discovery**: Adds ~50% overhead to patching

### GPU Memory
- Per model: ~8.8 GB (merged)
- Peak usage: ~12 GB (with activation caching)
- Recommended: 16+ GB VRAM

### Optimization Tips
```bash
# Run on subset of scenarios for testing
python scripts/mech_interp/run_patching.py --scenarios CC_temptation

# Skip circuit discovery for faster results
python scripts/mech_interp/run_patching.py --no-circuits

# Analyze specific models only
python scripts/mech_interp/run_dla.py --models PT2_COREDe PT3_COREDe
```

## Output Directory Structure

```
mech_interp_outputs/
├── prompt_datasets/
│   └── ipd_eval_prompts.json              # 15 evaluation prompts
├── logit_lens/
│   ├── logit_lens_comparison_*.png        # Model comparisons per scenario
│   ├── logit_lens_grid.png                # All scenarios grid
│   ├── final_preferences_heatmap.png      # Final layer heatmap
│   ├── statistics.csv                     # Per-model statistics
│   └── trajectories.json                  # Full trajectory data
├── dla/
│   ├── dla_heads_*.png                    # Head heatmaps per scenario
│   ├── dla_mlps_*.png                     # MLP contributions per scenario
│   ├── dla_top_components_*.png           # Top-20 components per model
│   ├── dla_full_results.csv               # Complete attribution data
│   ├── dla_summary_stats.csv              # Statistical summary
│   └── dla_top_components.csv             # Ranked components
└── patching/
    ├── patch_heatmap_*.png                # Patching effect heatmaps
    ├── patch_top_components_*.png         # Top components per experiment
    ├── circuit_discovery_*.png            # Minimal circuits
    ├── patch_consistency_*.png            # Cross-scenario consistency
    ├── patch_results_*.csv                # Full patching data per experiment
    ├── patch_summary_*.csv                # Statistical summary per experiment
    ├── circuits_*.csv                     # Discovered circuits per experiment
    ├── top_components_*.csv               # Ranked components per experiment
    └── cross_experiment_summary.csv       # Cross-experiment comparison
```

## Citation

This implementation is based on methods from:

```bibtex
@article{anthropic2024moral,
  title={Moral Reasoning in LLMs: A Mechanistic Interpretability Study},
  url={https://arxiv.org/html/2410.01639},
  year={2024}
}
```

## Development

### Adding New Scenarios

```python
# In mech_interp/prompt_generator.py
SCENARIOS["new_scenario"] = {
    "state_self": "Cooperate",
    "state_opp": "Defect",
    "description": "Your scenario description"
}
```

### Custom Patching Experiments

```python
from mech_interp import ActivationPatcher

# Define custom source-target pairs
experiments = [
    ("base", "PT2_COREDe"),  # Base → Strategic
    ("base", "PT3_COREDe"),  # Base → Deontological
]

# Run with custom settings
run_full_analysis(
    experiments=experiments,
    scenarios_to_analyze=["CC_temptation"],
    discover_circuits=False  # Skip for speed
)
```

## Troubleshooting

### CUDA out of memory
```python
# Reduce batch size or clear cache between models
torch.cuda.empty_cache()
```

### Logit reconstruction error too large
```python
# Ensure RMSNorm is applied before unembedding
# This should already be handled in HookedGemmaModel.unembed()
```

### Missing dependencies
```bash
uv pip install transformer-lens circuitsvis einops peft
```

## Next Steps

1. Run full analysis pipeline (Tasks 2-4)
2. Analyze results to answer RQs
3. Update MECH_INTERP_RESEARCH_LOG.md with findings
4. Generate publication-quality figures
5. Write up results for paper
