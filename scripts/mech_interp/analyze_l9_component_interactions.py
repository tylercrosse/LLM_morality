"""
Experiment 3: Component Interaction Focus on L9

Analyzes Layer 9 MLP and attention connectivity patterns from pre-computed
correlation matrices to understand how Strategic vs Deontological models
wire L9 differently.
"""

import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

OUTPUT_DIR = REPO_ROOT / "mech_interp_outputs" / "l9_investigation" / "component_interactions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CORRELATION_DIR = REPO_ROOT / "mech_interp_outputs" / "component_interactions"


def load_correlation_matrix(model_id: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load pre-computed correlation matrix for a model.

    Returns:
        correlation_matrix: (n_components, n_components) array
        component_names: List of component names
    """
    corr_file = CORRELATION_DIR / f"correlation_matrix_{model_id}.npz"

    if not corr_file.exists():
        raise FileNotFoundError(f"Correlation matrix not found: {corr_file}")

    data = np.load(corr_file)
    correlation_matrix = data['correlation_matrix']
    component_names = data['component_names'].tolist()

    print(f"Loaded correlation matrix for {model_id}: {correlation_matrix.shape}")
    return correlation_matrix, component_names


def find_l9_components(component_names: List[str]) -> Dict[str, List[int]]:
    """
    Find indices of Layer 9 components (MLP and attention heads).

    Returns:
        Dict with 'L9_MLP' and 'L9_ATTN' keys mapping to component indices
    """
    l9_components = {
        'L9_MLP': [],
        'L9_ATTN': [],
    }

    for i, name in enumerate(component_names):
        if 'L9' in name or 'layer_9' in name.lower():
            if 'MLP' in name or 'mlp' in name.lower():
                l9_components['L9_MLP'].append(i)
            elif 'ATTN' in name or 'attn' in name.lower() or 'head' in name.lower():
                l9_components['L9_ATTN'].append(i)

    print(f"Found {len(l9_components['L9_MLP'])} L9 MLP components")
    print(f"Found {len(l9_components['L9_ATTN'])} L9 attention components")

    return l9_components


def extract_l9_correlations(
    correlation_matrix: np.ndarray,
    component_names: List[str],
    l9_indices: List[int]
) -> pd.DataFrame:
    """
    Extract all correlations involving Layer 9 components.

    Returns:
        DataFrame with columns: [component_1, component_2, correlation]
    """
    records = []

    for l9_idx in l9_indices:
        l9_name = component_names[l9_idx]

        for other_idx, other_name in enumerate(component_names):
            if other_idx == l9_idx:
                continue

            corr = correlation_matrix[l9_idx, other_idx]

            records.append({
                'l9_component': l9_name,
                'other_component': other_name,
                'correlation': corr,
                'abs_correlation': abs(corr),
            })

    return pd.DataFrame(records)


def compare_models(
    model1_df: pd.DataFrame,
    model2_df: pd.DataFrame,
    top_k: int = 20
) -> pd.DataFrame:
    """
    Compare L9 connectivity between two models (Deontological vs Utilitarian).

    Returns:
        DataFrame with columns: [l9_component, other_component, De_corr, Ut_corr, difference]
    """
    # Merge on component pairs
    merged = model1_df.merge(
        model2_df,
        on=['l9_component', 'other_component'],
        suffixes=('_De', '_Ut')
    )

    # Calculate difference
    merged['difference'] = merged['correlation_De'] - merged['correlation_Ut']
    merged['abs_difference'] = merged['difference'].abs()

    # Sort by absolute difference
    merged = merged.sort_values('abs_difference', ascending=False)

    return merged.head(top_k)


def plot_l9_connectivity_heatmap(
    correlation_matrix: np.ndarray,
    component_names: List[str],
    l9_indices: List[int],
    model_id: str,
    output_path: Path
):
    """
    Plot heatmap showing L9 connectivity to all other components.
    """
    # Extract L9 rows
    l9_corr = correlation_matrix[l9_indices, :]

    # Get L9 component names
    l9_names = [component_names[i] for i in l9_indices]

    plt.figure(figsize=(20, max(6, len(l9_indices) * 0.5)))

    sns.heatmap(
        l9_corr,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        yticklabels=l9_names,
        xticklabels=component_names,
        cbar_kws={'label': 'Correlation'}
    )

    plt.xlabel('All Components', fontsize=12)
    plt.ylabel('Layer 9 Components', fontsize=12)
    plt.title(f'{model_id}: Layer 9 Connectivity Matrix', fontsize=14)
    plt.xticks(rotation=90, ha='right', fontsize=6)
    plt.yticks(fontsize=8)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved L9 connectivity heatmap to {output_path}")
    plt.close()


def plot_top_differential_connections(
    comparison_df: pd.DataFrame,
    output_path: Path
):
    """
    Plot bar chart of top differential connections.
    """
    # Take top 15 for readability
    top_15 = comparison_df.head(15).copy()

    # Create labels
    top_15['label'] = top_15['l9_component'] + ' ↔ ' + top_15['other_component']

    plt.figure(figsize=(14, 8))

    # Create grouped bar plot
    x = range(len(top_15))
    width = 0.35

    plt.barh([i - width/2 for i in x], top_15['correlation_De'], width,
             label='Deontological', color='blue', alpha=0.7)
    plt.barh([i + width/2 for i in x], top_15['correlation_Ut'], width,
             label='Utilitarian', color='green', alpha=0.7)

    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.yticks(x, top_15['label'], fontsize=9)
    plt.xlabel('Correlation', fontsize=12)
    plt.title('Top 15 Differential L9 Connections\n(Deontological vs Utilitarian)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved top differential connections plot to {output_path}")
    plt.close()


def plot_difference_heatmap(
    comparison_df: pd.DataFrame,
    component_names: List[str],
    l9_component_names: List[str],
    output_path: Path
):
    """
    Plot heatmap showing correlation differences (PT2 - PT3).
    """
    # Create matrix of differences
    n_l9 = len(l9_component_names)
    n_components = len(component_names)

    diff_matrix = np.zeros((n_l9, n_components))

    for _, row in comparison_df.iterrows():
        l9_name = row['l9_component']
        other_name = row['other_component']

        if l9_name in l9_component_names:
            l9_idx = l9_component_names.index(l9_name)
            other_idx = component_names.index(other_name)

            diff_matrix[l9_idx, other_idx] = row['difference']

    plt.figure(figsize=(20, max(6, n_l9 * 0.5)))

    sns.heatmap(
        diff_matrix,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        yticklabels=l9_component_names,
        xticklabels=component_names,
        cbar_kws={'label': 'Difference (De - Ut)'}
    )

    plt.xlabel('All Components', fontsize=12)
    plt.ylabel('Layer 9 Components', fontsize=12)
    plt.title('Layer 9 Connectivity Differences\n(Red = De more correlated, Blue = Ut more correlated)', fontsize=14)
    plt.xticks(rotation=90, ha='right', fontsize=6)
    plt.yticks(fontsize=8)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved difference heatmap to {output_path}")
    plt.close()


def main():
    """Run Experiment 3: Component Interaction Focus on L9."""
    print("="*80)
    print("Experiment 3: Component Interaction Focus on L9")
    print("="*80)

    # Models to compare
    models = {
        'PT3_COREDe': 'Deontological',
        'PT3_COREUt': 'Utilitarian',
    }

    # Load correlation matrices
    print("\nLoading correlation matrices...")
    matrices = {}
    component_names = {}
    l9_components = {}

    for model_id in models.keys():
        corr_matrix, comp_names = load_correlation_matrix(model_id)
        matrices[model_id] = corr_matrix
        component_names[model_id] = comp_names

        # Find L9 components
        l9_comps = find_l9_components(comp_names)
        l9_components[model_id] = l9_comps

    # Extract L9 correlations for each model
    print("\nExtracting L9 correlations...")
    l9_correlations = {}

    for model_id in models.keys():
        # Combine L9_MLP and L9_ATTN indices
        all_l9_indices = (l9_components[model_id]['L9_MLP'] +
                         l9_components[model_id]['L9_ATTN'])

        l9_corr_df = extract_l9_correlations(
            matrices[model_id],
            component_names[model_id],
            all_l9_indices
        )

        l9_correlations[model_id] = l9_corr_df

        # Save detailed correlations
        output_csv = OUTPUT_DIR / f"l9_correlations_{model_id}.csv"
        l9_corr_df.to_csv(output_csv, index=False)
        print(f"Saved L9 correlations for {model_id} to {output_csv}")

    # Compare models
    print("\n" + "="*60)
    print("COMPARING MODELS")
    print("="*60)

    comparison_df = compare_models(
        l9_correlations['PT3_COREDe'],
        l9_correlations['PT3_COREUt'],
        top_k=50
    )

    # Save comparison
    comparison_csv = OUTPUT_DIR / "l9_differential_connections.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"\nSaved differential connections to {comparison_csv}")

    # Display top 15
    print("\nTop 15 Differential L9 Connections:")
    print(comparison_df[['l9_component', 'other_component', 'correlation_De', 'correlation_Ut', 'difference']].head(15).to_string(index=False))

    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    # L2_MLP and L6_MLP connections (from RQ2 findings)
    print("\nL9_MLP connections to key routing hubs:")
    for target in ['L2_MLP', 'L6_MLP']:
        target_matches = comparison_df[comparison_df['other_component'].str.contains(target, case=False)]
        if len(target_matches) > 0:
            print(f"\n{target}:")
            print(target_matches[['l9_component', 'other_component', 'correlation_De', 'correlation_Ut', 'difference']].to_string(index=False))

    # Generate plots
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    # Connectivity heatmaps for each model
    for model_id in models.keys():
        all_l9_indices = (l9_components[model_id]['L9_MLP'] +
                         l9_components[model_id]['L9_ATTN'])

        plot_l9_connectivity_heatmap(
            matrices[model_id],
            component_names[model_id],
            all_l9_indices,
            model_id,
            OUTPUT_DIR / f"l9_connectivity_{model_id}.png"
        )

    # Top differential connections
    plot_top_differential_connections(
        comparison_df,
        OUTPUT_DIR / "l9_top_differential_connections.png"
    )

    # Difference heatmap
    all_l9_names_de = [component_names['PT3_COREDe'][i] for i in
                        (l9_components['PT3_COREDe']['L9_MLP'] +
                         l9_components['PT3_COREDe']['L9_ATTN'])]

    plot_difference_heatmap(
        comparison_df,
        component_names['PT3_COREDe'],
        all_l9_names_de,
        OUTPUT_DIR / "l9_connectivity_difference_heatmap.png"
    )

    print("\n" + "="*80)
    print("EXPERIMENT 3 COMPLETE")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
