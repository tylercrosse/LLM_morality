"""
Experiment 1: Model Comparison at Layer 9

Analyzes the Layer 9 cooperative dip across models and scenarios.
Extracts trajectory statistics and performs comparative analysis.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple

# Repository root
REPO_ROOT = Path(__file__).parent.parent.parent
LOGIT_LENS_DIR = REPO_ROOT / "mech_interp_outputs" / "logit_lens"
OUTPUT_DIR = REPO_ROOT / "mech_interp_outputs" / "l9_investigation" / "model_comparisons"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_trajectories() -> Dict:
    """Load logit lens trajectories from JSON file."""
    trajectories_path = LOGIT_LENS_DIR / "trajectories.json"
    with open(trajectories_path, 'r') as f:
        return json.load(f)


def extract_layer_deltas(trajectories: Dict) -> pd.DataFrame:
    """
    Extract key layer deltas for all models and scenarios.

    Returns DataFrame with columns:
    - scenario: Game scenario
    - model: Model name
    - L0_delta: Layer 0 delta
    - L3_delta: Layer 3 delta
    - L9_delta: Layer 9 delta
    - L16_delta: Layer 16 delta
    - final_delta: Layer 25 delta
    - dip_magnitude: L3_delta - L9_delta (more negative = larger dip)
    - dip_direction: "cooperative" if L9 < L3, else "defective"
    """
    records = []

    for scenario_name, scenario_data in trajectories.items():
        for model_name, trajectory in scenario_data.items():
            # Extract key layers (0-indexed in list, so layer N is at index N)
            L0_delta = trajectory[0] if len(trajectory) > 0 else np.nan
            L3_delta = trajectory[3] if len(trajectory) > 3 else np.nan
            L9_delta = trajectory[9] if len(trajectory) > 9 else np.nan
            L16_delta = trajectory[16] if len(trajectory) > 16 else np.nan
            final_delta = trajectory[25] if len(trajectory) > 25 else np.nan

            # Compute dip magnitude
            dip_magnitude = L3_delta - L9_delta
            dip_direction = "cooperative" if L9_delta < L3_delta else "defective"

            records.append({
                'scenario': scenario_name,
                'model': model_name,
                'L0_delta': L0_delta,
                'L3_delta': L3_delta,
                'L9_delta': L9_delta,
                'L16_delta': L16_delta,
                'final_delta': final_delta,
                'dip_magnitude': dip_magnitude,
                'dip_direction': dip_direction,
            })

    return pd.DataFrame(records)


def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics by model across scenarios.

    Returns DataFrame with:
    - model
    - mean_L9_dip: Mean dip magnitude
    - std_L9_dip: Std dev of dip magnitude
    - min_L9_delta: Most cooperative L9 value
    - max_L9_delta: Least cooperative L9 value
    - mean_final_delta: Mean final decision
    """
    stats_records = []

    for model in df['model'].unique():
        model_df = df[df['model'] == model]

        stats_records.append({
            'model': model,
            'mean_L0_delta': model_df['L0_delta'].mean(),
            'std_L0_delta': model_df['L0_delta'].std(),
            'mean_L3_delta': model_df['L3_delta'].mean(),
            'std_L3_delta': model_df['L3_delta'].std(),
            'mean_L9_delta': model_df['L9_delta'].mean(),
            'std_L9_delta': model_df['L9_delta'].std(),
            'mean_L9_dip_magnitude': model_df['dip_magnitude'].mean(),
            'std_L9_dip_magnitude': model_df['dip_magnitude'].std(),
            'min_L9_delta': model_df['L9_delta'].min(),
            'max_L9_delta': model_df['L9_delta'].max(),
            'mean_final_delta': model_df['final_delta'].mean(),
            'std_final_delta': model_df['final_delta'].std(),
        })

    return pd.DataFrame(stats_records)


def statistical_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform pairwise statistical comparisons of L9 dip magnitude.

    Uses t-test to compare dip magnitude between models.
    """
    models = df['model'].unique()
    comparison_records = []

    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            model1_dips = df[df['model'] == model1]['dip_magnitude']
            model2_dips = df[df['model'] == model2]['dip_magnitude']

            t_stat, p_value = stats.ttest_ind(model1_dips, model2_dips)

            comparison_records.append({
                'model1': model1,
                'model2': model2,
                'model1_mean_dip': model1_dips.mean(),
                'model2_mean_dip': model2_dips.mean(),
                'difference': model1_dips.mean() - model2_dips.mean(),
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': 'Yes' if p_value < 0.05 else 'No',
            })

    return pd.DataFrame(comparison_records)


def plot_dip_comparison(df: pd.DataFrame, output_path: Path):
    """
    Create box plot comparing L9 dip magnitude across models.
    """
    plt.figure(figsize=(12, 6))

    # Create box plot
    sns.boxplot(data=df, x='model', y='dip_magnitude', palette='Set2')

    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='No dip')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('L9 Dip Magnitude (L3 - L9)', fontsize=12)
    plt.title('Layer 9 Cooperative Dip Across Models\n(More positive = larger cooperative dip)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved box plot to {output_path}")
    plt.close()


def plot_trajectory_comparison(df: pd.DataFrame, trajectories: Dict, output_path: Path):
    """
    Plot full trajectories for all models in a single scenario.
    """
    # Choose CC_continue as representative scenario
    scenario = "CC_continue"
    scenario_data = trajectories[scenario]

    plt.figure(figsize=(14, 8))

    colors = {
        'base': 'black',
        'PT2_COREDe': 'red',
        'PT3_COREDe': 'blue',
        'PT3_COREUt': 'green',
        'PT4_COREDe': 'purple',
    }

    for model_name, trajectory in scenario_data.items():
        color = colors.get(model_name, 'gray')
        label = model_name
        if model_name == 'PT2_COREDe':
            label += ' (Strategic)'
        elif model_name == 'PT3_COREDe':
            label += ' (Deontological)'
        elif model_name == 'PT3_COREUt':
            label += ' (Utilitarian)'

        plt.plot(range(len(trajectory)), trajectory, label=label, color=color, linewidth=2)

    # Highlight key layers
    plt.axvline(x=0, color='gray', linestyle=':', alpha=0.3, label='Layer 0')
    plt.axvline(x=3, color='gray', linestyle=':', alpha=0.3, label='Layer 3')
    plt.axvline(x=9, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Layer 9 (dip)')
    plt.axvline(x=16, color='gray', linestyle=':', alpha=0.3, label='Layer 16')

    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Delta (Defect - Cooperate)', fontsize=12)
    plt.title(f'Logit Lens Trajectories: {scenario}\n(Negative = prefer Cooperate, Positive = prefer Defect)', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved trajectory plot to {output_path}")
    plt.close()


def plot_layer_comparison_heatmap(df: pd.DataFrame, output_path: Path):
    """
    Create heatmap showing layer deltas for all models and scenarios.
    """
    # Pivot data for heatmap
    pivot_data = df.pivot_table(
        index=['model', 'scenario'],
        values=['L0_delta', 'L3_delta', 'L9_delta', 'L16_delta', 'final_delta'],
        aggfunc='mean'
    )

    plt.figure(figsize=(10, 12))
    sns.heatmap(
        pivot_data,
        cmap='RdBu_r',
        center=0,
        annot=True,
        fmt='.2f',
        cbar_kws={'label': 'Delta (D - C)'}
    )
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Model / Scenario', fontsize=12)
    plt.title('Layer-wise Delta Values Across Models and Scenarios', fontsize=14)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {output_path}")
    plt.close()


def main():
    """Run Experiment 1: Model Comparison at Layer 9."""
    print("="*80)
    print("Experiment 1: Model Comparison at Layer 9")
    print("="*80)

    # Load data
    print("\nLoading trajectories...")
    trajectories = load_trajectories()

    # Extract layer deltas
    print("Extracting layer deltas...")
    df = extract_layer_deltas(trajectories)

    # Save detailed data
    detailed_csv = OUTPUT_DIR / "l9_dip_detailed.csv"
    df.to_csv(detailed_csv, index=False)
    print(f"\nSaved detailed data to {detailed_csv}")

    # Compute summary statistics
    print("\nComputing summary statistics...")
    stats_df = compute_statistics(df)
    stats_csv = OUTPUT_DIR / "l9_dip_summary_statistics.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"Saved summary statistics to {stats_csv}")

    # Display summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(stats_df.to_string(index=False))

    # Statistical comparisons
    print("\n\nPerforming statistical comparisons...")
    comparison_df = statistical_comparison(df)
    comparison_csv = OUTPUT_DIR / "l9_dip_statistical_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"Saved statistical comparison to {comparison_csv}")

    print("\n" + "="*80)
    print("PAIRWISE COMPARISONS")
    print("="*80)
    print(comparison_df.to_string(index=False))

    # Generate plots
    print("\n\nGenerating visualizations...")
    plot_dip_comparison(df, OUTPUT_DIR / "l9_dip_boxplot.png")
    plot_trajectory_comparison(df, trajectories, OUTPUT_DIR / "trajectory_comparison_CC_continue.png")
    plot_layer_comparison_heatmap(df, OUTPUT_DIR / "layer_comparison_heatmap.png")

    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    # Find models with largest dip
    avg_dip_by_model = df.groupby('model')['dip_magnitude'].mean().sort_values(ascending=False)
    print("\nModels ranked by L9 dip magnitude (larger = more cooperative dip):")
    for model, dip in avg_dip_by_model.items():
        print(f"  {model}: {dip:.3f}")

    # Check if all models show dip
    models_with_dip = df.groupby('model')['dip_direction'].apply(
        lambda x: (x == 'cooperative').mean()
    )
    print("\nProportion of scenarios with cooperative dip (L9 < L3):")
    for model, prop in models_with_dip.items():
        print(f"  {model}: {prop*100:.1f}%")

    # Final decision comparison
    avg_final = df.groupby('model')['final_delta'].mean().sort_values()
    print("\nFinal layer delta (more negative = more cooperative):")
    for model, delta in avg_final.items():
        print(f"  {model}: {delta:.3f}")

    print("\n" + "="*80)
    print("EXPERIMENT 1 COMPLETE")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
