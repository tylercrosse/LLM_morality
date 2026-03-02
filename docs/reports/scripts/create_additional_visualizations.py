#!/usr/bin/env python3
"""
Create additional visualizations for routing/pathway analysis.

Generates 5 visualizations:
1. Network graph of top pathways
2. Scatter plot: De correlations vs Ut correlations
3. Layer-wise aggregation of differences
4. Pathway-specific focus on L2_MLP and L6_MLP
5. Sankey diagram of information flow
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
import re
from collections import defaultdict

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
DATA_DIR = Path("/root/LLM_morality/mech_interp_outputs/component_interactions")
OUTPUT_DIR = Path("/root/LLM_morality/mech_interp_outputs/component_interactions/additional_viz")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load data
print("Loading data...")
significant_df = pd.read_csv(DATA_DIR / "significant_pathways_De_vs_Ut.csv")
key_pathways_df = pd.read_csv(DATA_DIR / "key_component_pathways_De_vs_Ut.csv")
all_interactions_df = pd.read_csv(DATA_DIR / "interaction_comparison_De_vs_Ut.csv")

print(f"Loaded {len(significant_df)} significant pathways")
print(f"Loaded {len(all_interactions_df)} total interactions")

# Helper function to extract layer number
def get_layer_number(component_name):
    """Extract layer number from component name like L2_MLP or L13_ATTN"""
    match = re.match(r'L(\d+)', component_name)
    return int(match.group(1)) if match else -1

# Helper function to parse component type
def get_component_type(component_name):
    """Get component type (MLP, ATTN, or HEAD)"""
    if '_MLP' in component_name:
        return 'MLP'
    elif '_ATTN' in component_name or 'H' in component_name:
        return 'ATTN'
    return 'OTHER'

print("\n" + "="*80)
print("VISUALIZATION 1: Network Graph of Top Pathways")
print("="*80)

# Take top 25 pathways for network graph
top_pathways = significant_df.head(25).copy()

# Create directed graph
G = nx.DiGraph()

# Add nodes and edges
for _, row in top_pathways.iterrows():
    comp1, comp2 = row['component_1'], row['component_2']
    weight = abs(row['correlation_diff'])

    # Add nodes with layer info
    G.add_node(comp1, layer=get_layer_number(comp1), type=get_component_type(comp1))
    G.add_node(comp2, layer=get_layer_number(comp2), type=get_component_type(comp2))

    # Add edge with weight
    G.add_edge(comp1, comp2, weight=weight,
               de_corr=row['Deontological_corr'],
               ut_corr=row['Utilitarian_corr'])

print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# Create layout based on layer numbers
pos = {}
for node in G.nodes():
    layer = G.nodes[node]['layer']
    # Group nodes by layer for y-position
    nodes_in_layer = [n for n in G.nodes() if G.nodes[n]['layer'] == layer]
    idx = nodes_in_layer.index(node)
    pos[node] = (layer, -idx * 0.5)  # x = layer, y = staggered by index

# Create figure
fig, ax = plt.subplots(figsize=(16, 10))

# Color nodes by type
node_colors = []
for node in G.nodes():
    if G.nodes[node]['type'] == 'MLP':
        node_colors.append('#FF6B6B')  # Red for MLPs
    else:
        node_colors.append('#4ECDC4')  # Teal for attention

# Size nodes by degree
node_sizes = [300 + 200 * G.degree(node) for node in G.nodes()]

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                       alpha=0.8, ax=ax)

# Draw edges with thickness based on weight
edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.6,
                       edge_color='gray', arrows=True,
                       arrowsize=15, ax=ax, connectionstyle='arc3,rad=0.1')

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=7, font_weight='bold', ax=ax)

ax.set_title('Network Graph: Top 25 Pathway Differences (Deontological vs Utilitarian)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Layer Number', fontsize=12)
ax.axis('on')
ax.grid(True, alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#FF6B6B', label='MLP'),
    Patch(facecolor='#4ECDC4', label='Attention'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'viz1_network_graph.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: viz1_network_graph.png")
plt.close()

print("\n" + "="*80)
print("VISUALIZATION 2: Scatter Plot - De vs Ut Correlations")
print("="*80)

# Use all interactions for scatter plot
fig, ax = plt.subplots(figsize=(12, 12))

# Separate pathways that flipped sign
all_interactions_df['flipped'] = (
    (all_interactions_df['Deontological_corr'] * all_interactions_df['Utilitarian_corr']) < 0
)
all_interactions_df['significant'] = all_interactions_df['abs_diff'] > 0.3

# Plot all pathways
scatter_colors = []
for _, row in all_interactions_df.iterrows():
    if row['significant'] and row['flipped']:
        scatter_colors.append('#E74C3C')  # Red for significant + flipped
    elif row['significant']:
        scatter_colors.append('#F39C12')  # Orange for significant
    elif row['flipped']:
        scatter_colors.append('#9B59B6')  # Purple for flipped
    else:
        scatter_colors.append('#BDC3C7')  # Gray for others

ax.scatter(all_interactions_df['Deontological_corr'],
          all_interactions_df['Utilitarian_corr'],
          c=scatter_colors, alpha=0.4, s=10)

# Add diagonal line (y=x)
lim = max(abs(all_interactions_df['Deontological_corr'].max()),
          abs(all_interactions_df['Utilitarian_corr'].max()))
ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3, linewidth=2, label='y=x (identical)')

# Add axes lines
ax.axhline(y=0, color='k', linestyle='-', alpha=0.2, linewidth=1)
ax.axvline(x=0, color='k', linestyle='-', alpha=0.2, linewidth=1)

# Annotate top pathways
top_for_annotation = significant_df.head(10)
for _, row in top_for_annotation.iterrows():
    de_val = row['Deontological_corr']
    ut_val = row['Utilitarian_corr']
    label = f"{row['component_1']}→{row['component_2']}"
    ax.annotate(label, (de_val, ut_val), fontsize=6, alpha=0.7,
                xytext=(5, 5), textcoords='offset points')

ax.set_xlabel('Deontological Correlation', fontsize=12, fontweight='bold')
ax.set_ylabel('Utilitarian Correlation', fontsize=12, fontweight='bold')
ax.set_title('Pathway Correlations: Deontological vs Utilitarian\n(Points below diagonal = flipped sign)',
             fontsize=14, fontweight='bold', pad=20)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C',
           markersize=8, label='Significant & Flipped'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#F39C12',
           markersize=8, label='Significant'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#9B59B6',
           markersize=8, label='Flipped'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#BDC3C7',
           markersize=8, label='Other'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'viz2_scatter_de_vs_ut.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: viz2_scatter_de_vs_ut.png")
plt.close()

print("\n" + "="*80)
print("VISUALIZATION 3: Layer-wise Aggregation")
print("="*80)

# Aggregate differences by layer
layer_stats = defaultdict(lambda: {'count': 0, 'total_diff': 0, 'max_diff': 0})

for _, row in significant_df.iterrows():
    layer1 = get_layer_number(row['component_1'])
    layer2 = get_layer_number(row['component_2'])
    diff = row['abs_diff']

    # Add to both layers
    for layer in [layer1, layer2]:
        layer_stats[layer]['count'] += 1
        layer_stats[layer]['total_diff'] += diff
        layer_stats[layer]['max_diff'] = max(layer_stats[layer]['max_diff'], diff)

# Convert to dataframe
layer_df = pd.DataFrame([
    {
        'layer': layer,
        'count': stats['count'],
        'avg_diff': stats['total_diff'] / stats['count'],
        'max_diff': stats['max_diff']
    }
    for layer, stats in sorted(layer_stats.items())
])

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Number of significant pathways per layer
bars1 = ax1.bar(layer_df['layer'], layer_df['count'], color='steelblue', alpha=0.8)
ax1.set_xlabel('Layer Number', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Significant Pathways', fontsize=12, fontweight='bold')
ax1.set_title('Pathway Differences by Layer (Count)', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Highlight L2 and L6
for bar, layer in zip(bars1, layer_df['layer']):
    if layer in [2, 6]:
        bar.set_color('#E74C3C')
        bar.set_alpha(1.0)

# Plot 2: Average difference magnitude per layer
bars2 = ax2.bar(layer_df['layer'], layer_df['avg_diff'], color='darkgreen', alpha=0.8)
ax2.set_xlabel('Layer Number', fontsize=12, fontweight='bold')
ax2.set_ylabel('Average |Correlation Difference|', fontsize=12, fontweight='bold')
ax2.set_title('Pathway Differences by Layer (Magnitude)', fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Highlight L2 and L6
for bar, layer in zip(bars2, layer_df['layer']):
    if layer in [2, 6]:
        bar.set_color('#E74C3C')
        bar.set_alpha(1.0)

# Add text annotations for L2 and L6
for ax in [ax1, ax2]:
    ax.text(2, ax.get_ylim()[1] * 0.95, 'L2_MLP', ha='center', fontsize=9,
            fontweight='bold', color='#E74C3C')
    ax.text(6, ax.get_ylim()[1] * 0.95, 'L6_MLP', ha='center', fontsize=9,
            fontweight='bold', color='#E74C3C')

plt.suptitle('Layer-wise Analysis of Routing Differences', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'viz3_layerwise_aggregation.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: viz3_layerwise_aggregation.png")
plt.close()

print("\n" + "="*80)
print("VISUALIZATION 4: Pathway-Specific Focus (L2_MLP and L6_MLP)")
print("="*80)

# Filter for L2_MLP and L6_MLP pathways
l2_pathways = all_interactions_df[
    (all_interactions_df['component_1'] == 'L2_MLP') |
    (all_interactions_df['component_2'] == 'L2_MLP')
].copy()

l6_pathways = all_interactions_df[
    (all_interactions_df['component_1'] == 'L6_MLP') |
    (all_interactions_df['component_2'] == 'L6_MLP')
].copy()

# Get layer numbers for connected components
def get_connected_layer(row, hub_component):
    """Get the layer number of the component connected to the hub"""
    if row['component_1'] == hub_component:
        return get_layer_number(row['component_2'])
    else:
        return get_layer_number(row['component_1'])

l2_pathways['connected_layer'] = l2_pathways.apply(lambda r: get_connected_layer(r, 'L2_MLP'), axis=1)
l6_pathways['connected_layer'] = l6_pathways.apply(lambda r: get_connected_layer(r, 'L6_MLP'), axis=1)

# Sort by layer
l2_pathways = l2_pathways.sort_values('connected_layer')
l6_pathways = l6_pathways.sort_values('connected_layer')

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# L2_MLP: Deontological correlations
ax = axes[0, 0]
layers = l2_pathways['connected_layer'].values
de_corrs = l2_pathways['Deontological_corr'].values
colors = ['#3498DB' if c > 0 else '#E74C3C' for c in de_corrs]
ax.barh(range(len(layers)), de_corrs, color=colors, alpha=0.7)
ax.set_yticks(range(len(layers)))
ax.set_yticklabels([f'L{l}' for l in layers], fontsize=8)
ax.axvline(x=0, color='k', linestyle='-', linewidth=1, alpha=0.3)
ax.set_xlabel('Correlation', fontsize=10)
ax.set_title('L2_MLP Connections: Deontological Model', fontsize=11, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# L2_MLP: Utilitarian correlations
ax = axes[0, 1]
ut_corrs = l2_pathways['Utilitarian_corr'].values
colors = ['#3498DB' if c > 0 else '#E74C3C' for c in ut_corrs]
ax.barh(range(len(layers)), ut_corrs, color=colors, alpha=0.7)
ax.set_yticks(range(len(layers)))
ax.set_yticklabels([f'L{l}' for l in layers], fontsize=8)
ax.axvline(x=0, color='k', linestyle='-', linewidth=1, alpha=0.3)
ax.set_xlabel('Correlation', fontsize=10)
ax.set_title('L2_MLP Connections: Utilitarian Model', fontsize=11, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# L6_MLP: Deontological correlations
ax = axes[1, 0]
layers = l6_pathways['connected_layer'].values
de_corrs = l6_pathways['Deontological_corr'].values
colors = ['#3498DB' if c > 0 else '#E74C3C' for c in de_corrs]
ax.barh(range(len(layers)), de_corrs, color=colors, alpha=0.7)
ax.set_yticks(range(len(layers)))
ax.set_yticklabels([f'L{l}' for l in layers], fontsize=8)
ax.axvline(x=0, color='k', linestyle='-', linewidth=1, alpha=0.3)
ax.set_xlabel('Correlation', fontsize=10)
ax.set_title('L6_MLP Connections: Deontological Model', fontsize=11, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# L6_MLP: Utilitarian correlations
ax = axes[1, 1]
ut_corrs = l6_pathways['Utilitarian_corr'].values
colors = ['#3498DB' if c > 0 else '#E74C3C' for c in ut_corrs]
ax.barh(range(len(layers)), ut_corrs, color=colors, alpha=0.7)
ax.set_yticks(range(len(layers)))
ax.set_yticklabels([f'L{l}' for l in layers], fontsize=8)
ax.axvline(x=0, color='k', linestyle='-', linewidth=1, alpha=0.3)
ax.set_xlabel('Correlation', fontsize=10)
ax.set_title('L6_MLP Connections: Utilitarian Model', fontsize=11, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

plt.suptitle('Routing Hub Analysis: L2_MLP and L6_MLP Connectivity Patterns',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'viz4_hub_specific_pathways.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: viz4_hub_specific_pathways.png")
plt.close()

print("\n" + "="*80)
print("VISUALIZATION 5: Sankey Diagram of Information Flow")
print("="*80)

# Create simplified Sankey showing flow through key hubs
# We'll show: Early layers -> L2_MLP/L6_MLP -> L8/L9 -> Late layers

# Aggregate pathways into flow categories
def categorize_layer(layer_num):
    """Categorize layers into groups"""
    if layer_num <= 1:
        return 'Early (L0-L1)'
    elif 2 <= layer_num <= 5:
        return 'L2-L5'
    elif layer_num == 6:
        return 'L6_MLP'
    elif 7 <= layer_num <= 10:
        return 'L7-L10'
    elif layer_num in [8, 9]:
        return f'L{layer_num}_MLP'
    elif 11 <= layer_num <= 20:
        return 'Mid (L11-L20)'
    else:
        return 'Late (L21-L25)'

# Calculate flows for Deontological model (using top pathways)
flows = defaultdict(float)
for _, row in significant_df.head(30).iterrows():
    comp1, comp2 = row['component_1'], row['component_2']
    layer1, layer2 = get_layer_number(comp1), get_layer_number(comp2)

    # Special handling for L2_MLP and L6_MLP
    if comp1 == 'L2_MLP':
        source = 'L2_MLP'
    elif comp1 == 'L6_MLP':
        source = 'L6_MLP'
    else:
        source = categorize_layer(layer1)

    if comp2 == 'L2_MLP':
        target = 'L2_MLP'
    elif comp2 == 'L6_MLP':
        target = 'L6_MLP'
    elif comp2 == 'L9_MLP':
        target = 'L9_MLP'
    elif comp2 == 'L8_MLP':
        target = 'L8_MLP'
    else:
        target = categorize_layer(layer2)

    flow_key = (source, target)
    flows[flow_key] += abs(row['correlation_diff'])

# Create figure using matplotlib (simple arrow-based flow diagram)
fig, ax = plt.subplots(figsize=(14, 10))

# Define node positions (x, y)
node_positions = {
    'Early (L0-L1)': (0, 4),
    'L2_MLP': (2, 5),
    'L2-L5': (2, 3),
    'L6_MLP': (4, 4),
    'L7-L10': (4, 2),
    'L8_MLP': (6, 5),
    'L9_MLP': (6, 3),
    'Mid (L11-L20)': (8, 4),
    'Late (L21-L25)': (10, 4),
}

# Draw nodes
for node, (x, y) in node_positions.items():
    is_hub = node in ['L2_MLP', 'L6_MLP', 'L8_MLP', 'L9_MLP']
    color = '#E74C3C' if is_hub else '#3498DB'
    size = 2500 if is_hub else 1500
    ax.scatter(x, y, s=size, c=color, alpha=0.7, edgecolors='black', linewidths=2, zorder=10)
    ax.text(x, y, node, ha='center', va='center', fontsize=9 if is_hub else 8,
            fontweight='bold' if is_hub else 'normal', zorder=11)

# Draw edges (flows)
for (source, target), weight in sorted(flows.items(), key=lambda x: x[1], reverse=True)[:20]:
    if source in node_positions and target in node_positions:
        x1, y1 = node_positions[source]
        x2, y2 = node_positions[target]

        # Draw arrow
        arrow_width = weight * 2
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=arrow_width, alpha=0.4,
                                 color='gray', connectionstyle='arc3,rad=0.2'))

ax.set_xlim(-1, 11)
ax.set_ylim(1, 6)
ax.set_title('Information Flow Through Routing Hubs\n(Thickness = Pathway Difference Magnitude)',
             fontsize=14, fontweight='bold', pad=20)
ax.axis('off')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#E74C3C', label='Key Routing Hubs', alpha=0.7),
    Patch(facecolor='#3498DB', label='Layer Groups', alpha=0.7),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'viz5_sankey_flow.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: viz5_sankey_flow.png")
plt.close()

print("\n" + "="*80)
print("ALL VISUALIZATIONS COMPLETE!")
print("="*80)
print(f"\nOutput directory: {OUTPUT_DIR}")
print("\nGenerated files:")
print("  1. viz1_network_graph.png - Network graph of top 25 pathways")
print("  2. viz2_scatter_de_vs_ut.png - Scatter plot comparing correlations")
print("  3. viz3_layerwise_aggregation.png - Layer-wise analysis")
print("  4. viz4_hub_specific_pathways.png - L2_MLP and L6_MLP connections")
print("  5. viz5_sankey_flow.png - Information flow diagram")
print("\n" + "="*80)
