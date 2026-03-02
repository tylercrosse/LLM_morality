import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def run():
    base_dir = "mech_interp_outputs/component_interactions"
    de_path = os.path.join(base_dir, "component_activations_PT3_COREDe.json")
    ut_path = os.path.join(base_dir, "component_activations_PT3_COREUt.json")
    
    de_data = load_data(de_path)
    ut_data = load_data(ut_path)
    
    # We want to pick a concrete pair we know changed a lot.
    # From WRITE_UP: L19_ATTN and L22_MLP (or L24_MLP, etc) had a large shift.
    # Let's find the pair with the actual max absolute difference in correlation.
    
    components = list(de_data[0]['component_activations'].keys())
    n_scenarios = len(de_data)
    
    de_acts = np.zeros((len(components), n_scenarios))
    ut_acts = np.zeros((len(components), n_scenarios))
    
    for i, d in enumerate(de_data):
        for j, c in enumerate(components):
            de_acts[j, i] = d['component_activations'][c]
            
    for i, d in enumerate(ut_data):
        for j, c in enumerate(components):
            ut_acts[j, i] = d['component_activations'][c]
            
    # Calculate correlation matrices
    de_corr = np.corrcoef(de_acts)
    ut_corr = np.corrcoef(ut_acts)
    
    diff = de_corr - ut_corr
    np.fill_diagonal(diff, 0)
    
    # Find max absolute difference
    max_idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
    c1, c2 = components[max_idx[0]], components[max_idx[1]]
    print(f"Max diff pair: {c1} & {c2}, De corr: {de_corr[max_idx]:.3f}, Ut corr: {ut_corr[max_idx]:.3f}, Diff: {diff[max_idx]:.3f}")
    
    # Also find a specific known hub connection like L19_ATTN to something
    l19_idx = components.index('L19_ATTN')
    l19_diffs = np.abs(diff[l19_idx, :])
    best_l19_partner_idx = np.argmax(l19_diffs)
    c3 = components[best_l19_partner_idx]
    
    print(f"L19_ATTN max diff partner: {c3}, De corr: {de_corr[l19_idx, best_l19_partner_idx]:.3f}, Ut corr: {ut_corr[l19_idx, best_l19_partner_idx]:.3f}")
    
    # Let's plot L19_ATTN and its best partner
    comp_A_idx, comp_B_idx = l19_idx, best_l19_partner_idx
    comp_A, comp_B = 'L19_ATTN', c3
    
    # We will standardize the activations for plotting so they fit on the same y-axis visually
    def standardize(x):
        return (x - np.mean(x)) / (np.std(x) + 1e-8)
        
    de_A = standardize(de_acts[comp_A_idx])
    de_B = standardize(de_acts[comp_B_idx])
    
    ut_A = standardize(ut_acts[comp_A_idx])
    ut_B = standardize(ut_acts[comp_B_idx])
    
    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    sns.set_theme(style="whitegrid")
    
    x = np.arange(1, n_scenarios + 1)
    
    axes[0].plot(x, de_A, marker='o', label=comp_A, color='#d62728', linewidth=2)
    axes[0].plot(x, de_B, marker='s', label=comp_B, color='#1f77b4', linewidth=2)
    axes[0].set_title(f"Deontological Model (r = {de_corr[comp_A_idx, comp_B_idx]:.2f})", fontsize=14)
    axes[0].set_xlabel("Evaluation Scenario", fontsize=12)
    axes[0].set_ylabel("Standardized Activation\n(Z-Score)", fontsize=12)
    axes[0].set_xticks(x)
    
    axes[1].plot(x, ut_A, marker='o', label=comp_A, color='#d62728', linewidth=2)
    axes[1].plot(x, ut_B, marker='s', label=comp_B, color='#1f77b4', linewidth=2)
    axes[1].set_title(f"Utilitarian Model (r = {ut_corr[comp_A_idx, comp_B_idx]:.2f})", fontsize=14)
    axes[1].set_xlabel("Evaluation Scenario", fontsize=12)
    axes[1].set_xticks(x)
    axes[1].legend(loc='upper right')
    
    plt.suptitle(f"Concrete Example of Network Rewiring\nInteraction between {comp_A} and {comp_B}", fontsize=16, y=1.05)
    plt.tight_layout()
    
    out_path = os.path.join(base_dir, "additional_viz", "viz6_concrete_rewiring_example.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    run()
