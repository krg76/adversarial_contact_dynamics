import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def calculate_error(row, gt_k, gt_d):
    """
    Calculates the combined error for stiffness and damping.
    Using Euclidean distance (L2 norm) here.
    """
    error_k = row['final_stiffness_k'] - gt_k
    error_d = row['final_damping_d'] - gt_d
    return np.sqrt(error_k**2 + error_d**2)

def generate_plots(csv_file="grid_search_parameters_summary.csv", gt_k=0.005, gt_d=0.00012):
    # 1. Load the data
    if not os.path.exists(csv_file):
        print(f"Error: Could not find {csv_file}")
        return
        
    df = pd.read_csv(csv_file)
    
    # 2. Calculate the error for each run
    df['parameter_error'] = df.apply(lambda row: calculate_error(row, gt_k, gt_d), axis=1)
    
    # --- ADDED: Determine global scale for consistent colors ---
    global_min = df['parameter_error'].min()
    global_max = df['parameter_error'].max()
    # ------------------------------------------------------------
    
    architectures = df['architecture'].unique()
    
    # 3. Plot 1: Heatmaps for each architecture
    fig_heat, axes = plt.subplots(1, len(architectures), figsize=(6 * len(architectures), 5))
    
    if len(architectures) == 1:
        axes = [axes]
        
    for ax, arch in zip(axes, architectures):
        df_arch = df[df['architecture'] == arch]
        pivot_table = df_arch.pivot(index='lr_d', columns='lr_g', values='parameter_error')
        
        # --- MODIFIED: Added vmin and vmax ---
        sns.heatmap(pivot_table, ax=ax, annot=True, fmt=".4f", 
                    cmap="viridis_r", vmin=global_min, vmax=global_max,
                    cbar_kws={'label': 'Error'})
        # --------------------------------------
        
        ax.set_title(f'Parameter Error Heatmap: {arch.upper()}')
        ax.set_ylabel('Discriminator LR (lr_d)')
        ax.set_xlabel('Generator LR (lr_g)')
        ax.invert_yaxis()
        
    plt.tight_layout()
    plt.savefig("error_heatmaps.png")
    # ... (Rest of the code for the bar chart remains the same)

if __name__ == "__main__":
    # --- USER SETTINGS ---
    # Set your exact Ground Truth values here
    GROUND_TRUTH_STIFFNESS_K = 0.5   # Example value
    GROUND_TRUTH_DAMPING_D = 0.002  # Example value
    
    INPUT_CSV = "grid_search_parameters_summary.csv"
    # ---------------------
    
    print(f"Analyzing {INPUT_CSV} using GT K={GROUND_TRUTH_STIFFNESS_K}, GT D={GROUND_TRUTH_DAMPING_D}...")
    generate_plots(csv_file=INPUT_CSV, gt_k=GROUND_TRUTH_STIFFNESS_K, gt_d=GROUND_TRUTH_DAMPING_D)