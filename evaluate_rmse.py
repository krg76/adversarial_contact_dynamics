import argparse
import pandas as pd
import numpy as np
import os

# Import the existing pipeline from your adversarial training script
import adversarial_training as at

def compute_rmse(real_trajectories, fake_trajectories):
    """Computes the Root Mean Squared Error between two sets of trajectories."""
    mse = np.mean((real_trajectories - fake_trajectories) ** 2)
    return np.sqrt(mse)

def main(csv_files, output_file):
    # 1. Load the default configuration to ensure consistency with training
    config = at.get_default_config()
    config["num_goals"] = 1
    config["goal_dist_std"] = [0.0, 0.0, 0.0],
    
    # 2. Setup fixed noise and goals for a fair deterministic evaluation
    print("Setting up evaluation environment...")
    np.random.seed(42)  # Fixed seed for goal sampling
    goals = np.array(config["goal_dist_mean"])[np.newaxis, :]#at.sample_goals(config)
    
    rng = np.random.default_rng(seed=42)
    fixed_noise = np.zeros((int(config["mppi_samples"]), 3), dtype=np.float32)
    fixed_noise[:, [0, 2]] = rng.normal(
        0, config["mppi_noise_sigma"], size=(int(config["mppi_samples"]), 2)
    ).astype(np.float32)

    # 3. Pre-compute Ground Truth (Real) trajectories
    print(f"Collecting Ground Truth trajectories for {len(goals)} goals...")
    real_trajs = at.collect_trajectories(
        config=config, 
        goals=goals, 
        k=config["gt_k"], 
        d=config["gt_d"],
        use_comfree=config["use_com_free_for_gt"], 
        fixed_noise=fixed_noise
    )

    all_results = []

    # 4. Loop through provided CSV files
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"Warning: File not found {csv_file}")
            continue
            
        print(f"\nProcessing {csv_file}...")
        df = pd.read_csv(csv_file)
        
        # 5. Evaluate each run in the CSV
        for index, row in df.iterrows():
            run_name = row['run_name']
            
            # Extract the final k and d parameters
            k_fake = [row['final_k1']]
            d_fake = [row['final_d1']]
            
            print(f"  Evaluating {run_name} | k: {k_fake} | d: {d_fake}")
            
            # Generate trajectories with the optimized parameters
            fake_trajs = at.collect_trajectories(
                config=config, 
                goals=goals, 
                k=k_fake, 
                d=d_fake,
                use_comfree=True, 
                fixed_noise=fixed_noise
            )
            
            # Compute RMSE
            rmse = compute_rmse(real_trajs, fake_trajs)
            print(f"    -> RMSE: {rmse:.6f}")
            
            # Append result row
            row_dict = row.to_dict()
            row_dict['rmse'] = rmse
            all_results.append(row_dict)

    # 6. Save final consolidated results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_file, index=False)
        print(f"\nEvaluation complete. Results saved with RMSE to: {output_file}")
    else:
        print("\nNo data processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RMSE of GAN Grid Search Results")
    parser.add_argument(
        "--csv_files", 
        nargs="+", 
        required=True, 
        help="List of CSV files to process (e.g., --csv_files grid1.csv grid2.csv)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="grid_search_rmse_results.csv", 
        help="Output CSV filename"
    )
    
    args = parser.parse_args()
    main(args.csv_files, args.output)