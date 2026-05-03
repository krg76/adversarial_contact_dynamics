import os
import re
import pandas as pd
from pathlib import Path

def summarize_grid_search(base_directory=".", output_csv="grid_search_summary.csv"):
    """
    Scans the base directory for run folders, extracts hyperparameters from 
    the folder names, and fetches the final stiffness and damping parameters.
    """
    
    # Regular expression to match the folder names and capture groups for hyperparameters
    # Example match: "run_cnn_lr_d0.0005_lr_g0.05"
    folder_pattern = re.compile(r"run_(?P<arch>cnn|mlp|lstm)_lr_d(?P<lr_d>[\d.]+)_lr_g(?P<lr_g>[\d.]+)")
    
    summary_data = []
    base_path = Path(base_directory)
    
    # Iterate through all items in the base directory
    for item in base_path.iterdir():
        if not item.is_dir():
            continue
            
        match = folder_pattern.search(item.name)
        if match:
            # Extract hyperparameters from the folder name
            arch = match.group("arch")
            lr_d = float(match.group("lr_d"))
            lr_g = float(match.group("lr_g"))
            
            history_path = item / "training_history.csv"
            
            # Check if the training history file exists in this directory
            if history_path.exists():
                try:
                    # Read the CSV file
                    df = pd.read_csv(history_path)
                    
                    if not df.empty:
                        # Get the final row for losses and iteration count
                        final_row = df.iloc[-1]
                        
                        run_info = {
                            "run_name": item.name,
                            "architecture": arch,
                            "lr_d": lr_d,
                            "lr_g": lr_g,
                            "final_iteration": final_row.get("iteration"),
                            "final_d_loss": final_row.get("d_loss"),
                            "final_g_loss": final_row.get("g_loss"),
                        }

                        # Define all potential parameter columns for both types
                        param_cols = ["stiffness_k", "damping_d", "k1", "k2", "k3", "d1", "d2", "d3"]
                        for col in param_cols:
                            if col in df.columns:
                                # Extract the last non-null value (useful if params stop being logged)
                                val = df[col].dropna().iloc[-1] if not df[col].dropna().empty else None
                                run_info[f"final_{col}"] = val

                        summary_data.append(run_info)
                    else:
                        print(f"Warning: {history_path} is empty.")
                        
                except Exception as e:
                    print(f"Error reading {history_path}: {e}")
            else:
                print(f"Warning: 'training_history.csv' not found inside {item.name}")

    # Export the collected data to a new CSV file
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_csv, index=False)
        print(f"\nSuccess! Analyzed {len(summary_data)} runs.")
        print(f"Results saved to: {output_csv}")
    else:
        print("\nNo valid runs found. Please check your folder names and directory path.")

if __name__ == "__main__":
    # You can change '.' to the absolute path of the directory containing your run folders
    # e.g., target_dir = "/path/to/your/experiments/"
    target_dir = "results/nonliear_results" 
    
    summarize_grid_search(
        base_directory=target_dir, 
        output_csv="grid_search_parameters_summary.csv"
    )