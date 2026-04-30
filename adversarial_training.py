import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm

# Import your custom modules
import random_shooting_warp as rs
import discriminator as disc
import learn_params as lp
import mujoco

# ─── CONFIGURATION & GRID SEARCH SETUP ────────────────────────────────────────

def get_default_config():
    """Default configuration dictionary for easy grid search integration."""
    return {
        "env_xml": "bouncing_ball.xml",
        "duration": 2.0,
        "mppi_noise_sigma": 1.0,
        "mppi_samples": 1000,
        "num_goals": 10,                 # Number of goals to sample per GAN iteration
        "goal_dist_mean": [0.0, 0.0, 0.0],
        "goal_dist_std": [0.5, 0.0, 0.0], # e.g., vary X and Y, keep Z flat
        "gan_iterations": 10,           # Outer loops
        "d_epochs": 200,                 # Discriminator training epochs per loop
        "d_lr": 0.001,
        "d_batch_size": 16,
        "g_optim_algo": "Powell",       # Scipy optimizer (Powell, Nelder-Mead, L-BFGS-B)
        "g_max_iters": 10,#50,              # Max function evaluations per G-step
        "init_k": 0.5,
        "init_d": 0.1,
        "gt_k": 0.2,                    # Ground truth for standard Mujoco simulation
        "gt_d": 0.001,
        "use_com_free_for_gt":True,
        "output_dir": "./gan_results"
    }

# ─── TRAJECTORY COLLECTION ────────────────────────────────────────────────────

def sample_goals(config):
    """1) Sample goal positions from a normal distribution."""
    goals = np.random.normal(
        loc=config["goal_dist_mean"], 
        scale=config["goal_dist_std"], 
        size=(config["num_goals"], 3)
    )
    return goals

def collect_trajectories(config, goals, k, d, use_comfree, fixed_noise):
    """2 & 3) Run MPPI on models and record trajectories."""
    mj_model = mujoco.MjModel.from_xml_path(config["env_xml"])
    mj_data = mujoco.MjData(mj_model)
    
    starting_pos = mj_data.qpos[:3].copy()
    trajectories = []
    
    for goal in goals:
        base_qvel = goal - starting_pos
        
        # Use the fixed noise instead of resampling
        optimal_qvel = lp.get_iterative_mppi_qvel(
            mj_model, mj_data, base_qvel, fixed_noise, config["duration"], k, d
        )
        
        # Simulate final trajectory using the optimal velocity
        pos_batch, _ = rs.simulate_trajectories_parallel(
            mj_model, mj_data, optimal_qvel[np.newaxis, :], 
            config["duration"], k, d, use_comfree=use_comfree
        )
        trajectories.append(pos_batch[0]) # (TimeSteps, 3)
        
    return np.array(trajectories)

# ─── DISCRIMINATOR TRAINING ───────────────────────────────────────────────────

def train_discriminator(D, optimizer, real_trajs, fake_trajs, config):
    """4) Train the discriminator for a number of epochs."""
    criterion = nn.BCELoss()
    device = next(D.parameters()).device
    
    # Create dataset: Real=1, Fake=0
    X = np.vstack((real_trajs, fake_trajs))
    y = np.vstack((np.ones((len(real_trajs), 1)), np.zeros((len(fake_trajs), 1))))
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config["d_batch_size"], shuffle=True)
    
    D.train()
    total_loss = 0
    for epoch in range(config["d_epochs"]):
        epoch_loss = 0
        for seqs, labels in loader:
            optimizer.zero_grad()
            outputs = D(seqs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        total_loss += epoch_loss / len(loader)
        
    return total_loss / config["d_epochs"]

# ─── GENERATOR / PARAMETER OPTIMIZATION ───────────────────────────────────────

def optimize_parameters(D, config, goals, current_k, current_d, fixed_noise):
    """5) GAN style optimization using the discriminator to score trajectories."""
    device = next(D.parameters()).device
    mj_model = mujoco.MjModel.from_xml_path(config["env_xml"])
    mj_data = mujoco.MjData(mj_model)
    starting_pos = mj_data.qpos[:3].copy()
    
    D.eval()
    
    # We optimize in log-space to ensure positivity
    initial_params = np.log([current_k, current_d])

    def objective(params):
        k, d = np.exp(params)
        generated_trajs = []
        
        # Generate trajectories for all goals using the SAME fixed noise
        for goal in goals:
            base_qvel = goal - starting_pos
            opt_qvel = lp.get_iterative_mppi_qvel(
                mj_model, mj_data, base_qvel, fixed_noise, config["duration"], k, d
            )
            pos_batch, _ = rs.simulate_trajectories_parallel(
                mj_model, mj_data, opt_qvel[np.newaxis, :], 
                config["duration"], k, d, use_comfree=True
            )
            generated_trajs.append(pos_batch[0])
            
        generated_trajs = torch.tensor(np.array(generated_trajs), dtype=torch.float32).to(device)
        
        with torch.no_grad():
            d_scores = D(generated_trajs)
            loss = torch.mean(1.0 - d_scores).item()
            
        return loss

    res = minimize(
        objective, 
        initial_params, 
        method=config["g_optim_algo"],                
        bounds=[(-10, 10), (-10, 10)],
        options={'maxfev': config["g_max_iters"], 'xatol': 1e-4} if config["g_optim_algo"] == 'Nelder-Mead' else {'maxfev': config["g_max_iters"]}
    )
    
    best_k, best_d = np.exp(res.x)
    return best_k, best_d, res.fun

# ─── MAIN GAN LOOP ────────────────────────────────────────────────────────────

def run_gan_optimization(config):
    """6 & 7) Main loop, running until convergence and saving artifacts."""
    os.makedirs(config["output_dir"], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")
    
    # Initialize Discriminator
    D = disc.LSTMDiscriminator(input_size=3, hidden_size=64, num_layers=2).to(device)
    optimizer = optim.Adam(D.parameters(), lr=config["d_lr"])
    
    current_k = config["init_k"]
    current_d = config["init_d"]
    
    # Initialize fixed noise ONCE for the entire training run
    #fixed_noise = np.random.normal(0, config["mppi_noise_sigma"], size=(config["mppi_samples"], 3))
    rng = np.random.default_rng(seed=42)
    fixed_noise = np.zeros((int(config["mppi_samples"]), 3), dtype=np.float32)
    fixed_noise[:, [0, 2]] = rng.normal(0, config["mppi_noise_sigma"], size=(int(config["mppi_samples"]), 2)).astype(np.float32)
    
    history = []
    
    for iteration in range(config["gan_iterations"]):
        print(f"\n=== GAN Iteration {iteration+1}/{config['gan_iterations']} ===")
        
        # 1. Sample goals
        goals = sample_goals(config)
        print(f"Sampled {len(goals)} new goals.")
        
        # 2. Collect Real and Fake Data (Pass fixed_noise)
        print("Collecting Ground Truth (Standard MuJoCo) trajectories...")
        real_trajs = collect_trajectories(config, goals, config["gt_k"], config["gt_d"], use_comfree=False, fixed_noise=fixed_noise)
        
        print("Collecting Generated (ComFree_Warp) trajectories...")
        fake_trajs = collect_trajectories(config, goals, current_k, current_d, use_comfree=True, fixed_noise=fixed_noise)
        
        # 3. Train Discriminator
        print(f"Training Discriminator for {config['d_epochs']} epochs...")
        d_loss = train_discriminator(D, optimizer, real_trajs, fake_trajs, config)
        print(f"Discriminator Loss: {d_loss:.4f}")
        
        # 4. Train Generator (Pass fixed_noise)
        print("Optimizing ComFree parameters to fool the Discriminator...")
        new_goals = sample_goals(config) 
        best_k, best_d, g_loss = optimize_parameters(D, config, new_goals, current_k, current_d, fixed_noise)
        print(f"Generator Loss: {g_loss:.4f} | Updated Params -> K: {best_k:.5f}, D: {best_d:.5f}")
        
        current_k, current_d = best_k, best_d
        
        # Record metrics
        history.append({
            "iteration": iteration + 1,
            "d_loss": d_loss,
            "g_loss": g_loss,
            "stiffness_k": current_k,
            "damping_d": current_d
        })
        
    # 7. Save outputs
    save_results(history, D, config)
    return history

def save_results(history, D, config):
    """Saves CSV, Model Weights, and Plots."""
    df = pd.DataFrame(history)
    csv_path = os.path.join(config["output_dir"], "training_history.csv")
    df.to_csv(csv_path, index=False)
    
    weights_path = os.path.join(config["output_dir"], "discriminator_weights.pth")
    torch.save(D.state_dict(), weights_path)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot Losses
    ax1.plot(df["iteration"], df["d_loss"], label="D Loss (BCE)", marker='o')
    ax1.plot(df["iteration"], df["g_loss"], label="G Loss (1 - D_score)", marker='o')
    ax1.set_title("GAN Losses")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)
    
    # Plot Parameters
    ax2.plot(df["iteration"], df["stiffness_k"], label="Learned Stiffness (k)", marker='x', color='green')
    ax2.axhline(config["gt_k"], color='green', linestyle='--', label="GT Stiffness")
    ax2.plot(df["iteration"], df["damping_d"], label="Learned Damping (d)", marker='x', color='red')
    ax2.axhline(config["gt_d"], color='red', linestyle='--', label="GT Damping")
    ax2.set_title("Parameter Evolution")
    ax2.set_xlabel("GAN Iteration")
    ax2.set_ylabel("Parameter Value")
    ax2.set_yscale("log") # Log scale helps view damping and stiffness simultaneously
    ax2.legend()
    ax2.grid(True)
    
    plot_path = os.path.join(config["output_dir"], "training_curves.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"\nTraining complete. Artifacts saved to: {config['output_dir']}")

# ─── EXECUTION ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example Grid Search usage
    algorithms = ["Powell"] # Add "Nelder-Mead", "L-BFGS-B" to test others
    
    for algo in algorithms:
        print(f"\n{'='*50}\nStarting Optimization with {algo}\n{'='*50}")
        config = get_default_config()
        config["g_optim_algo"] = algo
        config["output_dir"] = f"./gan_results_{algo.lower()}"
        
        # For a faster dry-run, you might lower these:
        #config["gan_iterations"] = 5
        #config["mppi_samples"] = 100 
        
        run_gan_optimization(config)