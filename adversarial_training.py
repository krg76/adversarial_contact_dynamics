import os
import gc
import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm

import random_shooting_warp as rs
import discriminator as disc
import learn_params as lp
import mujoco
import warp as wp
import mujoco_warp as mw
import comfree_warp as cf_mjwarp

#wp.config.verify_cuda = True # Checks for OOB memory access synchronously
#wp.config.verify_fp = True   # Checks for NaNs or Infinities in physics

# ─── CONFIGURATION & GRID SEARCH SETUP ────────────────────────────────────────
#python adversarial_training.py --disc_type "cnn"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_lr", type=float, default=0.00025)
    parser.add_argument("--g_lr", type=float, default=0.0025)
    parser.add_argument("--g_l2_weight", type=float, default=1e-5)
    parser.add_argument("--g_reg", type=float, default=0.1)
    parser.add_argument("--disc_type", type=str, choices=["lstm", "cnn", "mlp"], default="lstm")
    parser.add_argument("--output_dir", type=str, default="./gan_results")
    parser.add_argument("--gan_iterations", type=int, default=20)
    # Add any other config keys you wish to tune here
    return parser.parse_args()

def get_default_config():
    return {
        "env_xml": "bouncing_ball.xml",
        "duration": 1.0,
        "mppi_noise_sigma": 5.0,
        "mppi_samples": 256,
        "num_goals": 10,
        "num_goals_gen_train": 5,
        "goal_dist_mean": [0.0, 0.0, 0.0],
        "goal_dist_std": [1.0, 0.0, 0.0],
        "gan_iterations": 100,
        "disc_type": "lstm",
        "d_epochs": 10,
        "d_lr": 0.00025,
        "d_batch_size": 16,
        "d_r1_gamma": 1e-3,          # ← NEW: set to 0.0 to disable R1 reg
        "g_optim_algo": "GD",
        "g_max_iters": 10,#10,
        "g_lr": 0.0025,
        "g_eps": 0.0001,
        "g_reg": 1e-1,
        "g_l2_weight": 1e-5,
        "init_k": 0.5,#[0.4, 0.0001, 0.00005],
        "init_d": 0.001,#[0.01, 0.0002, 0.00002],
        "gt_k": 0.5,#[0.5, 0.0005, 0.00005],
        "gt_d": 0.001,#[0.001, 0.0001, 0.00001],
        "use_com_free_for_gt": False,
        "output_dir": "./gan_comfree_tests_results"
    }

# ─── TRAJECTORY COLLECTION ────────────────────────────────────────────────────

def sample_goals(config):
    goals = np.random.normal(
        loc=config["goal_dist_mean"],
        scale=config["goal_dist_std"],
        size=(config["num_goals"], 3)
    )
    return goals

def collect_trajectories(config, goals, k, d, use_comfree, fixed_noise):
    mj_model = mujoco.MjModel.from_xml_path(config["env_xml"])
    mj_data = mujoco.MjData(mj_model)
    starting_pos = mj_data.qpos[:3].copy()
    trajectories = []

    for goal in goals:
        base_qvel = goal - starting_pos
        optimal_qvel = lp.get_iterative_mppi_qvel(
            mj_model, mj_data, base_qvel, config["duration"], k, d, goal, fixed_noise=fixed_noise
        )
        pos_batch, _ = rs.simulate_trajectories_parallel(
            mj_model, mj_data, optimal_qvel[np.newaxis, :],
            config["duration"], k, d, use_comfree=use_comfree
        )
        trajectories.append(pos_batch[0])

    return np.array(trajectories)

# ─── DISCRIMINATOR TRAINING ───────────────────────────────────────────────────

def train_discriminator(D, optimizer, real_trajs, fake_trajs, config):
    """Train the discriminator with BCE loss + R1 gradient penalty on real samples."""
    criterion = nn.BCEWithLogitsLoss()#nn.BCELoss()
    device = next(D.parameters()).device
    r1_gamma = config.get("d_r1_gamma", 0.0)

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

            # ── Standard BCE loss ─────────────────────────────────────────────
            outputs = D(seqs)
            loss = criterion(outputs, labels)

            # ── R1 gradient penalty on real samples ───────────────────────────
            
            if r1_gamma > 0.0:
                # Corrected R1 logic
                real_mask = (labels == 1).squeeze()
                if real_mask.any():
                    real_seqs = seqs[real_mask].detach().requires_grad_(True)
                    with torch.backends.cudnn.flags(enabled=False):
                        real_scores = D(real_seqs)
                    # ... rest of the penalty calculation

                # CuDNN RNNs don't support double backward, so disable it for this forward pass only
                with torch.backends.cudnn.flags(enabled=False):
                    real_scores = D(real_seqs)

                grads = torch.autograd.grad(
                    outputs=real_scores.sum(),
                    inputs=real_seqs,
                    create_graph=True,
                )[0]

                r1_penalty = (r1_gamma / 2.0) * grads.pow(2).sum([1, 2]).mean()
                loss = loss + r1_penalty

            loss.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        total_loss += epoch_loss / len(loader)

    return total_loss / config["d_epochs"]

# ─── GENERATOR / PARAMETER OPTIMIZATION ───────────────────────────────────────

def optimize_parameters(D, config, goals, current_k, current_d, fixed_noise):
    device = next(D.parameters()).device
    mj_model = mujoco.MjModel.from_xml_path(config["env_xml"])
    mj_data = mujoco.MjData(mj_model)
    starting_pos = mj_data.qpos[:3].copy()

    # Pre-calculate Ground Truth trajectories for the new goals
    gt_trajs = collect_trajectories(config, goals, config["gt_k"], config["gt_d"],
                                    use_comfree=config["use_com_free_for_gt"], fixed_noise=fixed_noise)

    D.eval()

    # Create a 2-element array: [k, d]
    init_p = np.array([current_k, current_d], dtype=np.float64)
    log_params = torch.tensor(np.log(init_p), requires_grad=False)
    log_params.grad = torch.zeros_like(log_params)
    adam = torch.optim.SGD([log_params], lr=config["g_lr"])
    fd_eps = config["g_eps"]

    def objective(log_p: np.ndarray) -> float:
        p = np.exp(log_p)
        # Extract as single scalars
        k, d = p[0], p[1] 
        
        generated_trajs = []
        for goal in goals:
            base_qvel = goal - starting_pos
            # Pass scalars to your simulation functions
            opt_qvel = lp.get_iterative_mppi_qvel(
                mj_model, mj_data, base_qvel, config["duration"],
                k, d, goal, fixed_noise=fixed_noise,
            )
            pos_batch, _ = rs.simulate_trajectories_parallel(
                mj_model, mj_data, opt_qvel[np.newaxis, :],
                config["duration"], k, d, use_comfree=True,
            )
            generated_trajs.append(pos_batch[0])

        traj_array = np.array(generated_trajs)
        l2_penalty = np.mean((traj_array - gt_trajs)**2)
        if np.isnan(l2_penalty).any():
            l2_penalty = 1e10
        

        traj_tensor = torch.tensor(traj_array, dtype=torch.float32).to(device)
        with torch.no_grad():
            d_logits = D(traj_tensor)
            d_scores = torch.sigmoid(d_logits)
            gan_loss = torch.mean(1.0 - d_scores).item()
            reg_loss = (config["g_reg"] * torch.mean(1.0 / torch.cosh(d_logits))).item()
            
        total_loss = gan_loss + reg_loss + config.get("g_l2_weight", 1.0) * l2_penalty
        return total_loss

    best_loss = float("inf")
    best_log_p = log_params.detach().numpy().copy()

    for step in range(config["g_max_iters"]):
        log_p_np = log_params.detach().numpy()
        #print(f"DEBUG: Trying parameters (k, d): {np.exp(log_p_np)}")
        f0 = objective(log_p_np)

        grad_np = np.zeros_like(log_p_np)
        for i in range(len(log_p_np)):
            p_plus = log_p_np.copy(); p_plus[i] += fd_eps
            grad_np[i] = (objective(p_plus) - f0) / (fd_eps)

        log_params.grad.copy_(torch.tensor(grad_np, dtype=torch.float64))
        torch.nn.utils.clip_grad_norm_([log_params], max_norm=1.0)
        adam.step()

        with torch.no_grad():
            log_params.clamp_(-25.0, 1.0)

        gc.collect()

        p_curr = np.exp(log_p_np)
        k_str = ", ".join([f"{v:.2e}" for v in p_curr[:1]])
        d_str = ", ".join([f"{v:.2e}" for v in p_curr[1:]])
        print(f"  Step {step+1:>3d} | loss={f0:.6f} | k=[{k_str}], d=[{d_str}]")

        if f0 < best_loss:
            best_loss = f0
            best_log_p = log_p_np.copy()

    best_p = np.exp(best_log_p)
    # Return as scalars
    return best_p[0], best_p[1], best_loss, gt_trajs

# ─── MAIN GAN LOOP ────────────────────────────────────────────────────────────

def run_gan_optimization(cmd_args):
    # Start with your hardcoded defaults
    config = get_default_config()
    
    # Convert Namespace to dict if it's not already a dict
    # This allows the function to handle BOTH dicts and argparse objects
    if not isinstance(cmd_args, dict):
        cmd_args_dict = vars(cmd_args)
    else:
        cmd_args_dict = cmd_args

    # Update config with whatever was passed in via command line/Slurm
    for key, value in cmd_args_dict.items():
        config[key] = value
        
    os.makedirs(config["output_dir"], exist_ok=True)
    
    os.makedirs(config["output_dir"], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model Selection
    if config["disc_type"] == "lstm":
        D = disc.LSTMDiscriminator(input_size=3, hidden_size=16).to(device)
    elif config["disc_type"] == "cnn":
        D = disc.CNN1DDiscriminator(input_size=3, hidden_size=16).to(device)
    elif config["disc_type"] == "mlp":
        # We need a sample trajectory to find the sequence length for the MLP flatten layer
        sample_goals_dummy = sample_goals(config)[:1]
        sample_traj = collect_trajectories(config, sample_goals_dummy, config["init_k"], config["init_d"], True, None)
        seq_len = sample_traj.shape[1]
        D = disc.MLPDiscriminator(input_size=3, seq_length=seq_len).to(device)

    #optimizer = optim.Adam(D.parameters(), lr=config["d_lr"])
    os.makedirs(config["output_dir"], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    #D = disc.LSTMDiscriminator(input_size=3, hidden_size=16, num_layers=2).to(device)
    optimizer = optim.Adam(D.parameters(), lr=config["d_lr"])

    current_k = config["init_k"]
    current_d = config["init_d"]

    rng = np.random.default_rng(seed=42)
    fixed_noise = np.zeros((int(config["mppi_samples"]), 3), dtype=np.float32)
    fixed_noise[:, [0, 2]] = rng.normal(0, config["mppi_noise_sigma"], size=(int(config["mppi_samples"]), 2)).astype(np.float32)

    history = []
    goals = sample_goals(config)
    print(f"Sampled {len(goals)} new goals.")

    print("Collecting Ground Truth (Standard MuJoCo) trajectories...")
    real_trajs = collect_trajectories(config, goals, config["gt_k"], config["gt_d"],
                                      use_comfree=config["use_com_free_for_gt"], fixed_noise=fixed_noise)

    for iteration in range(config["gan_iterations"]):
        print(f"\n=== GAN Iteration {iteration+1}/{config['gan_iterations']} ===")

        print("Collecting Generated (ComFree_Warp) trajectories...")
        fake_trajs = collect_trajectories(config, goals, current_k, current_d,
                                          use_comfree=True, fixed_noise=fixed_noise)

        print(f"Training Discriminator for {config['d_epochs']} epochs...")
        d_loss = train_discriminator(D, optimizer, real_trajs, fake_trajs, config)
        print(f"Discriminator Loss: {d_loss:.4f}")

        print("Optimizing ComFree parameters...")
        new_goals = sample_goals(config)[:config["num_goals_gen_train"]]
        best_k, best_d, g_loss, new_gt_trajs = optimize_parameters(D, config, new_goals, current_k, current_d, fixed_noise)
        print(f"Generator Loss: {g_loss:.4f} | Updated Params -> K: {best_k:.2e}, D: {best_d:.2e}")

        # Update goals and real trajectories buffer
        goals = np.vstack([goals, new_goals])
        real_trajs = np.vstack([real_trajs, new_gt_trajs])

        current_k, current_d = best_k, best_d
        history.append({
            "iteration": iteration + 1,
            "d_loss": d_loss,
            "g_loss": g_loss,
            "k1": current_k, # Store the scalar
            "d1": current_d  # Store the scalar
        })

    save_results(history, D, config)
    return history

def save_results(history, D, config):
    df = pd.DataFrame(history)
    csv_path = os.path.join(config["output_dir"], "training_history.csv")
    df.to_csv(csv_path, index=False)

    weights_path = os.path.join(config["output_dir"], "discriminator_weights.pth")
    torch.save(D.state_dict(), weights_path)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(df["iteration"], df["d_loss"], label="D Loss (BCE + R1)", marker='o')
    ax1.plot(df["iteration"], df["g_loss"], label="G Loss (1 - D_score)", marker='o')
    ax1.set_title("GAN Losses")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(df["iteration"], df["k1"], label="k (estimated)", marker='x', color='green')
    ax2.axhline(config["gt_k"], color='green', linestyle='--', label="GT k")

    ax2.plot(df["iteration"], df["d1"], label="d (estimated)", marker='x', color='red')
    ax2.axhline(config["gt_d"], color='red', linestyle='--', label="GT d")
    
    ax2.set_title("Parameter Evolution")
    ax2.set_xlabel("GAN Iteration")
    ax2.set_ylabel("Parameter Value")
    ax2.legend()
    ax2.grid(True)

    plot_path = os.path.join(config["output_dir"], "training_curves.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"\nTraining complete. Artifacts saved to: {config['output_dir']}")

# ─── EXECUTION ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Parse the command line arguments
    args = get_args()
    
    # 2. Pass the Namespace object (args) to the function
    # This matches the new run_gan_optimization(cmd_args) signature
    run_gan_optimization(args)