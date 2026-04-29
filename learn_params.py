import os
os.environ["MUJOCO_GL"] = "egl"   # must be set before importing mujoco

import mujoco
import comfree_warp as cf_mjwarp

import warp as wp
import numpy as np
import mediapy as media
import math

import random_shooting_warp as rs 

# ── Paths ─────────────────────────────────────────────────────────────────────
XML_PATH = "bouncing_ball.xml"

# MPPI Parameters
TARGET_POS   = np.array([0.0, 0.0, 0.0])
NUM_SAMPLES  = 1000
NOISE_SIGMA  = 5.0

# Comfree Warp parameters
CF_GT_STIFFNESS = 0.2
CF_GT_DAMPING = 0.001

# Optimization Hyperparameters
LEARNING_RATE = 0.5
FINITE_DIFF_EPS = 1e-4
MAX_ITERS = 20

def main() -> None:
    duration = 2.0
    fps = 30

    mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
    mj_data  = mujoco.MjData(mj_model)

    starting_pos  = mj_data.qpos[:3].copy()
    base_qvel     = TARGET_POS - starting_pos
    sampled_qvels = (base_qvel + np.random.normal(0, NOISE_SIGMA, size=(NUM_SAMPLES, 3))).astype(np.float32)

    # 1. Generate target trajectory using Ground Truth parameters
    # We first find an optimal velocity with GT parameters to ensure contact happens
    optimal_qvel = rs.run_MPPI(
        mj_model, mj_data, sampled_qvels, duration, CF_GT_STIFFNESS, CF_GT_DAMPING
    )
    
    # Generate the reference (target) path using the optimal velocity and GT params
    gt_pos_batch, _ = rs.simulate_trajectories_parallel(
        mj_model, mj_data, optimal_qvel[np.newaxis, :], duration, CF_GT_STIFFNESS, CF_GT_DAMPING
    )
    target_traj = gt_pos_batch[0]

    # 2. Setup Gradient Descent to learn parameters
    cf_stiffness = 0.5 
    cf_damping = 0.1

    def get_loss(k, d):
        # Evaluate current parameter set
        pred_pos_batch, _ = rs.simulate_trajectories_parallel(
            mj_model, mj_data, optimal_qvel[np.newaxis, :], duration, k, d
        )
        # Mean Squared Error between trajectories
        return np.mean((pred_pos_batch[0] - target_traj)**2)

    print(f"Starting Parameter Identification (Gradient Descent)...")
    print(f"GT: stiffness={CF_GT_STIFFNESS}, damping={CF_GT_DAMPING}\n")
    print(f"{'Iter':<5} | {'Loss':<12} | {'Stiffness':<10} | {'Damping':<10}")
    print("-" * 55)

    # 3. Optimization Loop
    for i in range(MAX_ITERS):
        current_loss = get_loss(cf_stiffness, cf_damping)
        
        # Finite differences for gradients
        # dLoss/dStiffness
        loss_k = get_loss(cf_stiffness + FINITE_DIFF_EPS, cf_damping)
        grad_k = (loss_k - current_loss) / FINITE_DIFF_EPS
        
        # dLoss/dDamping
        loss_d = get_loss(cf_stiffness, cf_damping + FINITE_DIFF_EPS)
        grad_d = (loss_d - current_loss) / FINITE_DIFF_EPS
        
        # Parameter updates
        cf_stiffness -= LEARNING_RATE * grad_k
        cf_damping   -= LEARNING_RATE * grad_d
        
        # Enforcement of positivity (stiffness/damping cannot be negative)
        cf_stiffness = max(1e-6, cf_stiffness)
        cf_damping   = max(1e-6, cf_damping)

        print(f"{i:<5} | {current_loss:<12.8f} | {cf_stiffness:<10.4f} | {cf_damping:<10.4f}")
        if current_loss < 1e-9: break

    # 4. Render final learned trajectory for verification ─────────────────────
    renderer = mujoco.Renderer(mj_model, height=480, width=640)
    frames = rs.render_trajectory(mj_model, mj_data, renderer, optimal_qvel, duration, fps)
    media.write_video("learning_results.mp4", frames, fps=fps)
    print(f"\nOptimization complete. Video saved to learning_results.mp4")

if __name__ == "__main__":
    main()