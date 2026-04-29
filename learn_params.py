import os
os.environ["MUJOCO_GL"] = "egl"   # must be set before importing mujoco

import mujoco
import comfree_warp as cf_mjwarp

import warp as wp
import numpy as np
import mediapy as media
import math
from scipy.optimize import minimize

import random_shooting_warp as rs 

# ── Paths ─────────────────────────────────────────────────────────────────────
XML_PATH = "bouncing_ball.xml"

# MPPI Parameters
TARGET_POS   = np.array([0.0, 0.0, 0.0])
NUM_SAMPLES  = 1000
NOISE_SIGMA  = 1.0

# Comfree Warp parameters
CF_GT_STIFFNESS = 0.2
CF_GT_DAMPING = 0.001

MAX_ITERS = 1000

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

    # 2. Setup optimization to learn parameters
    # We optimize in log-space to ensure parameters stay positive and to 
    # better handle sensitivity across different scales.
    initial_params = np.log([0.5, 0.1])

    def objective(params):
        log_k, log_d = params
        k, d = np.exp(log_k), np.exp(log_d)

        # Evaluate current parameter set
        pred_pos_batch, _ = rs.simulate_trajectories_parallel(
            mj_model, mj_data, optimal_qvel[np.newaxis, :], duration, k, d
        )
        # Mean Squared Error between trajectories
        loss = np.mean((pred_pos_batch[0] - target_traj)**2)
        
        objective.iter_count += 1
        print(f"{objective.iter_count:<5} | {loss:<12.8f} | {k:<10.6f} | {d:<10.6f}")
        return loss
    
    objective.iter_count = 0

    print(f"Starting Parameter Identification (Scipy minimize in log-space)...")
    print(f"GT: stiffness={CF_GT_STIFFNESS}, damping={CF_GT_DAMPING}\n")
    print(f"{'Iter':<5} | {'Loss':<12} | {'Stiffness':<10} | {'Damping':<10}")
    print("-" * 55)

    # 3. Optimization Loop using SciPy
    # res = minimize(objective, initial_params, method='Nelder-Mead',
    #                options={'maxfev': MAX_ITERS, 
    #                'xatol':1e-5})
    res = minimize(objective, initial_params, method='Powell',
                   options={'maxfev': MAX_ITERS, 
                   'xtol':1e-5
                   })
    cf_stiffness, cf_damping = np.exp(res.x)
    print("Final Parameters:",cf_stiffness, cf_damping)

if __name__ == "__main__":
    main()