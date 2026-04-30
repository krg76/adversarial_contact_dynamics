import os
os.environ["MUJOCO_GL"] = "egl"   # must be set before importing mujoco

import mujoco
import mujoco_warp as mw
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

MPPI_ITERS = 3  # Number of MPPI refinement loops
MAX_ITERS = 1000
OPTIM_ALGO = "Powell"#"L-BFGS-B"
W_RUNNING_POS  = 1.0
W_RUNNING_VEL  = 0.0
W_TERMINAL_POS = 10.0
W_TERMINAL_VEL = 5.0

def get_iterative_mppi_qvel(mj_model, mj_data, base_qvel, duration, k, d, goal, 
                            temp = 1.0, 
                            num_samples=NUM_SAMPLES, 
                            noise_sigma=NOISE_SIGMA, 
                            fixed_noise=None, 
                            cost_coeffs = [W_RUNNING_POS, W_RUNNING_VEL,W_TERMINAL_POS,W_TERMINAL_VEL,]):
    """Runs multiple loops of MPPI to refine the initial velocity for specific parameters."""
    sampling_args = {
        "Noise_Sigma": noise_sigma,
        "Num_Samples": num_samples,
        "MPPI_Iters": MPPI_ITERS
    }
    b_min = np.array([0.0 - 1.0 + goal[0], 0.0 - 1.0 + goal[1], 0.0])
    b_max = np.array([0.0 + 1.0 + goal[0], 0.0 + 1.0 + goal[1], 1.0])
    costs_args = {
        "Box_Center":goal,
        "Box_Min":b_min,
        "Box_Max":b_max,
        "Temp":temp,
        "Cost_Coeff":cost_coeffs
    },
    return rs.run_MPPI(mj_model, mj_data, base_qvel, duration, k, d, sampling_args=sampling_args, costs_args=costs_args, fixed_noise=fixed_noise)

def main() -> None:
    duration = 2.0

    mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
    mj_data  = mujoco.MjData(mj_model)

    starting_pos  = mj_data.qpos[:3].copy()
    base_qvel     = TARGET_POS - starting_pos

    # 1. Generate target trajectory using Ground Truth parameters
    print(f"Generating target trajectory with {MPPI_ITERS} MPPI loops...")
    optimal_qvel = get_iterative_mppi_qvel(
        mj_model, mj_data, base_qvel, duration, 
        CF_GT_STIFFNESS, CF_GT_DAMPING
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

        # NEW: Re-optimize velocity for the current parameters being evaluated
        current_opt_qvel = get_iterative_mppi_qvel(
            mj_model, mj_data, base_qvel, duration, k, d
        )

        pred_pos_batch, _ = rs.simulate_trajectories_parallel(
            mj_model, mj_data, current_opt_qvel[np.newaxis, :], duration, k, d
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
    if OPTIM_ALGO == 'Powell':
        res = minimize(objective, initial_params, method='Powell',                
                    bounds = [(-10,10),(-10,10)],#[(1e-10,None),(1e-10,None)],
                    options={'maxfev': MAX_ITERS, 
                    'xtol':1e-5
                    }
        )
    elif OPTIM_ALGO == 'Nelder-Mead':
        res = minimize(objective, initial_params, method='Nelder-Mead',                
            bounds = [(-10,10),(-10,10)],#[(1e-10,None),(1e-10,None)],
            options={'maxfev': MAX_ITERS, 
            'xatol':1e-5}
        )
    elif OPTIM_ALGO == 'L-BFGS-B':
        res = minimize(objective, initial_params, method='L-BFGS-B',                
            bounds = [(-10,10),(-10,10)],#[(1e-10,None),(1e-10,None)],
            options={'maxfun': MAX_ITERS,
            'eps':1e-4,
            'ftol':1e-5}
        )
    cf_stiffness, cf_damping = np.exp(res.x)
    print("Final Parameters:",cf_stiffness, cf_damping)

if __name__ == "__main__":
    main()