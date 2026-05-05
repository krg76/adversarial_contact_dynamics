import os
os.environ["MUJOCO_GL"] = "egl"

import warp as wp
import mujoco
import mujoco_warp as mw
import comfree_warp as cf_mjwarp
import comfree_warp.comfree_core._src.forward as cf_mj_src

import numpy as np
import math
import argparse
import matplotlib.pyplot as plt

# ── Paths & MPPI Parameters ───────────────────────────────────────────────────
XML_PATH = "bouncing_ball.xml"

NUM_SAMPLES  = 1000
NOISE_SIGMA  = 5.0
TEMP         = 1
MPPI_ITERS   = 2  

BASKET_MOCAP_POS = np.array([0.0, 0.0, 0.0])
BOX_MIN = np.array([-1.0, -1.0, 0.0]) + BASKET_MOCAP_POS
BOX_MAX = np.array([1.0, 1.0, 1.0]) + BASKET_MOCAP_POS
TARGET_POS   = BASKET_MOCAP_POS

W_RUNNING_POS  = 0.05
W_RUNNING_VEL  = 0.001
W_TERMINAL_POS = 100.0
W_TERMINAL_VEL = 0.01

def simulate_trajectories_parallel(
    mj_model, mj_data, initial_qvels, duration,
    cf_stiffness, cf_damping,
    use_comfree=True,
    warp_model=None,
    warp_data=None,
):
    nworld = initial_qvels.shape[0]
    num_steps = math.ceil(duration / mj_model.opt.timestep)

    rebuild_model = warp_model is None
    if rebuild_model:
        engine = cf_mjwarp if use_comfree else mw
        if use_comfree:
            warp_model = engine.put_model(mj_model,
                comfree_stiffness=cf_stiffness,
                comfree_damping=cf_damping)
        else:
            warp_model = engine.put_model(mj_model)
        warp_data = engine.put_data(mj_model, mj_data, nworld=nworld)
    else:
        engine = cf_mjwarp if use_comfree else mw

    qvel_np = np.zeros((nworld, mj_model.nv), dtype=np.float32)
    qvel_np[:, :3] = initial_qvels.astype(np.float32)
    warp_data.qvel.assign(qvel_np)
    
    qpos_np = np.tile(mj_data.qpos.astype(np.float32), (nworld, 1))
    warp_data.qpos.assign(qpos_np)

    all_positions  = np.empty((nworld, num_steps, 3), dtype=np.float32)
    all_velocities = np.empty((nworld, num_steps, 3), dtype=np.float32)

    for step in range(num_steps):
        engine.step(warp_model, warp_data)
        wp.synchronize()
        all_positions[:, step, :]  = warp_data.qpos.numpy()[:, :3]
        all_velocities[:, step, :] = warp_data.qvel.numpy()[:, :3]

    return all_positions, all_velocities

def run_MPPI(
    mj_model, mj_data, nominal_qvel, duration,
    cf_stiffness, cf_damping, use_comfree=False
):
    costs_args = {
        "Box_Center": TARGET_POS,
        "Box_Min": BOX_MIN,
        "Box_Max": BOX_MAX,
        "Temp": TEMP,
        "Cost_Coeff": [W_RUNNING_POS, W_RUNNING_VEL, W_TERMINAL_POS, W_TERMINAL_VEL]
    }
    
    rng = np.random.default_rng(seed=42)
    noise = np.zeros((NUM_SAMPLES, 3), dtype=np.float32)
    noise[:, [0, 2]] = rng.normal(0, NOISE_SIGMA, size=(NUM_SAMPLES, 2)).astype(np.float32)

    current_qvel = nominal_qvel.copy()

    for _ in range(MPPI_ITERS):
        sampled_qvels = (current_qvel + noise).astype(np.float32)
    
        all_positions, all_velocities = simulate_trajectories_parallel(
            mj_model, mj_data, sampled_qvels, duration, cf_stiffness, cf_damping, use_comfree
        )

        diff             = all_positions - costs_args["Box_Center"]
        cost_running_pos = np.sum(diff ** 2, axis=(1, 2))
        cost_running_vel = np.sum(all_velocities ** 2, axis=(1, 2))

        term_pos   = all_positions[:, -1, :]
        out_of_box = (np.maximum(0.0, costs_args["Box_Min"] - term_pos) +
                      np.maximum(0.0, term_pos - costs_args["Box_Max"]))
        cost_term_pos = np.sum(out_of_box ** 2, axis=1)
        cost_term_vel = np.sum(all_velocities[:, -1, :] ** 2, axis=1)

        costs = (costs_args["Cost_Coeff"][0] * cost_running_pos +
                 costs_args["Cost_Coeff"][1] * cost_running_vel +
                 costs_args["Cost_Coeff"][2] * cost_term_pos +
                 costs_args["Cost_Coeff"][3] * cost_term_vel)

        min_cost = np.min(costs)
        weights  = np.exp(-(costs - min_cost) / costs_args["Temp"])
        weights /= np.sum(weights)

        current_qvel = np.sum(weights[:, None] * sampled_qvels, axis=0)
        
    return current_qvel

def plot_comparisons(pos_cf, vel_cf, pos_mj, vel_mj, duration):
    """Generates the comparison plots for trajectories and velocities."""
    time_steps = np.linspace(0, duration, pos_cf.shape[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Trajectory (X vs Z)
    ax1.plot(pos_cf[:, 0], pos_cf[:, 2], label='ComFree Sim', color='#1f77b4', linewidth=2)
    ax1.plot(pos_mj[:, 0], pos_mj[:, 2], label='Standard MuJoCo', color='#ff7f0e', linestyle='--', linewidth=2)
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Z Position (m)')
    ax1.set_title('Optimal Trajectory Comparison (X-Z Plane)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Velocity Profile (Z-Velocity over time)
    ax2.plot(time_steps, vel_cf[:, 2], label='ComFree Sim Z-Vel', color='#1f77b4', linewidth=2)
    ax2.plot(time_steps, vel_mj[:, 2], label='Standard MuJoCo Z-Vel', color='#ff7f0e', linestyle='--', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Z Velocity (m/s)')
    ax2.set_title('Vertical Velocity Profile over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pipeline_comparison.png', dpi=300)
    print("Plots saved to 'pipeline_comparison.png'")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Compare MPPI performance between ComFree and Standard MuJoCo")
    parser.add_argument("--stiffness", type=float, default=0.5, help="Stiffness parameter for the contact model")
    parser.add_argument("--damping", type=float, default=0.001, help="Damping parameter for the contact model")
    args = parser.parse_args()

    duration = 2.0

    mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
    mj_data  = mujoco.MjData(mj_model)

    basket_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "basket_floating")
    mocap_idx = mj_model.body_mocapid[basket_body_id]
    mj_data.mocap_pos[mocap_idx] = BASKET_MOCAP_POS

    starting_pos = mj_data.qpos[:3].copy()
    base_qvel    = TARGET_POS - starting_pos

    # 1. Run MPPI Optimization
    print(f"Running MPPI with ComFree Sim (Stiffness: {args.stiffness}, Damping: {args.damping})...")
    opt_qvel_cf = run_MPPI(
        mj_model, mj_data, base_qvel, duration,
        cf_stiffness=args.stiffness, cf_damping=args.damping, use_comfree=True
    )

    print("Running MPPI with Standard MuJoCo...")
    opt_qvel_mj = run_MPPI(
        mj_model, mj_data, base_qvel, duration,
        cf_stiffness=args.stiffness, cf_damping=args.damping, use_comfree=False
    )

    # 2. Simulate single rollouts of the optimal trajectories to gather plotting data
    print("Simulating final trajectories for plotting...")
    pos_cf, vel_cf = simulate_trajectories_parallel(
        mj_model, mj_data, np.array([opt_qvel_cf]), duration,
        args.stiffness, args.damping, use_comfree=True
    )
    
    pos_mj, vel_mj = simulate_trajectories_parallel(
        mj_model, mj_data, np.array([opt_qvel_mj]), duration,
        args.stiffness, args.damping, use_comfree=False
    )

    # 3. Generate the visualization
    plot_comparisons(pos_cf[0], vel_cf[0], pos_mj[0], vel_mj[0], duration)


if __name__ == "__main__":
    main()