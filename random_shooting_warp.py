import os
os.environ["MUJOCO_GL"] = "egl"   # must be set before importing mujoco

import mujoco
import comfree_warp as cf_mjwarp

import warp as wp
import numpy as np
import mediapy as media
import math


# ── Paths ─────────────────────────────────────────────────────────────────────
XML_PATH = "bouncing_ball.xml"
OUT_PATH = "bouncing_ball.mp4"

# MPPI Parameters
TARGET_POS   = np.array([0.0, 0.0, 0.0])
NUM_SAMPLES  = 1000
NOISE_SIGMA  = 5.0
TEMP         = 1
MPPI_ITERS = 3  # Number of MPPI refinement loops

# Box constraints for terminal position [x, y, z]
BOX_MIN = np.array([0.0, 0.0, 0.0])
BOX_MAX = np.array([1.0, 1.0, 0.5])

# Cost Weights
W_RUNNING_POS  = 1.0
W_RUNNING_VEL  = 0.0
W_TERMINAL_POS = 10.0
W_TERMINAL_VEL = 5.0

# Comfree Warp parameters (if applicable)
CF_STIFFNESS = 0.2
CF_DAMPING   = 0.001


def simulate_trajectories_parallel(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,       # <-- add this
    initial_qvels: np.ndarray,
    duration: float,
    cf_stiffness: float,
    cf_damping: float,
) -> tuple[np.ndarray, np.ndarray]:
    nworld    = initial_qvels.shape[0]
    num_steps = math.ceil(duration / mj_model.opt.timestep)

    # ── Reset: re-upload CPU data to get a clean GPU state ───────────────────
    model = cf_mjwarp.put_model(mj_model,
        comfree_stiffness=cf_stiffness,
        comfree_damping=cf_damping)
    data  = cf_mjwarp.put_data(mj_model, mj_data, nworld=nworld)  # replaces mj_resetData

    # ── Set per-world initial velocities ──────────────────────────────────────
    qvel_np = np.zeros((nworld, mj_model.nv), dtype=np.float32)
    qvel_np[:, :3] = initial_qvels.astype(np.float32)
    data.qvel.assign(qvel_np)

    # ── Pre-allocate output buffers ───────────────────────────────────────────
    all_positions  = np.empty((nworld, num_steps, 3), dtype=np.float32)
    all_velocities = np.empty((nworld, num_steps, 3), dtype=np.float32)

    for step in range(num_steps):
        cf_mjwarp.step(model, data)
        all_positions[:, step, :]  = data.qpos.numpy()[:, :3]
        all_velocities[:, step, :] = data.qvel.numpy()[:, :3]

    return all_positions, all_velocities


def render_trajectory(
    mj_model: mujoco.MjModel,
    mj_data:  mujoco.MjData,
    renderer: mujoco.Renderer,
    initial_qvel: np.ndarray,
    duration: float,
    fps: int,
) -> list[np.ndarray]:
    """
    Render a single trajectory with the standard CPU MuJoCo renderer.
    MuJoCo Warp does not expose its own renderer, so we fall back here.
    """
    steps_per_frame = max(1, round(1.0 / (fps * mj_model.opt.timestep)))
    num_steps       = math.ceil(duration / mj_model.opt.timestep)
    frames          = []

    mujoco.mj_resetData(mj_model, mj_data)
    mj_data.qvel[:3] = initial_qvel

    for step in range(num_steps):
        mujoco.mj_step(mj_model, mj_data)
        if step % steps_per_frame == 0:
            renderer.update_scene(mj_data)
            frames.append(renderer.render())

    return frames

def run_MPPI(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    nominal_qvel: np.ndarray,
    duration: float,
    cf_stiffness: float,
    cf_damping: float,
    costs_args = {
        "Box_Center":TARGET_POS,
        "Box_Min":BOX_MIN,
        "Box_Max":BOX_MAX,
        "Temp":TEMP,
        "Cost_Coeff":[W_RUNNING_POS,
            W_RUNNING_VEL,
            W_TERMINAL_POS,
            W_TERMINAL_VEL,]
    },
    sampling_args = {
        "Noise_Sigma":NOISE_SIGMA,
        "Num_Samples":NUM_SAMPLES,
        "MPPI_Iters":MPPI_ITERS
    }
) -> np.ndarray:
    """
    Runs the MPPI algorithm to find the optimal initial velocity.
    
    Args:
        mj_model: The MuJoCo model.
        mj_data: The MuJoCo data.
        sampled_qvels: An array of sampled initial velocities for each trajectory.
        duration: The duration of each simulation trajectory.
        cf_stiffness: Stiffness parameter for comfree_warp.
        cf_damping: Damping parameter for comfree_warp.
        
    Returns:
        The optimal initial velocity determined by MPPI.
    """
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0, sampling_args["Noise_Sigma"], size=(sampling_args["Num_Samples"], 3)).astype(np.float32) # Consider removing the Y velocity

    for _ in range(sampling_args["MPPI_Iters"]):
        sampled_qvels = (nominal_qvel + noise).astype(np.float32)
    
        all_positions, all_velocities = simulate_trajectories_parallel(
            mj_model, mj_data, sampled_qvels, duration, cf_stiffness, cf_damping
        )

        diff             = all_positions - costs_args["Box_Center"]          # broadcast over (N, T, 3)
        cost_running_pos = np.sum(diff ** 2,              axis=(1, 2))  # (N,)
        cost_running_vel = np.sum(all_velocities ** 2,    axis=(1, 2))  # (N,)

        # Terminal position: penalty for leaving the box
        term_pos   = all_positions[:, -1, :]                           # (N, 3)
        out_of_box = (np.maximum(0.0, costs_args["Box_Min"] - term_pos) +
                    np.maximum(0.0, term_pos - costs_args["Box_Max"]))
        cost_term_pos = np.sum(out_of_box ** 2,           axis=1)      # (N,)

        # Terminal velocity: penalty for arriving fast
        cost_term_vel = np.sum(all_velocities[:, -1, :] ** 2, axis=1)  # (N,)

        costs = (costs_args["Cost_Coeff"][0]  * cost_running_pos +
                costs_args["Cost_Coeff"][1]  * cost_running_vel +
                costs_args["Cost_Coeff"][2] * cost_term_pos    +
                costs_args["Cost_Coeff"][3] * cost_term_vel)

        # 6. MPPI weights
        min_cost = np.min(costs)
        weights  = np.exp(-(costs - min_cost) / costs_args["Temp"])
        weights /= np.sum(weights)

        optimal_qvel = np.sum(weights[:, None] * sampled_qvels, axis=0)
    #print(f"Optimal Velocity: {optimal_qvel}")
    return optimal_qvel

def main() -> None:
    fps      = 30
    duration = 2.0

    # ── 1. Build CPU model/data (needed for XML parsing & rendering) ──────────
    mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
    mj_data  = mujoco.MjData(mj_model)

    # 2. Sample initial velocities ────────────────────────────────────────────
    starting_pos  = mj_data.qpos[:3].copy()
    base_qvel     = TARGET_POS - starting_pos

    # 3. Run MPPI to find the optimal velocity ────────────────────────────────
    optimal_qvel = run_MPPI(
        mj_model, mj_data, base_qvel, duration, CF_STIFFNESS, CF_DAMPING
    )

    # 4. Render optimal trajectory (CPU renderer) ─────────────────────────────
    renderer = mujoco.Renderer(mj_model, height=480, width=640)
    frames   = render_trajectory(mj_model, mj_data, renderer, optimal_qvel, duration, fps)

    media.write_video(OUT_PATH, frames, fps=fps)
    print(f"Video saved → {OUT_PATH}")


if __name__ == "__main__":
    main()