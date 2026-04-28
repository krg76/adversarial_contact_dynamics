import mujoco
import numpy as np
import mediapy as media
from scipy.interpolate import CubicSpline
import math

# ── Paths ─────────────────────────────────────────────────────────────────────
XML_PATH = "bouncing_ball.xml"
OUT_PATH = "bouning_ball.mp4"

# MPPI Parameters
TARGET_POS = np.array([0.0, 0.0, 0.0]) # Target for the ball to reach
NUM_SAMPLES = 500                        # Number of random trajectories to sample
NOISE_SIGMA = 5.0                       # Variance for velocity sampling
TEMP = 0.5                           # Temperature parameter for MPPI weights

# Box constraints for terminal position [x, y, z]
BOX_MIN = np.array([0.0, 0.0, 0.0])
BOX_MAX = np.array([1, 1, 0.5])

# Cost Weights
W_RUNNING_POS  = 1.0
W_RUNNING_VEL  = 0.0  # Penalty for moving fast
W_TERMINAL_POS = 10.0  # Penalty for ending outside box
W_TERMINAL_VEL = 5.0   # Penalty for ending with high speed

def simulate_trajectory(model,data,
                        initial_qvel,
                        duration,
                        renderer=None,steps_per_frame=30):
    # Calculate the total number of simulation steps
    num_steps = math.ceil(duration / model.opt.timestep)

    frames = []
    # Pre-allocate memory for positions and velocities
    positions = np.zeros((num_steps, 3), dtype=np.float64)
    velocities = np.zeros((num_steps, 3), dtype=np.float64)
    step = 0
    mujoco.mj_resetData(model, data)
    data.qvel[:3] = initial_qvel
    while data.time < duration:
        mujoco.mj_step(model, data)
        positions[step] = data.qpos[:3].copy()
        velocities[step] = data.qvel[:3].copy()
        step += 1
        if step % steps_per_frame == 0 and not renderer is None:
            renderer.update_scene(data)
            frames.append(renderer.render())
    return positions, velocities, frames

# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    model    = mujoco.MjModel.from_xml_path(XML_PATH)
    data     = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)

    fps             = 30
    duration        = 2.0
    steps_per_frame = max(1, round(1.0 / (fps * model.opt.timestep)))

    # 1. Sample trajectories
    starting_pos = data.qpos[:3].copy()
    base_qvel = TARGET_POS - starting_pos
    sampled_qvels = base_qvel + np.random.normal(0, NOISE_SIGMA, size=(NUM_SAMPLES, 3))
    costs = np.zeros(NUM_SAMPLES)

    print(f"Sampling {NUM_SAMPLES} trajectories...")
    for i in range(NUM_SAMPLES):
        print(i)
        pos, vel, __ = simulate_trajectory(model, data, sampled_qvels[i], duration)

        # 1. Running Position Cost (Standard L2 to target)
        cost_running_pos = np.sum((pos - TARGET_POS) ** 2)

        # 2. Running Velocity Cost (Low velocities throughout)
        cost_running_vel = np.sum(vel ** 2)

        # 3. Terminal Position Cost (Penalty if outside the defined box)
        term_pos = pos[-1]
        out_of_box = np.maximum(0, BOX_MIN - term_pos) + np.maximum(0, term_pos - BOX_MAX)
        cost_term_pos = np.sum(out_of_box ** 2)

        # 4. Terminal Velocity Cost (Small velocity at the final step)
        cost_term_vel = np.sum(vel[-1] ** 2)

        # Total weighted cost
        costs[i] = (W_RUNNING_POS * cost_running_pos + 
                    W_RUNNING_VEL * cost_running_vel + 
                    W_TERMINAL_POS * cost_term_pos + 
                    W_TERMINAL_VEL * cost_term_vel)

    # 2. Compute MPPI Weights
    # We use (costs - min_costs) for numerical stability in the exponential
    min_cost = np.min(costs)
    weights = np.exp(-(costs - min_cost) / TEMP)
    weights /= np.sum(weights)

    # 3. Calculate optimal initial velocity
    optimal_qvel = np.sum(weights[:, None] * sampled_qvels, axis=0)
    print(f"Optimal Velocity: {optimal_qvel}")

    # 4. Render final trajectory with optimal velocity
    mujoco.mj_resetData(model, data)
    data.qvel[:3] = optimal_qvel
    _,_,frames = simulate_trajectory(model, data, optimal_qvel, duration, renderer, steps_per_frame)

    media.write_video(OUT_PATH, frames, fps=fps)
    print(f"Video saved → {OUT_PATH}")


if __name__ == "__main__":
    main()