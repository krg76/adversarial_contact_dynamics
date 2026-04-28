import mujoco
import warp as wp
import mujoco_warp
import numpy as np
import mediapy as media
from scipy.interpolate import CubicSpline
import math

# ── Paths ─────────────────────────────────────────────────────────────────────
XML_PATH = "bouncing_ball.xml"
OUT_PATH = "bouning_ball_gpu.mp4"

# MPPI Parameters
TARGET_POS = np.array([0.0, 0.0, 0.0])
NUM_SAMPLES = 2048                        # Increased for GPU parallelization
NOISE_SIGMA = 5.0
TEMP = 0.5

# Box constraints for terminal position [x, y, z]
BOX_MIN = np.array([0.0, 0.0, 0.0])
BOX_MAX = np.array([1.0, 1.0, 0.5])

# Cost Weights
W_RUNNING_POS  = 1.0
W_RUNNING_VEL  = 0.0  
W_TERMINAL_POS = 10.0  
W_TERMINAL_VEL = 5.0   

@wp.kernel
def init_state_kernel(qpos: wp.array(dtype=float), 
                      qvel: wp.array(dtype=float), 
                      nq: int, nv: int, 
                      qpos0: wp.array(dtype=float), 
                      sampled_vels: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    # Initialize each sample with the model's base configuration
    for i in range(nq):
        qpos[tid * nq + i] = qpos0[i]
    # Set initial velocity for the ball (indices 0, 1, 2)
    qvel[tid * nv + 0] = sampled_vels[tid][0]
    qvel[tid * nv + 1] = sampled_vels[tid][1]
    qvel[tid * nv + 2] = sampled_vels[tid][2]
    # Zero out other degrees of freedom (like angular velocity)
    for i in range(3, nv):
        qvel[tid * nv + i] = 0.0

@wp.kernel
def compute_running_costs_kernel(qpos: wp.array(dtype=float), 
                                 qvel: wp.array(dtype=float), 
                                 nq: int, nv: int, 
                                 target: wp.vec3, 
                                 costs: wp.array(dtype=float), 
                                 w_pos: float, w_vel: float):
    tid = wp.tid()
    p = wp.vec3(qpos[tid * nq + 0], qpos[tid * nq + 1], qpos[tid * nq + 2])
    v = wp.vec3(qvel[tid * nv + 0], qvel[tid * nv + 1], qvel[tid * nv + 2])
    diff = p - target
    costs[tid] += (wp.dot(diff, diff) * w_pos + wp.dot(v, v) * w_vel)

@wp.kernel
def compute_terminal_costs_kernel(qpos: wp.array(dtype=float), 
                                  qvel: wp.array(dtype=float), 
                                  nq: int, nv: int, 
                                  box_min: wp.vec3, box_max: wp.vec3, 
                                  costs: wp.array(dtype=float), 
                                  w_pos: float, w_vel: float):
    tid = wp.tid()
    p = wp.vec3(qpos[tid * nq + 0], qpos[tid * nq + 1], qpos[tid * nq + 2])
    v = wp.vec3(qvel[tid * nv + 0], qvel[tid * nv + 1], qvel[tid * nv + 2])
    
    # Penalty if outside defined box
    out_x = wp.max(0.0, box_min[0] - p[0]) + wp.max(0.0, p[0] - box_max[0])
    out_y = wp.max(0.0, box_min[1] - p[1]) + wp.max(0.0, p[1] - box_max[1])
    out_z = wp.max(0.0, box_min[2] - p[2]) + wp.max(0.0, p[2] - box_max[2])
    
    cost_term_pos = (out_x*out_x + out_y*out_y + out_z*out_z) * w_pos
    cost_term_vel = wp.dot(v, v) * w_vel
    costs[tid] += (cost_term_pos + cost_term_vel)

def main():
    wp.init()
    device = "cuda" if wp.is_cuda_available() else "cpu"
    
    mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
    wp_model = mujoco_warp.Model(mj_model, device=device)
    num_steps = math.ceil(2.0 / mj_model.opt.timestep)
    
    wp_state = wp_model.state(NUM_SAMPLES)
    wp_state_next = wp_model.state(NUM_SAMPLES)
    costs_wp = wp.zeros(NUM_SAMPLES, dtype=float, device=device)
    
    # 1. Sample trajectories
    base_qvel = TARGET_POS - mj_model.qpos0[:3]
    sampled_qvels_np = (base_qvel + np.random.normal(0, NOISE_SIGMA, size=(NUM_SAMPLES, 3))).astype(np.float32)
    sampled_qvels_wp = wp.from_numpy(sampled_qvels_np, dtype=wp.vec3, device=device)
    
    # Initialize batch state
    wp.launch(init_state_kernel, dim=NUM_SAMPLES, inputs=[wp_state.qpos, wp_state.qvel, wp_model.nq, wp_model.nv, wp.from_numpy(mj_model.qpos0.astype(np.float32), device=device), sampled_qvels_wp], device=device)
    
    print(f"Simulating {NUM_SAMPLES} trajectories in parallel on {device}...")
    for _ in range(num_steps):
        wp_model.step(wp_state, wp_state_next)
        wp_state, wp_state_next = wp_state_next, wp_state # Double buffer swap
        wp.launch(compute_running_costs_kernel, dim=NUM_SAMPLES, inputs=[wp_state.qpos, wp_state.qvel, wp_model.nq, wp_model.nv, wp.vec3(*TARGET_POS), costs_wp, float(W_RUNNING_POS), float(W_RUNNING_VEL)], device=device)
    
    wp.launch(compute_terminal_costs_kernel, dim=NUM_SAMPLES, inputs=[wp_state.qpos, wp_state.qvel, wp_model.nq, wp_model.nv, wp.vec3(*BOX_MIN), wp.vec3(*BOX_MAX), costs_wp, float(W_TERMINAL_POS), float(W_TERMINAL_VEL)], device=device)

    # 2. Compute MPPI Weights
    costs_np = costs_wp.numpy()
    weights = np.exp(-(costs_np - np.min(costs_np)) / TEMP)
    weights /= np.sum(weights)
    optimal_qvel = np.sum(weights[:, None] * sampled_qvels_np, axis=0)
    print(f"Optimal Velocity: {optimal_qvel}")

    # 3. Render final trajectory with standard mujoco for visualization
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=480, width=640)
    frames = []
    mj_data.qvel[:3] = optimal_qvel
    for i in range(num_steps):
        mujoco.mj_step(mj_model, mj_data)
        if i % max(1, round(1.0 / (30 * mj_model.opt.timestep))) == 0:
            renderer.update_scene(mj_data)
            frames.append(renderer.render())
    media.write_video(OUT_PATH, frames, fps=30)
    print(f"Video saved → {OUT_PATH}")

if __name__ == "__main__":
    main()
