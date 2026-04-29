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

def main() -> None:
    mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
    mj_data  = mujoco.MjData(mj_model)

    starting_pos  = mj_data.qpos[:3].copy()
    base_qvel     = TARGET_POS - starting_pos
    sampled_qvels = (base_qvel + np.random.normal(0, NOISE_SIGMA, size=(NUM_SAMPLES, 3))).astype(np.float32)

    cf_stiffness = 0.5 #<- PARAM TO LEARN
    cf_dampling = 0.1 #<- Param to Learn

    optimal_qvel = rs.run_MPPI(
        mj_model, mj_data, sampled_qvels, duration, CF_GT_STIFFNESS, CF_GT_DAMPING
    )

    # 4. Render optimal trajectory (CPU renderer) ─────────────────────────────
    renderer = mujoco.Renderer(mj_model, height=480, width=640)
    frames   = rs.render_trajectory(mj_model, mj_data, renderer, optimal_qvel, duration, fps)


if __name__ == "__main__":
    main()