import os
os.environ["MUJOCO_GL"] = "egl"   # must be set before importing mujoco

import mujoco
from mujoco import mjx

import jax
import jax.numpy as jnp
import numpy as np
import optax
import mediapy as media
import math
from functools import partial

# ── Paths ─────────────────────────────────────────────────────────────────────
XML_PATH = "bouncing_ball.xml"
OUT_PATH = "bouncing_ball.mp4"

# ── MPPI Parameters ────────────────────────────────────────────────────────────
TARGET_POS  = jnp.array([0.0, 0.0, 0.0])
NUM_SAMPLES = 1000
NOISE_SIGMA = 5.0
TEMP        = 0.5

# ── Box constraints for terminal position [x, y, z] ──────────────────────────
BOX_MIN = jnp.array([0.0, 0.0, 0.0])
BOX_MAX = jnp.array([1.0, 1.0, 0.5])

# ── Cost weights ──────────────────────────────────────────────────────────────
W_RUNNING_POS  = 1.0
W_RUNNING_VEL  = 0.0
W_TERMINAL_POS = 10.0
W_TERMINAL_VEL = 5.0

# ── Gradient-based refinement (post-MPPI) ─────────────────────────────────────
REFINE_STEPS = 100
REFINE_LR    = 1e-2


# ═════════════════════════════════════════════════════════════════════════════
#  Core: single-trajectory rollout via jax.lax.scan
#  Everything here is pure JAX → fully differentiable
# ═════════════════════════════════════════════════════════════════════════════

def rollout(
    mjx_model: mjx.Model,
    mjx_data_init: mjx.Data,
    initial_qvel: jax.Array,   # shape (3,)
    num_steps: int,
) -> tuple[jax.Array, jax.Array]:
    """
    Simulate a single trajectory starting from `mjx_data_init` with the
    given initial translational velocity.

    Returns
    -------
    positions  : (num_steps, 3)  – ball (x, y, z) at every step
    velocities : (num_steps, 3)  – ball (vx, vy, vz) at every step
    """
    # Overwrite only the first 3 dofs of qvel; everything else from init data.
    data = mjx_data_init.replace(
        qvel=mjx_data_init.qvel.at[:3].set(initial_qvel)
    )

    def step_fn(data: mjx.Data, _):
        data = mjx.step(mjx_model, data)
        return data, (data.qpos[:3], data.qvel[:3])

    # jax.lax.scan unrolls the loop symbolically → gradients flow through time
    _, (positions, velocities) = jax.lax.scan(step_fn, data, None, length=num_steps)
    return positions, velocities   # (T, 3), (T, 3)


# ── Vectorise over a batch of initial velocities ──────────────────────────────
# We close over (mjx_model, mjx_data_init, num_steps) so vmap only maps qvels.
def make_batched_rollout(mjx_model, mjx_data_init, num_steps):
    single = partial(rollout, mjx_model, mjx_data_init, num_steps=num_steps)
    return jax.jit(jax.vmap(single))   # (N, 3) → (N, T, 3), (N, T, 3)


# ═════════════════════════════════════════════════════════════════════════════
#  Differentiable cost
# ═════════════════════════════════════════════════════════════════════════════

def trajectory_cost(
    positions: jax.Array,    # (T, 3)
    velocities: jax.Array,   # (T, 3)
) -> jax.Array:
    """Scalar cost for a single trajectory. Fully differentiable."""
    # ── Running terms ─────────────────────────────────────────────────────────
    diff             = positions - TARGET_POS            # broadcast (T, 3)
    cost_running_pos = jnp.sum(diff ** 2)
    cost_running_vel = jnp.sum(velocities ** 2)

    # ── Terminal position: L2 distance outside the box ─────────────────────
    term_pos   = positions[-1]
    out_of_box = (jnp.maximum(0.0, BOX_MIN - term_pos) +
                  jnp.maximum(0.0, term_pos - BOX_MAX))
    cost_term_pos = jnp.sum(out_of_box ** 2)

    # ── Terminal velocity ─────────────────────────────────────────────────────
    cost_term_vel = jnp.sum(velocities[-1] ** 2)

    return (W_RUNNING_POS  * cost_running_pos +
            W_RUNNING_VEL  * cost_running_vel +
            W_TERMINAL_POS * cost_term_pos    +
            W_TERMINAL_VEL * cost_term_vel)


def batched_cost(
    all_positions: jax.Array,    # (N, T, 3)
    all_velocities: jax.Array,   # (N, T, 3)
) -> jax.Array:
    """Vectorised cost over a batch – returns (N,)."""
    return jax.vmap(trajectory_cost)(all_positions, all_velocities)


# ── End-to-end scalar cost as a function of a single initial velocity ─────────
# This is what you differentiate with jax.grad / jax.value_and_grad.
def cost_from_qvel(
    mjx_model: mjx.Model,
    mjx_data_init: mjx.Data,
    initial_qvel: jax.Array,   # (3,)
    num_steps: int,
) -> jax.Array:
    positions, velocities = rollout(mjx_model, mjx_data_init, initial_qvel, num_steps)
    return trajectory_cost(positions, velocities)


# ═════════════════════════════════════════════════════════════════════════════
#  MPPI
# ═════════════════════════════════════════════════════════════════════════════

def mppi(
    mjx_model: mjx.Model,
    mjx_data_init: mjx.Data,
    base_qvel: np.ndarray,
    num_steps: int,
    key: jax.Array,
) -> jax.Array:
    """
    Draw NUM_SAMPLES perturbations, run them in parallel, and return the
    MPPI-weighted optimal velocity.
    """
    key, subkey = jax.random.split(key)
    noise       = jax.random.normal(subkey, shape=(NUM_SAMPLES, 3)) * NOISE_SIGMA
    sampled     = jnp.array(base_qvel) + noise           # (N, 3)

    # Parallel rollout on GPU via vmap + jit
    batched_rollout = make_batched_rollout(mjx_model, mjx_data_init, num_steps)
    all_positions, all_velocities = batched_rollout(sampled)

    costs    = batched_cost(all_positions, all_velocities)   # (N,)
    min_cost = jnp.min(costs)
    weights  = jnp.exp(-(costs - min_cost) / TEMP)
    weights /= jnp.sum(weights)

    optimal_qvel = jnp.sum(weights[:, None] * sampled, axis=0)
    return optimal_qvel, key


# ═════════════════════════════════════════════════════════════════════════════
#  Gradient-based refinement  (the differentiable payoff)
# ═════════════════════════════════════════════════════════════════════════════

def gradient_refine(
    mjx_model: mjx.Model,
    mjx_data_init: mjx.Data,
    init_qvel: jax.Array,
    num_steps: int,
) -> jax.Array:
    """
    Polish the MPPI solution with Adam gradient descent through the physics.

    MJX's contact solver uses jax.lax.while_loop internally, which is
    incompatible with reverse-mode AD (jax.grad).  Forward-mode AD
    (jax.jacfwd) works through while_loop and is equally efficient here
    because the control input is only 3-dimensional — it requires exactly
    3 JVP passes regardless of trajectory length.
    """
    scalar_cost = jax.jit(
        lambda qv: cost_from_qvel(mjx_model, mjx_data_init, qv, num_steps)
    )

    # jacfwd on a scalar output returns the gradient (shape matches input).
    # Wrapping value + grad together avoids a second forward pass.
    @jax.jit
    def value_and_grad_fwd(qv: jax.Array):
        loss = scalar_cost(qv)
        grads = jax.jacfwd(scalar_cost)(qv)   # (3,) — forward-mode, works through while_loop
        return loss, grads

    optimiser = optax.adam(REFINE_LR)
    opt_state = optimiser.init(init_qvel)
    qvel      = init_qvel

    for i in range(REFINE_STEPS):
        loss, grads        = value_and_grad_fwd(qvel)
        updates, opt_state = optimiser.update(grads, opt_state)
        qvel               = optax.apply_updates(qvel, updates)
        if i % 20 == 0:
            print(f"  [refine {i:3d}] cost = {float(loss):.4f}  |grad| = {float(jnp.linalg.norm(grads)):.4f}")

    return qvel


# ═════════════════════════════════════════════════════════════════════════════
#  CPU rendering (MJX has no renderer – fall back to mujoco.Renderer)
# ═════════════════════════════════════════════════════════════════════════════

def render_trajectory(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    renderer: mujoco.Renderer,
    initial_qvel: np.ndarray,
    duration: float,
    fps: int,
) -> list[np.ndarray]:
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


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    fps      = 30
    duration = 2.0

    # ── 1. Build CPU model / data ─────────────────────────────────────────────
    mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
    mj_data  = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)

    num_steps = math.ceil(duration / mj_model.opt.timestep)

    # ── 2. Upload to GPU (single-world MJX model + data) ─────────────────────
    mjx_model     = mjx.put_model(mj_model)
    mjx_data_base = mjx.put_data(mj_model, mj_data)

    # ── 3. Base velocity: aim toward target ───────────────────────────────────
    starting_pos = np.array(mj_data.qpos[:3])
    base_qvel    = (np.array(TARGET_POS) - starting_pos).astype(np.float32)

    # ── 4. MPPI ───────────────────────────────────────────────────────────────
    print(f"MPPI: sampling {NUM_SAMPLES} trajectories in parallel with MJX + vmap ...")
    key          = jax.random.PRNGKey(0)
    optimal_qvel, _ = mppi(mjx_model, mjx_data_base, base_qvel, num_steps, key)
    print(f"MPPI optimal velocity : {np.array(optimal_qvel)}")

    # ── 5. Gradient-based refinement through physics ─────────────────────────
    print("\nRefining with gradient descent through MJX physics (differentiable) ...")
    refined_qvel = gradient_refine(mjx_model, mjx_data_base, optimal_qvel, num_steps)
    print(f"Refined velocity      : {np.array(refined_qvel)}")

    # ── 6. Render refined trajectory (CPU renderer) ───────────────────────────
    renderer = mujoco.Renderer(mj_model, height=480, width=640)
    frames   = render_trajectory(
        mj_model, mj_data, renderer, np.array(refined_qvel), duration, fps
    )

    media.write_video(OUT_PATH, frames, fps=fps)
    print(f"\nVideo saved → {OUT_PATH}")

    # ── 7. Demo: gradient of final cost w.r.t. initial velocity ──────────────
    # Forward-mode AD works through MJX's while_loop contact solver.
    # jacfwd on a scalar → gradient vector, same shape as input.
    scalar_cost  = jax.jit(lambda qv: cost_from_qvel(mjx_model, mjx_data_base, qv, num_steps))
    grad_fwd     = jax.jit(jax.jacfwd(scalar_cost))
    grad_at_solution = grad_fwd(refined_qvel)
    print(f"∂cost/∂qvel at solution: {np.array(grad_at_solution)}  (should be ≈ 0)")


if __name__ == "__main__":
    main()