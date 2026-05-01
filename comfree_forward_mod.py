# Copyright (c) 2026 ASU IRIS
# Licensed for noncommercial academic research use only.
# See comfree_warp/comfree_core/LICENSE for terms.
# -----------------------------------------------------------------------------
import warp as wp
import comfree_warp as cf_mjwarp

@wp.kernel
def _compute_qfrc_constraint(
  # Model:
  opt_timestep: wp.array(dtype=float),
  # comfree model parameter
  comfree_stiffness: wp.array(dtype=float),
  comfree_damping: wp.array(dtype=float),
  # Data in: 
  J: wp.array3d(dtype=float),
  efc_dist: wp.array2d(dtype=float),
  efc_mass: wp.array2d(dtype=float),
  qvel_smooth_pred: wp.array2d(dtype=float),
  nv: int,
  nefc: wp.array(dtype=int),
  # Out: 
  efc_force: wp.array2d(dtype=float),
  qfrc_constraint: wp.array2d(dtype=float),
):
  worldid, efcid = wp.tid()
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]

  if efcid >= nefc[worldid]:
    return
  
  efc_vel = float(0.0)
  for i in range(nv):
    efc_vel += J[worldid, efcid, i] * qvel_smooth_pred[worldid, i]

  # 3rd order non-linear parameters: 3 coefficients per system
  num_param_sets = comfree_stiffness.shape[0] // 3
  base_idx = (worldid % num_param_sets) * 3

  k1, k2, k3 = comfree_stiffness[base_idx], comfree_stiffness[base_idx + 1], comfree_stiffness[base_idx + 2]
  d1, d2, d3 = comfree_damping[base_idx], comfree_damping[base_idx + 1], comfree_damping[base_idx + 2]

  # predictive penetration with smoothing velocity
  efc_penetration = efc_vel * timestep + efc_dist[worldid, efcid]

  # 3rd order non-linear stiffness and damping calculation
  # acc = -( (k1*p + k2*p^2 + k3*p^3)/dt + (d1*v + d2*v^2 + d3*v^3)/dt )
  acc_k = (k1 * efc_penetration + k2 * efc_penetration * efc_penetration + k3 * efc_penetration * efc_penetration * efc_penetration) / timestep
  acc_d = (d1 * efc_vel + d2 * efc_vel * efc_vel + d3 * efc_vel * efc_vel * efc_vel) / timestep
  
  efc_acc = -acc_k - acc_d

  efc_frc=  efc_mass[worldid, efcid] * efc_acc
  efc_frc= wp.max(efc_frc, 0.0)

  # output
  efc_force[worldid, efcid] =efc_frc
  for i in range(nv):
    # qfrc_constraint[worldid][i] +=  J[worldid, efcid, i] * efc_frc
    wp.atomic_add(qfrc_constraint, worldid, i, J[worldid, efcid, i] * efc_frc)
