# adversarial_contact_dynamics

Collecting Generated (ComFree_Warp) trajectories...
Training Discriminator for 10 epochs...
Discriminator Loss: 0.6934
Optimizing ComFree parameters to fool the Discriminator...
  Step   1 | loss=0.595874 | k=0.38802 (log_k=-0.91629), d=0.04979 (log_d=-4.60517)
  Step   2 | loss=1.197111 | k=0.65056 (log_k=-0.94669), d=4.25205 (log_d=-3.00000)
  Step   3 | loss=447534.000000 | k=0.04979 (log_k=-0.42992), d=0.04979 (log_d=1.44740)
  Step   4 | loss=0.823755 | k=0.04979 (log_k=-3.00000), d=0.06867 (log_d=-3.00000)
  Step   5 | loss=2.885920 | k=0.04979 (log_k=-3.00000), d=1.26270 (log_d=-2.67843)
  Step   6 | loss=111.259987 | k=0.25674 (log_k=-3.00000), d=0.04979 (log_d=0.23325)
  Step   7 | loss=1.052487 | k=18.77917 (log_k=-1.35968), d=20.08554 (log_d=-3.00000)
/home/kyle/Desktop/Projects/adversarial_contact_dynamics/adversarial_training.py:170: RuntimeWarning: overflow encountered in square
  l2_penalty = np.mean((traj_array - gt_trajs)**2)
  Step   8 | loss=nan | k=nan (log_k=2.93275), d=nan (log_d=3.00000)