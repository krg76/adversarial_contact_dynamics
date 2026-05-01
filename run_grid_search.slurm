#!/bin/bash
#SBATCH --array=0-26
#SBATCH --job-name=gan_robotic_tuning
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --time=04:00:00

cd /home/kg878/project_pi_amd78/kg878/Projects/adversarial_contact_dynamics
module load miniconda
conda activate MJW

# 1. Define your parameter grids
MODELS=("lstm" "cnn" "mlp")
LRS_D=("0.0005" "0.00025" "0.000225")
LRS_G=("0.05" "0.0025" "0.000125")

# 2. Map the Task ID to specific indices
# Indexing math:
# model_idx  = task_id % 3
# temp_id    = task_id / 3
# lr_idx     = temp_id % 3
# l2_idx     = temp_id / 3

MODEL_IDX=$((SLURM_ARRAY_TASK_ID % 3))
TEMP_ID=$((SLURM_ARRAY_TASK_ID / 3))
LR_IDX=$((TEMP_ID % 3))
L2_IDX=$((TEMP_ID / 3))

# 3. Assign the actual values
SEL_MODEL=${MODELS[$MODEL_IDX]}
SEL_LR_D=${LRS_D[$LR_IDX]}
SEL_LR_G=${LRS_G[$L2_IDX]}

# 4. Create a unique output directory for this specific run
OUT_DIR="/home/kg878/scratch_pi_amd78/kg878/gan_training/results/run_${SEL_MODEL}_lr_d${SEL_LR_D}_lr_g${SEL_LR_G}"
mkdir -p $OUT_DIR

echo "Starting Task $SLURM_ARRAY_TASK_ID: Model=$SEL_MODEL, LR_D=$SEL_LR_D, LR_G=$SEL_LR_G"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# 5. Execute the training script
python /home/kg878/project_pi_amd78/kg878/Projects/adversarial_contact_dynamics/adversarial_training.py \
    --disc_type "$SEL_MODEL" \
    --d_lr "$SEL_LR_D" \
    --g_lr "$SEL_LR_G" \
    --output_dir "$OUT_DIR" \
    --gan_iterations 20