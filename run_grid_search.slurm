#!/bin/bash
#SBATCH --job-name=gan_robotic_tuning
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err
#SBATCH --array=0-17                # 3 models * 3 LRs * 2 L2_weights - 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

# 1. Define your parameter grids
MODELS=("lstm" "cnn" "mlp")
LRS=("0.001" "0.0005" "0.0001")
L2_WEIGHTS=("1e-4" "1e-5")

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
SEL_LR=${LRS[$LR_IDX]}
SEL_L2=${L2_WEIGHTS[$L2_IDX]}

# 4. Create a unique output directory for this specific run
OUT_DIR="./results/run_${SEL_MODEL}_lr${SEL_LR}_l2${SEL_L2}"
mkdir -p $OUT_DIR

echo "Starting Task $SLURM_ARRAY_TASK_ID: Model=$SEL_MODEL, LR=$SEL_LR, L2=$SEL_L2"

# 5. Execute the training script
python adversarial_training.py \
    --disc_type "$SEL_MODEL" \
    --d_lr "$SEL_LR" \
    --g_l2_weight "$SEL_SEL_L2" \
    --output_dir "$OUT_DIR" \
    --gan_iterations 25