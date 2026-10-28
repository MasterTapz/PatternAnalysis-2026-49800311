#!/bin/bash
# Train the ticker-conditioned TimeGAN on all five stocks on Rangpur, then audit it.
# Queued by rangpur_submit.sh after rangpur_setup.sh, or on its own from the project folder:
# sbatch rangpur_train.sh. DATA (the LOBSTER folder) and JOINT_STEPS can be set with sbatch --export.
#SBATCH --job-name=timegan-5stocks
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "Job $SLURM_JOB_ID on $(hostname), start $(date)"
nvidia-smi

source "$HOME/miniconda3/bin/activate"
conda activate torch
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

DATA=${DATA:-$HOME/comp3710/data/LOBSTER}
JOINT_STEPS=${JOINT_STEPS:-25000}
RUN=5stocks_1s_T504_v24_structured_timegan_cfull_mf_sn8_j$((JOINT_STEPS / 1000))k_s0

python -u src/train.py --data-dir "$DATA" --tickers AAPL AMZN GOOG INTC MSFT --clock-seconds 1 --seq-len 504 \
    --eval-stride 24 --batch-size 65 --move-flag --static-noise-dim 8 --condition full \
    --joint-steps "$JOINT_STEPS" --run-name "$RUN"
# The audit runs on the CPU so its numbers match the laptop audits.
python src/predict.py --data-dir "$DATA" --device cpu --run "runs/$RUN" \
    --out-dir "runs/comparison_5stocks_cfull_j$((JOINT_STEPS / 1000))k" --gif

echo "Finished $(date)"
