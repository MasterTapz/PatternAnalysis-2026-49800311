#!/bin/bash
# One training run and its audit on Rangpur, with any train.py options:
#   sbatch rangpur_job.sh --tickers AAPL AMZN GOOG INTC MSFT --condition full ...
# The run gets train.py's default name; its audit and GIFs go to runs/<run>/predict/ and,
# for several tickers, the comparison to runs/comparison_<run>/. rangpur_queue.sh submits a batch.
#SBATCH --job-name=timegan
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
echo "Options: $*"
nvidia-smi

source "$HOME/miniconda3/bin/activate"
conda activate torch
DATA=${DATA:-$HOME/comp3710/data/LOBSTER}

# train.py names the run from its options; ask it for the name so the audit can find the folder.
RUN=$(cd src && python -c "import sys, train; print(train.parse_args(sys.argv[1:]).run_name)" --data-dir "$DATA" "$@")
echo "Run $RUN"
python -u src/train.py --data-dir "$DATA" --run-name "$RUN" "$@"
# The audit runs on the CPU so its numbers match the laptop audits.
python src/predict.py --data-dir "$DATA" --device cpu --run "runs/$RUN" --out-dir "runs/comparison_$RUN" --gif

echo "Finished $(date)"
