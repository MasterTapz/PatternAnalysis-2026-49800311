#!/bin/bash
# Queue the setup job on a CPU node, then the training job, which starts only if setup succeeds.
# Run on the Rangpur login node: bash rangpur_submit.sh
set -euo pipefail
cd "$(dirname "$0")"
setup=$(sbatch --parsable rangpur_setup.sh)
train=$(sbatch --parsable --dependency=afterok:"$setup" rangpur_train.sh)
echo "Queued setup job $setup and training job $train (waits for setup)"
squeue --me
