#!/bin/bash
# Install what the TimeGAN code needs into the torch conda env, then check every import and the data.
# Runs on a CPU node so no A100 time is spent on installs; rangpur_submit.sh queues it first.
#SBATCH --job-name=timegan-setup
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=slurm-setup-%j.out
#SBATCH --error=slurm-setup-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "Job $SLURM_JOB_ID on $(hostname), start $(date)"

# pip unpacks into the temp folder, which is small on the nodes.
export TMPDIR=$HOME/tmp TMP=$HOME/tmp TEMP=$HOME/tmp
mkdir -p "$TMPDIR"
source "$HOME/miniconda3/bin/activate"
conda activate torch
python --version
pip install --no-cache-dir --quiet numpy pandas scipy scikit-learn scikit-image matplotlib pillow
(cd src && python -c "import train, predict; print('imports OK')")

DATA=${DATA:-$HOME/comp3710/data/LOBSTER}
for T in AAPL AMZN GOOG INTC MSFT; do
    # AMZN sits at the top of the folder, the others in a folder of their own.
    if ! compgen -G "$DATA/${T}_*_orderbook_10.csv" > /dev/null && ! compgen -G "$DATA/$T/${T}_*_orderbook_10.csv" > /dev/null; then
        echo "No $T order book under $DATA"
        exit 1
    fi
done
echo "Data for all five tickers found in $DATA"
echo "Finished $(date)"
