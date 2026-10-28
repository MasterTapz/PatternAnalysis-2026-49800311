#!/bin/bash
# Queue the 29 Oct batch on Rangpur, one GPU job each (run on the login node: bash rangpur_queue.sh):
#   1    the conditional model with the move-rate loss and VWAP in the checkpoint score,
#        15k joint steps at a constant learning rate
#   2-3  seeds 1 and 2 of the conditional model at 15k joint steps. Seed 0 is the 25k run of 28 Oct,
#        whose best checkpoint (12.5k) falls inside 15k steps at the same constant learning rate.
#   4-6  the raw-ladder penalty arms, lambda 1, 10 and 100, on the original AMZN setup
#        (event clock, 64 steps), against raw_timegan_s0 without the penalty
set -euo pipefail
cd "$(dirname "$0")"
FIVE="--tickers AAPL AMZN GOOG INTC MSFT --condition full --clock-seconds 1 --seq-len 504 --eval-stride 24
      --batch-size 65 --move-flag --static-noise-dim 8 --joint-steps 15000"

sbatch --job-name=timegan-fixes rangpur_job.sh $FIVE --move-rate-weight 100 --score-vwap
for SEED in 1 2; do
    sbatch --job-name=timegan-s$SEED rangpur_job.sh $FIVE --seed $SEED
done
for W in 1 10 100; do
    sbatch --job-name=timegan-bw$W --time=01:00:00 rangpur_job.sh --representation raw --book-weight $W
done
squeue --me
