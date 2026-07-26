#!/usr/bin/env bash
# Reproduce the three-game sweep: train Pgx's exact AZNet architecture with
# search-free PPO, then verify against the published Gumbel AlphaZero baselines.
#
#   uv sync --extra cuda
#   bash scripts/run_sweep.sh othello   # ~10h on one RTX PRO 6000
#   bash scripts/run_sweep.sh hex       # ~18h
#   bash scripts/run_sweep.sh go        # ~12.5h
set -euo pipefail
cd "$(dirname "$0")/.."

POOL="--opp-pool-frac 0.5 --pool-size 12 --pool-every 100"
COMMON="--arch aznet --width 128 --depth 6 --num-envs 4096 --rollout-len 32
        --reset-pool 8192 --iters 8000 --eval-every 1000 --ckpt-every 1000 --seed 0"

case "${1:?usage: run_sweep.sh othello|hex|go}" in
  othello)
    uv run python -m pgx4.train --game othello  $COMMON $POOL --reset-max-depth 50 --out runs/oth-aznet
    uv run python -m pgx4.verify --game othello --model othello_v0 --ckpt runs/oth-aznet/ckpt_final.msgpack
    ;;
  hex)
    uv run python -m pgx4.train --game hex      $COMMON $POOL --reset-max-depth 40 --out runs/hex-aznet
    uv run python -m pgx4.verify --game hex --model hex_v0 --ckpt runs/hex-aznet/ckpt_final.msgpack
    ;;
  go)
    uv run python -m pgx4.train --game go_9x9   $COMMON $POOL --reset-max-depth 40 --out runs/go9-aznet
    uv run python -m pgx4.verify --game go_9x9 --model go_9x9_v0 --ckpt runs/go9-aznet/ckpt_final.msgpack --opening-plies 20
    ;;
  *) echo "unknown game: $1"; exit 1;;
esac
