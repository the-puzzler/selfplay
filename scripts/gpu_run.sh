#!/usr/bin/env bash
# GPU experiment runner. On a fresh CUDA box (Lambda/RunPod/vast, 1x 4090 or
# A100), from the repo root:
#
#   curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
#   uv sync --extra cuda
#   bash scripts/gpu_run.sh
#
# Runs the core ablation sequentially: diverse (conditional random spawns)
# vs fixed (canonical standing starts), same budget. ~500M env steps each.
set -euo pipefail

ITERS=${ITERS:-1000}
NUM_ENVS=${NUM_ENVS:-4096}
ROLLOUT=${ROLLOUT:-128}
SEED=${SEED:-0}

echo "== sanity: JAX sees the GPU =="
uv run python -c "import jax; print(jax.devices()); assert jax.devices()[0].platform == 'gpu'"

for MODE in diverse fixed; do
  OUT="runs/gpu-${MODE}-s${SEED}"
  echo "== training ${MODE} -> ${OUT} =="
  uv run python -m fighter2d.train \
    --reset-mode "${MODE}" \
    --num-envs "${NUM_ENVS}" \
    --rollout-len "${ROLLOUT}" \
    --iters "${ITERS}" \
    --ckpt-every 100 \
    --seed "${SEED}" \
    --out "${OUT}"
  uv run python -m fighter2d.plot --run "${OUT}"
done

echo "== head-to-head: diverse vs fixed =="
uv run python -m fighter2d.eval \
  --a "runs/gpu-diverse-s${SEED}/ckpt_final.msgpack" \
  --b "runs/gpu-fixed-s${SEED}/ckpt_final.msgpack" \
  --episodes 512 | tee "runs/h2h-s${SEED}.txt"

echo "== render money-shot videos =="
uv run python -m fighter2d.render --ckpt "runs/gpu-diverse-s${SEED}/ckpt_final.msgpack" \
  --reset-mode fixed --episodes 5 --out "runs/gpu-diverse-s${SEED}/standing.mp4"
uv run python -m fighter2d.render --ckpt "runs/gpu-diverse-s${SEED}/ckpt_final.msgpack" \
  --reset-mode diverse --episodes 20 --out "runs/gpu-diverse-s${SEED}/diverse.mp4"
echo "done. sync runs/ back to your machine."
