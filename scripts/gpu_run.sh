#!/usr/bin/env bash
# GPU experiment runner. On a fresh CUDA box (Lambda/RunPod/vast, 1x 4090 or
# A100), from the repo root:
#
#   curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
#   uv sync --extra cuda
#   bash scripts/gpu_run.sh
#
# One big diverse-spawn self-play run, ~500M env steps at defaults.
set -euo pipefail

ITERS=${ITERS:-1000}
NUM_ENVS=${NUM_ENVS:-4096}
ROLLOUT=${ROLLOUT:-128}
SEED=${SEED:-0}
OUT="runs/gpu-diverse-s${SEED}"

echo "== sanity: JAX sees the GPU =="
uv run python -c "import jax; print(jax.devices()); assert jax.devices()[0].platform == 'gpu'"

echo "== training -> ${OUT} =="
uv run python -m fighter2d.train \
  --reset-mode diverse \
  --num-envs "${NUM_ENVS}" \
  --rollout-len "${ROLLOUT}" \
  --iters "${ITERS}" \
  --ckpt-every 100 \
  --seed "${SEED}" \
  --out "${OUT}"
uv run python -m fighter2d.plot --run "${OUT}"

echo "== skill check: final vs mid-training checkpoint =="
MID=$(printf 'ckpt_%05d.msgpack' $((ITERS / 2)))
uv run python -m fighter2d.eval \
  --a "${OUT}/ckpt_final.msgpack" \
  --b "${OUT}/${MID}" \
  --reset-mode diverse --episodes 512 | tee "${OUT}/skill_check.txt"

echo "== render money-shot videos =="
uv run python -m fighter2d.render --ckpt "${OUT}/ckpt_final.msgpack" \
  --reset-mode fixed --episodes 5 --out "${OUT}/standing.mp4"
uv run python -m fighter2d.render --ckpt "${OUT}/ckpt_final.msgpack" \
  --reset-mode diverse --episodes 20 --out "${OUT}/diverse.mp4"
echo "done. sync runs/ back to your machine."
