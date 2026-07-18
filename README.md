# selfplay — OmniReset × self-play

Testbed for extending [OmniReset](https://weirdlabuw.github.io/omnireset/)-style
diverse resets to **competitive self-play with purely sparse rewards**.
The thesis: you don't need reward shaping or curricula to train fighting
agents end-to-end — reset the fighters into massively diverse states and let
PPO solve the easy states first (opponent already falling) and propagate
outward to the hard ones (getting up, off-balance recovery, actual fighting).

Endgame: 3D MuJoCo MMA with limb health. Current phase: validate the
machinery in a cheap 2D env.

## The 2D env (`fighter2d/`)

Two planar stick-figure bipeds (torso, head, 2 legs, 2 arms — 13 dof / 10
actuators each) in an MJX arena. You lose by falling (torso center below
0.25m — crouched stances are legal) or leaving the arena. Reward is **purely sparse and zero-sum**: ±1 at episode
end, nothing else. No shaping of any kind.

- `model.py` — MuJoCo XML builder
- `env.py` — MJX env; `reset_mode="fixed"` (canonical standing) vs
  `"diverse"` (OmniReset-style: random poses, heights, orientations,
  velocities, positions — including fallen and mid-air states)
- `ppo.py` — PureJaxRL-style self-play PPO; one shared policy controls both
  fighters, every env contributes 2 agent-slots per step
- `train.py` / `eval.py` / `render.py` — CLI entrypoints

## Quickstart

```bash
uv sync

# The core ablation:
uv run python -m fighter2d.train --reset-mode diverse --out runs/diverse
uv run python -m fighter2d.train --reset-mode fixed   --out runs/fixed

# Who wins?
uv run python -m fighter2d.eval --a runs/diverse/ckpt_final.msgpack \
                                --b runs/fixed/ckpt_final.msgpack

# Watch a fight
uv run python -m fighter2d.render --ckpt runs/diverse/ckpt_final.msgpack --out fight.mp4
```

Runs on CPU (slow, fine for smoke tests); the same code jits to GPU/TPU
unchanged — rent an NVIDIA box and crank `--num-envs`.

## Roadmap

1. ✅ 2D self-play skeleton, sparse reward, diverse resets, ablation flags
2. Real training runs on GPU: does `diverse` beat `fixed` head-to-head?
3. Opponent checkpoint pool (avoid strategy cycling), visited-state archive
   resets (the self-play analog of OmniReset's reset distribution)
4. 3D humanoids + limb-health game mechanic (damage weakens actuators)
5. Blog post with interactive replays
