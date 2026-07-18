"""Render stickfight episodes to mp4.

Segments colored by HP (full color -> dark as HP drops); destroyed limbs
vanish. HP bars for head/torso above the arena. --slowmo N for N-times
slower smooth playback.

Usage:
  uv run python -m stickfight.render --ckpt runs/stick-v1/ckpt_final.msgpack --out fight.mp4
  uv run python -m stickfight.render --random --episodes 3 --out ragdoll.mp4
"""

import argparse
from pathlib import Path

import imageio
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from flax import serialization

from stickfight.env import (ARENA_X, EPISODE_LEN, HEAD, N_SEGS, OBS_DIM, SEG_A,
                            SEG_B, TORSO, StickFightEnv, _seg_alive)

BASE = ["#ff5a5a", "#5aa0ff"]


def draw(ax, s):
    ax.clear()
    ax.set_xlim(-ARENA_X - 0.1, ARENA_X + 0.1)
    ax.set_ylim(-0.15, 2.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.axhline(0, color="#555566", lw=2)
    pts = np.asarray(s.pts)
    hp = np.asarray(s.hp)
    eff = np.asarray(_seg_alive(jnp.asarray(hp)))
    for i in range(2):
        base = np.array(mcolors.to_rgb(BASE[i]))
        for k in range(N_SEGS):
            if hp[i, k] <= 0:
                continue  # destroyed: limb gone
            a, b = pts[i, int(SEG_A[k])], pts[i, int(SEG_B[k])]
            frac = 0.25 + 0.75 * hp[i, k]
            col = tuple(base * frac) if eff[i, k] else (0.45, 0.45, 0.5)
            ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=4,
                    solid_capstyle="round")
        # head circle
        if hp[i, HEAD] > 0:
            ax.add_patch(plt.Circle(pts[i, 0], 0.09,
                                    color=tuple(base * (0.25 + 0.75 * hp[i, HEAD]))))
        # HP bars (head + torso)
        x0 = -2.6 + 5.2 * i - 0.5 * i
        for row, seg in enumerate((HEAD, TORSO)):
            ax.plot([x0, x0 + 0.6], [2.45 - 0.12 * row] * 2, color="#333344", lw=5)
            ax.plot([x0, x0 + 0.6 * max(hp[i, seg], 1e-3)],
                    [2.45 - 0.12 * row] * 2, color=BASE[i], lw=5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--ckpt-b", type=str, default=None)
    p.add_argument("--random", action="store_true", help="random policy")
    p.add_argument("--out", type=str, default="stickfight.mp4")
    p.add_argument("--reset-mode", choices=["fixed", "diverse"], default="fixed")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--slowmo", type=float, default=1.0)
    args = p.parse_args()

    env = StickFightEnv(reset_mode=args.reset_mode)
    from stickfight.ppo import ActorCritic, policy_act
    network = ActorCritic()
    params = network.init(jax.random.PRNGKey(0), jnp.zeros((1, OBS_DIM)))
    if args.ckpt:
        params = serialization.from_bytes(params, Path(args.ckpt).read_bytes())
    params_b = params
    if args.ckpt_b:
        params_b = serialization.from_bytes(params, Path(args.ckpt_b).read_bytes())

    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#14141c")
    ax.set_facecolor("#14141c")
    step = jax.jit(env.step)
    rng = jax.random.PRNGKey(args.seed)
    frames = []
    for _ in range(args.episodes):
        rng, k = jax.random.split(rng)
        state, obs = env.reset(k)
        for _t in range(EPISODE_LEN):
            if args.random:
                rng, ka = jax.random.split(rng)
                action = jax.random.uniform(ka, (2, 8), minval=-1, maxval=1)
            else:
                a0 = policy_act(network, params, obs[0:1])[0]
                a1 = policy_act(network, params_b, obs[1:2])[0]
                action = jnp.stack([a0, a1])
            rng, k = jax.random.split(rng)
            prev_state = state
            state, obs, reward, done, _ = step(k, state, action)
            shown = jax.tree.map(lambda a, b: np.where(np.asarray(done), a, b),
                                 prev_state, state) if bool(done) else state
            n_sub = max(1, round(2 * args.slowmo))
            for sub in range(n_sub):
                f = (sub + 1) / n_sub
                interp = jax.tree.map(
                    lambda a, b: np.asarray(a) * (1 - f) + np.asarray(b) * f,
                    prev_state, shown)
                draw(ax, interp)
                fig.canvas.draw()
                frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
            if bool(done):
                for _ in range(4 * n_sub):
                    frames.append(frames[-1])
                break
    imageio.mimsave(args.out, frames, fps=20)
    print(f"wrote {len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
