"""Render dogfight episodes to mp4: neon ships, trails, bullets, hit flash.

Usage:
  uv run python -m dogfight.render --ckpt runs/dog-v0/ckpt_final.msgpack --out dog.mp4
"""

import argparse
from pathlib import Path

import imageio
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flax import serialization

from dogfight.env import ARENA, EPISODE_LEN, OBS_DIM, SHIP_R, DogfightEnv
from dogfight.ppo import ActorCritic, policy_act

COLORS = ["#ff5a5a", "#5aa0ff"]
TRAIL = 25


def draw_frame(ax, s, trails, flash):
    ax.clear()
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(plt.Rectangle((-ARENA, -ARENA), 2 * ARENA, 2 * ARENA,
                               fill=False, ec="#3a3a4a", lw=2))
    pos, th = np.asarray(s.pos), np.asarray(s.th)
    for i in range(2):
        tr = np.array(trails[i])
        if len(tr) > 1:
            for k in range(1, len(tr)):
                ax.plot(tr[k - 1 : k + 1, 0], tr[k - 1 : k + 1, 1],
                        color=COLORS[i], alpha=0.5 * k / len(tr), lw=1.5)
        d = np.array([np.cos(th[i]), np.sin(th[i])])
        p = np.array([np.cos(th[i] + 2.5), np.sin(th[i] + 2.5)])
        q = np.array([np.cos(th[i] - 2.5), np.sin(th[i] - 2.5)])
        tri = np.stack([pos[i] + 1.6 * SHIP_R * d,
                        pos[i] + 1.2 * SHIP_R * p,
                        pos[i] + 1.2 * SHIP_R * q])
        ax.add_patch(plt.Polygon(tri, color=COLORS[i]))
        bpos, bttl = np.asarray(s.bpos[i]), np.asarray(s.bttl[i])
        live = bttl > 0
        ax.scatter(bpos[live, 0], bpos[live, 1], s=14, color=COLORS[i],
                   edgecolors="white", linewidths=0.4, zorder=5)
        if flash == i:
            ax.add_patch(plt.Circle(pos[i], 4 * SHIP_R, fill=False,
                                    ec="#ffd24a", lw=3))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--out", type=str, default="dogfight.mp4")
    p.add_argument("--reset-mode", choices=["fixed", "diverse"], default="fixed")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = DogfightEnv(reset_mode=args.reset_mode)
    network = ActorCritic()
    params = network.init(jax.random.PRNGKey(0), jnp.zeros((1, OBS_DIM)))
    if args.ckpt:
        params = serialization.from_bytes(params, Path(args.ckpt).read_bytes())

    fig, ax = plt.subplots(figsize=(6, 6), facecolor="#101018")
    ax.set_facecolor("#101018")
    step = jax.jit(env.step)

    rng = jax.random.PRNGKey(args.seed)
    frames = []
    for _ in range(args.episodes):
        rng, k = jax.random.split(rng)
        state, obs = env.reset(k)
        trails = [[np.asarray(state.pos[i]).copy()] for i in range(2)]
        for _t in range(EPISODE_LEN):
            action = jnp.asarray(policy_act(network, params, obs))
            rng, k = jax.random.split(rng)
            prev = state
            state, obs, reward, done, _ = step(k, prev, action)
            r = np.asarray(reward)
            flash = int(np.argmin(r)) if r[0] != r[1] else None
            # state is post-auto-reset on done; draw the pre-reset world.
            shown = jax.tree.map(lambda a, b: np.where(done, a, b), prev, state) if done else state
            for i in range(2):
                trails[i].append(np.asarray(shown.pos[i]).copy())
                trails[i][:] = trails[i][-TRAIL:]
            draw_frame(ax, shown, trails, flash)
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            frames.append(frame)
            if bool(done):
                for _ in range(6):  # linger on the kill
                    frames.append(frame)
                break
    imageio.mimsave(args.out, frames, fps=15)
    print(f"wrote {len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
