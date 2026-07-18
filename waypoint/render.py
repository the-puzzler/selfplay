"""Render waypoint parkour: camera-following stickman on terrain, flag at
the waypoint, reach flash. --slowmo N for slower smooth playback.

Usage:
  uv run python -m waypoint.render --ckpt runs/wp-v1/ckpt_final.msgpack --out parkour.mp4
  uv run python -m waypoint.render --random --episodes 2 --out sanity.mp4
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

from waypoint.env import (EPISODE_LEN, GRID, OBS_DIM, SEG_AI, SEG_BI,
                          WaypointEnv)

BODY = "#5ad0a0"
BG = "#14141c"


def draw(ax, s, flash, cam_x):
    ax.clear()
    ax.set_xlim(cam_x - 5.5, cam_x + 5.5)
    ax.set_ylim(-3.2, 3.4)
    ax.set_aspect("equal")
    ax.axis("off")
    g = np.asarray(GRID)
    h = np.asarray(s.h)
    ax.fill_between(g, h, -3.2, color="#26263a", zorder=1)
    ax.plot(g, h, color="#3d3d5c", lw=2, zorder=2)
    pts = np.asarray(s.pts)
    for a, b in zip(SEG_AI, SEG_BI):
        ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]],
                color=BODY, lw=4, solid_capstyle="round", zorder=4)
    ax.add_patch(plt.Circle(pts[0], 0.11, color=BODY, zorder=4))
    wp = np.asarray(s.wp)
    ax.plot([wp[0], wp[0]], [wp[1] - 0.6, wp[1] + 0.5], color="#d9d9e8", lw=2, zorder=3)
    ax.add_patch(plt.Polygon([[wp[0], wp[1] + 0.5], [wp[0] + 0.45, wp[1] + 0.33],
                              [wp[0], wp[1] + 0.16]], color="#ff5a7a", zorder=3))
    if flash:
        ax.add_patch(plt.Circle(wp, 0.7, fill=False, ec="#ffd24a", lw=4, zorder=5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--random", action="store_true")
    p.add_argument("--out", type=str, default="parkour.mp4")
    p.add_argument("--reset-mode", choices=["fixed", "diverse"], default="fixed")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--slowmo", type=float, default=1.0)
    args = p.parse_args()

    env = WaypointEnv(reset_mode=args.reset_mode)
    from waypoint.ppo import HIDDEN, RecurrentAC, policy_act
    network = RecurrentAC()
    params = network.init(jax.random.PRNGKey(0), jnp.zeros((1, HIDDEN)),
                          jnp.zeros((1, OBS_DIM)), jnp.zeros((1,)))
    if args.ckpt:
        params = serialization.from_bytes(params, Path(args.ckpt).read_bytes())

    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=BG)
    ax.set_facecolor(BG)
    step = jax.jit(env.step)
    rng = jax.random.PRNGKey(args.seed)
    frames = []
    n_sub = max(1, round(args.slowmo))
    for _ in range(args.episodes):
        rng, k = jax.random.split(rng)
        state, obs = env.reset(k)
        h = jnp.zeros((1, HIDDEN))
        dp = jnp.ones((1,))
        for _t in range(EPISODE_LEN):
            if args.random:
                rng, ka = jax.random.split(rng)
                action = jax.random.uniform(ka, (8,), minval=-1, maxval=1)
            else:
                h, a = policy_act(network, params, h, obs[None], dp)
                action = jnp.asarray(a[0])
                dp = jnp.zeros((1,))
            rng, k = jax.random.split(rng)
            prev = state
            state, obs, reward, done, _ = step(k, prev, action)
            shown = jax.tree.map(lambda a, b: np.where(np.asarray(done), a, b),
                                 prev, state) if bool(done) else state
            flash = bool(np.asarray(reward) > 0)
            cam = float(np.asarray(shown.pts)[2, 0])
            for sub in range(n_sub):
                f = (sub + 1) / n_sub
                interp = jax.tree.map(lambda a, b: np.asarray(a) * (1 - f) + np.asarray(b) * f,
                                      prev, shown)
                draw(ax, interp, flash and sub == n_sub - 1, cam)
                fig.canvas.draw()
                frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
            if bool(done):
                for _ in range(5 * n_sub):
                    frames.append(frames[-1])
                break
    imageio.mimsave(args.out, frames, fps=20)
    print(f"wrote {len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
