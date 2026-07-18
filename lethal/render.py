"""Render Lethal League episodes: ball trail heats up with speed, swing
arcs, hit-freeze flashes, live speed readout (the visible skill meter).

Usage:
  uv run python -m lethal.render --ckpt runs/ll-v1/ckpt_final.msgpack --out rally.mp4
  uv run python -m lethal.render --random --episodes 3 --out sanity.mp4
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

from lethal.env import (AX, AY, BALL_VMAX, BUNT_ACT, EPISODE_LEN, HIT_R,
                        N_ACTIONS, OBS_DIM, SW_ACTIVE, LethalEnv)

COLORS = ["#ff4d6d", "#4dc3ff"]
BG = "#0d0d14"


def speed_color(f):
    """white -> yellow -> orange -> red as speed fraction rises."""
    stops = np.array([[0.95, 0.95, 1.0], [1.0, 0.9, 0.3],
                      [1.0, 0.55, 0.1], [1.0, 0.12, 0.25]])
    x = np.clip(f, 0, 1) * 3
    i = int(np.clip(np.floor(x), 0, 2))
    w = x - i
    return tuple(stops[i] * (1 - w) + stops[i + 1] * w)


def draw(ax, s, trail, flash):
    ax.clear()
    ax.set_xlim(-AX - 0.2, AX + 0.2)
    ax.set_ylim(-0.2, AY + 0.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(plt.Rectangle((-AX, 0), 2 * AX, AY, fill=False,
                               ec="#3c3c50", lw=3))
    p = np.asarray(s.p)
    phase = np.asarray(s.phase)
    speed = float(np.linalg.norm(np.asarray(s.bv)))
    f = speed / BALL_VMAX
    for i in range(2):
        c = COLORS[i]
        x, y = p[i, 0], p[i, 1]
        ax.plot([x, x], [y + 0.1, y + 0.55], color=c, lw=6, solid_capstyle="round")
        ax.add_patch(plt.Circle((x, y + 0.72), 0.13, color=c))
        if phase[i] == SW_ACTIVE:
            ax.add_patch(plt.Circle((x, y + 0.35), HIT_R, fill=False,
                                    ec=c, lw=2, alpha=0.8))
        if phase[i] == BUNT_ACT:
            ax.add_patch(plt.Circle((x, y + 0.35), HIT_R * 0.8, fill=False,
                                    ec=c, lw=2, ls=":", alpha=0.8))
    # ball trail
    for k in range(1, len(trail)):
        a = 0.7 * k / len(trail)
        ax.plot([trail[k - 1][0], trail[k][0]], [trail[k - 1][1], trail[k][1]],
                color=speed_color(f), lw=1.5 + 4 * f, alpha=a)
    b = np.asarray(s.b)
    owner = int(np.asarray(s.owner))
    ec = COLORS[owner] if owner >= 0 else "#888899"
    ax.add_patch(plt.Circle(b, 0.14, color=speed_color(f), ec=ec, lw=2))
    if flash:
        ax.add_patch(plt.Circle(b, 0.5, fill=False, ec="#ffffff", lw=3, alpha=0.9))
    ax.text(0, AY + 0.15, f"{speed:4.1f}", color=speed_color(f),
            ha="center", fontsize=14, family="monospace", weight="bold")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--ckpt-b", type=str, default=None)
    p.add_argument("--random", action="store_true")
    p.add_argument("--out", type=str, default="lethal.mp4")
    p.add_argument("--reset-mode", choices=["fixed", "diverse"], default="fixed")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--slowmo", type=float, default=1.0)
    args = p.parse_args()

    env = LethalEnv(reset_mode=args.reset_mode)
    from lethal.grpo import Policy, policy_act
    network = Policy()
    params = network.init(jax.random.PRNGKey(0), jnp.zeros((1, OBS_DIM)))
    if args.ckpt:
        params = serialization.from_bytes(params, Path(args.ckpt).read_bytes())
    params_b = params
    if args.ckpt_b:
        params_b = serialization.from_bytes(params, Path(args.ckpt_b).read_bytes())

    fig, axp = plt.subplots(figsize=(8, 3.6), facecolor=BG)
    axp.set_facecolor(BG)
    step = jax.jit(env.step)
    rng = jax.random.PRNGKey(args.seed)
    frames = []
    n_sub = max(1, round(args.slowmo))
    for _ in range(args.episodes):
        rng, k = jax.random.split(rng)
        state, obs = env.reset(k)
        trail = [np.asarray(state.b).copy()]
        for _t in range(EPISODE_LEN):
            if args.random:
                rng, ka = jax.random.split(rng)
                action = jax.random.randint(ka, (2,), 0, N_ACTIONS)
            else:
                a0 = policy_act(network, params, obs[0:1])[0]
                a1 = policy_act(network, params_b, obs[1:2])[0]
                action = jnp.array([a0, a1])
            rng, k = jax.random.split(rng)
            prev = state
            state, obs, reward, done, info = step(k, prev, action)
            shown = jax.tree.map(lambda a, b: np.where(np.asarray(done), a, b),
                                 prev, state) if bool(done) else state
            trail.append(np.asarray(shown.b).copy())
            trail[:] = trail[-14:]
            flash = bool(np.asarray(reward).any())
            for _ in range(n_sub):
                draw(axp, shown, trail, flash)
                fig.canvas.draw()
                frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
            if bool(done):
                for _ in range(6 * n_sub):
                    frames.append(frames[-1])
                break
    imageio.mimsave(args.out, frames, fps=20)
    print(f"wrote {len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
