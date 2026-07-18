"""Render Nidhogg duels to mp4: flat arena, stick fencers, stance-height
swords, lunge lean, blood flash on kills, exit gates, tug-of-war camera.

Usage:
  uv run python -m nidhogg.render --ckpt runs/nid-v1/ckpt_final.msgpack --out duel.mp4
  uv run python -m nidhogg.render --random --episodes 2 --out sanity.mp4
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

from nidhogg.env import (ACTIVE, DEAD, EPISODE_LEN, JUMP_H, JUMP_TICKS,
                         N_ACTIONS, OBS_DIM, RECOVER, STAGGER, STARTUP, X_MAX,
                         NidhoggEnv)

COLORS = ["#ff8c1a", "#ffd21a"]  # nidhogg orange & yellow
BG = "#2a1a3a"
HEIGHTS = [1.4, 1.0, 0.6]


def draw(ax, s, flash):
    ax.clear()
    ax.set_xlim(-X_MAX - 0.5, X_MAX + 0.5)
    ax.set_ylim(-0.3, 3.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.axhline(0, color="#120a1c", lw=6)
    for gx, c in ((-X_MAX, COLORS[1]), (X_MAX, COLORS[0])):
        ax.axvline(gx, color=c, lw=3, alpha=0.7)
    x = np.asarray(s.x)
    air = np.asarray(s.air)
    stance = np.asarray(s.stance)
    phase = np.asarray(s.phase)
    for i in range(2):
        c = COLORS[i]
        dead = phase[i] == DEAD
        y0 = JUMP_H * np.sin(np.pi * (JUMP_TICKS - air[i]) / JUMP_TICKS) if air[i] > 0 else 0.0
        d = 1.0 if i == 0 else -1.0  # facing/exit direction
        if dead:
            ax.plot([x[i] - 0.5, x[i] + 0.5], [0.15, 0.15], color=c, lw=5, alpha=0.35)
            continue
        lean = 0.25 * d if phase[i] in (STARTUP, ACTIVE) else (-0.15 * d if phase[i] == STAGGER else 0.0)
        hip = np.array([x[i], y0 + 0.7])
        neck = hip + np.array([lean, 0.75])
        ax.plot([hip[0], neck[0]], [hip[1], neck[1]], color=c, lw=5, solid_capstyle="round")
        ax.add_patch(plt.Circle(neck + [0.06 * d, 0.18], 0.14, color=c))
        for leg in (-0.28, 0.28):
            ax.plot([hip[0], x[i] + leg], [hip[1], y0], color=c, lw=4, solid_capstyle="round")
        # sword at stance height
        sh = y0 + HEIGHTS[stance[i]]
        ext = 1.05 if phase[i] == ACTIVE else (0.55 if phase[i] != RECOVER else 0.4)
        sx = x[i] + 0.15 * d
        ax.plot([sx, sx + ext * d], [sh, sh], color="#e8e8f0", lw=2.5)
        ax.plot([sx + 0.12 * d] * 2, [sh - 0.08, sh + 0.08], color="#e8e8f0", lw=2)
    if flash is not None:
        ax.add_patch(plt.Circle((flash[0], flash[1] + 1.0), 0.5, color="#d40f30", alpha=0.85))
    # progress marker
    mid = np.clip(0.5 * (x[0] + x[1]) / X_MAX, -1, 1)
    ax.plot([mid * (X_MAX - 1)], [3.0], marker="v", color="#e8e8f0", ms=8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--ckpt-b", type=str, default=None)
    p.add_argument("--random", action="store_true")
    p.add_argument("--out", type=str, default="nidhogg.mp4")
    p.add_argument("--reset-mode", choices=["fixed", "diverse"], default="fixed")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--slowmo", type=float, default=1.0)
    args = p.parse_args()

    env = NidhoggEnv(reset_mode=args.reset_mode)
    from nidhogg.grpo import Policy, policy_act
    network = Policy()
    params = network.init(jax.random.PRNGKey(0), jnp.zeros((1, OBS_DIM)))
    if args.ckpt:
        params = serialization.from_bytes(params, Path(args.ckpt).read_bytes())
    params_b = params
    if args.ckpt_b:
        params_b = serialization.from_bytes(params, Path(args.ckpt_b).read_bytes())

    fig, ax = plt.subplots(figsize=(10, 2.6), facecolor=BG)
    ax.set_facecolor(BG)
    step = jax.jit(env.step)
    rng = jax.random.PRNGKey(args.seed)
    frames = []
    fps_sub = max(1, round(args.slowmo))
    for _ in range(args.episodes):
        rng, k = jax.random.split(rng)
        state, obs = env.reset(k)
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
            flash = None
            if float(np.asarray(info["kills"])) > 0:
                px = np.asarray(shown.x)
                dp = np.asarray(shown.phase)
                for i in range(2):
                    if dp[i] == DEAD:
                        flash = (px[i], 0.0)
            for _ in range(fps_sub):
                draw(ax, shown, flash)
                fig.canvas.draw()
                frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
            if bool(done):
                for _ in range(6 * fps_sub):
                    frames.append(frames[-1])
                break
    imageio.mimsave(args.out, frames, fps=15)
    print(f"wrote {len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
