"""Render a fight to mp4 using a trained checkpoint (plain MuJoCo, CPU).

Usage:
  uv run python -m fighter2d.render --ckpt runs/diverse/ckpt_final.msgpack --out fight.mp4
"""

import argparse
from pathlib import Path

import imageio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from flax import serialization

from fighter2d import model as fmodel
from fighter2d.env import DOWN_Z, EPISODE_LEN, N_FRAMES, OBS_DIM, FighterEnv
from fighter2d.ppo import ActorCritic, policy_act


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None, help="omit for a random policy")
    p.add_argument("--out", type=str, default="fight.mp4")
    p.add_argument("--reset-mode", choices=["fixed", "diverse"], default="fixed")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=3)
    args = p.parse_args()

    env = FighterEnv(reset_mode=args.reset_mode)
    m = env.mj_model
    d = mujoco.MjData(m)
    network = ActorCritic()
    params = network.init(jax.random.PRNGKey(0), jnp.zeros((1, OBS_DIM)))
    if args.ckpt:
        params = serialization.from_bytes(params, Path(args.ckpt).read_bytes())

    renderer = mujoco.Renderer(m, height=480, width=854)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 0.9]
    cam.distance = 5.5
    cam.elevation = -12
    cam.azimuth = 90

    rng = jax.random.PRNGKey(args.seed)
    frames = []
    for ep in range(args.episodes):
        rng, k = jax.random.split(rng)
        qpos, qvel = env.reset_qpos_qvel(k)
        d.qpos[:] = np.asarray(qpos)
        d.qvel[:] = np.asarray(qvel)
        mujoco.mj_forward(m, d)
        for t in range(EPISODE_LEN):
            obs = env._obs(jnp.asarray(d.qpos), jnp.asarray(d.qvel), jnp.asarray(t))
            action = policy_act(network, params, obs)
            d.ctrl[:] = action.reshape(-1)
            for _ in range(N_FRAMES):
                mujoco.mj_step(m, d)
            renderer.update_scene(d, camera=cam)
            frames.append(renderer.render())
            z = fmodel.TORSO_INIT_Z + np.array([d.qpos[0 + 1], d.qpos[13 + 1]])
            x = np.array(fmodel.INIT_X) + np.array([d.qpos[0], d.qpos[13]])
            if (z < DOWN_Z).any() or (np.abs(x) > fmodel.ARENA_HALF).any():
                break
    imageio.mimsave(args.out, frames, fps=25)
    print(f"wrote {len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
