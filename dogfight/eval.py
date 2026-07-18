"""Head-to-head dogfight evaluation: policy A (ship 0) vs policy B (ship 1).

Runs N episodes, then N with sides swapped. Use --a/--b 'random' for a
random-init policy.

Usage:
  uv run python -m dogfight.eval --a runs/dog-v1/ckpt_00200.msgpack \\
                                 --b runs/dog-v1/ckpt_00050.msgpack
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import serialization

from dogfight.env import OBS_DIM, DogfightEnv
from dogfight.ppo import ActorCritic


def load_params(spec, network):
    params = network.init(jax.random.PRNGKey(0), jnp.zeros((1, OBS_DIM)))
    if spec != "random":
        params = serialization.from_bytes(params, Path(spec).read_bytes())
    return params


def head_to_head(env, network, params0, params1, rng, n_episodes):
    def act(params, obs):
        logits, _ = network.apply(params, obs)
        return jnp.argmax(logits, axis=-1)

    def run_episode(rng):
        rng, k = jax.random.split(rng)
        state, obs = env.reset(k)

        def cond(carry):
            return ~carry[3]

        def body(carry):
            rng, state, obs, _, _ = carry
            rng, k = jax.random.split(rng)
            action = jnp.stack([act(params0, obs[0]), act(params1, obs[1])])
            state, obs, reward, done, _ = env.step(k, state, action)
            return rng, state, obs, done, reward

        carry = (rng, state, obs, jnp.array(False), jnp.zeros(2))
        _, _, _, _, reward = jax.lax.while_loop(cond, body, carry)
        return reward

    rewards = jax.jit(jax.vmap(run_episode))(jax.random.split(rng, n_episodes))
    wins0 = int((rewards[:, 0] > 0).sum())
    wins1 = int((rewards[:, 1] > 0).sum())
    return wins0, wins1, n_episodes - wins0 - wins1


def match(a, b, episodes=128, seed=0, reset_mode="fixed"):
    env = DogfightEnv(reset_mode=reset_mode)
    network = ActorCritic()
    pa, pb = load_params(a, network), load_params(b, network)
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    a_w, b_w, d = head_to_head(env, network, pa, pb, k1, episodes)
    b_w2, a_w2, d2 = head_to_head(env, network, pb, pa, k2, episodes)
    n = 2 * episodes
    return (a_w + a_w2) / n, (b_w + b_w2) / n, (d + d2) / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--episodes", type=int, default=128, help="per side")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reset-mode", choices=["fixed", "diverse"], default="fixed")
    args = p.parse_args()
    ra, rb, rd = match(args.a, args.b, args.episodes, args.seed, args.reset_mode)
    print(f"A={args.a}\nB={args.b}")
    print(f"A wins {ra:.1%} | B wins {rb:.1%} | draws {rd:.1%}")


if __name__ == "__main__":
    main()
