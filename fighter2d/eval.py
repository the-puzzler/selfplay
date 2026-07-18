"""Head-to-head evaluation: policy A (fighter 0) vs policy B (fighter 1).

Runs N episodes from canonical standing starts, then N with sides swapped,
and reports win rates. Use --b random to fight a random policy.

Usage:
  uv run python -m fighter2d.eval --a runs/diverse/ckpt_final.msgpack --b random
  uv run python -m fighter2d.eval --a runs/diverse/ckpt_final.msgpack --b runs/fixed/ckpt_final.msgpack
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from fighter2d.env import OBS_DIM, FighterEnv
from fighter2d.ppo import ActorCritic


def load_params(spec, network):
    params = network.init(jax.random.PRNGKey(0), jnp.zeros((1, OBS_DIM)))
    if spec != "random":
        params = serialization.from_bytes(params, Path(spec).read_bytes())
    return params


def head_to_head(env, network, params0, params1, rng, n_episodes):
    """params0 controls fighter 0, params1 fighter 1. Returns (wins0, wins1, draws)."""

    def act(params, obs):
        mean, _, _ = network.apply(params, obs)
        return jnp.clip(mean, -1, 1)

    def run_episode(rng):
        rng, k = jax.random.split(rng)
        state, obs = env.reset(k)

        def cond(carry):
            _, _, _, done, _ = carry
            return ~done

        def body(carry):
            rng, state, obs, _, _ = carry
            rng, k = jax.random.split(rng)
            action = jnp.stack([act(params0, obs[0]), act(params1, obs[1])])
            state, obs, reward, done, _ = env.step(k, state, action)
            return rng, state, obs, done, reward

        carry = (rng, state, obs, jnp.array(False), jnp.zeros(2))
        _, _, _, _, reward = jax.lax.while_loop(cond, body, carry)
        return reward

    keys = jax.random.split(rng, n_episodes)
    rewards = jax.jit(jax.vmap(run_episode))(keys)  # (N, 2)
    wins0 = (rewards[:, 0] > 0).sum()
    wins1 = (rewards[:, 1] > 0).sum()
    draws = n_episodes - wins0 - wins1
    return int(wins0), int(wins1), int(draws)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="checkpoint path or 'random'")
    p.add_argument("--b", required=True, help="checkpoint path or 'random'")
    p.add_argument("--episodes", type=int, default=128, help="per side")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = FighterEnv(reset_mode="fixed")
    network = ActorCritic()
    pa = load_params(args.a, network)
    pb = load_params(args.b, network)
    rng = jax.random.PRNGKey(args.seed)
    k1, k2 = jax.random.split(rng)

    a_w, b_w, d = head_to_head(env, network, pa, pb, k1, args.episodes)
    b_w2, a_w2, d2 = head_to_head(env, network, pb, pa, k2, args.episodes)
    n = 2 * args.episodes
    print(f"A={args.a}\nB={args.b}")
    print(
        f"A wins {a_w + a_w2}/{n} ({(a_w + a_w2) / n:.1%}) | "
        f"B wins {b_w + b_w2}/{n} ({(b_w + b_w2) / n:.1%}) | "
        f"draws {d + d2}/{n} ({(d + d2) / n:.1%})"
    )


if __name__ == "__main__":
    main()
