"""Self-play PPO (continuous Gaussian actions) for the stickfight env.

Same shape as dogfight/ppo.py: shared actor-critic drives both fighters,
resets drawn from a per-iteration pre-sampled pool (EnvState pytree).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from stickfight.env import ACT_DIM, OBS_DIM, StickFightEnv


class ActorCritic(nn.Module):
    act_dim: int = ACT_DIM

    @nn.compact
    def __call__(self, x):
        a = x
        for _ in range(2):
            a = nn.tanh(nn.Dense(256)(a))
        mean = nn.Dense(self.act_dim, kernel_init=nn.initializers.orthogonal(0.01))(a)
        log_std = self.param("log_std", nn.initializers.constant(-0.5), (self.act_dim,))
        v = x
        for _ in range(2):
            v = nn.tanh(nn.Dense(256)(v))
        value = nn.Dense(1)(v)
        return mean, log_std, value.squeeze(-1)


class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray


def gaussian_log_prob(mean, log_std, action):
    var = jnp.exp(2 * log_std)
    return (-0.5 * ((action - mean) ** 2 / var + 2 * log_std + jnp.log(2 * jnp.pi))).sum(-1)


def make_train_iter(env: StickFightEnv, cfg):
    network = ActorCritic()
    v_step = jax.vmap(env.step)

    def rollout_step(carry, _):
        params, env_state, obs, rng, pool = carry
        E = cfg["num_envs"]
        rng, k_act, k_step, k_idx = jax.random.split(rng, 4)
        mean, log_std, value = network.apply(params, obs)
        action = mean + jnp.exp(log_std) * jax.random.normal(k_act, mean.shape)
        log_prob = gaussian_log_prob(mean, log_std, action)
        step_keys = jax.random.split(k_step, E)
        idx = jax.random.randint(k_idx, (E,), 0, cfg["reset_pool"])
        reset_to = jax.tree.map(lambda x: x[idx], pool)
        env_state, next_obs, reward, done, info = v_step(
            step_keys, env_state, action, reset_to
        )
        trans = Transition(obs, action, log_prob, value, reward, done)
        return (params, env_state, next_obs, rng, pool), (trans, info)

    def compute_gae(traj, last_value):
        def scan_fn(carry, t):
            gae, next_value = carry
            done = t.done[:, None].astype(jnp.float32)
            delta = t.reward + cfg["gamma"] * next_value * (1 - done) - t.value
            gae = delta + cfg["gamma"] * cfg["gae_lambda"] * (1 - done) * gae
            return (gae, t.value), gae

        (_, _), advantages = jax.lax.scan(
            scan_fn, (jnp.zeros_like(last_value), last_value), traj, reverse=True
        )
        return advantages, advantages + traj.value

    def loss_fn(params, batch):
        obs, action, old_log_prob, old_value, adv, target = batch
        mean, log_std, value = network.apply(params, obs)
        log_prob = gaussian_log_prob(mean, log_std, action)
        ratio = jnp.exp(log_prob - old_log_prob)
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg1 = ratio * adv_n
        pg2 = jnp.clip(ratio, 1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * adv_n
        pg_loss = -jnp.minimum(pg1, pg2).mean()
        v_clipped = old_value + jnp.clip(value - old_value, -cfg["clip_eps"], cfg["clip_eps"])
        v_loss = 0.5 * jnp.maximum((value - target) ** 2, (v_clipped - target) ** 2).mean()
        entropy = (log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e)).sum()
        loss = pg_loss + cfg["vf_coef"] * v_loss - cfg["ent_coef"] * entropy
        return loss, {"pg_loss": pg_loss, "v_loss": v_loss, "entropy": entropy}

    def train_iter(runner_state):
        params, opt_state, env_state, obs, rng = runner_state
        rng, k_pool = jax.random.split(rng)
        pool = jax.vmap(env.reset_state)(jax.random.split(k_pool, cfg["reset_pool"]))

        (params_, env_state, last_obs, rng, pool), (traj, infos) = jax.lax.scan(
            rollout_step, (params, env_state, obs, rng, pool), None,
            length=cfg["rollout_len"],
        )
        _, _, last_value = network.apply(params, last_obs)
        adv, target = compute_gae(traj, last_value)

        def flat(x):
            return x.reshape((-1,) + x.shape[3:]) if x.ndim > 2 else x.reshape(-1)

        batch = (
            flat(traj.obs), flat(traj.action), flat(traj.log_prob),
            flat(traj.value), flat(adv), flat(target),
        )
        n = batch[0].shape[0]

        def epoch(carry, _):
            params, opt_state, rng = carry
            rng, k = jax.random.split(rng)
            perm = jax.random.permutation(k, n)

            def minibatch(carry, idx):
                params, opt_state = carry
                mb = jax.tree.map(lambda x: x[idx], batch)
                (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, mb)
                updates, opt_state = tx.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), aux

            idxs = perm.reshape(cfg["num_minibatches"], -1)
            (params, opt_state), aux = jax.lax.scan(minibatch, (params, opt_state), idxs)
            return (params, opt_state, rng), aux

        (params, opt_state, rng), aux = jax.lax.scan(
            epoch, (params, opt_state, rng), None, length=cfg["update_epochs"]
        )

        n_done = infos["win0"].sum() + infos["win1"].sum() + infos["draw"].sum()
        metrics = {
            "episodes": n_done,
            "win0_rate": infos["win0"].sum() / jnp.maximum(n_done, 1),
            "win1_rate": infos["win1"].sum() / jnp.maximum(n_done, 1),
            "draw_rate": infos["draw"].sum() / jnp.maximum(n_done, 1),
            "timeout_rate": infos["timeout"].sum() / jnp.maximum(n_done, 1),
            "mean_ep_len": infos["ep_len"].sum() / jnp.maximum(n_done, 1),
            "pg_loss": aux["pg_loss"].mean(),
            "v_loss": aux["v_loss"].mean(),
            "entropy": aux["entropy"].mean(),
        }
        return (params, opt_state, env_state, last_obs, rng), metrics

    tx = optax.chain(
        optax.clip_by_global_norm(cfg["max_grad_norm"]),
        optax.adam(cfg["lr"], eps=1e-5),
    )

    def init(rng):
        rng, k_net, k_reset = jax.random.split(rng, 3)
        params = network.init(k_net, jnp.zeros((1, OBS_DIM)))
        opt_state = tx.init(params)
        reset_keys = jax.random.split(k_reset, cfg["num_envs"])
        env_state, obs = jax.vmap(env.reset)(reset_keys)
        return params, opt_state, env_state, obs, rng

    return init, jax.jit(train_iter), network


DEFAULT_CFG = {
    "num_envs": 1024,
    "rollout_len": 128,
    "num_minibatches": 8,
    "update_epochs": 4,
    "lr": 3e-4,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "vf_coef": 0.5,
    "ent_coef": 0.003,
    "max_grad_norm": 0.5,
    "reset_pool": 1024,
}


def policy_act(network, params, obs, deterministic=True, rng=None):
    mean, log_std, _ = network.apply(params, obs)
    if deterministic:
        return np.clip(np.asarray(mean), -1, 1)
    a = mean + jnp.exp(log_std) * jax.random.normal(rng, mean.shape)
    return np.clip(np.asarray(a), -1, 1)
