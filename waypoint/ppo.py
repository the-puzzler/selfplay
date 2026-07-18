"""Recurrent PPO (GRU) for the waypoint parkour env — single agent.

The policy is recurrent (user-specified: memory over terrain/gait state).
Rollouts carry the GRU hidden state, gated to zero at episode boundaries.
Updates replay stored sequences through the GRU with the same gating;
minibatches are over ENVS (whole sequences), never over shuffled steps.
The rollout-start hidden state is reused across epochs (standard stale-
hidden approximation).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from waypoint.env import ACT_DIM, OBS_DIM, WaypointEnv

HIDDEN = 128


class RecurrentAC(nn.Module):
    act_dim: int = ACT_DIM

    @nn.compact
    def __call__(self, h, x, reset):
        """One step. h (B,H), x (B,O), reset (B,) -> h', mean, log_std, value."""
        h = h * (1.0 - reset[:, None])
        e = nn.tanh(nn.Dense(HIDDEN)(x))
        h, out = nn.GRUCell(features=HIDDEN)(h, e)
        a = nn.tanh(nn.Dense(128)(out))
        mean = nn.Dense(self.act_dim, kernel_init=nn.initializers.orthogonal(0.01))(a)
        log_std = jnp.clip(
            self.param("log_std", nn.initializers.constant(-0.5), (self.act_dim,)),
            -4.0, 1.0)
        v = nn.tanh(nn.Dense(128)(out))
        value = nn.Dense(1)(v).squeeze(-1)
        return h, mean, log_std, value


class Transition(NamedTuple):
    obs: jnp.ndarray  # (E, O)
    action: jnp.ndarray  # (E, A)
    log_prob: jnp.ndarray  # (E,)
    value: jnp.ndarray  # (E,)
    reward: jnp.ndarray  # (E,)
    done: jnp.ndarray  # (E,)  done AFTER this step
    done_prev: jnp.ndarray  # (E,)  episode boundary BEFORE this step


def gaussian_log_prob(mean, log_std, action):
    var = jnp.exp(2 * log_std)
    return (-0.5 * ((action - mean) ** 2 / var + 2 * log_std + jnp.log(2 * jnp.pi))).sum(-1)


def make_train_iter(env: WaypointEnv, cfg):
    network = RecurrentAC()
    v_step = jax.vmap(env.step)
    E = cfg["num_envs"]

    def rollout_step(carry, _):
        params, env_state, obs, done_prev, h, rng, pool = carry
        rng, k_act, k_step, k_idx = jax.random.split(rng, 4)
        h, mean, log_std, value = network.apply(params, h, obs, done_prev.astype(jnp.float32))
        action = mean + jnp.exp(log_std) * jax.random.normal(k_act, mean.shape)
        log_prob = gaussian_log_prob(mean, log_std, action)
        step_keys = jax.random.split(k_step, E)
        idx = jax.random.randint(k_idx, (E,), 0, cfg["reset_pool"])
        reset_to = jax.tree.map(lambda x: x[idx], pool)
        env_state, next_obs, reward, done, info = v_step(step_keys, env_state, action, reset_to)
        tr = Transition(obs, action, log_prob, value, reward, done, done_prev)
        return (params, env_state, next_obs, done, h, rng, pool), (tr, info)

    def compute_gae(traj, last_value):
        def scan_fn(carry, t):
            gae, next_value = carry
            nd = 1.0 - t.done.astype(jnp.float32)
            delta = t.reward + cfg["gamma"] * next_value * nd - t.value
            gae = delta + cfg["gamma"] * cfg["gae_lambda"] * nd * gae
            return (gae, t.value), gae

        (_, _), adv = jax.lax.scan(scan_fn, (jnp.zeros_like(last_value), last_value),
                                   traj, reverse=True)
        return adv, adv + traj.value

    def loss_fn(params, mb):
        obs, action, old_lp, old_v, adv, target, done_prev, h0 = mb

        def scan_net(h, inp):
            o, dp = inp
            h, mean, log_std, value = network.apply(params, h, o, dp)
            return h, (mean, jnp.broadcast_to(log_std, mean.shape), value)

        _, (mean, log_std, value) = jax.lax.scan(
            scan_net, h0, (obs, done_prev.astype(jnp.float32)))
        log_prob = gaussian_log_prob(mean, log_std, action)
        ratio = jnp.exp(jnp.clip(log_prob - old_lp, -20.0, 20.0))
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg1 = ratio * adv_n
        pg2 = jnp.clip(ratio, 1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * adv_n
        pg_loss = -jnp.minimum(pg1, pg2).mean()
        v_clip = old_v + jnp.clip(value - old_v, -cfg["clip_eps"], cfg["clip_eps"])
        v_loss = 0.5 * jnp.maximum((value - target) ** 2, (v_clip - target) ** 2).mean()
        entropy = (log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e)).sum(-1).mean()
        return pg_loss + cfg["vf_coef"] * v_loss - cfg["ent_coef"] * entropy, \
            {"pg_loss": pg_loss, "v_loss": v_loss, "entropy": entropy}

    def train_iter(runner_state):
        params, opt_state, env_state, obs, done_prev, h, rng = runner_state
        rng, k_pool = jax.random.split(rng)
        pool = jax.vmap(env.reset_state)(jax.random.split(k_pool, cfg["reset_pool"]))

        h0_rollout = h
        (params_, env_state, last_obs, last_done, h, rng, pool), (traj, infos) = jax.lax.scan(
            rollout_step, (params, env_state, obs, done_prev, h, rng, pool), None,
            length=cfg["rollout_len"])
        _, _, _, last_value = network.apply(
            params, h, last_obs, last_done.astype(jnp.float32))
        adv, target = compute_gae(traj, last_value)

        def epoch(carry, _):
            params, opt_state, rng = carry
            rng, k = jax.random.split(rng)
            perm = jax.random.permutation(k, E).reshape(cfg["num_minibatches"], -1)

            def minibatch(carry, eidx):
                params, opt_state = carry
                mb = (
                    traj.obs[:, eidx], traj.action[:, eidx], traj.log_prob[:, eidx],
                    traj.value[:, eidx], adv[:, eidx], target[:, eidx],
                    traj.done_prev[:, eidx], h0_rollout[eidx],
                )
                (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, mb)
                updates, opt_state = tx.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), aux

            (params, opt_state), aux = jax.lax.scan(minibatch, (params, opt_state), perm)
            return (params, opt_state, rng), aux

        (params, opt_state, rng), aux = jax.lax.scan(
            epoch, (params, opt_state, rng), None, length=cfg["update_epochs"])

        n_done = (infos["reach"] | infos["timeout"]).sum()
        metrics = {
            "episodes": n_done,
            "win0_rate": infos["reach"].sum() / jnp.maximum(n_done, 1),  # reach rate
            "win1_rate": jnp.zeros(()),
            "draw_rate": infos["timeout"].sum() / jnp.maximum(n_done, 1),
            "timeout_rate": infos["timeout"].sum() / jnp.maximum(n_done, 1),
            "mean_ep_len": infos["ep_len"].sum() / jnp.maximum(n_done, 1),
            "final_dist": infos["final_dist"].sum() / jnp.maximum(n_done, 1),
            "pg_loss": aux["pg_loss"].mean(),
            "v_loss": aux["v_loss"].mean(),
            "entropy": aux["entropy"].mean(),
        }
        return (params, opt_state, env_state, last_obs, last_done, h, rng), metrics

    tx = optax.chain(
        optax.clip_by_global_norm(cfg["max_grad_norm"]),
        optax.adam(cfg["lr"], eps=1e-5),
    )

    def init(rng):
        rng, k_net, k_reset = jax.random.split(rng, 3)
        params = network.init(k_net, jnp.zeros((1, HIDDEN)), jnp.zeros((1, OBS_DIM)),
                              jnp.zeros((1,)))
        opt_state = tx.init(params)
        reset_keys = jax.random.split(k_reset, cfg["num_envs"])
        env_state, obs = jax.vmap(env.reset)(reset_keys)
        h = jnp.zeros((cfg["num_envs"], HIDDEN))
        done_prev = jnp.ones((cfg["num_envs"],), bool)  # fresh episodes
        return params, opt_state, env_state, obs, done_prev, h, rng

    return init, jax.jit(train_iter), network


DEFAULT_CFG = {
    "num_envs": 512,
    "rollout_len": 128,
    "num_minibatches": 4,
    "update_epochs": 2,
    "lr": 3e-4,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "vf_coef": 0.5,
    "ent_coef": 0.003,
    "max_grad_norm": 0.5,
    "reset_pool": 512,
}


def policy_act(network, params, h, obs, done_prev, deterministic=True, rng=None):
    """Single-step recurrent act for eval/render. obs (B,O)."""
    h, mean, log_std, _ = network.apply(params, h, obs, done_prev)
    if deterministic:
        return h, np.clip(np.asarray(mean), -1, 1)
    a = mean + jnp.exp(log_std) * jax.random.normal(rng, mean.shape)
    return h, np.clip(np.asarray(a), -1, 1)
