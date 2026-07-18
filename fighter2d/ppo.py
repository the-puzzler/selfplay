"""Self-play PPO for the two-fighter env, PureJaxRL-style.

One shared actor-critic controls both fighters (parameter sharing); each
fighter's egocentric observation is a separate batch row, so PPO sees
2 * num_envs agent-slots per env step. Everything inside one training
iteration is jitted; the outer python loop handles logging/checkpoints.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from fighter2d.env import ACT_DIM, OBS_DIM, FighterEnv


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
    obs: jnp.ndarray  # (E, 2, O)
    action: jnp.ndarray  # (E, 2, A)
    log_prob: jnp.ndarray  # (E, 2)
    value: jnp.ndarray  # (E, 2)
    reward: jnp.ndarray  # (E, 2)
    done: jnp.ndarray  # (E,)


def gaussian_log_prob(mean, log_std, action):
    var = jnp.exp(2 * log_std)
    return (-0.5 * ((action - mean) ** 2 / var + 2 * log_std + jnp.log(2 * jnp.pi))).sum(-1)


def make_train_iter(env: FighterEnv, cfg):
    """Returns a jitted function running one iteration: rollout + PPO update."""
    network = ActorCritic()
    v_step = jax.vmap(env.step)

    def rollout_step(carry, _):
        params, env_state, obs, rng, pool, archive = carry
        pool_qpos, pool_qvel = pool
        arch_qpos, arch_qvel = archive
        E = cfg["num_envs"]
        rng, k_act, k_step, k_idx, k_wr, k_slot = jax.random.split(rng, 6)
        mean, log_std, value = network.apply(params, obs)  # obs (E,2,O)
        action = mean + jnp.exp(log_std) * jax.random.normal(k_act, mean.shape)
        log_prob = gaussian_log_prob(mean, log_std, action)
        step_keys = jax.random.split(k_step, E)
        # Episodes ending this step reset to a random pre-sampled pool state
        # (spawn sampling is expensive; the pool amortizes it per iteration).
        idx = jax.random.randint(k_idx, (E,), 0, cfg["reset_pool"])
        env_state, next_obs, reward, done, info = v_step(
            step_keys, env_state, action, pool_qpos[idx], pool_qvel[idx]
        )
        # Harvest visited-state snapshots of live envs into the archive.
        wr = jax.random.bernoulli(k_wr, cfg["archive_write_prob"], (E,)) & ~done
        slots = jax.random.randint(k_slot, (E,), 0, cfg["archive_size"])
        arch_qpos = arch_qpos.at[slots].set(
            jnp.where(wr[:, None], env_state.data.qpos, arch_qpos[slots])
        )
        arch_qvel = arch_qvel.at[slots].set(
            jnp.where(wr[:, None], env_state.data.qvel, arch_qvel[slots])
        )
        trans = Transition(obs, action, log_prob, value, reward, done)
        return (params, env_state, next_obs, rng, pool, (arch_qpos, arch_qvel)), (trans, info)

    def compute_gae(traj: Transition, last_value):
        def scan_fn(carry, t):
            gae, next_value = carry
            done = t.done[:, None].astype(jnp.float32)  # (E,1) -> broadcast over agents
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
        params, opt_state, env_state, obs, rng, archive = runner_state

        # Build this iteration's reset pool: fresh procedural spawns, with an
        # optional fraction of archived visited states mixed in.
        rng, k_pool, k_mix, k_aidx = jax.random.split(rng, 4)
        P = cfg["reset_pool"]
        pool_qpos, pool_qvel = jax.vmap(env.reset_qpos_qvel)(jax.random.split(k_pool, P))
        if cfg["archive_frac"] > 0:
            arch_qpos, arch_qvel = archive
            mix = jax.random.bernoulli(k_mix, cfg["archive_frac"], (P,))
            aidx = jax.random.randint(k_aidx, (P,), 0, cfg["archive_size"])
            pool_qpos = jnp.where(mix[:, None], arch_qpos[aidx], pool_qpos)
            pool_qvel = jnp.where(mix[:, None], arch_qvel[aidx], pool_qvel)
        pool = (pool_qpos, pool_qvel)

        (params_, env_state, last_obs, rng, pool, archive), (traj, infos) = jax.lax.scan(
            rollout_step, (params, env_state, obs, rng, pool, archive), None,
            length=cfg["rollout_len"],
        )
        _, _, last_value = network.apply(params, last_obs)
        adv, target = compute_gae(traj, last_value)

        # Flatten (T, E, 2, ...) -> (T*E*2, ...)
        def flat(x):
            return x.reshape((-1,) + x.shape[3:]) if x.ndim > 2 else x.reshape(-1)

        batch = (
            flat(traj.obs),
            flat(traj.action),
            flat(traj.log_prob),
            flat(traj.value),
            flat(adv),
            flat(target),
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
        return (params, opt_state, env_state, last_obs, rng, archive), metrics

    tx = optax.chain(
        optax.clip_by_global_norm(cfg["max_grad_norm"]),
        optax.adam(cfg["lr"], eps=1e-5),
    )

    def init(rng):
        rng, k_net, k_reset, k_arch = jax.random.split(rng, 4)
        params = network.init(k_net, jnp.zeros((1, OBS_DIM)))
        opt_state = tx.init(params)
        reset_keys = jax.random.split(k_reset, cfg["num_envs"])
        env_state, obs = jax.vmap(env.reset)(reset_keys)
        # Archive starts as procedural spawns; play overwrites it with
        # genuinely visited states (tangles, clinches, mid-throws included).
        arch_keys = jax.random.split(k_arch, cfg["archive_size"])
        arch_qpos, arch_qvel = jax.vmap(env.reset_qpos_qvel)(arch_keys)
        return params, opt_state, env_state, obs, rng, (arch_qpos, arch_qvel)

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
    # Resets are drawn from a pool pre-sampled once per iteration.
    "reset_pool": 1024,
    # Visited-state archive resets, mixed into the pool (0.0 = disabled;
    # spawns stay purely procedural).
    "archive_frac": 0.0,
    "archive_size": 4096,
    "archive_write_prob": 0.01,
}


def policy_act(network, params, obs, deterministic=True, rng=None):
    """Numpy-friendly single call for eval/render."""
    mean, log_std, _ = network.apply(params, obs)
    if deterministic:
        return np.clip(np.asarray(mean), -1, 1)
    a = mean + jnp.exp(log_std) * jax.random.normal(rng, mean.shape)
    return np.clip(np.asarray(a), -1, 1)
