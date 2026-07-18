"""GRPO self-play for the Lethal League env (discrete actions, no critic).

Group-Relative Policy Optimization adapted to zero-sum self-play:
each iteration samples n_groups reset states; each state is branched into
group_size complete episodes. A trajectory's advantage is its episode
return minus the mean return of its group (same start state, same player
slot), normalized by the group std — the group IS the baseline, no value
network. Both player slots of every episode train the one shared policy
(their returns are mirrored). PPO-style clipping keeps updates stable.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from lethal.env import ACT_DIM, EPISODE_LEN, OBS_DIM, LethalEnv


class Policy(nn.Module):
    act_dim: int = ACT_DIM

    @nn.compact
    def __call__(self, x):
        for _ in range(2):
            x = nn.tanh(nn.Dense(256)(x))
        return nn.Dense(self.act_dim, kernel_init=nn.initializers.orthogonal(0.01))(x)


class Traj(NamedTuple):
    obs: jnp.ndarray  # (E, 2, O)
    action: jnp.ndarray  # (E, 2)
    log_prob: jnp.ndarray  # (E, 2)
    reward: jnp.ndarray  # (E, 2)
    done: jnp.ndarray  # (E,)


def make_train_iter(env: LethalEnv, cfg):
    network = Policy()
    v_step = jax.vmap(env.step)
    E = cfg["n_groups"] * cfg["group_size"]

    def rollout_step(carry, _):
        params, env_state, obs, rng, group_reset = carry
        rng, k_act, k_step = jax.random.split(rng, 3)
        logits = network.apply(params, obs)
        action = jax.random.categorical(k_act, logits)
        log_prob = jnp.take_along_axis(
            jax.nn.log_softmax(logits), action[..., None], axis=-1).squeeze(-1)
        step_keys = jax.random.split(k_step, E)
        env_state, next_obs, reward, done, info = v_step(
            step_keys, env_state, action, group_reset)
        tr = Traj(obs, action, log_prob, reward, done)
        return (params, env_state, next_obs, rng, group_reset), (tr, info)

    def loss_fn(params, batch):
        obs, action, old_log_prob, adv, w = batch
        logits = network.apply(params, obs)
        logp_all = jax.nn.log_softmax(logits)
        log_prob = jnp.take_along_axis(logp_all, action[..., None], axis=-1).squeeze(-1)
        ratio = jnp.exp(log_prob - old_log_prob)
        pg1 = ratio * adv
        pg2 = jnp.clip(ratio, 1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * adv
        wsum = w.sum() + 1e-8
        pg_loss = -(jnp.minimum(pg1, pg2) * w).sum() / wsum
        entropy = (-(jnp.exp(logp_all) * logp_all).sum(-1) * w).sum() / wsum
        return pg_loss - cfg["ent_coef"] * entropy, {"pg_loss": pg_loss, "entropy": entropy}

    def train_iter(params, opt_state, rng):
        rng, k_reset, k_init = jax.random.split(rng, 3)
        # One reset state per group, tiled group_size times.
        g_states = jax.vmap(env.reset_state)(jax.random.split(k_reset, cfg["n_groups"]))
        group_reset = jax.tree.map(
            lambda x: jnp.repeat(x, cfg["group_size"], axis=0), g_states)
        env_state = group_reset
        obs = jax.vmap(env._obs)(env_state)

        (params_, _, _, rng, _), (traj, infos) = jax.lax.scan(
            rollout_step, (params, env_state, obs, rng, group_reset), None,
            length=EPISODE_LEN,
        )

        # First episode per env only: steps up to and including first done.
        done_before = jnp.cumsum(traj.done, axis=0) - traj.done  # (T, E)
        valid = (done_before == 0)  # (T, E)
        w = valid[..., None].astype(jnp.float32) * jnp.ones((1, 1, 2))  # (T,E,2)
        # Episode return per env/slot = reward at the first done step.
        ret = (traj.reward * valid[..., None]).sum(axis=0)  # (E, 2)
        # Group-relative advantage.
        ret_g = ret.reshape(cfg["n_groups"], cfg["group_size"], 2)
        mean = ret_g.mean(axis=1, keepdims=True)
        std = ret_g.std(axis=1, keepdims=True)
        adv_g = (ret_g - mean) / (std + 1e-4)
        adv = adv_g.reshape(E, 2)  # broadcast over time below

        T = EPISODE_LEN
        batch = (
            traj.obs.reshape(T * E * 2, -1),
            traj.action.reshape(T * E * 2),
            traj.log_prob.reshape(T * E * 2),
            jnp.broadcast_to(adv[None], (T, E, 2)).reshape(T * E * 2),
            w.reshape(T * E * 2),
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
            epoch, (params, opt_state, rng), None, length=cfg["update_epochs"])

        # Metrics over first episodes.
        first_done = traj.done & valid
        n_eps = first_done.sum()
        ep_len = (valid.sum(axis=0)).mean()
        win0 = ((ret[:, 0] > 0)).sum()
        win1 = ((ret[:, 1] > 0)).sum()
        draws = E - win0 - win1
        kills = (infos["kills"] * valid).sum() / jnp.maximum(E, 1)
        metrics = {
            "episodes": n_eps,
            "win0_rate": win0 / E,
            "win1_rate": win1 / E,
            "draw_rate": draws / E,
            "timeout_rate": draws / E,  # draws are timeouts in this game
            "mean_ep_len": ep_len,
            "kills_per_ep": kills,
            "pg_loss": aux["pg_loss"].mean(),
            "v_loss": jnp.zeros(()),  # no critic in GRPO; keeps dashboard happy
            "entropy": aux["entropy"].mean(),
        }
        return params, opt_state, rng, metrics

    tx = optax.chain(
        optax.clip_by_global_norm(cfg["max_grad_norm"]),
        optax.adam(cfg["lr"], eps=1e-5),
    )

    def init(rng):
        rng, k_net = jax.random.split(rng)
        params = network.init(k_net, jnp.zeros((1, OBS_DIM)))
        opt_state = tx.init(params)
        return params, opt_state, rng

    return init, jax.jit(train_iter), network


DEFAULT_CFG = {
    "n_groups": 128,
    "group_size": 8,
    "num_minibatches": 8,
    "update_epochs": 2,
    "lr": 3e-4,
    "clip_eps": 0.2,
    "ent_coef": 0.005,
    "max_grad_norm": 0.5,
}


def policy_act(network, params, obs, deterministic=True, rng=None):
    logits = network.apply(params, obs)
    if deterministic:
        return np.asarray(jnp.argmax(logits, axis=-1))
    return np.asarray(jax.random.categorical(rng, logits))
