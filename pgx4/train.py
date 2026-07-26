"""Self-play PPO on a Pgx board game with OmniReset-style diverse resets.

The thesis carried over from fighter2d: you don't need curricula. Reset the
agents into massively diverse states and let PPO solve the easy states first
(here: positions one move from a win) and propagate value outward to the hard
opening play. In a turn-based board game the diverse reset is a random legal
mid-game position instead of the empty board.

One shared policy plays both seats (self-play). Reward is purely sparse and
zero-sum: Pgx hands out +1/-1/0 at terminal, nothing else. Credit is assigned
negamax-style: value is always from the perspective of the player to move, so
the bootstrap flips sign every ply.

  uv run python -m pgx4.train --out runs/c4
"""

import argparse
import json
import time
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pgx
from flax import linen as nn
from flax import serialization

NEG_INF = -1e9


class ActorCritic(nn.Module):
    n_actions: int
    width: int = 256
    depth: int = 2
    arch: str = "mlp"  # mlp | conv | transformer | aznet

    @nn.compact
    def __call__(self, x, train: bool = False):
        # x: (..., H, W, 2) float board, current-player-relative planes.
        if self.arch == "aznet":
            return self._aznet(x, train)
        if self.arch == "conv":
            return self._conv(x)
        if self.arch == "transformer":
            return self._transformer(x)
        # MLP over the flattened board (cuBLAS only, no cuDNN).
        h = x.reshape(x.shape[:-3] + (-1,))
        for _ in range(self.depth):
            h = nn.relu(nn.Dense(self.width)(h))
        logits = nn.Dense(self.n_actions,
                          kernel_init=nn.initializers.orthogonal(0.01))(h)
        v = h
        for _ in range(self.depth):
            v = nn.relu(nn.Dense(self.width)(v))
        value = nn.Dense(1)(v).squeeze(-1)
        return logits, value

    def _aznet(self, x, train):
        """Exact Flax port of Pgx's AZNet v0 baseline (pgx/_src/baseline.py):
        ResNet-v2, width channels x depth blocks, BatchNorm(momentum .9)
        everywhere, tanh value head. With width=128, depth=6 this is
        parameter-for-parameter the published othello_v0/hex_v0 baseline net."""
        conv = lambda c, k=3: nn.Conv(c, (k, k), padding="SAME")
        bn = lambda z: nn.BatchNorm(use_running_average=not train, momentum=0.9)(z)
        x = conv(self.width)(x)
        for _ in range(self.depth):
            i = x
            x = jax.nn.relu(bn(x))
            x = conv(self.width)(x)
            x = jax.nn.relu(bn(x))
            x = conv(self.width)(x)
            x = x + i
        x = jax.nn.relu(bn(x))
        # policy head (BatchNorm over 2 channels is per-channel over the batch
        # axis — fine, unlike LayerNorm which would be degenerate here)
        p = jax.nn.relu(bn(conv(2, 1)(x)))
        p = p.reshape(p.shape[:-3] + (-1,))
        logits = nn.Dense(self.n_actions)(p)
        # value head
        v = jax.nn.relu(bn(conv(1, 1)(x)))
        v = v.reshape(v.shape[:-3] + (-1,))
        v = jax.nn.relu(nn.Dense(self.width)(v))
        value = jnp.tanh(nn.Dense(1)(v)).squeeze(-1)
        return logits, value

    def _transformer(self, x):
        # Board as a sequence of H*W cell tokens + a CLS token, self-attention
        # throughout. Policy reads per-cell logits (cell i scores action i, the
        # natural mapping for placement games) with the pass logit from CLS; a
        # giant flattened readout is deliberately avoided — one Dense over all
        # tokens can sharpen the policy near-instantly and collapse entropy.
        # Value reads from CLS through its own private 2-layer MLP.
        lead, (H, W, C) = x.shape[:-3], x.shape[-3:]
        n, d = H * W, self.width
        tok = x.reshape(lead + (n, C))
        h = nn.Dense(d)(tok)
        h = h + self.param("pos", nn.initializers.normal(0.02), (n, d))
        cls = jnp.broadcast_to(self.param("cls", nn.initializers.normal(0.02), (1, d)),
                               lead + (1, d))
        h = jnp.concatenate([cls, h], axis=-2)  # (..., n+1, d)
        for _ in range(self.depth):
            a = nn.MultiHeadDotProductAttention(num_heads=4)(nn.LayerNorm()(h))
            h = h + a
            m = nn.LayerNorm()(h)
            m = nn.Dense(2 * d)(m)
            m = nn.gelu(m)
            m = nn.Dense(d)(m)
            h = h + m
        h = nn.LayerNorm()(h)
        cls_out, cells = h[..., 0, :], h[..., 1:, :]
        if self.n_actions == n + 1:  # placement game: cells + pass
            cell_logits = nn.Dense(
                1, kernel_init=nn.initializers.orthogonal(0.01))(cells)[..., 0]
            pass_logit = nn.Dense(
                1, kernel_init=nn.initializers.orthogonal(0.01))(cls_out)
            logits = jnp.concatenate([cell_logits, pass_logit], axis=-1)
        else:  # generic action space: read from CLS only
            logits = nn.Dense(self.n_actions,
                              kernel_init=nn.initializers.orthogonal(0.01))(cls_out)
        v = cls_out
        for _ in range(2):
            v = nn.relu(nn.Dense(d)(v))
        value = nn.Dense(1)(v).squeeze(-1)
        return logits, value

    def _conv(self, x):
        # width = channels, depth = residual blocks; the standard board-game net.
        # LayerNorm in each block keeps the deep residual tower trainable under
        # Adam (without it, it barely learns).
        conv = lambda c, k=3: nn.Conv(c, (k, k), padding="SAME")
        norm = lambda z: nn.LayerNorm()(z)
        h = nn.relu(norm(conv(self.width)(x)))
        for _ in range(self.depth):
            y = nn.relu(norm(conv(self.width)(h)))
            y = norm(conv(self.width)(y))
            h = nn.relu(h + y)
        # Heads carry no norm on narrow layers: LayerNorm over 1-2 channels is
        # degenerate (1-channel LN outputs a constant; 2-channel keeps a sign).
        # Policy: private 1x1 projection per cell.
        ph = nn.relu(conv(4, 1)(h))
        ph = ph.reshape(ph.shape[:-3] + (-1,))
        logits = nn.Dense(self.n_actions,
                          kernel_init=nn.initializers.orthogonal(0.01))(ph)
        # Value: private full-width tower (2 conv blocks), so the critic has
        # capacity of its own instead of leaning entirely on the shared trunk.
        vh = h
        for _ in range(2):
            vh = nn.relu(norm(conv(self.width)(vh)))
        vh = nn.relu(conv(4, 1)(vh))
        vh = vh.reshape(vh.shape[:-3] + (-1,))
        vh = nn.relu(nn.Dense(self.width)(vh))
        value = nn.Dense(1)(vh).squeeze(-1)
        return logits, value


class Transition(NamedTuple):
    obs: jnp.ndarray       # (E, 6, 7, 2)
    mask: jnp.ndarray      # (E, A) legal-action mask at decision time
    action: jnp.ndarray    # (E,)
    log_prob: jnp.ndarray  # (E,)
    value: jnp.ndarray     # (E,) value for the player to move
    reward: jnp.ndarray    # (E,) terminal reward for the player who just moved
    done: jnp.ndarray      # (E,)
    weight: jnp.ndarray    # (E,) 1 for the learner's own moves, 0 for a frozen opponent's


def masked_logits(logits, mask):
    return jnp.where(mask, logits, NEG_INF)


def categorical_log_prob(logits, action):
    return jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]


def categorical_entropy(logits):
    logp = jax.nn.log_softmax(logits)
    return -(jnp.exp(logp) * logp).sum(-1)


def _tree_select(cond, a, b):
    """Pick leaves from a where cond (per-env) is True, else from b."""
    def pick(x, y):
        c = cond.reshape(cond.shape + (1,) * (x.ndim - cond.ndim))
        return jnp.where(c, x, y)
    return jax.tree.map(pick, a, b)


def make_pgx_core(env: pgx.Env, cfg):
    n_actions = env.num_actions
    network = ActorCritic(n_actions=n_actions, width=cfg["width"],
                          depth=cfg["depth"], arch=cfg["arch"])
    v_step = jax.vmap(env.step)

    def obs_of(state):
        return state.observation.astype(jnp.float32)

    def reward_of_mover(state, next_state):
        # next_state.rewards is indexed by absolute player id; credit the player
        # who moved (state.current_player).
        E = state.current_player.shape[0]
        return next_state.rewards[jnp.arange(E), state.current_player]

    # ---- diverse-reset sampler: a random legal mid-game position --------------
    def random_midgame(key):
        """Play depth ~ U[0, max_depth] uniformly-random legal moves from the
        empty board, stopping before any move that would end the game. Returns a
        non-terminal state (depth 0 == the true opening)."""
        k0, kd, kp = jax.random.split(key, 3)
        state = env.init(k0)
        depth = jax.random.randint(kd, (), 0, cfg["reset_max_depth"] + 1)

        def body(carry, i):
            state, frozen, kk = carry
            kk, ka = jax.random.split(kk)
            logits = jnp.where(state.legal_action_mask, 0.0, NEG_INF)
            a = jax.random.categorical(ka, logits)
            nstate = env.step(state, a)
            # Stop advancing once we've hit target depth, already froze, or the
            # next move would terminate (we never sit on a terminal state).
            stop = frozen | (i >= depth) | nstate.terminated | nstate.truncated
            state = jax.tree.map(lambda n, s: jnp.where(~stop, n, s), nstate, state)
            return (state, frozen | stop, kk), None

        (state, _, _), _ = jax.lax.scan(
            body, (state, jnp.bool_(False), kp), jnp.arange(cfg["reset_max_depth"]))
        return state

    def sample_pool(key):
        return jax.vmap(random_midgame)(jax.random.split(key, cfg["reset_pool"]))

    def draw_from_pool(pool, key, n):
        idx = jax.random.randint(key, (n,), 0, cfg["reset_pool"])
        return jax.tree.map(lambda x: x[idx], pool)

    # ---- rollout -------------------------------------------------------------
    def rollout_step(carry, _):
        variables, state, rng, pool, frozen, learner_seat, is_pool_env = carry
        E = cfg["num_envs"]
        rng, k_act, k_opp, k_reset = jax.random.split(rng, 4)
        obs = obs_of(state)
        mask = state.legal_action_mask
        logits, value = network.apply(variables, obs)       # current policy
        logits = masked_logits(logits, mask)
        a_cur = jax.random.categorical(k_act, logits)
        # frozen opponent from the pool (only used on opponent seats of pool envs)
        opp_logits, _ = network.apply(frozen, obs)
        opp_logits = masked_logits(opp_logits, mask)
        a_opp = jax.random.categorical(k_opp, opp_logits)

        mover_is_learner = state.current_player == learner_seat
        use_opp = is_pool_env & (~mover_is_learner)          # frozen opponent moves here
        action = jnp.where(use_opp, a_opp, a_cur)
        # train only on the learner's own moves; frozen-opponent plies get weight 0
        weight = jnp.where(use_opp, 0.0, 1.0)
        log_prob = categorical_log_prob(logits, action)      # under current policy

        next_state = v_step(state, action)
        reward = reward_of_mover(state, next_state)
        done = next_state.terminated | next_state.truncated

        # Auto-reset finished games to a fresh reset-pool position.
        fresh = draw_from_pool(pool, k_reset, E)
        next_state = _tree_select(done, fresh, next_state)

        trans = Transition(obs, mask, action, log_prob, value, reward, done, weight)
        info = {"done": done, "reward": reward}
        return (variables, next_state, rng, pool, frozen, learner_seat, is_pool_env), (trans, info)

    def compute_gae(traj, last_value):
        """Negamax GAE: value/return are from the mover's perspective, so the
        next state (opponent to move) contributes with a flipped sign."""
        def scan_fn(carry, t):
            gae, next_value = carry
            notdone = 1.0 - t.done.astype(jnp.float32)
            delta = t.reward + cfg["gamma"] * (-next_value) * notdone - t.value
            gae = delta + cfg["gamma"] * cfg["gae_lambda"] * (-1.0) * notdone * gae
            return (gae, t.value), gae

        (_, _), adv = jax.lax.scan(
            scan_fn, (jnp.zeros_like(last_value), last_value), traj, reverse=True)
        return adv, adv + traj.value

    has_bn = cfg["arch"] == "aznet"

    def loss_fn(params, bstats, batch):
        obs, mask, action, old_log_prob, old_value, adv, target, weight = batch
        if has_bn:
            (logits, value), new_state = network.apply(
                {"params": params, "batch_stats": bstats}, obs,
                train=True, mutable=["batch_stats"])
            new_bstats = new_state["batch_stats"]
        else:
            logits, value = network.apply({"params": params}, obs)
            new_bstats = bstats
        logits = masked_logits(logits, mask)
        log_prob = categorical_log_prob(logits, action)
        ratio = jnp.exp(log_prob - old_log_prob)
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg1 = ratio * adv_n
        pg2 = jnp.clip(ratio, 1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * adv_n
        # weight masks out frozen-opponent plies so we never train toward them
        w = weight
        wsum = jnp.maximum(w.sum(), 1.0)
        pg_loss = -(jnp.minimum(pg1, pg2) * w).sum() / wsum
        v_clipped = old_value + jnp.clip(value - old_value,
                                         -cfg["clip_eps"], cfg["clip_eps"])
        v_loss = 0.5 * (jnp.maximum((value - target) ** 2,
                                    (v_clipped - target) ** 2) * w).sum() / wsum
        entropy = (categorical_entropy(logits) * w).sum() / wsum
        loss = pg_loss + cfg["vf_coef"] * v_loss - cfg["ent_coef"] * entropy
        return loss, ({"pg_loss": pg_loss, "v_loss": v_loss, "entropy": entropy},
                      new_bstats)

    tx = optax.chain(
        optax.clip_by_global_norm(cfg["max_grad_norm"]),
        optax.adam(cfg["lr"], eps=1e-5),
    )

    def train_iter(runner_state):
        variables, opt_state, state, rng, opp_pool, push_count, it_count = runner_state
        params = variables["params"]
        bstats = variables.get("batch_stats", {})
        E = cfg["num_envs"]
        rng, k_pool, k_opp, k_seat, k_env = jax.random.split(rng, 5)
        pool = sample_pool(k_pool)  # fresh diverse resets each iter

        # sample one frozen opponent from the snapshot pool; a fraction of envs
        # play their opponent seat with it instead of the live policy.
        K = cfg["pool_size"]
        n_avail = jnp.maximum(jnp.minimum(push_count, K), 1)
        opp_idx = jax.random.randint(k_opp, (), 0, n_avail)
        frozen = jax.tree.map(lambda x: x[opp_idx], opp_pool)
        learner_seat = jax.random.randint(k_seat, (E,), 0, 2)
        is_pool_env = jax.random.bernoulli(k_env, cfg["opp_pool_frac"], (E,))

        init_carry = (variables, state, rng, pool, frozen, learner_seat, is_pool_env)
        (variables, state, rng, pool, _, _, _), (traj, info) = jax.lax.scan(
            rollout_step, init_carry, None, length=cfg["rollout_len"])
        _, last_value = network.apply(variables, obs_of(state))
        adv, target = compute_gae(traj, last_value)

        def flat(x):
            return x.reshape((-1,) + x.shape[2:]) if x.ndim > 1 else x.reshape(-1)

        batch = tuple(flat(x) for x in
                      (traj.obs, traj.mask, traj.action, traj.log_prob,
                       traj.value, adv, target, traj.weight))
        n = batch[0].shape[0]

        def epoch(carry, _):
            params, bstats, opt_state, rng = carry
            rng, k = jax.random.split(rng)
            perm = jax.random.permutation(k, n)

            def minibatch(carry, idx):
                params, bstats, opt_state = carry
                mb = tuple(x[idx] for x in batch)
                (loss, (aux, bstats)), grads = jax.value_and_grad(
                    loss_fn, has_aux=True)(params, bstats, mb)
                updates, opt_state = tx.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return (params, bstats, opt_state), aux

            idxs = perm.reshape(cfg["num_minibatches"], -1)
            (params, bstats, opt_state), aux = jax.lax.scan(
                minibatch, (params, bstats, opt_state), idxs)
            return (params, bstats, opt_state, rng), aux

        (params, bstats, opt_state, rng), aux = jax.lax.scan(
            epoch, (params, bstats, opt_state, rng), None, length=cfg["update_epochs"])
        variables = {"params": params, "batch_stats": bstats} if has_bn else {"params": params}

        # push the updated policy into the snapshot pool every pool_every iters
        it_count = it_count + 1
        should_push = (it_count % cfg["pool_every"]) == 0
        ptr = push_count % cfg["pool_size"]
        opp_pool = jax.tree.map(
            lambda P, pn: P.at[ptr].set(jnp.where(should_push, pn, P[ptr])),
            opp_pool, variables)
        push_count = push_count + should_push.astype(jnp.int32)

        done = info["done"]
        n_done = jnp.maximum(done.sum(), 1)
        decisive = (done & (info["reward"] != 0)).sum()
        metrics = {
            "episodes": done.sum(),
            "decisive_rate": decisive / n_done,
            "draw_rate": (done & (info["reward"] == 0)).sum() / n_done,
            "pg_loss": aux["pg_loss"].mean(),
            "v_loss": aux["v_loss"].mean(),
            "entropy": aux["entropy"].mean(),
        }
        return (variables, opt_state, state, rng, opp_pool, push_count, it_count), metrics

    def init(rng):
        rng, k_net, k_pool, k_state = jax.random.split(rng, 4)
        dummy = jnp.zeros((1,) + env.observation_shape, jnp.float32)
        variables = network.init(k_net, dummy)
        opt_state = tx.init(variables["params"])
        pool = sample_pool(k_pool)
        state = draw_from_pool(pool, k_state, cfg["num_envs"])
        # snapshot pool: K slots, all seeded with the initial variables
        opp_pool = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (cfg["pool_size"],) + x.shape).copy(),
            variables)
        return (variables, opt_state, state, rng, opp_pool,
                jnp.int32(0), jnp.int32(0))

    return init, jax.jit(train_iter), network


# ---- vs-random correctness eval (runs in the python loop, not jitted hot) ----
def evaluate_vs_random(env, network, params, key, n_games):
    """Greedy policy vs a uniform-random opponent, playing both seats equally.
    Returns (win_rate, draw_rate, loss_rate) from the policy's perspective."""
    v_step = jax.vmap(env.step)

    @jax.jit
    def run(key):
        k_init, k_play = jax.random.split(key)
        state = jax.vmap(env.init)(jax.random.split(k_init, n_games))
        # Half the games the policy is player 0, half player 1.
        policy_seat = (jnp.arange(n_games) % 2).astype(jnp.int32)

        def cond(carry):
            state, _, _ = carry
            return ~(state.terminated | state.truncated).all()

        def body(carry):
            state, kk, results = carry
            kk, kr = jax.random.split(kk)
            obs = state.observation.astype(jnp.float32)
            logits, _ = network.apply(params, obs)
            logits = jnp.where(state.legal_action_mask, logits, NEG_INF)
            greedy = jnp.argmax(logits, axis=-1)
            rand_logits = jnp.where(state.legal_action_mask, 0.0, NEG_INF)
            rand = jax.random.categorical(kr, rand_logits)
            use_policy = state.current_player == policy_seat
            action = jnp.where(use_policy, greedy, rand)
            done_before = state.terminated | state.truncated
            nstate = v_step(state, action)
            newly = (nstate.terminated | nstate.truncated) & ~done_before
            r = nstate.rewards[jnp.arange(n_games), policy_seat]
            results = jnp.where(newly, r, results)
            return nstate, kk, results

        results = jnp.zeros(n_games)
        state, _, results = jax.lax.while_loop(
            cond, body, (state, k_play, results))
        return results

    r = run(key)
    return (float((r > 0).mean()), float((r == 0).mean()), float((r < 0).mean()))


DEFAULT_CFG = {
    "game": "connect_four",
    "num_envs": 4096,
    "rollout_len": 64,
    "num_minibatches": 8,
    "update_epochs": 3,
    "lr": 3e-4,
    "gamma": 0.997,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "vf_coef": 0.5,
    "ent_coef": 0.01,
    "max_grad_norm": 0.5,
    "width": 256,
    "depth": 2,
    "arch": "mlp",
    # Diverse resets: a pool of random mid-game positions, resampled each iter.
    "reset_pool": 4096,
    "reset_max_depth": 20,
    # Opponent snapshot pool: fraction of envs whose opponent seat is played by a
    # sampled past policy (0.0 = pure self-play). Snapshots pushed every
    # pool_every iters into a ring of pool_size.
    "opp_pool_frac": 0.0,
    "pool_size": 8,
    "pool_every": 50,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", type=str, default=DEFAULT_CFG["game"])
    p.add_argument("--num-envs", type=int, default=DEFAULT_CFG["num_envs"])
    p.add_argument("--rollout-len", type=int, default=DEFAULT_CFG["rollout_len"])
    p.add_argument("--reset-max-depth", type=int, default=DEFAULT_CFG["reset_max_depth"],
                   help="max random plies for a diverse-reset position")
    p.add_argument("--reset-pool", type=int, default=DEFAULT_CFG["reset_pool"],
                   help="number of pre-sampled reset positions, resampled each iter")
    p.add_argument("--width", type=int, default=DEFAULT_CFG["width"])
    p.add_argument("--depth", type=int, default=DEFAULT_CFG["depth"])
    p.add_argument("--arch", choices=["mlp", "conv", "transformer", "aznet"], default=DEFAULT_CFG["arch"])
    p.add_argument("--lr", type=float, default=DEFAULT_CFG["lr"])
    p.add_argument("--ent-coef", type=float, default=DEFAULT_CFG["ent_coef"])
    p.add_argument("--opp-pool-frac", type=float, default=DEFAULT_CFG["opp_pool_frac"],
                   help="fraction of envs whose opponent seat is a sampled past policy")
    p.add_argument("--pool-size", type=int, default=DEFAULT_CFG["pool_size"])
    p.add_argument("--pool-every", type=int, default=DEFAULT_CFG["pool_every"])
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-games", type=int, default=2048)
    p.add_argument("--out", type=str, default="runs/c4-dev")
    args = p.parse_args()

    cfg = dict(DEFAULT_CFG, game=args.game, num_envs=args.num_envs,
               rollout_len=args.rollout_len, reset_max_depth=args.reset_max_depth,
               reset_pool=args.reset_pool, width=args.width, depth=args.depth,
               arch=args.arch, opp_pool_frac=args.opp_pool_frac,
               pool_size=args.pool_size, pool_every=args.pool_every,
               lr=args.lr, ent_coef=args.ent_coef)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps({**cfg, "seed": args.seed}, indent=2))

    env = pgx.make(cfg["game"])
    init, train_iter, network = make_pgx_core(env, cfg)
    runner_state = init(jax.random.PRNGKey(args.seed))

    steps_per_iter = cfg["num_envs"] * cfg["rollout_len"]
    log_path = out / "metrics.jsonl"
    print(f"game={cfg['game']} devices={jax.devices()} "
          f"max_depth={cfg['reset_max_depth']} steps/iter={steps_per_iter}")

    def save(params, name):
        (out / name).write_bytes(serialization.to_bytes(params))

    eval_key = jax.random.PRNGKey(args.seed + 10_000)
    t0 = time.time()
    with log_path.open("a") as log:
        for it in range(1, args.iters + 1):
            t_it = time.time()
            runner_state, metrics = train_iter(runner_state)
            metrics = {k: float(np.asarray(v)) for k, v in metrics.items()}
            metrics.update(iter=it, env_steps=it * steps_per_iter,
                           sps=steps_per_iter / (time.time() - t_it),
                           wall=time.time() - t0)
            if it % args.eval_every == 0 or it == 1 or it == args.iters:
                eval_key, k = jax.random.split(eval_key)
                w, d, l = evaluate_vs_random(env, network, runner_state[0], k,
                                             args.eval_games)
                metrics.update(vs_random_win=w, vs_random_draw=d, vs_random_loss=l)
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            vr = (f" | vs_random W{metrics['vs_random_win']:.2f}"
                  f"/D{metrics['vs_random_draw']:.2f}/L{metrics['vs_random_loss']:.2f}"
                  if "vs_random_win" in metrics else "")
            print(f"it {it:4d} | steps {metrics['env_steps']:.2e}"
                  f" | sps {metrics['sps']:8.0f} | dec {metrics['decisive_rate']:.2f}"
                  f" | draw {metrics['draw_rate']:.2f} | ent {metrics['entropy']:.3f}{vr}")
            if it % args.ckpt_every == 0 or it == args.iters:
                save(runner_state[0], f"ckpt_{it:05d}.msgpack")
    save(runner_state[0], "ckpt_final.msgpack")
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
