"""Self-play Elo across training checkpoints.

Round-robins the saved checkpoints against each other and fits Elo ratings from
the pairwise results, so you can see whether skill actually climbed over training
and where it plateaued. Games start from varied random mid-game openings (so
greedy play differs game to game) and each opening is played with both seat
assignments to cancel first-move bias.

  uv run python -m pgx4.elo --run runs/c4-diverse-s0 --openings 256
"""

import argparse
import glob
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pgx
from flax import serialization

from pgx4.train import ActorCritic, masked_logits

NEG = -1e9


def random_start(env, key, max_depth):
    """A non-terminal position reached by up to max_depth random legal plies."""
    k0, kd, kp = jax.random.split(key, 3)
    state = env.init(k0)
    depth = jax.random.randint(kd, (), 0, max_depth + 1)

    def body(carry, i):
        state, frozen, kk = carry
        kk, ka = jax.random.split(kk)
        a = jax.random.categorical(ka, jnp.where(state.legal_action_mask, 0.0, NEG))
        ns = env.step(state, a)
        stop = frozen | (i >= depth) | ns.terminated | ns.truncated
        state = jax.tree.map(lambda n, s: jnp.where(~stop, n, s), ns, state)
        return (state, frozen | stop, kk), None

    (state, _, _), _ = jax.lax.scan(
        body, (state, jnp.bool_(False), kp), jnp.arange(max_depth))
    return state


def load_ckpts(run: Path, ckpts=None):
    if ckpts:
        paths = [p for pat in ckpts for p in sorted(glob.glob(pat))]
    else:
        paths = sorted(glob.glob(str(run / "ckpt_*.msgpack")))
        paths = [p for p in paths if re.search(r"ckpt_\d+\.msgpack$", p)]
    labels, params_list = [], []
    env = pgx.make("connect_four")
    net = ActorCritic(n_actions=env.num_actions)
    template = net.init(jax.random.PRNGKey(0), jnp.zeros((1,) + env.observation_shape))
    for p in paths:
        m = re.search(r"ckpt_(\d+)", p)
        # label by iteration, disambiguated by parent run dir when mixing runs
        run_tag = Path(p).parent.name.replace("c4-", "")
        labels.append(f"{run_tag}:{m.group(1)}" if m else Path(p).stem)
        params_list.append(serialization.from_bytes(template, Path(p).read_bytes()))
    return env, net, labels, params_list


def fit_elo(wins, games, anchor_idx=0):
    """MLE Elo from a wins matrix (wins[i,j] = points i scored vs j, out of
    games[i,j]). Ratings on the standard 400-point logistic scale."""
    wins = jnp.asarray(wins)
    games = jnp.asarray(games)
    off = ~jnp.eye(wins.shape[0], dtype=bool)
    scale = 400.0 / jnp.log(10.0)

    def loss(r):
        p = jax.nn.sigmoid((r[:, None] - r[None, :]) / scale)
        ll = wins * jnp.log(p + 1e-9) + (games - wins) * jnp.log(1 - p + 1e-9)
        return -(jnp.where(off, ll, 0.0)).sum() / games.sum()

    r = jnp.zeros(wins.shape[0])
    opt = optax.adam(8.0)
    st = opt.init(r)
    for _ in range(4000):
        g = jax.grad(loss)(r)
        upd, st = opt.update(g, st)
        r = optax.apply_updates(r, upd)
    r = r - r[anchor_idx]
    return np.asarray(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None, help="run dir (uses all its numbered checkpoints)")
    p.add_argument("--ckpts", nargs="+", default=None,
                   help="explicit checkpoint paths/globs (overrides --run); can mix runs")
    p.add_argument("--openings", type=int, default=256,
                   help="distinct random openings per pair (each played both seat orders)")
    p.add_argument("--max-depth", type=int, default=10,
                   help="max random plies for opening positions")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not args.run and not args.ckpts:
        p.error("provide --run or --ckpts")
    run = Path(args.run) if args.run else Path(args.ckpts[0]).parent
    env, net, labels, params_list = load_ckpts(run, args.ckpts)
    K = len(labels)
    v_step = jax.vmap(env.step)
    print(f"checkpoints: {labels}")

    def greedy(params, obs, mask):
        logits, _ = net.apply(params, obs)
        return jnp.argmax(masked_logits(logits, mask), -1)

    @jax.jit
    def play(pA, pB, starts, a_seat):
        N = a_seat.shape[0]

        def cond(c):
            s, _ = c
            return ~(s.terminated | s.truncated).all()

        def body(c):
            s, res = c
            obs = s.observation.astype(jnp.float32)
            aA = greedy(pA, obs, s.legal_action_mask)
            aB = greedy(pB, obs, s.legal_action_mask)
            action = jnp.where(s.current_player == a_seat, aA, aB)
            db = s.terminated | s.truncated
            ns = v_step(s, action)
            newly = (ns.terminated | ns.truncated) & ~db
            r = ns.rewards[jnp.arange(N), a_seat]
            return ns, jnp.where(newly, r, res)

        _, res = jax.lax.while_loop(cond, body, (starts, jnp.zeros(N)))
        return res  # +1 A win, -1 A loss, 0 draw

    # one shared pool of openings, reused for every pair
    key = jax.random.PRNGKey(args.seed)
    key, ks = jax.random.split(key)
    starts = jax.vmap(lambda k: random_start(env, k, args.max_depth))(
        jax.random.split(ks, args.openings))
    seat0 = jnp.zeros(args.openings, jnp.int32)
    seat1 = jnp.ones(args.openings, jnp.int32)

    n_per_pair = 2 * args.openings
    wins = np.zeros((K, K))
    games = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            # i as A, both seat orders
            r0 = np.asarray(play(params_list[i], params_list[j], starts, seat0))
            r1 = np.asarray(play(params_list[i], params_list[j], starts, seat1))
            r = np.concatenate([r0, r1])
            pts_i = float((r > 0).sum() + 0.5 * (r == 0).sum())
            wins[i, j] += pts_i
            wins[j, i] += n_per_pair - pts_i
            games[i, j] += n_per_pair
            games[j, i] += n_per_pair

    elo = fit_elo(wins, games, anchor_idx=0)
    order = np.argsort(elo)
    print(f"\nSelf-play Elo ({n_per_pair} games/pair, anchored at {labels[0]}=0):")
    top = int(np.argmax(elo))
    for idx in order:
        # score rate vs the strongest checkpoint for context
        if idx == top:
            ctx = "(strongest)"
        else:
            wr = wins[idx, top] / games[idx, top]
            ctx = f"(score vs top {wr*100:4.1f}%)"
        print(f"  ckpt_{labels[idx]:>5}  Elo {elo[idx]:+7.1f}   {ctx}")
    (run / "elo.json").write_text(
        __import__("json").dumps(
            {labels[i]: float(elo[i]) for i in range(K)}, indent=2))
    print(f"\nwrote {run/'elo.json'}")


if __name__ == "__main__":
    main()
