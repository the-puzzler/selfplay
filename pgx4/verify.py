"""Multi-seed verification: our checkpoint vs a published Pgx AlphaZero baseline.

Plays 2 x openings games per seed (both seat assignments), greedy raw policy on
both sides, and reports the pooled match score with a standard error.

  uv run python -m pgx4.verify --game othello --model othello_v0 \
      --ckpt results/checkpoints/oth-aznet-s0-final.msgpack \
      --config results/checkpoints/oth-aznet-s0-config.json
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pgx
from flax import serialization

from pgx4.az_baseline import load_model, make_move_fn
from pgx4.elo import random_start
from pgx4.train import ActorCritic

NEG = -1e9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", required=True)
    p.add_argument("--model", required=True, help="baseline id, e.g. othello_v0")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", default=None,
                   help="run config.json (default: config.json next to the ckpt)")
    p.add_argument("--openings", type=int, default=1024)
    p.add_argument("--opening-plies", type=int, default=10)
    p.add_argument("--seeds", type=int, default=5)
    args = p.parse_args()

    env = pgx.make(args.game)
    v_step = jax.vmap(env.step)
    base = make_move_fn(load_model(args.model))
    cfg_path = Path(args.config) if args.config else Path(args.ckpt).parent / "config.json"
    cfg = json.loads(cfg_path.read_text())
    net = ActorCritic(n_actions=env.num_actions, width=cfg["width"],
                      depth=cfg["depth"], arch=cfg["arch"])
    tmpl = net.init(jax.random.PRNGKey(0), jnp.zeros((1,) + env.observation_shape))
    params = serialization.from_bytes(tmpl, Path(args.ckpt).read_bytes())
    M = args.openings

    @jax.jit
    def run(key):
        ks, _ = jax.random.split(key)
        op = jax.vmap(lambda k: random_start(env, k, args.opening_plies))(
            jax.random.split(ks, M))
        st = jax.tree.map(lambda x: jnp.concatenate([x, x], 0), op)
        n = 2 * M
        seat = jnp.concatenate([jnp.zeros(M, jnp.int32), jnp.ones(M, jnp.int32)])

        def cond(c):
            s, _ = c
            return ~(s.terminated | s.truncated).all()

        def body(c):
            s, res = c
            obs = s.observation.astype(jnp.float32)
            a = jnp.where(s.current_player == seat,
                          jnp.argmax(jnp.where(s.legal_action_mask,
                                               net.apply(params, obs)[0], NEG), -1),
                          base(obs, s.legal_action_mask))
            db = s.terminated | s.truncated
            ns = v_step(s, a)
            newly = (ns.terminated | ns.truncated) & ~db
            return ns, jnp.where(newly, ns.rewards[jnp.arange(n), seat], res)

        _, res = jax.lax.while_loop(cond, body, (st, jnp.zeros(n)))
        return res

    allr = []
    for seed in range(args.seeds):
        r = np.asarray(run(jax.random.PRNGKey(seed + 1)))
        allr.append(r)
        s = (r > 0).mean() + 0.5 * (r == 0).mean()
        print(f"seed {seed+1}: W{(r>0).mean():.3f}/D{(r==0).mean():.3f}/"
              f"L{(r<0).mean():.3f}  score {s:.3f}", flush=True)
    allr = np.concatenate(allr)
    pts = np.array([(x > 0) + 0.5 * (x == 0) for x in allr])
    print(f"POOLED {len(allr)} games: score {pts.mean():.4f} "
          f"+/- {pts.std()/np.sqrt(len(allr)):.4f} (1SE)  "
          f"W{(allr>0).mean():.3f}/D{(allr==0).mean():.3f}/L{(allr<0).mean():.3f}")


if __name__ == "__main__":
    main()
