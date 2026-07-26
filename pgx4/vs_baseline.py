"""Head-to-head: our PPO + diverse-resets agent vs Pgx's official AlphaZero
baseline. Runs on CPU (the baseline is a conv-net; cuDNN is broken on this GPU).

  JAX_PLATFORMS=cpu uv run python -m pgx4.vs_baseline \
      --ckpt runs/othello-s0/ckpt_final.msgpack --model othello_v0 --games 512
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pgx
from flax import serialization

from pgx4.az_baseline import MODEL_GAME, load_model, make_move_fn
from pgx4.elo import random_start
from pgx4.train import ActorCritic

NEG = -1e9


def load_agent(ckpt, env):
    cfg_path = Path(ckpt).parent / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    net = ActorCritic(n_actions=env.num_actions,
                      width=cfg.get("width", 256), depth=cfg.get("depth", 2),
                      arch=cfg.get("arch", "mlp"))
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1,) + env.observation_shape))
    params = serialization.from_bytes(params, Path(ckpt).read_bytes())
    return net, params, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model", default="othello_v0", choices=list(MODEL_GAME))
    p.add_argument("--openings", type=int, default=256,
                   help="distinct random openings; each played with both seat assignments")
    p.add_argument("--opening-plies", type=int, default=10,
                   help="random legal plies to vary the start (fixed init -> deterministic games otherwise)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = pgx.make(MODEL_GAME[args.model])
    net, params, cfg = load_agent(args.ckpt, env)
    baseline = make_move_fn(load_model(args.model))
    v_step = jax.vmap(env.step)

    def agent_move(obs, mask):
        logits, _ = net.apply(params, obs)
        return jnp.argmax(jnp.where(mask, logits, NEG), -1)

    @jax.jit
    def run(key):
        m = args.openings
        ks, _ = jax.random.split(key)
        openings = jax.vmap(lambda k: random_start(env, k, args.opening_plies))(
            jax.random.split(ks, m))
        # play each opening under both seat assignments -> 2m games, bias cancels
        state = jax.tree.map(lambda x: jnp.concatenate([x, x], 0), openings)
        n = 2 * m
        agent_seat = jnp.concatenate([jnp.zeros(m, jnp.int32),
                                      jnp.ones(m, jnp.int32)])
        results = jnp.zeros(n)

        def cond(c):
            s, _ = c
            return ~(s.terminated | s.truncated).all()

        def body(c):
            s, res = c
            obs = s.observation.astype(jnp.float32)
            a = jnp.where(s.current_player == agent_seat,
                          agent_move(obs, s.legal_action_mask),
                          baseline(obs, s.legal_action_mask))
            db = s.terminated | s.truncated
            ns = v_step(s, a)
            newly = (ns.terminated | ns.truncated) & ~db
            r = ns.rewards[jnp.arange(n), agent_seat]
            return ns, jnp.where(newly, r, res)

        _, res = jax.lax.while_loop(cond, body, (state, results))
        return res

    r = np.asarray(run(jax.random.PRNGKey(args.seed)))
    w, d, l = float((r > 0).mean()), float((r == 0).mean()), float((r < 0).mean())
    score = w + 0.5 * d  # match score from the agent's perspective
    net_desc = f"{cfg.get('width', 256)}x{cfg.get('depth', 2)} MLP"
    print(f"agent ({net_desc}, {Path(args.ckpt).name}) vs {args.model}")
    print(f"  W {w:.3f} / D {d:.3f} / L {l:.3f}   match score {score:.3f}"
          f"   ({2*args.openings} games from {args.openings} openings"
          f", {args.opening_plies}-ply, both seats)")


if __name__ == "__main__":
    main()
