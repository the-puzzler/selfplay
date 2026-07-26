"""Adversarial audit of the sweep claims. Invariants + fresh-seed re-verification
with opening-clustered standard errors."""
import json, sys
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np, pgx
from flax import serialization
from pgx4.az_baseline import load_model, make_move_fn
from pgx4.elo import random_start
from pgx4.train import ActorCritic
NEG=-1e9

GAMES = {
  "othello": ("othello_v0", "results/checkpoints/oth-aznet-s0-final.msgpack", 10),
  "hex": ("hex_v0", "results/checkpoints/hex-aznet-s0-final.msgpack", 10),
  "go_9x9": ("go_9x9_v0", "results/checkpoints/go9-aznet-s0-final.msgpack", 20),
}

def load_ours(env, ckpt):
    cfg = json.loads((Path(ckpt).parent / (Path(ckpt).name.replace("-final.msgpack","-config.json"))).read_text())
    net = ActorCritic(n_actions=env.num_actions, width=cfg["width"], depth=cfg["depth"], arch=cfg["arch"])
    tmpl = net.init(jax.random.PRNGKey(0), jnp.zeros((1,)+env.observation_shape))
    params = serialization.from_bytes(tmpl, Path(ckpt).read_bytes())
    return lambda obs, mask, key: jnp.argmax(jnp.where(mask, net.apply(params, obs)[0], NEG), -1)

def match(env, moveA, moveB, plies, M, seed):
    v_step = jax.vmap(env.step)
    def run(key):
        ks, kp = jax.random.split(key)
        op = jax.vmap(lambda k: random_start(env, k, plies))(jax.random.split(ks, M))
        st = jax.tree.map(lambda x: jnp.concatenate([x, x], 0), op); n = 2*M
        seat = jnp.concatenate([jnp.zeros(M, jnp.int32), jnp.ones(M, jnp.int32)])
        def cond(c): s,_,_ = c; return ~(s.terminated | s.truncated).all()
        def body(c):
            s, kk, res = c
            kk, k1, k2 = jax.random.split(kk, 3)
            obs = s.observation.astype(jnp.float32); mask = s.legal_action_mask
            a = jnp.where(s.current_player == seat, moveA(obs, mask, k1), moveB(obs, mask, k2))
            db = s.terminated | s.truncated; ns = v_step(s, a)
            newly = (ns.terminated | ns.truncated) & ~db
            return ns, kk, jnp.where(newly, ns.rewards[jnp.arange(n), seat], res)
        _,_,res = jax.lax.while_loop(cond, body, (st, kp, jnp.zeros(n)))
        return res
    r = np.asarray(jax.jit(run)(jax.random.PRNGKey(seed)))
    pts = (r > 0) + 0.5*(r == 0)
    pairs = (pts[:M] + pts[M:]) / 2          # cluster by opening (both seats)
    return pts.mean(), pairs.std(ddof=1)/np.sqrt(M), len(r)

for game, (model_id, ckpt, plies) in GAMES.items():
    env = pgx.make(game)
    base_apply = make_move_fn(load_model(model_id))
    base = lambda obs, mask, key: base_apply(obs, mask)
    ours = load_ours(env, ckpt)
    rand = lambda obs, mask, key: jax.random.categorical(key, jnp.where(mask, 0.0, NEG))
    print(f"== {game} ==", flush=True)
    for name, A, B, M, seeds in [
        ("ours  vs base ", ours, base, 512, (11,12,13)),
        ("ours  vs ours ", ours, ours, 256, (21,)),
        ("base  vs base ", base, base, 256, (22,)),
        ("base  vs random", base, rand, 128, (23,)),
        ("ours  vs random", ours, rand, 128, (24,)),
    ]:
        scores, ses, ns = [], [], 0
        for sd in seeds:
            m, se, n = match(env, A, B, plies, M, sd); scores.append(m); ses.append(se); ns += n
        s = np.mean(scores); se = np.sqrt(np.mean(np.array(ses)**2)/len(seeds))
        print(f"  {name}: {s:.4f} +/- {se:.4f} (clustered SE, {ns} games)", flush=True)
print("AUDIT DONE", flush=True)

# Forensic check: do the released checkpoints match the paper's stated training
# lengths? haiku BatchNorm EMA counters increment once per gradient step; their
# setup runs 262,144 frames/iter / 4,096 minibatch = 64 steps per iteration.
import pickle
for mid, iters in [("othello_v0", 100), ("hex_v0", 100), ("go_9x9_v0", 200)]:
    d = pickle.load(open(f"baselines/{mid}.ckpt", "rb"))
    counters = {int(np.asarray(v["counter"])) for v in d["state"].values() if "counter" in v}
    assert counters == {iters * 64}, (mid, counters)
    print(f"{mid}: BN EMA counter {counters} == {iters} iters x 64 steps  OK")
