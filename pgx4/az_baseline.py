"""Load Pgx's official AlphaZero baseline models under JAX >= 0.11.

The baselines are dm-haiku conv-nets. Two problems on this box:
  1. haiku 0.0.16 references jax.core symbols removed in JAX 0.11 -> we shim them.
  2. the nets are convolutional -> cuDNN fails on this GPU, so baseline inference
     must run on CPU. Callers should set JAX_PLATFORMS=cpu (or the baseline path
     runs on CPU while our MLP agent runs wherever).

  uv run python -m pgx4.az_baseline --model othello_v0 --games 256
"""

import argparse

import jax
import jax.extend

# --- shim jax.core aliases that dm-haiku 0.0.16 still imports/uses ------------
import jax.core as _jc

_SHIM = ("DropVar", "Literal", "Var", "Jaxpr", "JaxprEqn", "ClosedJaxpr",
         "get_opaque_trace_state", "new_main", "cur_sublevel", "Trace", "Tracer",
         "MainTrace", "extend_axis_env", "eval_jaxpr")
for _name in _SHIM:
    if not hasattr(_jc, _name):
        for _src in (jax.extend.core, getattr(jax, "_src", None)
                     and jax._src.core):
            if _src is not None and hasattr(_src, _name):
                setattr(_jc, _name, getattr(_src, _name))
                break

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pgx  # noqa: E402

NEG = -1e9

# model_id -> game id
MODEL_GAME = {
    "othello_v0": "othello",
    "hex_v0": "hex",
    "go_9x9_v0": "go_9x9",
    "animal_shogi_v0": "animal_shogi",
    "gardner_chess_v0": "gardner_chess",
}


def load_model(model_id, download_dir="baselines"):
    return pgx.make_baseline_model(model_id, download_dir=download_dir)


def make_move_fn(model):
    """Greedy legal move from the baseline's policy head."""
    def move(obs, mask):
        logits, _ = model(obs)
        return jnp.argmax(jnp.where(mask, logits, NEG), -1)
    return move


def _selftest(model_id, n_games, download_dir):
    """Baseline vs uniform-random, both seats, to confirm it plays strongly."""
    env = pgx.make(MODEL_GAME[model_id])
    model = load_model(model_id, download_dir)
    move = make_move_fn(model)
    v_step = jax.vmap(env.step)

    key = jax.random.PRNGKey(0)
    k_init, k_play = jax.random.split(key)
    state = jax.vmap(env.init)(jax.random.split(k_init, n_games))
    base_seat = (jnp.arange(n_games) % 2).astype(jnp.int32)
    results = jnp.zeros(n_games)

    def cond(c):
        s, _, _ = c
        return ~(s.terminated | s.truncated).all()

    def body(c):
        s, kk, res = c
        kk, kr = jax.random.split(kk)
        obs = s.observation.astype(jnp.float32)
        base_a = move(obs, s.legal_action_mask)
        rand_a = jax.random.categorical(kr, jnp.where(s.legal_action_mask, 0.0, NEG))
        a = jnp.where(s.current_player == base_seat, base_a, rand_a)
        db = s.terminated | s.truncated
        ns = v_step(s, a)
        newly = (ns.terminated | ns.truncated) & ~db
        r = ns.rewards[jnp.arange(n_games), base_seat]
        return ns, kk, jnp.where(newly, r, res)

    _, _, results = jax.lax.while_loop(cond, body, (state, k_play, results))
    r = np.asarray(results)
    print(f"{model_id} ({MODEL_GAME[model_id]}): obs {env.observation_shape}, "
          f"actions {env.num_actions}")
    print(f"baseline vs random: W {(r>0).mean():.3f} / D {(r==0).mean():.3f} / "
          f"L {(r<0).mean():.3f}  ({n_games} games)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="othello_v0", choices=list(MODEL_GAME))
    p.add_argument("--games", type=int, default=256)
    p.add_argument("--download-dir", default="baselines")
    args = p.parse_args()
    _selftest(args.model, args.games, args.download_dir)


if __name__ == "__main__":
    main()
