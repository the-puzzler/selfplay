"""Train GRPO self-play on the Nidhogg env.

Usage:
  uv run python -m lethal.train --reset-mode diverse --out runs/ll-v1
Dashboard: uv run python -m fighter2d.plot --run runs/ll-v1
"""

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np
from flax import serialization

from lethal.env import EPISODE_LEN, LethalEnv
from lethal.grpo import DEFAULT_CFG, make_train_iter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reset-mode", choices=["fixed", "diverse"], default="diverse")
    p.add_argument("--n-groups", type=int, default=DEFAULT_CFG["n_groups"])
    p.add_argument("--group-size", type=int, default=DEFAULT_CFG["group_size"])
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-every", type=int, default=25)
    p.add_argument("--out", type=str, default="runs/ll-dev")
    args = p.parse_args()

    cfg = dict(DEFAULT_CFG, n_groups=args.n_groups, group_size=args.group_size)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps({**cfg, "reset_mode": args.reset_mode, "seed": args.seed}, indent=2))

    env = LethalEnv(reset_mode=args.reset_mode)
    init, train_iter, network = make_train_iter(env, cfg)
    params, opt_state, rng = init(jax.random.PRNGKey(args.seed))

    steps_per_iter = cfg["n_groups"] * cfg["group_size"] * EPISODE_LEN
    print(f"[{args.reset_mode}] devices={jax.devices()} steps/iter={steps_per_iter}")

    def save(par, name):
        (out / name).write_bytes(serialization.to_bytes(par))

    t0 = time.time()
    with (out / "metrics.jsonl").open("a") as log:
        for it in range(1, args.iters + 1):
            t_it = time.time()
            params, opt_state, rng, metrics = train_iter(params, opt_state, rng)
            metrics = {k: float(np.asarray(v)) for k, v in metrics.items()}
            metrics.update(
                iter=it, env_steps=it * steps_per_iter,
                sps=steps_per_iter / (time.time() - t_it), wall=time.time() - t0)
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            print(
                f"it {it:4d} | steps {metrics['env_steps']:.2e} | sps {metrics['sps']:8.0f}"
                f" | draw {metrics['draw_rate']:.2f} | ep_len {metrics['mean_ep_len']:5.1f}"
                f" | kills/ep {metrics['kills_per_ep']:.2f} | ent {metrics['entropy']:5.2f}")
            if it % args.ckpt_every == 0 or it == args.iters:
                save(params, f"ckpt_{it:05d}.msgpack")
    save(params, "ckpt_final.msgpack")
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
