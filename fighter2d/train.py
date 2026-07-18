"""Train self-play PPO on the 2D fighter env.

Usage:
  uv run python -m fighter2d.train --reset-mode diverse --out runs/diverse
  uv run python -m fighter2d.train --reset-mode fixed --out runs/fixed
"""

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np
from flax import serialization

from fighter2d.env import FighterEnv
from fighter2d.ppo import DEFAULT_CFG, make_train_iter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reset-mode", choices=["fixed", "diverse"], default="diverse")
    p.add_argument("--num-envs", type=int, default=DEFAULT_CFG["num_envs"])
    p.add_argument("--rollout-len", type=int, default=DEFAULT_CFG["rollout_len"])
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument("--out", type=str, default="runs/dev")
    args = p.parse_args()

    cfg = dict(DEFAULT_CFG, num_envs=args.num_envs, rollout_len=args.rollout_len)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps({**cfg, "reset_mode": args.reset_mode, "seed": args.seed}, indent=2)
    )

    env = FighterEnv(reset_mode=args.reset_mode)
    init, train_iter, network = make_train_iter(env, cfg)
    runner_state = init(jax.random.PRNGKey(args.seed))

    steps_per_iter = cfg["num_envs"] * cfg["rollout_len"]
    log_path = out / "metrics.jsonl"
    print(f"[{args.reset_mode}] devices={jax.devices()} steps/iter={steps_per_iter}")

    def save(params, name):
        (out / name).write_bytes(serialization.to_bytes(params))

    t0 = time.time()
    with log_path.open("a") as log:
        for it in range(1, args.iters + 1):
            t_it = time.time()
            runner_state, metrics = train_iter(runner_state)
            metrics = {k: float(np.asarray(v)) for k, v in metrics.items()}
            metrics.update(
                iter=it,
                env_steps=it * steps_per_iter,
                sps=steps_per_iter / (time.time() - t_it),
                wall=time.time() - t0,
            )
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            print(
                f"it {it:4d} | steps {metrics['env_steps']:.2e} | sps {metrics['sps']:8.0f}"
                f" | eps {metrics['episodes']:6.0f} | draw {metrics['draw_rate']:.2f}"
                f" | timeout {metrics['timeout_rate']:.2f} | ep_len {metrics['mean_ep_len']:5.1f}"
                f" | ent {metrics['entropy']:6.2f}"
            )
            if it % args.ckpt_every == 0 or it == args.iters:
                save(runner_state[0], f"ckpt_{it:05d}.msgpack")
    save(runner_state[0], "ckpt_final.msgpack")
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
