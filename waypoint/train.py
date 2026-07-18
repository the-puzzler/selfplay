"""Train recurrent PPO on waypoint parkour.

Usage:
  uv run python -m waypoint.train --out runs/wp-v1
Dashboard: uv run python -m fighter2d.plot --run runs/wp-v1
(win0_rate on the dashboard = waypoint reach rate.)
"""

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np
from flax import serialization

from waypoint.env import WaypointEnv
from waypoint.ppo import DEFAULT_CFG, make_train_iter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reset-mode", choices=["fixed", "diverse"], default="diverse")
    p.add_argument("--num-envs", type=int, default=DEFAULT_CFG["num_envs"])
    p.add_argument("--rollout-len", type=int, default=DEFAULT_CFG["rollout_len"])
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-every", type=int, default=25)
    p.add_argument("--out", type=str, default="runs/wp-dev")
    args = p.parse_args()

    cfg = dict(DEFAULT_CFG, num_envs=args.num_envs, rollout_len=args.rollout_len)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps({**cfg, "reset_mode": args.reset_mode, "seed": args.seed}, indent=2))

    env = WaypointEnv(reset_mode=args.reset_mode)
    init, train_iter, network = make_train_iter(env, cfg)
    runner_state = init(jax.random.PRNGKey(args.seed))

    steps_per_iter = cfg["num_envs"] * cfg["rollout_len"]
    print(f"[{args.reset_mode}] devices={jax.devices()} steps/iter={steps_per_iter}")

    def save(par, name):
        (out / name).write_bytes(serialization.to_bytes(par))

    t0 = time.time()
    with (out / "metrics.jsonl").open("a") as log:
        for it in range(1, args.iters + 1):
            t_it = time.time()
            runner_state, metrics = train_iter(runner_state)
            metrics = {k: float(np.asarray(v)) for k, v in metrics.items()}
            metrics.update(
                iter=it, env_steps=it * steps_per_iter,
                sps=steps_per_iter / (time.time() - t_it), wall=time.time() - t0)
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            print(
                f"it {it:4d} | steps {metrics['env_steps']:.2e} | sps {metrics['sps']:7.0f}"
                f" | reach {metrics['win0_rate']:.2f} | ep_len {metrics['mean_ep_len']:5.1f}"
                f" | dist {metrics['final_dist']:4.2f} | ent {metrics['entropy']:5.2f}")
            if it % args.ckpt_every == 0 or it == args.iters:
                save(runner_state[0], f"ckpt_{it:05d}.msgpack")
    save(runner_state[0], "ckpt_final.msgpack")
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
