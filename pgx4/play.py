"""Play against a trained checkpoint in the terminal.

Each turn prints an ASCII board (plus a board.svg you can open in a browser for
the pretty version) and the legal moves as coordinates. The model plays its
moves greedily.

  uv run python -m pgx4.play --game othello \
      --ckpt results/checkpoints/oth-aznet-s0-final.msgpack \
      --config results/checkpoints/oth-aznet-s0-config.json \
      --human-seat 0

  --human-seat 0 moves first; 1 moves second; --selfplay watches model vs model.
Enter moves as e.g. "d3" (column letter + row number), or "pass"/"swap".
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pgx
from flax import serialization

from pgx4.train import ActorCritic

NEG = -1e9
STONES = {"othello": ("●", "○"), "go_9x9": ("●", "○"), "hex": ("●", "○")}


def board_hw(env):
    h, w = env.observation_shape[0], env.observation_shape[1]
    return h, w


def ascii_board(state, env):
    """Best-effort ASCII board. Obs planes 0/1 are current-player/opponent, so we
    map back to absolute seat colors via state.current_player."""
    h, w = board_hw(env)
    obs = np.asarray(state.observation)
    me, opp = obs[..., 0].astype(bool), obs[..., 1].astype(bool)
    cur = int(state.current_player)
    p0 = me if cur == 0 else opp   # seat-0 stones
    p1 = opp if cur == 0 else me
    s0, s1 = STONES.get(env.id, ("●", "○"))
    lines = ["   " + " ".join(chr(ord("a") + c) for c in range(w))]
    for r in range(h):
        row = " ".join(s0 if p0[r, c] else s1 if p1[r, c] else "·" for c in range(w))
        indent = " " * r if env.id == "hex" else ""   # shear hex for readability
        lines.append(f"{r+1:2d} {indent}{row}")
    return "\n".join(lines)


def action_name(a, env):
    h, w = board_hw(env)
    if a < h * w:
        return f"{chr(ord('a') + a % w)}{a // w + 1}"
    return "pass" if env.id != "hex" else "swap"


def parse_move(txt, env):
    txt = txt.strip().lower()
    h, w = board_hw(env)
    if txt in ("pass", "swap"):
        return h * w
    col = ord(txt[0]) - ord("a")
    row = int(txt[1:]) - 1
    return row * w + col


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", required=True, choices=["othello", "hex", "go_9x9"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--human-seat", type=int, default=0, choices=[0, 1])
    p.add_argument("--selfplay", action="store_true", help="model plays both seats")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = pgx.make(args.game)
    cfg_path = Path(args.config) if args.config else Path(args.ckpt).parent / "config.json"
    cfg = json.loads(cfg_path.read_text())
    net = ActorCritic(n_actions=env.num_actions, width=cfg["width"],
                      depth=cfg["depth"], arch=cfg["arch"])
    tmpl = net.init(jax.random.PRNGKey(0), jnp.zeros((1,) + env.observation_shape))
    params = serialization.from_bytes(tmpl, Path(args.ckpt).read_bytes())

    @jax.jit
    def model_move(obs, mask):
        logits, value = net.apply(params, obs[None])
        return jnp.argmax(jnp.where(mask, logits[0], NEG)), value[0]

    state = env.init(jax.random.PRNGKey(args.seed))
    move_no = 0
    while not (bool(state.terminated) or bool(state.truncated)):
        pgx.save_svg(state, "board.svg")
        cur = int(state.current_player)
        legal = np.nonzero(np.asarray(state.legal_action_mask))[0]
        print(f"\nmove {move_no + 1} — {'seat 0' if cur == 0 else 'seat 1'} to play")
        print(ascii_board(state, env))
        if args.selfplay or cur != args.human_seat:
            a, v = model_move(state.observation.astype(jnp.float32),
                              state.legal_action_mask)
            a = int(a)
            print(f"model plays {action_name(a, env)}  (its value estimate: {float(v):+.2f})")
        else:
            print("legal:", " ".join(action_name(a, env) for a in legal))
            while True:
                try:
                    a = parse_move(input("your move> "), env)
                    if a in legal:
                        break
                except (ValueError, IndexError):
                    pass
                except EOFError:
                    print("\n(input closed — exiting)")
                    return
                print("illegal — try again")
        state = env.step(state, a)
        move_no += 1

    pgx.save_svg(state, "board.svg")
    print("\nfinal position:")
    print(ascii_board(state, env))
    r = np.asarray(state.rewards)
    if r[0] == r[1]:
        print("result: draw")
    else:
        winner = int(np.argmax(r))
        who = "model" if (args.selfplay or winner != args.human_seat) else "you"
        print(f"result: seat {winner} wins ({who})")


if __name__ == "__main__":
    main()
