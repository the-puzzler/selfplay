"""Lethal League demake: one ball, two players, speed compounds per hit.

Official core mechanics (established game, not ours):
  - The ball only harms players who don't own it; hitting it claims it,
    aims it (flat / 45 up / 45 down toward the opponent) and multiplies
    its speed. Walls/floor/ceiling reflect it losslessly.
  - Swings have startup/active/recovery: timing is the skill. Bunting
    pops the ball up slow and NEUTRAL (harmless) — tempo control.
  - Simultaneous swings on the ball clash: neutral vertical pop.
  - Hit-freeze after every strike (scales with speed): the signature
    beat that makes high-speed rallies followable and punishes panic.
  - First tag ends the episode: +1 owner / -1 tagged, 0 on timeout.

State is small and fully discrete/continuous-simple: every random reset
(players anywhere, any swing phase, ball anywhere at ANY speed, any owner)
is a valid game state by construction — including "ball inbound at mach 3".
"""

import jax
import jax.numpy as jnp
from flax import struct

DT = 0.05  # 20 Hz control
EPISODE_LEN = 600  # 30 seconds
AX, AY = 4.0, 3.0  # arena half-width, height (x in [-AX, AX], y in [0, AY])
RUN = 6.0
JUMP_V = 8.5
GRAV = 24.0
P_R = 0.30  # player radius (ball tag distance)
HIT_R = 0.85  # swing reach
BALL_R = 0.12
BALL_V0 = 2.0
BALL_MULT = 1.2
BALL_VMAX = 36.0
N_SUB = 12  # ball substeps per tick (continuous collision)

NEUTRAL, SW_START, SW_ACTIVE, SW_REC, BUNT_ACT = 0, 1, 2, 3, 4
START_T, ACTIVE_T, REC_T, BUNT_T = 1, 4, 8, 3
FREEZE_BASE, FREEZE_MAX = 2, 12

N_ACTIONS = 18  # move{-1,0,1} x verb{none,jump,swingH,swingUp,swingDown,bunt}
OBS_DIM = 36
ACT_DIM = N_ACTIONS


@struct.dataclass
class EnvState:
    p: jnp.ndarray  # (2, 2) player positions
    vy: jnp.ndarray  # (2,) vertical velocities
    phase: jnp.ndarray  # (2,) int32
    timer: jnp.ndarray  # (2,) int32
    kind: jnp.ndarray  # (2,) int32 swing kind 0 h / 1 up / 2 down
    b: jnp.ndarray  # (2,) ball position
    bv: jnp.ndarray  # (2,) ball velocity (direction * speed)
    owner: jnp.ndarray  # scalar int32: -1 neutral, 0, 1
    freeze: jnp.ndarray  # scalar int32 hit-freeze ticks
    t: jnp.ndarray  # scalar int32


class LethalEnv:
    def __init__(self, reset_mode: str = "diverse"):
        assert reset_mode in ("fixed", "diverse")
        self.reset_mode = reset_mode

    # -------------------------------------------------------------- resets

    def reset_state(self, rng: jax.Array) -> EnvState:
        if self.reset_mode == "fixed":
            e = jax.random.uniform(rng, (2,), minval=-0.05, maxval=0.05)
            return EnvState(
                p=jnp.array([[-2.5, 0.0], [2.5, 0.0]]) + e[:, None] * 0,
                vy=jnp.zeros(2),
                phase=jnp.zeros(2, jnp.int32), timer=jnp.zeros(2, jnp.int32),
                kind=jnp.zeros(2, jnp.int32),
                b=jnp.array([0.0, 2.4]), bv=jnp.array([0.0, -BALL_V0]),
                owner=jnp.array(-1, jnp.int32), freeze=jnp.zeros((), jnp.int32),
                t=jnp.zeros((), jnp.int32),
            )
        kp, kv, kph, kt, kb, kbd, ks, ko = jax.random.split(rng, 8)
        p = jax.random.uniform(kp, (2, 2)) * jnp.array([2 * AX, AY - 0.4]) \
            - jnp.array([AX, 0.0])
        vy = 3.0 * jax.random.normal(kv, (2,))
        phase = jax.random.randint(kph, (2,), 0, 5)
        maxt = jnp.array([1, START_T, ACTIVE_T, REC_T, BUNT_T])
        timer = jax.random.randint(kt, (2,), 0, 8) % jnp.maximum(maxt[phase], 1)
        kind = jax.random.randint(kt, (2,), 0, 3)
        b = jax.random.uniform(kb, (2,)) * jnp.array([2 * AX - 0.4, AY - 0.4]) \
            + jnp.array([-AX + 0.2, 0.2])
        ang = jax.random.uniform(kbd, minval=-jnp.pi, maxval=jnp.pi)
        # log-uniform speed: slow rallies to near-cap chaos
        speed = jnp.exp(jax.random.uniform(
            ks, minval=jnp.log(BALL_V0), maxval=jnp.log(BALL_VMAX)))
        bv = speed * jnp.array([jnp.cos(ang), jnp.sin(ang)])
        owner = jax.random.randint(ko, (), -1, 2)
        # Don't spawn the ball already tagging someone: nudge it off players.
        for i in range(2):
            d = b - p[i]
            dist = jnp.linalg.norm(d) + 1e-6
            need = jnp.maximum((P_R + BALL_R + 0.05) - dist, 0.0)
            b = b + d / dist * need
        b = jnp.clip(b, jnp.array([-AX + BALL_R, BALL_R]),
                     jnp.array([AX - BALL_R, AY - BALL_R]))
        return EnvState(p=p, vy=vy, phase=phase.astype(jnp.int32),
                        timer=timer.astype(jnp.int32), kind=kind.astype(jnp.int32),
                        b=b, bv=bv, owner=owner.astype(jnp.int32),
                        freeze=jnp.zeros((), jnp.int32), t=jnp.zeros((), jnp.int32))

    def reset(self, rng):
        s = self.reset_state(rng)
        return s, self._obs(s)

    # ---------------------------------------------------------------- obs

    def _obs(self, s: EnvState) -> jax.Array:
        time_left = 1.0 - s.t.astype(jnp.float32) / EPISODE_LEN

        def one(i):
            j = 1 - i

            def player(k):
                return jnp.concatenate([
                    s.p[k] / jnp.array([AX, AY]),
                    jnp.array([s.vy[k] / 10.0, (s.p[k, 1] <= 0.0) * 1.0]),
                    jax.nn.one_hot(s.phase[k], 5),
                    jnp.array([s.timer[k] / 8.0]),
                    jax.nn.one_hot(s.kind[k], 3),
                ])  # 13

            own_ball = (s.owner == i) * 1.0
            their_ball = (s.owner == j) * 1.0
            neutral = (s.owner == -1) * 1.0
            ball = jnp.concatenate([
                (s.b - s.p[i]) / jnp.array([AX, AY]),
                s.bv / BALL_VMAX,
                jnp.array([jnp.linalg.norm(s.bv) / BALL_VMAX,
                           own_ball, their_ball, neutral,
                           s.freeze / FREEZE_MAX]),
            ])  # 9
            return jnp.concatenate([player(i), player(j), ball,
                                    jnp.array([time_left])])

        return jnp.stack([one(0), one(1)])

    # ---------------------------------------------------------------- step

    def step(self, rng, s: EnvState, action, reset_to: EnvState | None = None):
        """action: (2,) int32 in [0, 18)."""
        move = action % 3  # 0 left, 1 none, 2 right
        verb = action // 3  # 0 none 1 jump 2 swingH 3 swingUp 4 swingDown 5 bunt

        grounded = s.p[:, 1] <= 0.0
        can_act = s.phase == NEUTRAL
        do_jump = can_act & grounded & (verb == 1)
        do_swing = can_act & (verb >= 2) & (verb <= 4)
        do_bunt = can_act & (verb == 5)

        phase = jnp.where(do_swing, SW_START,
                          jnp.where(do_bunt, BUNT_ACT, s.phase))
        timer = jnp.where(do_swing, START_T,
                          jnp.where(do_bunt, BUNT_T, s.timer))
        kind = jnp.where(do_swing, verb - 2, s.kind)

        # Player kinematics.
        vx = (move.astype(jnp.float32) - 1.0) * RUN
        vy = jnp.where(do_jump, JUMP_V, s.vy) - GRAV * DT
        newp = s.p + jnp.stack([vx, vy], axis=-1) * DT
        on_floor = newp[:, 1] <= 0.0
        newp = newp.at[:, 1].set(jnp.maximum(newp[:, 1], 0.0))
        vy = jnp.where(on_floor, 0.0, vy)
        newp = newp.at[:, 0].set(jnp.clip(newp[:, 0], -AX + P_R, AX - P_R))
        newp = newp.at[:, 1].set(jnp.minimum(newp[:, 1], AY - 0.5))

        # Phase timers.
        timer = jnp.maximum(timer - 1, 0)
        expire = timer == 0
        nxt = jnp.array([NEUTRAL, SW_ACTIVE, SW_REC, NEUTRAL, SW_REC])
        nxt_t = jnp.array([0, ACTIVE_T, REC_T, 0, REC_T])
        phase2 = jnp.where(expire, nxt[phase], phase)
        timer = jnp.where(expire, nxt_t[phase], timer)
        phase = phase2

        # Ball flight with substeps; swings/bunts/tags resolved continuously.
        active = phase == SW_ACTIVE
        bunting = phase == BUNT_ACT
        frozen = s.freeze > 0
        b, bv, owner = s.b, s.bv, s.owner
        hit_by = jnp.array([False, False])
        tagged = jnp.array([False, False])
        clash = jnp.array(False)

        speed = jnp.linalg.norm(bv) + 1e-6
        sub_dt = DT / N_SUB
        for _ in range(N_SUB):
            b2 = jnp.where(frozen, b, b + bv * sub_dt)
            # wall reflections
            lo = jnp.array([-AX + BALL_R, BALL_R])
            hi = jnp.array([AX - BALL_R, AY - BALL_R])
            under = b2 < lo
            over = b2 > hi
            bv = jnp.where(under | over, -bv, bv)
            b2 = jnp.clip(b2, lo, hi)
            d = jnp.linalg.norm(b2 - newp, axis=-1)  # (2,)
            # swing connect (active swing & ball in reach & not already hit this tick)
            connect = active & (d < HIT_R) & ~hit_by & ~frozen
            both = connect.all()
            clash = clash | both
            # single hit: last writer wins deterministically via priority 0 then 1
            for i in range(2):
                only = connect[i] & ~both
                dirx = jnp.sign(newp[1 - i, 0] - newp[i, 0] + 1e-6)
                aim = jnp.stack([
                    jnp.where(kind[i] == 0, dirx, dirx * 0.7071),
                    jnp.where(kind[i] == 0, 0.0,
                              jnp.where(kind[i] == 1, 0.7071, -0.7071)),
                ])
                nspeed = jnp.minimum(speed * BALL_MULT, BALL_VMAX)
                bv = jnp.where(only, aim / (jnp.linalg.norm(aim) + 1e-6) * nspeed, bv)
                b2 = jnp.where(only, newp[i] + aim * (P_R + BALL_R + 0.08), b2)
                owner = jnp.where(only, i, owner)
                hit_by = hit_by.at[i].set(hit_by[i] | only)
            # clash: neutral pop straight up, speed kept
            bv = jnp.where(both, jnp.array([0.0, 1.0]) * speed, bv)
            owner = jnp.where(both, -1, owner)
            # bunt
            bunted = bunting & (d < HIT_R) & ~hit_by & ~frozen & ~connect.any()
            for i in range(2):
                bslow = jnp.maximum(speed * 0.35, BALL_V0)
                bdir = jnp.array([0.35 * jnp.sign(newp[1 - i, 0] - newp[i, 0] + 1e-6), 1.0])
                bv = jnp.where(bunted[i], bdir / jnp.linalg.norm(bdir) * bslow, bv)
                owner = jnp.where(bunted[i], -1, owner)
                hit_by = hit_by.at[i].set(hit_by[i] | bunted[i])
            # tags: ball harms non-owners (neutral ball harms no one)
            can_tag = (owner >= 0) & ~frozen
            for i in range(2):
                is_victim = can_tag & (owner != i) & (d[i] < P_R + BALL_R) & ~hit_by[i]
                tagged = tagged.at[i].set(tagged[i] | is_victim)
            b = b2
            speed = jnp.linalg.norm(bv) + 1e-6

        any_hit = hit_by.any() | clash
        freeze = jnp.where(
            any_hit,
            jnp.clip(FREEZE_BASE + (speed / BALL_VMAX * FREEZE_MAX).astype(jnp.int32),
                     0, FREEZE_MAX),
            jnp.maximum(s.freeze - 1, 0),
        )

        t = s.t + 1
        timeout = t >= EPISODE_LEN
        both_tagged = tagged.all()
        tagged = tagged & ~both_tagged  # simultaneous: draw
        done = tagged.any() | timeout | both_tagged
        reward = jnp.where(
            tagged.any(),
            jnp.array([
                tagged[1].astype(jnp.float32) - tagged[0].astype(jnp.float32),
                tagged[0].astype(jnp.float32) - tagged[1].astype(jnp.float32),
            ]),
            jnp.zeros(2),
        )

        new = EnvState(p=newp, vy=vy, phase=phase.astype(jnp.int32),
                       timer=timer.astype(jnp.int32), kind=kind.astype(jnp.int32),
                       b=b, bv=bv, owner=owner.astype(jnp.int32),
                       freeze=freeze.astype(jnp.int32), t=t)
        if reset_to is None:
            reset_to = self.reset_state(rng)
        s2 = jax.tree.map(lambda r, n: jnp.where(done, r, n), reset_to, new)
        obs = self._obs(s2)
        info = {
            "win0": (reward[0] > 0) & done,
            "win1": (reward[1] > 0) & done,
            "draw": done & (reward[0] == 0),
            "timeout": timeout & ~tagged.any() & ~both_tagged,
            "ep_len": jnp.where(done, t, 0),
            "kills": tagged.sum(),
            "ball_speed": speed,
        }
        return s2, obs, reward, done, info
