"""Nidhogg-style fencing duel: three stances, one-hit kills, tug-of-war.

A faithful demake of Nidhogg's core loop (established mechanics, not ours):
  - Sword at high/mid/low. A stab at a height the defender is NOT guarding
    kills; a stab into a matching guard is PARRIED (attacker staggered).
  - Stabs have startup/active/recovery — whiffing leaves you punishable.
  - One hit = death. The killer earns the right to RUN toward their exit;
    the victim respawns shortly, dropped in ahead of the runner to block.
  - Standing bodies block; jumping crosses over an opponent.
  - You win ONLY by reaching your exit end of the multi-screen strip.

Reward is purely sparse and zero-sum: +1 escape / -1 opponent escaped,
0 timeout. Kills themselves pay nothing — they are instrumental.

Player 0's exit is +X_MAX, player 1's is -X_MAX. Observations are
egocentric-mirrored so one shared policy plays both sides identically.
All-discrete state: every random reset (any positions, stances, phases,
timers) is a valid game state by construction.
"""

import jax
import jax.numpy as jnp
from flax import struct

DT = 1.0 / 15.0
EPISODE_LEN = 900  # 60 seconds
X_MAX = 10.0  # exit lines at +-X_MAX
WALK = 3.2  # units/s
BACK_WALK = 2.4
REACH = 1.05  # stab reach
BLOCK_DIST = 0.55  # bodies closer than this can't pass on the ground
PUSHBACK = 0.55  # on parry / clash
JUMP_TICKS = 8
JUMP_VX = 3.8
JUMP_H = 1.3

# phase enum
NEUTRAL, STARTUP, ACTIVE, RECOVER, STAGGER, DEAD = 0, 1, 2, 3, 4, 5
STARTUP_T, ACTIVE_T, RECOVER_T, STAGGER_T, DEAD_T = 2, 2, 6, 5, 12
RESPAWN_AHEAD = 3.0  # victim drops in this far ahead of the killer

N_ACTIONS = 27  # move{none,fwd,back} x stance{high,mid,low} x verb{none,stab,jump}
OBS_DIM = 26
ACT_DIM = N_ACTIONS


@struct.dataclass
class EnvState:
    x: jnp.ndarray  # (2,)
    air: jnp.ndarray  # (2,) int32 ticks remaining airborne (0 = grounded)
    stance: jnp.ndarray  # (2,) int32 0 high 1 mid 2 low
    phase: jnp.ndarray  # (2,) int32
    timer: jnp.ndarray  # (2,) int32 ticks left in phase
    t: jnp.ndarray  # scalar int32


DIRS = jnp.array([1.0, -1.0])  # each player's exit direction


class NidhoggEnv:
    def __init__(self, reset_mode: str = "diverse"):
        assert reset_mode in ("fixed", "diverse")
        self.reset_mode = reset_mode

    # -------------------------------------------------------------- resets

    def reset_state(self, rng: jax.Array) -> EnvState:
        if self.reset_mode == "fixed":
            e = jax.random.uniform(rng, (2,), minval=-0.1, maxval=0.1)
            return EnvState(
                x=jnp.array([-1.2, 1.2]) + e,
                air=jnp.zeros(2, jnp.int32),
                stance=jnp.ones(2, jnp.int32),
                phase=jnp.zeros(2, jnp.int32),
                timer=jnp.zeros(2, jnp.int32),
                t=jnp.zeros((), jnp.int32),
            )
        kx, ks, kp, kt, ka = jax.random.split(rng, 5)
        x = jnp.sort(jax.random.uniform(kx, (2,), minval=-X_MAX + 0.5,
                                        maxval=X_MAX - 0.5))
        # sorted then randomly swapped keeps arbitrary relative order
        swap = jax.random.bernoulli(ka)
        x = jnp.where(swap, x[::-1], x)
        stance = jax.random.randint(ks, (2,), 0, 3)
        phase = jax.random.randint(kp, (2,), 0, 6)
        maxt = jnp.array([1, STARTUP_T, ACTIVE_T, RECOVER_T, STAGGER_T, DEAD_T])
        timer = jax.random.randint(kt, (2,), 0, 8) % jnp.maximum(maxt[phase], 1)
        air = jnp.where(
            jax.random.bernoulli(ka, 0.15, (2,)),
            jax.random.randint(kt, (2,), 1, JUMP_TICKS), 0
        ).astype(jnp.int32)
        air = jnp.where(phase == DEAD, 0, air)  # dead fighters aren't airborne
        return EnvState(x=x, air=air, stance=stance, phase=phase.astype(jnp.int32),
                        timer=timer.astype(jnp.int32), t=jnp.zeros((), jnp.int32))

    def reset(self, rng):
        s = self.reset_state(rng)
        return s, self._obs(s)

    # ----------------------------------------------------------------- obs

    def _obs(self, s: EnvState) -> jax.Array:
        time_left = 1.0 - s.t.astype(jnp.float32) / EPISODE_LEN

        def one(i):
            j = 1 - i
            d = DIRS[i]  # mirror so "my exit" is always +x

            def fighter(k):
                return jnp.concatenate([
                    jnp.array([d * s.x[k] / X_MAX,
                               s.air[k] / JUMP_TICKS]),
                    jax.nn.one_hot(s.stance[k], 3),
                    jax.nn.one_hot(s.phase[k], 6),
                    jnp.array([s.timer[k] / 12.0]),
                ])  # 12

            rel = jnp.array([d * (s.x[j] - s.x[i]) / X_MAX])
            return jnp.concatenate([fighter(i), fighter(j), rel,
                                    jnp.array([time_left])])  # 12+12+1+1+1...

        return jnp.stack([one(0), one(1)])

    # ---------------------------------------------------------------- step

    def step(self, rng, s: EnvState, action, reset_to: EnvState | None = None):
        """action: (2,) int32 in [0, 27)."""
        move = action % 3          # 0 none, 1 toward exit, 2 away
        stance_a = (action // 3) % 3
        verb = action // 9         # 0 none, 1 stab, 2 jump

        dead = s.phase == DEAD
        can_act = (s.phase == NEUTRAL) & ~dead
        grounded = s.air == 0

        # Stance: freely adjustable while alive & not mid-move commitment.
        stance = jnp.where(can_act, stance_a, s.stance)

        # Start actions.
        do_stab = can_act & grounded & (verb == 1)
        do_jump = can_act & grounded & (verb == 2)
        phase = jnp.where(do_stab, STARTUP, s.phase)
        timer = jnp.where(do_stab, STARTUP_T, s.timer)
        air = jnp.where(do_jump, JUMP_TICKS, s.air)

        # Movement.
        mdir = jnp.where(move == 1, DIRS, jnp.where(move == 2, -DIRS, 0.0))
        speed = jnp.where(move == 1, WALK, BACK_WALK)
        vx = jnp.where(can_act | (s.phase == RECOVER), mdir * speed, 0.0)
        vx = jnp.where(air > 0, jnp.sign(vx) * JUMP_VX, vx)  # committed air speed
        vx = jnp.where(dead, 0.0, vx)
        newx = s.x + vx * DT

        # Body blocking: grounded live fighters can't pass through each other.
        both_ground = (air == 0).all() & (~dead).all()
        gap = newx[1] - newx[0]
        orig_sign = jnp.sign(s.x[1] - s.x[0] + 1e-6)
        crossed = jnp.sign(gap + 1e-6) != orig_sign
        too_close = jnp.abs(gap) < BLOCK_DIST
        violate = both_ground & (crossed | too_close)
        mid = 0.5 * (newx[0] + newx[1])
        blocked = jnp.stack([mid - 0.5 * BLOCK_DIST * orig_sign,
                             mid + 0.5 * BLOCK_DIST * orig_sign])
        newx = jnp.where(violate, blocked, newx)
        newx = jnp.clip(newx, -X_MAX - 0.2, X_MAX + 0.2)

        # Tick phase timers.
        timer = jnp.maximum(timer - 1, 0)
        expire = timer == 0
        nxt = jnp.array([NEUTRAL, ACTIVE, RECOVER, NEUTRAL, NEUTRAL, NEUTRAL])
        nxt_t = jnp.array([0, ACTIVE_T, RECOVER_T, 0, 0, 0])
        phase2 = jnp.where(expire, nxt[phase], phase)
        timer = jnp.where(expire, nxt_t[phase], timer)
        phase = phase2
        air = jnp.maximum(air - 1, 0)

        # Combat: active blades touching the opponent.
        heights = jnp.array([1.4, 1.0, 0.6])
        y = jnp.where(air > 0, JUMP_H * jnp.sin(jnp.pi * (JUMP_TICKS - air) / JUMP_TICKS), 0.0)
        attacking = phase == ACTIVE
        dist = jnp.abs(newx[1] - newx[0])
        tip_h = heights[stance] + y  # attack height in world
        body_lo, body_hi = y + 0.2, y + 1.6  # vulnerable band
        opp = jnp.array([1, 0])
        in_range = dist < REACH
        hits_body = (tip_h > body_lo[opp]) & (tip_h < body_hi[opp])
        target_live = (phase[opp] != DEAD)
        landed = attacking & in_range & hits_body & target_live
        # Parry: defender grounded, neutral-ish, guarding the same height.
        guard = (~attacking[opp]) & (phase[opp] == NEUTRAL) & (air[opp] == 0) \
            & (stance[opp] == stance)
        parried = landed & guard
        killed_opp = landed & ~guard
        clash = attacking[0] & attacking[1] & in_range & (stance[0] == stance[1])
        killed_opp = killed_opp & ~clash
        parried = parried | (attacking & clash)

        # Apply parry stagger + pushback.
        phase = jnp.where(parried, STAGGER, phase)
        timer = jnp.where(parried, STAGGER_T, timer)
        newx = newx - jnp.where(parried, DIRS * 0.0, 0.0)  # placeholder keeps shape
        push = jnp.sign(newx - newx[opp] + 1e-6) * PUSHBACK
        newx = jnp.where(parried, newx + push, newx)

        # Deaths: victim enters DEAD, is teleported to a blocking respawn
        # position ahead of the killer, frozen for DEAD_T.
        victim = killed_opp[opp]  # victim[i]: player i was killed
        both_killed = victim.all()
        victim = victim & ~both_killed  # trade = both parry-priced, no deaths
        killer_x = newx[opp]
        spawn = jnp.clip(killer_x + DIRS[opp] * RESPAWN_AHEAD,
                         -X_MAX + 0.4, X_MAX - 0.4)
        newx = jnp.where(victim, spawn, newx)
        phase = jnp.where(victim, DEAD, phase)
        timer = jnp.where(victim, DEAD_T, timer)
        air = jnp.where(victim, 0, air)

        # Win: cross your exit line (alive, grounded or not).
        escaped = jnp.stack([newx[0] >= X_MAX, newx[1] <= -X_MAX]) & (phase != DEAD)
        t = s.t + 1
        timeout = t >= EPISODE_LEN
        done = escaped.any() | timeout
        reward = jnp.where(
            escaped.any(),
            jnp.array([
                escaped[0].astype(jnp.float32) - escaped[1].astype(jnp.float32),
                escaped[1].astype(jnp.float32) - escaped[0].astype(jnp.float32),
            ]),
            jnp.zeros(2),
        )

        new = EnvState(x=newx, air=air.astype(jnp.int32), stance=stance,
                       phase=phase.astype(jnp.int32), timer=timer.astype(jnp.int32), t=t)
        if reset_to is None:
            reset_to = self.reset_state(rng)
        s2 = jax.tree.map(lambda r, n: jnp.where(done, r, n), reset_to, new)
        obs = self._obs(s2)
        info = {
            "win0": (reward[0] > 0) & done,
            "win1": (reward[1] > 0) & done,
            "draw": done & (reward[0] == 0),
            "timeout": timeout & ~escaped.any(),
            "ep_len": jnp.where(done, t, 0),
            "kills": victim.sum(),
        }
        return s2, obs, reward, done, info
