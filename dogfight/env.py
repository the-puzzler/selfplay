"""Space dogfight: two ships, discrete actions, purely sparse reward.

Pure-JAX physics (no MuJoCo): rotate/thrust/fire in a walled 2D arena.
You get +1 for hitting the opponent with a bullet, -1 for being hit,
0 on timeout. Nothing else — no aim bonuses, no distance shaping.

Resets ("diverse") sample any valid game state: random positions,
headings, velocities, cooldowns, and live bullets already in flight —
so the curriculum includes point-blank kills, incoming-bullet dodges,
and neutral standoffs. "fixed" is the classic duel start: opposite
sides, facing each other, guns cold.

All functions are single-env and pure; vmap for batches.
Action space (per ship): 12 = rotate {none,left,right} x thrust {off,on}
x fire {no,yes}.
"""

import jax
import jax.numpy as jnp
from flax import struct

DT = 0.1
EPISODE_LEN = 240  # 24 seconds
ARENA = 1.0  # half-width; arena is [-1, 1]^2
SHIP_R = 0.045
ROT_RATE = 3.5  # rad/s
ACCEL = 1.4
DRAG = 0.25  # per second
MUZZLE = 1.5  # bullet speed relative to ship
N_BULLETS = 3  # live bullets per ship
BULLET_TTL = 14  # steps (~1.4 s)
COOLDOWN = 5  # steps between shots
HIT_R = 0.055  # bullet-to-ship kill distance
WALL_BOUNCE = 0.7  # velocity retained on wall hit

N_ACTIONS = 12
OBS_DIM = 45
ACT_DIM = N_ACTIONS


@struct.dataclass
class EnvState:
    pos: jnp.ndarray  # (2, 2) ship positions
    vel: jnp.ndarray  # (2, 2)
    th: jnp.ndarray  # (2,) headings
    cd: jnp.ndarray  # (2,) fire cooldown, int32 steps
    bpos: jnp.ndarray  # (2, N_BULLETS, 2) bullet positions per owner
    bvel: jnp.ndarray  # (2, N_BULLETS, 2)
    bttl: jnp.ndarray  # (2, N_BULLETS) int32; >0 means live
    t: jnp.ndarray  # scalar int32


def _fresh(pos, vel, th, cd, bpos, bvel, bttl):
    return EnvState(pos=pos, vel=vel, th=th, cd=cd, bpos=bpos, bvel=bvel,
                    bttl=bttl, t=jnp.zeros((), jnp.int32))


class DogfightEnv:
    def __init__(self, reset_mode: str = "diverse"):
        assert reset_mode in ("fixed", "diverse")
        self.reset_mode = reset_mode

    # ---------------------------------------------------------------- resets

    def _fixed_state(self, rng: jax.Array) -> EnvState:
        eps = 0.01 * jax.random.normal(rng, (2, 2))
        pos = jnp.array([[-0.7, 0.0], [0.7, 0.0]]) + eps
        th = jnp.array([0.0, jnp.pi])  # facing each other
        return _fresh(pos, jnp.zeros((2, 2)), th, jnp.zeros(2, jnp.int32),
                      jnp.zeros((2, N_BULLETS, 2)), jnp.zeros((2, N_BULLETS, 2)),
                      jnp.zeros((2, N_BULLETS), jnp.int32))

    def _diverse_state(self, rng: jax.Array) -> EnvState:
        kp, kv, kth, kcd, kb, kbv, kbt, ks = jax.random.split(rng, 8)
        # Ship positions: anywhere, not overlapping (resample-free: push apart).
        pos = jax.random.uniform(kp, (2, 2), minval=-0.92, maxval=0.92)
        d = pos[1] - pos[0]
        dist = jnp.linalg.norm(d) + 1e-6
        need = jnp.maximum(2.5 * SHIP_R - dist, 0.0)
        shift = 0.5 * need * d / dist
        pos = jnp.clip(pos.at[0].add(-shift).at[1].add(shift), -0.95, 0.95)
        vel_scale = jax.random.uniform(ks)
        vel = vel_scale * 0.8 * jax.random.normal(kv, (2, 2))
        th = jax.random.uniform(kth, (2,), minval=-jnp.pi, maxval=jnp.pi)
        cd = jax.random.randint(kcd, (2,), 0, COOLDOWN + 1)
        # Live bullets already in flight (any number 0..N per ship).
        bpos = jax.random.uniform(kb, (2, N_BULLETS, 2), minval=-0.95, maxval=0.95)
        bth = jax.random.uniform(kbv, (2, N_BULLETS), minval=-jnp.pi, maxval=jnp.pi)
        bvel = MUZZLE * jnp.stack([jnp.cos(bth), jnp.sin(bth)], axis=-1)
        bttl = jax.random.randint(kbt, (2, N_BULLETS), -BULLET_TTL, BULLET_TTL + 1)
        bttl = jnp.maximum(bttl, 0)  # ~half the slots empty on average
        # A bullet spawned on top of its target would end the game at t=0;
        # kill those so every episode is playable.
        tgt = pos[jnp.array([1, 0])]  # target of owner i is ship 1-i
        close = jnp.linalg.norm(bpos - tgt[:, None, :], axis=-1) < 2.0 * HIT_R
        bttl = jnp.where(close, 0, bttl)
        return _fresh(pos, vel, th, cd, bpos, bvel, bttl.astype(jnp.int32))

    def reset_state(self, rng: jax.Array) -> EnvState:
        if self.reset_mode == "fixed":
            return self._fixed_state(rng)
        return self._diverse_state(rng)

    def reset(self, rng: jax.Array) -> tuple[EnvState, jax.Array]:
        state = self.reset_state(rng)
        return state, self._obs(state)

    # ------------------------------------------------------------------ obs

    def _obs(self, s: EnvState) -> jax.Array:
        """Egocentric observations for both ships, (2, OBS_DIM)."""
        time_left = 1.0 - s.t.astype(jnp.float32) / EPISODE_LEN

        def one(i):
            j = 1 - i
            own = jnp.concatenate([
                s.pos[i], s.vel[i],
                jnp.array([jnp.cos(s.th[i]), jnp.sin(s.th[i]),
                           s.cd[i] / COOLDOWN]),
            ])  # 7
            opp = jnp.concatenate([
                s.pos[j] - s.pos[i], s.vel[j],
                jnp.array([jnp.cos(s.th[j]), jnp.sin(s.th[j]),
                           s.cd[j] / COOLDOWN]),
            ])  # 7
            def bullets(owner):
                alive = (s.bttl[owner] > 0).astype(jnp.float32)[:, None]
                rel = (s.bpos[owner] - s.pos[i]) * alive
                bv = s.bvel[owner] * alive
                return jnp.concatenate([rel, bv, alive], axis=-1).reshape(-1)  # 15
            return jnp.concatenate([own, opp, bullets(j), bullets(i),
                                    jnp.array([time_left])])

        return jnp.stack([one(0), one(1)])

    # ----------------------------------------------------------------- step

    def step(
        self,
        rng: jax.Array,
        state: EnvState,
        action: jax.Array,  # (2,) int32 in [0, 12)
        reset_to: EnvState | None = None,
    ) -> tuple[EnvState, jax.Array, jax.Array, jax.Array, dict]:
        rot = action % 3  # 0 none, 1 left, 2 right
        thrust = (action // 3) % 2
        fire = action // 6

        th = state.th + jnp.where(rot == 1, 1.0, jnp.where(rot == 2, -1.0, 0.0)) * ROT_RATE * DT
        dirv = jnp.stack([jnp.cos(th), jnp.sin(th)], axis=-1)
        vel = state.vel + thrust[:, None] * ACCEL * DT * dirv
        vel = vel * (1.0 - DRAG * DT)
        pos = state.pos + vel * DT
        # Wall bounce.
        hit_wall = jnp.abs(pos) > ARENA - SHIP_R
        vel = jnp.where(hit_wall, -WALL_BOUNCE * vel, vel)
        pos = jnp.clip(pos, -(ARENA - SHIP_R), ARENA - SHIP_R)

        # Bullets fly; die on walls or ttl.
        bpos = state.bpos + state.bvel * DT
        bttl = jnp.maximum(state.bttl - 1, 0)
        bttl = jnp.where(jnp.any(jnp.abs(bpos) > ARENA, axis=-1), 0, bttl)

        # Fire: needs cooldown 0 and a free slot.
        cd = jnp.maximum(state.cd - 1, 0)
        slot = jnp.argmin(bttl, axis=1)  # a dead slot if one exists
        slot_free = jnp.take_along_axis(bttl, slot[:, None], axis=1)[:, 0] == 0
        do_fire = (fire == 1) & (cd == 0) & slot_free
        nose = pos + (SHIP_R + 0.02) * dirv
        bvel_new = vel + MUZZLE * dirv
        owner = jnp.arange(2)
        bpos = bpos.at[owner, slot].set(
            jnp.where(do_fire[:, None], nose, bpos[owner, slot]))
        bvel = state.bvel.at[owner, slot].set(
            jnp.where(do_fire[:, None], bvel_new, state.bvel[owner, slot]))
        bttl = bttl.at[owner, slot].set(
            jnp.where(do_fire, BULLET_TTL, bttl[owner, slot]))
        cd = jnp.where(do_fire, COOLDOWN, cd)

        # Hits: owner i's live bullets vs ship 1-i.
        tgt = pos[jnp.array([1, 0])]
        dist = jnp.linalg.norm(bpos - tgt[:, None, :], axis=-1)
        hit_by = jnp.any((dist < HIT_R) & (bttl > 0), axis=1)  # [hit on 1, hit on 0]
        hit = hit_by[jnp.array([1, 0])]  # hit[i] = ship i was hit
        t = state.t + 1
        timeout = t >= EPISODE_LEN
        done = hit.any() | timeout

        reward = jnp.where(
            hit.any(),
            jnp.array([
                hit[1].astype(jnp.float32) - hit[0].astype(jnp.float32),
                hit[0].astype(jnp.float32) - hit[1].astype(jnp.float32),
            ]),
            jnp.zeros(2),
        )

        new = EnvState(pos=pos, vel=vel, th=th, cd=cd, bpos=bpos, bvel=bvel,
                       bttl=bttl, t=t)
        if reset_to is None:
            reset_to = self.reset_state(rng)
        state = jax.tree.map(lambda r, n: jnp.where(done, r, n), reset_to, new)
        obs = self._obs(state)
        info = {
            "win0": (reward[0] > 0) & done,
            "win1": (reward[1] > 0) & done,
            "draw": done & (reward[0] == 0),
            "timeout": timeout & ~hit.any(),
            "ep_len": jnp.where(done, t, 0),
        }
        return state, obs, reward, done, info
