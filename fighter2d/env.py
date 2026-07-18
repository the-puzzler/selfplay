"""MJX two-fighter env with sparse win/loss reward and OmniReset-style resets.

Reward is purely sparse and zero-sum: +1 to the winner, -1 to the loser at
episode end (a fighter loses by falling — torso below a height threshold —
or leaving the arena), 0 for draws/timeouts, 0 everywhere else.

Reset modes:
  "fixed":   both fighters standing at canonical positions (tiny noise).
  "diverse": OmniReset-style procedural diversity — each fighter is
             independently either "standing-ish" or fully randomized
             (arbitrary root height/orientation, joint angles anywhere in
             their ranges, random velocities, random arena positions,
             possibly overlapping the opponent). Easy states (opponent
             already nearly fallen) and hard states (self nearly fallen)
             both occur, forming an implicit curriculum.

All functions are single-env and pure; vmap them for batched training.
"""

from functools import partial

import jax
import jax.numpy as jnp
import mujoco
from flax import struct
from mujoco import mjx

from fighter2d import model as fmodel

N_FRAMES = 5  # control dt = 5 * 0.008 = 0.04s
EPISODE_LEN = 300  # 12 seconds
DOWN_Z = 0.5  # torso center below this => downed
NQ = fmodel.NQ_PER_FIGHTER  # 13 per fighter
NU = fmodel.NU_PER_FIGHTER  # 10 per fighter
OBS_DIM = 55
ACT_DIM = NU  # per agent


@struct.dataclass
class EnvState:
    data: mjx.Data
    t: jnp.ndarray  # scalar int32 step counter


class FighterEnv:
    def __init__(self, reset_mode: str = "diverse"):
        assert reset_mode in ("fixed", "diverse")
        self.reset_mode = reset_mode
        self.mj_model = mujoco.MjModel.from_xml_string(fmodel.build_xml())
        self.model = mjx.put_model(self.mj_model)
        self._template = mjx.make_data(self.mj_model)
        self.init_x = jnp.array(fmodel.INIT_X)  # absolute x of each root body
        self.base_z = fmodel.TORSO_INIT_Z
        self.arena_half = fmodel.ARENA_HALF
        # Limb joint ranges, one fighter's worth: qpos indices 3..12.
        jr = self.mj_model.jnt_range.copy()  # (njnt, 2), joint order == qpos order
        self.limb_lo = jnp.array(jr[3 : NQ, 0])
        self.limb_hi = jnp.array(jr[3 : NQ, 1])

    # ---------------------------------------------------------------- resets

    def _random_qpos(self, rng: jax.Array, fi: int) -> tuple[jax.Array, jax.Array]:
        """Fully randomized configuration for fighter fi: anywhere in the
        arena, any height from lying on the floor to mid-air, any root
        orientation, joints anywhere in their ranges, velocity intensity
        from stillness to violent tumbling. No curated sub-distributions."""
        kx, kz, ka, kj, ks, kv1, kv2 = jax.random.split(rng, 7)
        x_abs = jax.random.uniform(kx, minval=-2.4, maxval=2.4)
        z_abs = jax.random.uniform(kz, minval=0.15, maxval=1.8)
        angle = jax.random.uniform(ka, minval=-jnp.pi, maxval=jnp.pi)
        limbs = jax.random.uniform(kj, (NQ - 3,), minval=self.limb_lo, maxval=self.limb_hi)
        qpos = jnp.concatenate(
            [
                jnp.array([x_abs - self.init_x[fi], z_abs - self.base_z, angle]),
                limbs,
            ]
        )
        vel_scale = jax.random.uniform(ks)
        qvel = vel_scale * jnp.concatenate(
            [2.0 * jax.random.normal(kv1, (3,)), 4.0 * jax.random.normal(kv2, (NQ - 3,))]
        )
        return qpos, qvel

    def _fighter_reset(self, rng: jax.Array, fi: int) -> tuple[jax.Array, jax.Array]:
        k = jax.random.fold_in(rng, fi)
        if self.reset_mode == "fixed":
            qpos = 0.01 * jax.random.normal(k, (NQ,))
            return qpos, jnp.zeros(NQ)
        return self._random_qpos(k, fi)

    MIN_SEPARATION = 0.4  # torso centers; close enough for limb contact, no interpenetration

    def reset_qpos_qvel(self, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        r0, r1 = jax.random.split(rng)
        q0, v0 = self._fighter_reset(r0, 0)
        q1, v1 = self._fighter_reset(r1, 1)
        # If torsos spawned closer than MIN_SEPARATION, respread them around
        # their (arena-clamped) midpoint; otherwise leave positions untouched.
        x0 = self.init_x[0] + q0[0]
        x1 = self.init_x[1] + q1[0]
        d = x1 - x0
        s = jnp.where(d >= 0, 1.0, -1.0)
        too_close = jnp.abs(d) < self.MIN_SEPARATION
        c = jnp.clip(0.5 * (x0 + x1), -2.2, 2.2)
        half = 0.5 * self.MIN_SEPARATION
        new_x0 = jnp.where(too_close, c - s * half, x0)
        new_x1 = jnp.where(too_close, c + s * half, x1)
        q0 = q0.at[0].set(new_x0 - self.init_x[0])
        q1 = q1.at[0].set(new_x1 - self.init_x[1])
        return jnp.concatenate([q0, q1]), jnp.concatenate([v0, v1])

    def reset(self, rng: jax.Array) -> tuple[EnvState, jax.Array]:
        qpos, qvel = self.reset_qpos_qvel(rng)
        data = self._template.replace(qpos=qpos, qvel=qvel)
        state = EnvState(data=data, t=jnp.zeros((), jnp.int32))
        return state, self._obs(qpos, qvel, state.t)

    # ------------------------------------------------------------------ obs

    def _fighter_obs(self, qpos, qvel, fi: int, other_x: jax.Array) -> jax.Array:
        q = jax.lax.dynamic_slice_in_dim(qpos, fi * NQ, NQ)
        v = jax.lax.dynamic_slice_in_dim(qvel, fi * NQ, NQ)
        x_abs = self.init_x[fi] + q[0]
        z = self.base_z + q[1]
        return jnp.concatenate(
            [
                jnp.array([x_abs / self.arena_half, z, jnp.cos(q[2]), jnp.sin(q[2])]),
                q[3:],
                v * 0.1,
            ]
        )  # 4 + 10 + 13 = 27

    def _obs(self, qpos, qvel, t) -> jax.Array:
        """Observations for both agents, shape (2, OBS_DIM)."""
        x0 = self.init_x[0] + qpos[0]
        x1 = self.init_x[1] + qpos[NQ]
        o0 = self._fighter_obs(qpos, qvel, 0, x1)
        o1 = self._fighter_obs(qpos, qvel, 1, x0)
        time_left = 1.0 - t.astype(jnp.float32) / EPISODE_LEN
        # Each agent: [self obs (27), opponent obs with dx instead of abs x (27), time (1)]
        def pack(self_o, opp_o, self_x, opp_x):
            opp_rel = opp_o.at[0].set((opp_x - self_x) / self.arena_half)
            return jnp.concatenate([self_o, opp_rel, jnp.array([time_left])])

        return jnp.stack([pack(o0, o1, x0, x1), pack(o1, o0, x1, x0)])

    # ----------------------------------------------------------------- step

    def step(
        self, rng: jax.Array, state: EnvState, action: jax.Array
    ) -> tuple[EnvState, jax.Array, jax.Array, jax.Array, dict]:
        """action: (2, NU) in [-1, 1]. Returns (state', obs, reward(2,), done, info)."""
        ctrl = jnp.clip(action, -1.0, 1.0).reshape(-1)
        data = state.data.replace(ctrl=ctrl)
        for _ in range(N_FRAMES):
            data = mjx.step(self.model, data)
        t = state.t + 1

        qpos = data.qpos
        x = self.init_x + jnp.array([qpos[0], qpos[NQ]])
        z = self.base_z + jnp.array([qpos[1], qpos[NQ + 1]])
        downed = z < DOWN_Z
        out = jnp.abs(x) > self.arena_half
        lost = downed | out  # (2,)
        timeout = t >= EPISODE_LEN
        done = lost.any() | timeout

        # +1 if opponent lost and you didn't; -1 mirrored; 0 draw/timeout/mid-episode.
        reward = jnp.where(
            lost.any(),
            jnp.array(
                [
                    lost[1].astype(jnp.float32) - lost[0].astype(jnp.float32),
                    lost[0].astype(jnp.float32) - lost[1].astype(jnp.float32),
                ]
            ),
            jnp.zeros(2),
        )

        # Auto-reset on done.
        reset_qpos, reset_qvel = self.reset_qpos_qvel(rng)
        new_qpos = jnp.where(done, reset_qpos, data.qpos)
        new_qvel = jnp.where(done, reset_qvel, data.qvel)
        new_data = data.replace(
            qpos=new_qpos,
            qvel=new_qvel,
            qacc_warmstart=jnp.where(done, 0.0, data.qacc_warmstart),
        )
        new_t = jnp.where(done, 0, t)
        obs = self._obs(new_qpos, new_qvel, new_t)

        info = {
            "win0": (reward[0] > 0) & done,
            "win1": (reward[1] > 0) & done,
            "draw": done & (reward[0] == 0),
            "timeout": timeout & ~lost.any(),
            "ep_len": jnp.where(done, t, 0),
        }
        return EnvState(data=new_data, t=new_t), obs, reward, done, info
