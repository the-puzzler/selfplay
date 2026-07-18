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
# Torso center below this => downed. 0.25 was tried and produced a degenerate
# equilibrium: fighters dive into a passively-stable low brace and camp for
# the timeout draw. 0.5 makes floor-hugging a loss, forcing upright play.
DOWN_Z = 0.5
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
        # Upright-ish spawns: lean up to ~34 degrees, never sideways/inverted
        # (deliberate compute-focusing constraint for CPU-scale runs).
        angle = jax.random.uniform(ka, minval=-0.6, maxval=0.6)
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

    SPAWN_GAP = 0.02  # clearance between fighters' reach intervals and above floor

    def _fighter_frame(self, q: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Forward kinematics of one fighter's capsule endpoints, relative to
        the root. Returns (dx, dz, radius) arrays. Planar chain: a body-frame
        offset (bx, bz) rotated about +y by angle a maps to
        (bx*cos a + bz*sin a, -bx*sin a + bz*cos a)."""
        th = q[2]

        def rot(a, bx, bz):
            return bx * jnp.cos(a) + bz * jnp.sin(a), -bx * jnp.sin(a) + bz * jnp.cos(a)

        pts = []  # (dx, dz, r)
        for bz, r in ((0.2, 0.07), (-0.2, 0.07), (0.32, 0.09)):  # torso ends, head
            dx, dz = rot(th, 0.0, bz)
            pts.append((jnp.full(1, dx), jnp.full(1, dz), jnp.full(1, r)))

        def chain(base_bz, offsets_r):
            """offsets_r: [(angles, bx, bz, radius), ...] cumulative chain for
            both limbs of a pair at once (angles shape (2,))."""
            bx0, bz0 = rot(th, 0.0, base_bz)
            pos = (jnp.full(2, bx0), jnp.full(2, bz0))
            acc = th
            for ang, bx, bz, r in offsets_r:
                acc = acc + ang
                dx, dz = rot(acc, bx, bz)
                pos = (pos[0] + dx, pos[1] + dz)
                pts.append((pos[0], pos[1], jnp.full(2, r)))
            return pos

        hips, knees, ankles = q[jnp.array([3, 6])], q[jnp.array([4, 7])], q[jnp.array([5, 8])]
        # Feet: two endpoints; append the -x end manually after the chain.
        foot_pos = chain(-0.2, [
            (hips, 0.0, -0.45, 0.05),   # knee (thigh end)
            (knees, 0.0, -0.5, 0.045),  # ankle
            (ankles, 0.1, 0.0, 0.045),  # foot +x end
        ])
        foot_ang = th + hips + knees + ankles
        bdx, bdz = rot(foot_ang, -0.2, 0.0)  # from +x end back to -x end
        pts.append((foot_pos[0] + bdx, foot_pos[1] + bdz, jnp.full(2, 0.045)))

        shoulders, elbows = q[jnp.array([9, 11])], q[jnp.array([10, 12])]
        chain(0.15, [
            (shoulders, 0.0, -0.3, 0.04),  # elbow
            (elbows, 0.0, -0.28, 0.035),   # wrist
        ])

        dx = jnp.concatenate([p[0] for p in pts])
        dz = jnp.concatenate([p[1] for p in pts])
        r = jnp.concatenate([p[2] for p in pts])
        return dx, dz, r

    def reset_qpos_qvel(self, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        r0, r1, rplace = jax.random.split(rng, 3)
        q0, v0 = self._fighter_reset(r0, 0)
        q1, v1 = self._fighter_reset(r1, 1)

        dx0, dz0, rad0 = self._fighter_frame(q0)
        dx1, dz1, rad1 = self._fighter_frame(q1)

        # Lift each fighter so no limb starts below the floor.
        for i, (q, dz, rad) in enumerate(((q0, dz0, rad0), (q1, dz1, rad1))):
            min_dz = jnp.min(dz - rad)  # lowest point relative to root
            z_abs = self.base_z + q[1]
            z_abs = jnp.maximum(z_abs, self.SPAWN_GAP - min_dz)
            # Never spawn at game over: torso starts above the knockdown line
            # (the fighter may still be doomed, but the episode is playable).
            z_abs = jnp.maximum(z_abs, DOWN_Z + self.SPAWN_GAP)
            q = q.at[1].set(z_abs - self.base_z)
            if i == 0:
                q0 = q
            else:
                q1 = q

        if self.reset_mode == "diverse":
            # Place the pair with random separation (never overlapping),
            # random arena position, random side assignment. The minimum
            # separation is *directional*: the inner fighter's reach toward
            # the opponent, not the max reach in both directions — so poses
            # with limbs tucked/pointing away can spawn nearly torso-to-torso.
            right_reach = jnp.array([jnp.max(dx0 + rad0), jnp.max(dx1 + rad1)])
            left_reach = jnp.array([jnp.max(-dx0 + rad0), jnp.max(-dx1 + rad1)])
            ksep, kc, kside = jax.random.split(rplace, 3)
            side = jnp.where(jax.random.bernoulli(kside), 1.0, -1.0)
            # side=+1: fighter0 left of fighter1; side=-1: swapped.
            min_sep = jnp.where(
                side > 0,
                right_reach[0] + left_reach[1],
                right_reach[1] + left_reach[0],
            ) + self.SPAWN_GAP
            sep = jax.random.uniform(ksep, minval=min_sep, maxval=4.8)
            half = 0.5 * sep
            cmax = 2.4 - half
            c = jax.random.uniform(kc, minval=-cmax, maxval=cmax)
            new_x0 = c - side * half
            new_x1 = c + side * half
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
