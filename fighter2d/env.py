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

    SPAWN_GAP = 0.02  # floor clearance at spawn (and fallback-path reach gap)
    INTER_GAP = 0.01  # min surface distance between fighters at spawn
    FLOOR_GAP = 0.002  # min surface distance above floor for placed limbs
    K_TORSO = 48  # placement candidates for the torso stage
    K_LIMB = 24  # placement candidates per limb-joint stage

    # Capsule chain geometry (body-frame): mirrors model.py.
    _TORSO = ((0.0, -0.2), (0.0, 0.2), 0.07)
    _HEAD = ((0.0, 0.32), (0.0, 0.32), 0.09)
    _LEG = [((0.0, -0.45), 0.05), ((0.0, -0.5), 0.045)]  # thigh, shin
    _FOOT = (0.1, 0.045)  # half-length along foot frame x, radius
    _ARM = [((0.0, -0.3), 0.04), ((0.0, -0.28), 0.035)]  # upper, forearm

    @staticmethod
    def _rot(a, bx, bz):
        """Body-frame (bx, bz) rotated about +y by angle a (broadcasts)."""
        return bx * jnp.cos(a) + bz * jnp.sin(a), -bx * jnp.sin(a) + bz * jnp.cos(a)

    @staticmethod
    def _seg_dist(a1, a2, b1, b2):
        """Min distance between 2D segments a1-a2 and b1-b2 (broadcasts over
        leading dims; robust to zero-length segments)."""
        eps = 1e-12
        d1 = a2 - a1
        d2 = b2 - b1
        r = a1 - b1
        a = (d1 * d1).sum(-1)
        e = (d2 * d2).sum(-1)
        f = (d2 * r).sum(-1)
        c = (d1 * r).sum(-1)
        b = (d1 * d2).sum(-1)
        denom = a * e - b * b
        s = jnp.clip(jnp.where(denom > eps, (b * f - c * e) / jnp.where(denom > eps, denom, 1.0), 0.0), 0.0, 1.0)
        t = jnp.where(e > eps, (b * s + f) / jnp.where(e > eps, e, 1.0), 0.0)
        t = jnp.clip(t, 0.0, 1.0)
        s = jnp.clip(jnp.where(a > eps, (b * t - c) / jnp.where(a > eps, a, 1.0), 0.0), 0.0, 1.0)
        p = a1 + s[..., None] * d1
        q = b1 + t[..., None] * d2
        return jnp.sqrt(((p - q) ** 2).sum(-1) + eps)

    def _capsules(self, q_abs: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """World capsule segments for a fighter whose qpos slot 0 holds the
        ABSOLUTE root x. Returns (p1(12,2), p2(12,2), r(12,))."""
        root = jnp.array([q_abs[0], self.base_z + q_abs[1]])
        th = q_abs[2]
        segs = []

        def world(a, bx, bz):
            dx, dz = self._rot(a, bx, bz)
            return root + jnp.stack([dx, dz], axis=-1)

        segs.append((world(th, *self._TORSO[0]), world(th, *self._TORSO[1]), self._TORSO[2]))
        segs.append((world(th, *self._HEAD[0]), world(th, *self._HEAD[1]), self._HEAD[2]))
        for base, angs in ((-0.2, q_abs[jnp.array([3, 4, 5])]), (-0.2, q_abs[jnp.array([6, 7, 8])])):
            pos = world(th, 0.0, base)
            acc = th
            for (off, r), ang in zip(self._LEG, angs[:2]):
                acc = acc + ang
                nxt = pos + jnp.stack(self._rot(acc, *off), axis=-1)
                segs.append((pos, nxt, r))
                pos = nxt
            acc = acc + angs[2]
            hl, fr = self._FOOT
            e1 = pos + jnp.stack(self._rot(acc, -hl, 0.0), axis=-1)
            e2 = pos + jnp.stack(self._rot(acc, hl, 0.0), axis=-1)
            segs.append((e1, e2, fr))
        for angs in (q_abs[jnp.array([9, 10])], q_abs[jnp.array([11, 12])]):
            pos = world(th, 0.0, 0.15)
            acc = th
            for (off, r), ang in zip(self._ARM, angs):
                acc = acc + ang
                nxt = pos + jnp.stack(self._rot(acc, *off), axis=-1)
                segs.append((pos, nxt, r))
                pos = nxt
        p1 = jnp.stack([s[0] for s in segs])
        p2 = jnp.stack([s[1] for s in segs])
        r = jnp.array([s[2] for s in segs])
        return p1, p2, r

    def _cand_clear(self, p1, p2, r, caps):
        """Clearance of candidate segments p1/p2 (K,2) radius r against the
        opponent's capsules and the floor. >= 0 means placeable."""
        Ap1, Ap2, Ar = caps
        d = self._seg_dist(p1[:, None, :], p2[:, None, :], Ap1[None], Ap2[None])
        inter = (d - Ar[None]).min(axis=1) - r - self.INTER_GAP
        floor = jnp.minimum(p1[:, 1], p2[:, 1]) - r - self.FLOOR_GAP
        return jnp.minimum(inter, floor)

    @staticmethod
    def _choose(rng, clear):
        """Uniform pick among valid candidates (gumbel-max over the mask);
        falls back to the least-colliding candidate if none are valid."""
        valid = clear >= 0.0
        g = jax.random.gumbel(rng, clear.shape)
        pick = jnp.where(valid.any(), jnp.argmax(jnp.where(valid, g, -jnp.inf)), jnp.argmax(clear))
        return pick

    def _conditional_place(self, rng: jax.Array, caps) -> tuple[jax.Array, jax.Array]:
        """Place fighter B in the space left available by fighter A (whose
        world capsules are `caps`): torso first — anywhere collision-free in
        the arena, including above A — then each limb joint sampled uniformly
        from its non-colliding angles, worked outward along each chain.
        Returns (q_abs (13,), ok) where ok means fully collision-free."""
        ks = jax.random.split(rng, 12)
        K = self.K_TORSO
        kx, kz, ka = jax.random.split(ks[0], 3)
        xs = jax.random.uniform(kx, (K,), minval=-2.4, maxval=2.4)
        zs = jax.random.uniform(kz, (K,), minval=DOWN_Z + self.SPAWN_GAP, maxval=1.8)
        ths = jax.random.uniform(ka, (K,), minval=-0.6, maxval=0.6)
        roots = jnp.stack([xs, zs], axis=-1)  # (K,2) world torso centers

        def wpt(roots, angs, bx, bz):
            dx, dz = self._rot(angs, bx, bz)
            return roots + jnp.stack([dx, dz], axis=-1)

        t1 = wpt(roots, ths, *self._TORSO[0])
        t2 = wpt(roots, ths, *self._TORSO[1])
        h1 = wpt(roots, ths, *self._HEAD[0])
        clear = jnp.minimum(
            self._cand_clear(t1, t2, self._TORSO[2], caps),
            self._cand_clear(h1, h1, self._HEAD[2], caps),
        )
        pick = self._choose(ks[1], clear)
        root, th = roots[pick], ths[pick]
        min_clear = clear[pick]

        angles = []
        ki = 2
        Kl = self.K_LIMB
        for base_bz, chain_geom, lo_hi in (
            (-0.2, "leg", (0, 3)),
            (-0.2, "leg", (3, 6)),
            (0.15, "arm", (6, 8)),
            (0.15, "arm", (8, 10)),
        ):
            pos = root + jnp.stack(self._rot(th, 0.0, base_bz), axis=-1)
            acc = th
            geom = self._LEG if chain_geom == "leg" else self._ARM
            lo, hi = lo_hi
            for gi, (off, r) in enumerate(geom):
                j = lo + gi
                cand = jax.random.uniform(
                    ks[ki], (Kl,), minval=self.limb_lo[j], maxval=self.limb_hi[j]
                )
                accs = acc + cand
                nxt = pos[None] + jnp.stack(self._rot(accs, *off), axis=-1)
                clear = self._cand_clear(jnp.broadcast_to(pos, (Kl, 2)), nxt, r, caps)
                pick = self._choose(jax.random.fold_in(ks[ki], 1), clear)
                angles.append(cand[pick])
                min_clear = jnp.minimum(min_clear, clear[pick])
                acc = accs[pick]
                pos = nxt[pick]
                ki += 1
            if chain_geom == "leg":
                j = lo + 2
                cand = jax.random.uniform(
                    ks[ki], (Kl,), minval=self.limb_lo[j], maxval=self.limb_hi[j]
                )
                accs = acc + cand
                hl, fr = self._FOOT
                e1 = pos[None] + jnp.stack(self._rot(accs, -hl, 0.0), axis=-1)
                e2 = pos[None] + jnp.stack(self._rot(accs, hl, 0.0), axis=-1)
                clear = self._cand_clear(e1, e2, fr, caps)
                pick = self._choose(jax.random.fold_in(ks[ki], 1), clear)
                angles.append(cand[pick])
                min_clear = jnp.minimum(min_clear, clear[pick])
                ki += 1

        # angles collected in order: hipA kneeA ankleA hipB kneeB ankleB
        # shA elA shB elB — matches qpos layout.
        q_abs = jnp.concatenate(
            [jnp.array([root[0], root[1] - self.base_z, th]), jnp.stack(angles)]
        )
        return q_abs, min_clear >= 0.0

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
            kplace, kswap, kfall = jax.random.split(rplace, 3)
            # Fighter A: fully random (as sampled + lifted). Fighter B:
            # conditionally placed anywhere in the space A leaves available —
            # including directly above/interlocked with A.
            qA = q0.at[0].add(self.init_x[0])  # slot 0 -> absolute x
            capsA = self._capsules(qA)
            qB, ok = self._conditional_place(kplace, capsA)

            # Fallback for the rare unplaceable draw: directional-reach
            # separation (guaranteed non-overlapping by x-interval disjointness).
            right_reach = jnp.array([jnp.max(dx0 + rad0), jnp.max(dx1 + rad1)])
            left_reach = jnp.array([jnp.max(-dx0 + rad0), jnp.max(-dx1 + rad1)])
            ksep, kc = jax.random.split(kfall)
            min_sep = right_reach[0] + left_reach[1] + self.SPAWN_GAP
            sep = jax.random.uniform(ksep, minval=min_sep, maxval=4.8)
            half = 0.5 * sep
            c = jax.random.uniform(kc, minval=-(2.4 - half), maxval=2.4 - half)
            qA_fb = qA.at[0].set(c - half)
            qB_fb = q1.at[0].add(self.init_x[1]).at[0].set(c + half)

            qA = jnp.where(ok, qA, qA_fb)
            qB = jnp.where(ok, qB, qB_fb)
            # Random role swap so neither fighter is systematically the one
            # placed conditionally.
            swap = jax.random.bernoulli(kswap)
            qf0 = jnp.where(swap, qB, qA).at[0].add(-self.init_x[0])
            qf1 = jnp.where(swap, qA, qB).at[0].add(-self.init_x[1])
            return jnp.concatenate([qf0, qf1]), jnp.concatenate([v0, v1])
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
        self,
        rng: jax.Array,
        state: EnvState,
        action: jax.Array,
        reset_to_qpos: jax.Array | None = None,
        reset_to_qvel: jax.Array | None = None,
    ) -> tuple[EnvState, jax.Array, jax.Array, jax.Array, dict]:
        """action: (2, NU) in [-1, 1]. Returns (state', obs, reward(2,), done, info).

        If reset_to_* are given, an episode ending this step resets to that
        state (supplied by the caller, e.g. from a pre-sampled reset pool)
        instead of sampling a fresh spawn in the hot path."""
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
        if reset_to_qpos is None:
            reset_qpos, reset_qvel = self.reset_qpos_qvel(rng)
        else:
            reset_qpos, reset_qvel = reset_to_qpos, reset_to_qvel
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
