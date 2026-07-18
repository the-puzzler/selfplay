"""Waypoint parkour: one PBD stickman, random terrain, sparse reach reward.

The stickman (same verified physics as stickfight: 11 Verlet points, 10
rigid segments, 8 COM-preserving joint motors) spawns in a random pose on
procedurally random terrain — smooth hills, steps, cliffs, and gaps — and
must reach a waypoint placed a random distance ahead.

Reward is purely sparse: +1 when the pelvis gets within REACH of the
waypoint (episode ends), 0 at timeout. No forward-velocity shaping, no
distance shaping, no gait priors. Terrain randomness + spawn randomness +
waypoint-distance randomness IS the curriculum (OmniReset style): near/flat
waypoints are learned first, gnarly far ones as competence propagates.
"""

import jax
import jax.numpy as jnp
from flax import struct

# --- timing (same as stickfight)
SUBSTEPS = 6
DT = 1.0 / 60.0
EPISODE_LEN = 300  # 30 s at 10 Hz control
GRAVITY = 9.8
DAMPING = 0.995
V_MAX = 8.0

# --- body: stickfight stickman + real FEET (heel-ankle-toe rigid lines,
# ankle motors). Point-feet made static balance impossible; forward-only
# feet tip backward. A heel puts the support polygon around the ankle.
# pts: 0 head 1 neck 2 pelvis 3 elbL 4 handL 5 elbR 6 handR
#      7 kneeL 8 ankleL 9 kneeR 10 ankleR 11 toeL 12 toeR 13 heelL 14 heelR
N_PTS = 15
SEG_AI = (1, 2, 1, 3, 1, 5, 2, 7, 2, 9, 8, 10, 8, 10, 13, 14)
SEG_BI = (0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 11, 12)
SEG_A = jnp.array(SEG_AI)
SEG_B = jnp.array(SEG_BI)
# Foot = stiff TRIANGLE (ankle raised 0.05 above the heel-toe line);
# a collinear heel-ankle-toe line is singular in PBD and explodes.
SEG_LEN = jnp.array([0.25, 0.60, 0.32, 0.30, 0.32, 0.30,
                     0.45, 0.45, 0.45, 0.45,
                     0.13, 0.13, 0.1118, 0.1118, 0.22, 0.22])
JOINTS = ((1, 2, 1, 3), (2, 3, 3, 4), (1, 4, 1, 5), (4, 5, 5, 6),
          (1, 6, 2, 7), (6, 7, 7, 8), (1, 8, 2, 9), (8, 9, 9, 10),
          (7, 10, 8, 11), (9, 11, 10, 12))
# Points each motor rotates: the FULL downstream subtree (rotating only the
# immediate distal point tears the constraints of everything past it — the
# solver's violent repairs were shaking the body apart).
SUBTREES = ((3, 4), (4,), (5, 6), (6,),
            (7, 8, 11, 13), (8, 11, 13), (9, 10, 12, 14), (10, 12, 14),
            (11, 13), (12, 14))
N_JOINTS = 10
MOTOR_SPEED = 5.0
MOTOR_GAIN = 0.5

# --- terrain
NG = 141
GRID = jnp.linspace(-14.0, 21.0, NG)  # more room ahead than behind
N_BUMPS, N_STEPS, N_GAPS = 6, 4, 2
GAP_DEPTH = 3.0

# --- task
# Floor just above REACH: flags can spawn one lean away (the curriculum's
# first rung) but never inside the reach radius (no free wins).
WP_MIN, WP_MAX = 0.7, 9.0
WP_BEHIND_P = 0.35  # fraction of waypoints spawning behind the stickman
REACH = 0.6
# Uprightness rule (BipedalWalker-style): the instant the torso passes
# TIP_LIMIT the episode ends as a fail. No grace, no persistence window —
# v5's amnesty became a kamikaze-cartwheel license. Fair only because the
# body has feet (a support polygon) and spawns are supportive.
TIP_LIMIT = 1.3  # rad (~75 degrees)
# PARTIAL observation: terrain visible only within ~2 units of the body.
# Waypoints can be 9 units out — the terrain between must be discovered by
# traveling and REMEMBERED (the recurrent policy's job). The waypoint
# itself is a compass reading (signed distance), not a visual.
TERRAIN_OFFS = jnp.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])

OBS_DIM = 71  # 1 + 2*15 pts + 2*15 vels + 8 terrain + compass + time
ACT_DIM = N_JOINTS


@struct.dataclass
class EnvState:
    pts: jnp.ndarray  # (N_PTS, 2)
    prev: jnp.ndarray  # (N_PTS, 2)
    h: jnp.ndarray  # (NG,) terrain heights
    wp: jnp.ndarray  # (2,)
    t: jnp.ndarray  # scalar int32


def _height(h, x):
    return jnp.interp(x, GRID, h)


def _wrap(a):
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi


class WaypointEnv:
    def __init__(self, reset_mode: str = "diverse"):
        assert reset_mode in ("fixed", "diverse")
        self.reset_mode = reset_mode

    # ------------------------------------------------------------- terrain

    def _terrain(self, rng: jax.Array) -> jax.Array:
        kb, ks, kg, kgw, kgp = jax.random.split(rng, 5)
        x = GRID[None, :]
        # smooth bumps/hills
        ba, bc, bw = jax.random.split(kb, 3)
        A = jax.random.uniform(ba, (N_BUMPS, 1), minval=-1.4, maxval=2.0)
        C = jax.random.uniform(bc, (N_BUMPS, 1), minval=-12.0, maxval=19.0)
        W = jax.random.uniform(bw, (N_BUMPS, 1), minval=0.6, maxval=3.0)
        h = (A * jnp.exp(-((x - C) / W) ** 2)).sum(0)
        # steps/cliffs
        sa, sc = jax.random.split(ks)
        S = jax.random.uniform(sa, (N_STEPS, 1), minval=-1.2, maxval=1.2)
        SC = jax.random.uniform(sc, (N_STEPS, 1), minval=-12.0, maxval=19.0)
        h = h + (S * jax.nn.sigmoid((x - SC) / 0.08)).sum(0)
        # gaps (pits) — only ahead of spawn, sometimes
        gc = jax.random.uniform(kg, (N_GAPS, 1), minval=1.5, maxval=18.0)
        gw = jax.random.uniform(kgw, (N_GAPS, 1), minval=0.35, maxval=1.1)
        on = jax.random.bernoulli(kgp, 0.6, (N_GAPS, 1))
        pit = jax.nn.sigmoid((x - (gc - gw / 2)) / 0.04) - jax.nn.sigmoid((x - (gc + gw / 2)) / 0.04)
        h = h - GAP_DEPTH * (on * pit).sum(0)
        # zero at spawn
        return h - jnp.interp(0.0, GRID, h)

    # -------------------------------------------------------------- resets

    def _pose_pts(self, root, torso_ang, jangles):
        """jangles (10,): shL elL shR elR hipL knL hipR knR ankL ankR."""
        up = jnp.stack([-jnp.sin(torso_ang), jnp.cos(torso_ang)])

        def rot(v, a):
            c, s = jnp.cos(a), jnp.sin(a)
            return jnp.stack([c * v[0] - s * v[1], s * v[0] + c * v[1]])

        pelvis = root
        neck = pelvis + 0.60 * up
        head = neck + 0.25 * up
        pts = [head, neck, pelvis]
        for sh, el in ((0, 1), (2, 3)):
            ua = rot(up, jangles[sh])
            elbow = neck + 0.32 * ua
            pts += [elbow, elbow + 0.30 * rot(ua, jangles[el])]
        toes, heels = [], []
        for hip, kn, ank in ((4, 5, 8), (6, 7, 9)):
            th = rot(up, jangles[hip])
            knee = pelvis + 0.45 * th
            sh_dir = rot(th, jangles[kn])
            ankle = knee + 0.45 * sh_dir
            fdir = rot(sh_dir, jangles[ank] + jnp.pi / 2)
            fdown = 0.05 * sh_dir  # sole sits below the ankle
            pts += [knee, ankle]
            toes.append(ankle + 0.12 * fdir + fdown)
            heels.append(ankle - 0.10 * fdir + fdown)
        return jnp.stack(pts + toes + heels)

    def reset_state(self, rng: jax.Array) -> EnvState:
        kt, kp, ka, kj, kv, kc, kw = jax.random.split(rng, 7)
        h = self._terrain(kt)
        if self.reset_mode == "fixed":
            jang = jnp.array([jnp.pi, 0.0, jnp.pi, 0.0, jnp.pi, 0.0, jnp.pi, 0.0, 0.0, 0.0])
            pts = self._pose_pts(jnp.array([0.0, 0.95]), 0.0, jang)
            vel = jnp.zeros((N_PTS, 2))
            wp_dx = 5.0
        else:
            # Supportive spawn: torso upright, ARMS fully random, LEGS in a
            # downward cone (feet under the body) so balance is catchable.
            tang = jax.random.uniform(ka, minval=-0.25, maxval=0.25)
            kj1, kj2 = jax.random.split(kj)
            arms = jax.random.uniform(kj1, (4,), minval=-jnp.pi, maxval=jnp.pi)
            kj2a, kj2b = jax.random.split(kj2)
            legs = jax.random.uniform(kj2a, (4,), minval=-0.5, maxval=0.5)
            legs = legs.at[jnp.array([0, 2])].add(jnp.pi)  # hips point down
            ankles = jax.random.uniform(kj2b, (2,), minval=-0.4, maxval=0.4)
            jang = jnp.concatenate([arms, legs, ankles])
            z = jax.random.uniform(kp, minval=0.85, maxval=1.15)
            pts = self._pose_pts(jnp.array([0.0, z]), tang, jang)
            scale = 0.5 * jax.random.uniform(kc)
            vel = scale * (1.0 * jax.random.normal(kc, (2,))[None] +
                           0.7 * jax.random.normal(kv, (N_PTS, 2)))
            kw1, kw2 = jax.random.split(kw)
            side = jnp.where(jax.random.bernoulli(kw1, WP_BEHIND_P), -1.0, 1.0)
            wp_dx = side * jax.random.uniform(kw2, minval=WP_MIN, maxval=WP_MAX)
        # lift above terrain
        under = pts[:, 1] - _height(h, pts[:, 0])
        pts = pts.at[:, 1].add(jnp.maximum(0.02 - under.min(), 0.0))
        wp = jnp.array([wp_dx, 0.0])
        wp = wp.at[1].set(_height(h, wp_dx) + 0.6)
        prev = pts - vel * DT if self.reset_mode == "diverse" else pts
        return EnvState(pts=pts, prev=prev, h=h, wp=wp, t=jnp.zeros((), jnp.int32))

    def reset(self, rng):
        s = self.reset_state(rng)
        return s, self._obs(s)

    # ------------------------------------------------------------- physics

    def _motors(self, pts, prev, action):
        """Rotate each joint's subtree toward its target angle. The SAME
        rotation is applied to prev positions: motor moves carry no phantom
        Verlet velocity (teleporting only pts injects several m/s per
        substep and shakes the body apart)."""
        for j in range(N_JOINTS):
            pseg, cseg, pivot, _ = JOINTS[j]
            pdir = pts[SEG_BI[pseg]] - pts[SEG_AI[pseg]]
            cdir = pts[SEG_BI[cseg]] - pts[SEG_AI[cseg]]
            ang = jnp.arctan2(pdir[0] * cdir[1] - pdir[1] * cdir[0], (pdir * cdir).sum())
            target = action[j] * jnp.pi
            delta = jnp.clip(MOTOR_GAIN * _wrap(target - ang),
                             -MOTOR_SPEED * DT, MOTOR_SPEED * DT)
            c, s = jnp.cos(delta), jnp.sin(delta)
            sub = jnp.array(SUBTREES[j])
            for arr_name in (0, 1):
                arr = pts if arr_name == 0 else prev
                rel = arr[sub] - arr[pivot]
                rot = jnp.stack([c * rel[:, 0] - s * rel[:, 1],
                                 s * rel[:, 0] + c * rel[:, 1]], axis=-1)
                if arr_name == 0:
                    pts = pts.at[sub].set(pts[pivot] + rot)
                else:
                    prev = prev.at[sub].set(prev[pivot] + rot)
        return pts, prev

    def _constraints(self, pts):
        for _ in range(6):
            pa, pb = pts[SEG_A], pts[SEG_B]
            d = pb - pa
            dist = jnp.linalg.norm(d, axis=-1, keepdims=True) + 1e-8
            corr = 0.5 * (dist - SEG_LEN[:, None]) * d / dist
            pts = pts.at[SEG_A].add(corr)
            pts = pts.at[SEG_B].add(-corr)
        return pts

    def _substep(self, pts, prev, action, h):
        vel = (pts - prev) * DAMPING
        speed = jnp.linalg.norm(vel, axis=-1, keepdims=True) + 1e-8
        vel = vel * jnp.minimum(V_MAX * DT / speed, 1.0)
        new = pts + vel + jnp.array([0.0, -GRAVITY]) * DT * DT
        prev = pts
        new, prev = self._motors(new, prev, action)
        new = self._constraints(new)
        # terrain contact (vertical projection heightfield)
        ground = _height(h, new[:, 0])
        below = new[:, 1] < ground
        vy = new[:, 1] - prev[:, 1]
        vx = new[:, 0] - prev[:, 0]
        new = new.at[:, 1].set(jnp.where(below, ground, new[:, 1]))
        prev = prev.at[:, 0].set(jnp.where(below, new[:, 0] - 0.4 * vx, prev[:, 0]))
        prev = prev.at[:, 1].set(jnp.where(below, new[:, 1] + 0.2 * vy, prev[:, 1]))
        return new, prev

    # ---------------------------------------------------------------- obs

    def _obs(self, s: EnvState) -> jax.Array:
        pelvis = s.pts[2]
        vel = (s.pts - s.prev) / DT
        ground_here = _height(s.h, pelvis[0])
        terr = _height(s.h, pelvis[0] + TERRAIN_OFFS) - pelvis[1]
        return jnp.concatenate([
            jnp.array([pelvis[1] - ground_here]),
            (s.pts - pelvis).reshape(-1),
            0.1 * vel.reshape(-1),
            jnp.clip(terr, -4.0, 4.0) / 4.0,
            jnp.array([(s.wp[0] - pelvis[0]) / WP_MAX]),  # compass only
            jnp.array([1.0 - s.t / EPISODE_LEN]),
        ])

    # ---------------------------------------------------------------- step

    def step(self, rng, s: EnvState, action, reset_to: EnvState | None = None):
        """action: (N_JOINTS,) in [-1, 1]. Single agent."""
        action = jnp.clip(action, -1.0, 1.0)
        pts, prev = s.pts, s.prev
        for _ in range(SUBSTEPS):
            pts, prev = self._substep(pts, prev, action, s.h)
        t = s.t + 1
        reached = jnp.linalg.norm(pts[2] - s.wp) < REACH
        torso = pts[1] - pts[2]  # pelvis -> neck
        tipped = jnp.abs(jnp.arctan2(torso[0], torso[1])) > TIP_LIMIT
        tipped = tipped & ~reached
        timeout = t >= EPISODE_LEN
        done = reached | timeout | tipped
        reward = reached.astype(jnp.float32)

        new = EnvState(pts=pts, prev=prev, h=s.h, wp=s.wp, t=t)
        if reset_to is None:
            reset_to = self.reset_state(rng)
        s2 = jax.tree.map(lambda r, n: jnp.where(done, r, n), reset_to, new)
        obs = self._obs(s2)
        info = {
            "reach": reached & done,
            "timeout": timeout & ~reached & ~tipped,
            "tipped": tipped,
            "ep_len": jnp.where(done, t, 0),
            "final_dist": jnp.where(done, jnp.abs(pts[2, 0] - s.wp[0]), 0.0),
        }
        return s2, obs, reward, done, info
