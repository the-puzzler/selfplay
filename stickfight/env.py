"""Stickman fight: hand-rolled 2D ragdoll physics, limb health, sparse reward.

Each fighter is 11 point masses joined by 10 rigid segments (Verlet
integration + constraint projection — unconditionally stable, no solver).
Eight joint motors (shoulders/elbows/hips/knees) take continuous targets.
Segments have hit points; contact with an opposing segment deals damage
proportional to closing speed. A segment at 0 HP is destroyed: its
constraint is cut (the limb falls off / dangles dead and stops dealing or
taking damage). Destroying the opponent's HEAD or TORSO wins.

Reward is purely sparse and zero-sum: +1 winner / -1 loser at the end,
0 for timeout draws, 0 elsewhere. HP is observable state, never reward.

Soft penalty contacts mean ANY random spawn is physically fine — overlaps
gently push apart. Diverse resets sample random poses anywhere (including
airborne/floored), random velocities, and random per-segment HP (wounded
and pre-maimed states included; head/torso HP floored above zero so no
spawn is dead-on-arrival).
"""

import jax
import jax.numpy as jnp
from flax import struct

# --- timing
SUBSTEPS = 6
DT = 1.0 / 60.0  # physics substep; control dt = 0.1 s
EPISODE_LEN = 300  # 30 seconds
GRAVITY = 9.8
DAMPING = 0.995

# --- body: point indices
# 0 head, 1 neck, 2 pelvis, 3 elbowL, 4 handL, 5 elbowR, 6 handR,
# 7 kneeL, 8 footL, 9 kneeR, 10 footR
N_PTS = 11
SEG_AI = (1, 2, 1, 3, 1, 5, 2, 7, 2, 9)  # static python ints
SEG_BI = (0, 1, 3, 4, 5, 6, 7, 8, 9, 10)
SEG_A = jnp.array(SEG_AI)
SEG_B = jnp.array(SEG_BI)
SEG_LEN = jnp.array([0.25, 0.60, 0.32, 0.30, 0.32, 0.30, 0.45, 0.45, 0.45, 0.45])
SEG_NAMES = ["head", "torso", "uarmL", "farmL", "uarmR", "farmR",
             "thighL", "shinL", "thighR", "shinR"]
N_SEGS = 10
HEAD, TORSO = 0, 1
# Parent segment (torso is its own parent); a segment is only "effective"
# (colliding, damaging, damageable) while its chain to the torso is intact.
SEG_PAR = jnp.array([1, 1, 1, 2, 1, 4, 1, 6, 1, 8])
SEG_THICK = 0.06  # contact distance between segment centerlines

# --- motors: (parent_seg, child_seg, pivot_pt, distal_pt) — static tuples
JOINTS = (
    (1, 2, 1, 3),   # shoulder L
    (2, 3, 3, 4),   # elbow L
    (1, 4, 1, 5),   # shoulder R
    (4, 5, 5, 6),   # elbow R
    (1, 6, 2, 7),   # hip L
    (6, 7, 7, 8),   # knee L
    (1, 8, 2, 9),   # hip R
    (8, 9, 9, 10),  # knee R
)
N_JOINTS = 8
MOTOR_SPEED = 5.0  # rad/s max joint speed
MOTOR_GAIN = 0.5
V_MAX = 8.0  # hard speed cap per point (m/s)

# --- combat
HP0 = 1.0
DMG_SPEED_MIN = 1.0  # closing speed below this deals nothing
DMG_COEF = 0.04  # damage per substep per (m/s over threshold)
SHOVE = 0.35  # positional separation per substep on contact
ARENA_X = 3.0

# --- spaces
OBS_DIM = 111
ACT_DIM = N_JOINTS


@struct.dataclass
class EnvState:
    pts: jnp.ndarray  # (2, N_PTS, 2)
    prev: jnp.ndarray  # (2, N_PTS, 2) previous positions (Verlet velocity)
    hp: jnp.ndarray  # (2, N_SEGS)
    t: jnp.ndarray  # scalar int32


def _seg_alive(hp):
    """Effective mask: segment alive AND its chain to the torso intact."""
    alive = hp > 0.0
    par = alive[..., SEG_PAR]
    par2 = alive[..., SEG_PAR[SEG_PAR]]
    return alive & par & par2


def _seg_dist(a1, a2, b1, b2):
    """Min distance between 2D segments (broadcasts; robust to degenerate)."""
    eps = 1e-12
    d1, d2, r = a2 - a1, b2 - b1, a1 - b1
    a = (d1 * d1).sum(-1)
    e = (d2 * d2).sum(-1)
    f = (d2 * r).sum(-1)
    c = (d1 * r).sum(-1)
    b = (d1 * d2).sum(-1)
    denom = a * e - b * b
    s = jnp.clip(jnp.where(denom > eps, (b * f - c * e) / jnp.where(denom > eps, denom, 1.0), 0.0), 0, 1)
    t = jnp.clip(jnp.where(e > eps, (b * s + f) / jnp.where(e > eps, e, 1.0), 0.0), 0, 1)
    s = jnp.clip(jnp.where(a > eps, (b * t - c) / jnp.where(a > eps, a, 1.0), 0.0), 0, 1)
    p = a1 + s[..., None] * d1
    q = b1 + t[..., None] * d2
    return jnp.sqrt(((p - q) ** 2).sum(-1) + eps), p, q


def _wrap(a):
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi


class StickFightEnv:
    def __init__(self, reset_mode: str = "diverse"):
        assert reset_mode in ("fixed", "diverse")
        self.reset_mode = reset_mode

    # ------------------------------------------------------------- physics

    def _motors(self, pts, action, hp):
        """Rotate each joint's distal point toward the action's target angle.
        COM-preserving: motors reconfigure the body but cannot translate it —
        locomotion must come from pushing against the floor/opponent."""
        pts0 = pts
        eff = _seg_alive(hp)  # (2, N_SEGS)
        for j in range(N_JOINTS):
            pseg, cseg, pivot, distal = JOINTS[j]
            pdir = pts[:, SEG_BI[pseg]] - pts[:, SEG_AI[pseg]]
            cdir = pts[:, SEG_BI[cseg]] - pts[:, SEG_AI[cseg]]
            ang = jnp.arctan2(
                pdir[:, 0] * cdir[:, 1] - pdir[:, 1] * cdir[:, 0],
                (pdir * cdir).sum(-1),
            )
            target = action[:, j] * jnp.pi
            delta = jnp.clip(MOTOR_GAIN * _wrap(target - ang),
                             -MOTOR_SPEED * DT, MOTOR_SPEED * DT)
            delta = delta * eff[:, cseg] * eff[:, pseg]
            rel = pts[:, distal] - pts[:, pivot]
            cos, sin = jnp.cos(delta), jnp.sin(delta)
            rot = jnp.stack([
                cos * rel[:, 0] - sin * rel[:, 1],
                sin * rel[:, 0] + cos * rel[:, 1],
            ], axis=-1)
            pts = pts.at[:, distal].set(pts[:, pivot] + rot)
        com_drift = (pts - pts0).mean(axis=1, keepdims=True)
        return pts - com_drift

    def _constraints(self, pts, hp):
        """Project rigid segment lengths (skip destroyed segments)."""
        alive = (hp > 0.0).astype(jnp.float32)  # (2, N_SEGS)
        for _ in range(6):
            pa = pts[:, SEG_A]  # (2, N_SEGS, 2)
            pb = pts[:, SEG_B]
            d = pb - pa
            dist = jnp.linalg.norm(d, axis=-1, keepdims=True) + 1e-8
            corr = 0.5 * (dist - SEG_LEN[None, :, None]) * d / dist
            corr = corr * alive[..., None]
            pts = pts.at[:, SEG_A].add(corr)
            pts = pts.at[:, SEG_B].add(-corr)
        return pts

    def _contacts(self, pts, prev, hp):
        """Fighter-vs-fighter segment contacts: shove apart + damage by
        closing speed. Returns (pts offset, damage (2, N_SEGS))."""
        eff = _seg_alive(hp)
        # Segments of fighter 0 vs fighter 1: (N_SEGS, N_SEGS) pairs.
        a1, a2 = pts[0, SEG_A], pts[0, SEG_B]
        b1, b2 = pts[1, SEG_A], pts[1, SEG_B]
        dist, p, q = _seg_dist(a1[:, None], a2[:, None], b1[None, :], b2[None, :])
        pair_eff = eff[0][:, None] & eff[1][None, :]
        touching = (dist < SEG_THICK) & pair_eff
        n = (p - q) / (dist[..., None] + 1e-8)  # from fighter1 toward fighter0
        # Closing speed from segment-midpoint velocities.
        v0 = ((pts[0, SEG_A] + pts[0, SEG_B]) - (prev[0, SEG_A] + prev[0, SEG_B])) / (2 * DT)
        v1 = ((pts[1, SEG_A] + pts[1, SEG_B]) - (prev[1, SEG_A] + prev[1, SEG_B])) / (2 * DT)
        closing = jnp.maximum(-(((v0[:, None] - v1[None, :]) * n).sum(-1)), 0.0)
        dmg = DMG_COEF * jnp.maximum(closing - DMG_SPEED_MIN, 0.0) * touching
        dmg0 = dmg.sum(axis=1)  # damage to fighter0's segments
        dmg1 = dmg.sum(axis=0)
        # Shove: separate along the contact normal.
        overlap = jnp.maximum(SEG_THICK - dist, 0.0) * touching
        push = SHOVE * overlap[..., None] * n  # (S, S, 2)
        push0 = push.sum(axis=1) / 2.0  # applied to fighter0 segments
        push1 = -push.sum(axis=0) / 2.0
        off = jnp.zeros_like(pts)
        off = off.at[0, SEG_A].add(push0).at[0, SEG_B].add(push0)
        off = off.at[1, SEG_A].add(push1).at[1, SEG_B].add(push1)
        return pts + off, jnp.stack([dmg0, dmg1])

    def _substep(self, pts, prev, action, hp):
        vel = (pts - prev) * DAMPING
        speed = jnp.linalg.norm(vel, axis=-1, keepdims=True) + 1e-8
        vel = vel * jnp.minimum(V_MAX * DT / speed, 1.0)
        new = pts + vel + jnp.array([0.0, -GRAVITY]) * DT * DT
        prev = pts
        new = self._motors(new, action, hp)
        new = self._constraints(new, hp)
        new, dmg = self._contacts(new, prev, hp)
        # Floor: project up, damp tangential (friction) and normal (inelastic).
        below = new[..., 1] < 0.0
        vy = new[..., 1] - prev[..., 1]
        vx = new[..., 0] - prev[..., 0]
        new = new.at[..., 1].set(jnp.where(below, 0.0, new[..., 1]))
        prev = prev.at[..., 0].set(jnp.where(below, new[..., 0] - 0.4 * vx, prev[..., 0]))
        prev = prev.at[..., 1].set(jnp.where(below, new[..., 1] + 0.2 * vy, prev[..., 1]))
        # Soft walls.
        outside = jnp.abs(new[..., 0]) > ARENA_X
        new = new.at[..., 0].set(jnp.clip(new[..., 0], -ARENA_X, ARENA_X))
        prev = prev.at[..., 0].set(jnp.where(outside, new[..., 0], prev[..., 0]))
        return new, prev, dmg

    # ------------------------------------------------------------- resets

    def _pose_pts(self, root, torso_ang, jangles):
        """FK: build 11 points from pelvis pos, torso angle, 8 joint angles.
        jangles order matches JOINTS."""
        up = jnp.stack([-jnp.sin(torso_ang), jnp.cos(torso_ang)])

        def rot(v, a):
            c, s = jnp.cos(a), jnp.sin(a)
            return jnp.stack([c * v[0] - s * v[1], s * v[0] + c * v[1]])

        pelvis = root
        neck = pelvis + 0.60 * up
        head = neck + 0.25 * up
        pts = [head, neck, pelvis]
        # arms: shoulder angle rotates torso dir; elbow rotates upper-arm dir
        for sh, el, l1, l2 in ((0, 1, 0.32, 0.30), (2, 3, 0.32, 0.30)):
            ua = rot(up, jangles[sh])
            elbow = neck + l1 * ua
            hand = elbow + l2 * rot(ua, jangles[el])
            pts += [elbow, hand]
        for hip, kn, l1, l2 in ((4, 5, 0.45, 0.45), (6, 7, 0.45, 0.45)):
            th = rot(up, jangles[hip])
            knee = pelvis + l1 * th
            foot = knee + l2 * rot(th, jangles[kn])
            pts += [knee, foot]
        return jnp.stack(pts)  # (11, 2)

    def _fixed_fighter(self, x0):
        # Standing: arms hanging (shoulder ~pi), legs straight down (hip ~pi).
        jang = jnp.array([jnp.pi, 0.0, jnp.pi, 0.0, jnp.pi, 0.0, jnp.pi, 0.0])
        pts = self._pose_pts(jnp.array([x0, 0.95]), 0.0, jang)
        return pts

    def _diverse_fighter(self, rng):
        kx, kz, ka, kj, ks, kv, kc = jax.random.split(rng, 7)
        x = jax.random.uniform(kx, minval=-2.5, maxval=2.5)
        z = jax.random.uniform(kz, minval=0.4, maxval=1.8)
        tang = jax.random.uniform(ka, minval=-jnp.pi, maxval=jnp.pi)
        jang = jax.random.uniform(kj, (N_JOINTS,), minval=-jnp.pi, maxval=jnp.pi)
        pts = self._pose_pts(jnp.array([x, z]), tang, jang)
        # Lift above floor.
        pts = pts.at[:, 1].add(jnp.maximum(0.02 - pts[:, 1].min(), 0.0))
        # Velocities: common drift + per-point noise, random intensity.
        scale = jax.random.uniform(ks)
        v = scale * (1.2 * jax.random.normal(kc, (2,))[None, :]
                     + 0.8 * jax.random.normal(kv, (N_PTS, 2)))
        return pts, v

    def reset_state(self, rng: jax.Array) -> EnvState:
        if self.reset_mode == "fixed":
            e = 0.01 * jax.random.normal(rng, (2, N_PTS, 2))
            pts = jnp.stack([self._fixed_fighter(-0.8), self._fixed_fighter(0.8)]) + e
            return EnvState(pts=pts, prev=pts, hp=jnp.full((2, N_SEGS), HP0),
                            t=jnp.zeros((), jnp.int32))
        k0, k1, kh, kf = jax.random.split(rng, 4)
        p0, v0 = self._diverse_fighter(k0)
        p1, v1 = self._diverse_fighter(k1)
        pts = jnp.stack([p0, p1])
        prev = pts - jnp.stack([v0, v1]) * DT
        # HP: fully healthy with prob 0.5, else random per segment (wounded /
        # pre-maimed spawns). Head & torso floored: never spawn-dead.
        healthy = jax.random.bernoulli(kf, 0.5)
        rand_hp = jax.random.uniform(kh, (2, N_SEGS))
        hp = jnp.where(healthy, jnp.full((2, N_SEGS), HP0), rand_hp)
        hp = hp.at[:, HEAD].max(0.15).at[:, TORSO].max(0.15)
        return EnvState(pts=pts, prev=prev, hp=hp, t=jnp.zeros((), jnp.int32))

    def reset(self, rng: jax.Array):
        s = self.reset_state(rng)
        return s, self._obs(s)

    # ---------------------------------------------------------------- obs

    def _obs(self, s: EnvState) -> jax.Array:
        vel = (s.pts - s.prev) / DT  # (2, N_PTS, 2)
        time_left = 1.0 - s.t.astype(jnp.float32) / EPISODE_LEN

        def one(i):
            j = 1 - i
            own_pelvis = s.pts[i, 2]
            own_rel = (s.pts[i] - own_pelvis).reshape(-1)  # 22 (pelvis rel = 0, fine)
            opp_rel = (s.pts[j] - own_pelvis).reshape(-1)  # 22
            return jnp.concatenate([
                own_pelvis / jnp.array([ARENA_X, 2.0]),
                own_rel, 0.1 * vel[i].reshape(-1), s.hp[i],
                opp_rel, 0.1 * vel[j].reshape(-1), s.hp[j],
                jnp.array([time_left]),
            ])

        return jnp.stack([one(0), one(1)])

    # ---------------------------------------------------------------- step

    def step(self, rng, state: EnvState, action, reset_to: EnvState | None = None):
        """action: (2, N_JOINTS) in [-1, 1] (target joint angles / pi)."""
        action = jnp.clip(action, -1.0, 1.0)
        pts, prev, hp = state.pts, state.prev, state.hp
        dmg_total = jnp.zeros((2, N_SEGS))
        for _ in range(SUBSTEPS):
            pts, prev, dmg = self._substep(pts, prev, action, hp)
            dmg_total = dmg_total + dmg
        hp = jnp.maximum(hp - dmg_total, 0.0)

        dead = (hp[:, HEAD] <= 0.0) | (hp[:, TORSO] <= 0.0)  # (2,)
        t = state.t + 1
        timeout = t >= EPISODE_LEN
        done = dead.any() | timeout
        reward = jnp.where(
            dead.any(),
            jnp.array([
                dead[1].astype(jnp.float32) - dead[0].astype(jnp.float32),
                dead[0].astype(jnp.float32) - dead[1].astype(jnp.float32),
            ]),
            jnp.zeros(2),
        )

        new = EnvState(pts=pts, prev=prev, hp=hp, t=t)
        if reset_to is None:
            reset_to = self.reset_state(rng)
        state = jax.tree.map(lambda r, n: jnp.where(done, r, n), reset_to, new)
        obs = self._obs(state)
        info = {
            "win0": (reward[0] > 0) & done,
            "win1": (reward[1] > 0) & done,
            "draw": done & (reward[0] == 0),
            "timeout": timeout & ~dead.any(),
            "ep_len": jnp.where(done, t, 0),
        }
        return state, obs, reward, done, info
