"""MuJoCo model for a 2D (sagittal-plane) two-fighter arena.

Each fighter is a planar stick-figure: torso + head, two legs
(hip/knee/ankle) and two arms (shoulder/elbow), all hinging about the
y-axis, with a 3-dof planar root (slide x, slide z, hinge y).

Joint limits are symmetric so the body is direction-agnostic: the same
policy can fight facing left or right without observation mirroring.

Collision bitmasks disable self-collision within a fighter but keep
fighter-vs-fighter and fighter-vs-floor contacts:
  floor: contype=4 conaffinity=3
  f0:    contype=1 conaffinity=6
  f1:    contype=2 conaffinity=5
"""

# Per-fighter layout (13 dof, all 1-dof joints, so qpos index == joint index):
#   0 rootx (slide)  1 rootz (slide)  2 rooty (hinge)
#   3 hip_a   4 knee_a  5 ankle_a
#   6 hip_b   7 knee_b  8 ankle_b
#   9 shoulder_a  10 elbow_a  11 shoulder_b  12 elbow_b
NQ_PER_FIGHTER = 13
NU_PER_FIGHTER = 10
TORSO_INIT_Z = 1.25
INIT_X = (-0.75, 0.75)
ARENA_HALF = 2.5

_LIMB_JOINTS = [
    # (name, range, gear)
    ("hip_a", "-2.0 2.0", 70),
    ("knee_a", "-2.4 2.4", 60),
    ("ankle_a", "-1.0 1.0", 30),
    ("hip_b", "-2.0 2.0", 70),
    ("knee_b", "-2.4 2.4", 60),
    ("ankle_b", "-1.0 1.0", 30),
    ("shoulder_a", "-3.0 3.0", 40),
    ("elbow_a", "-2.4 2.4", 30),
    ("shoulder_b", "-3.0 3.0", 40),
    ("elbow_b", "-2.4 2.4", 30),
]


def _leg(prefix: str, side: str, col: str) -> str:
    p = f"{prefix}_"
    return f"""
        <body name="{p}thigh_{side}" pos="0 0 -0.2">
          <joint name="{p}hip_{side}" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
          <geom name="{p}thigh_{side}" type="capsule" fromto="0 0 0 0 0 -0.45" size="0.05" {col}/>
          <body name="{p}shin_{side}" pos="0 0 -0.45">
            <joint name="{p}knee_{side}" type="hinge" axis="0 1 0" range="-2.4 2.4"/>
            <geom name="{p}shin_{side}" type="capsule" fromto="0 0 0 0 0 -0.5" size="0.04" {col}/>
            <body name="{p}foot_{side}" pos="0 0 -0.5">
              <joint name="{p}ankle_{side}" type="hinge" axis="0 1 0" range="-1.0 1.0"/>
              <geom name="{p}foot_{side}" type="capsule" fromto="-0.1 0 0 0.1 0 0" size="0.045" {col}/>
            </body>
          </body>
        </body>"""


def _arm(prefix: str, side: str, col: str) -> str:
    p = f"{prefix}_"
    return f"""
        <body name="{p}upper_arm_{side}" pos="0 0 0.15">
          <joint name="{p}shoulder_{side}" type="hinge" axis="0 1 0" range="-3.0 3.0"/>
          <geom name="{p}upper_arm_{side}" type="capsule" fromto="0 0 0 0 0 -0.3" size="0.04" {col}/>
          <body name="{p}forearm_{side}" pos="0 0 -0.3">
            <joint name="{p}elbow_{side}" type="hinge" axis="0 1 0" range="-2.4 2.4"/>
            <geom name="{p}forearm_{side}" type="capsule" fromto="0 0 0 0 0 -0.28" size="0.035" {col}/>
          </body>
        </body>"""


def _fighter(prefix: str, x0: float, contype: int, conaffinity: int, rgba: str) -> str:
    p = f"{prefix}_"
    col = f'contype="{contype}" conaffinity="{conaffinity}" rgba="{rgba}"'
    return f"""
      <body name="{p}torso" pos="{x0} 0 {TORSO_INIT_Z}">
        <joint name="{p}rootx" type="slide" axis="1 0 0" limited="false" damping="0"/>
        <joint name="{p}rootz" type="slide" axis="0 0 1" limited="false" damping="0"/>
        <joint name="{p}rooty" type="hinge" axis="0 1 0" limited="false" damping="0"/>
        <geom name="{p}torso" type="capsule" fromto="0 0 -0.2 0 0 0.2" size="0.07" {col}/>
        <geom name="{p}head" type="sphere" pos="0 0 0.32" size="0.09" {col}/>
        {_leg(prefix, "a", col)}
        {_leg(prefix, "b", col)}
        {_arm(prefix, "a", col)}
        {_arm(prefix, "b", col)}
      </body>"""


def _actuators(prefix: str) -> str:
    lines = []
    for name, _, gear in _LIMB_JOINTS:
        lines.append(
            f'    <motor name="{prefix}_{name}" joint="{prefix}_{name}" gear="{gear}"/>'
        )
    return "\n".join(lines)


def build_xml() -> str:
    return f"""
<mujoco model="fighter2d">
  <option timestep="0.008" iterations="8" ls_iterations="8"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
  </visual>
  <default>
    <joint damping="0.4" armature="0.01" limited="true"/>
    <geom friction="1.0 0.1 0.1" density="1000"/>
    <motor ctrlrange="-1 1" ctrllimited="true"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="10 3 0.1" contype="4" conaffinity="3" rgba="0.85 0.85 0.85 1"/>
    {_fighter("f0", INIT_X[0], 1, 6, "0.85 0.25 0.2 1")}
    {_fighter("f1", INIT_X[1], 2, 5, "0.2 0.4 0.85 1")}
  </worldbody>
  <actuator>
{_actuators("f0")}
{_actuators("f1")}
  </actuator>
</mujoco>
"""
