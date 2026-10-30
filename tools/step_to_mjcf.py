"""Convert the CLAWER-4 STEP assembly into MuJoCo meshes + MJCF.

Bodies per leg (CAD pose = all joints at 0):
  torso  : 2 body plates + 4 hip-yaw servos (+ payload placeholder for battery/electronics)
  hip    : BASE bracket + hip-pitch servo         (joint: hip yaw,   vertical axis)
  thigh  : 2 MID plates + knee servo              (joint: hip pitch, horizontal axis)
  shin   : LEG                                    (joint: knee,      horizontal axis)

Joint axes come from each servo's horn (FK-AP-019) and idler disc (WJ-DP00) centres.
Positive pitch/knee rotation lifts the leg. All body frames stay world-aligned.

Usage: python tools/step_to_mjcf.py
"""
from pathlib import Path

import numpy as np
import trimesh
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Tool
from OCP.GProp import GProp_GProps
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDF import TDF_Label
from OCP.TDataStd import TDataStd_Name
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.collections import Sequence_TDF_Label

ROOT = Path(__file__).resolve().parents[1]
STEP = ROOT / "cad" / "CLAWER-4 based on ST3020 Servo.step"
OUT = ROOT / "model"
MESH_DIR = OUT / "meshes"

# --- physical guesses: MEASURE THE REAL ROBOT AND UPDATE THESE ---
SERVO_MASS = 0.097          # kg, Waveshare ST3020 product page (wiki gives no weight)
PRINT_DENSITY = 0.8e-6      # kg/mm^3, 3D-printed PLA at ~50% infill (solid PLA = 1.24e-6)
TORSO_PAYLOAD = 0.30        # kg, battery + driver board (not in the CAD)

# --- Waveshare ST3020 wiki (https://www.waveshare.com/wiki/ST3020_Servo) ---
SERVO_TORQUE = 2.45         # N*m, 25 kg*cm stall @ 12 V
SERVO_SPEED = 6.28          # rad/s, 0.167 s/60deg (60 rpm) no-load @ 12 V
SERVO_CMD_SPEED = 4.71      # rad/s, max commandable speed 3073 steps/s (4096 steps/rev)
# Encoder: 4096 steps/rev (0.088 deg), centre 2047, no mechanical stop.
# Real joint limits come from the printed frame, not the servo.
KP, KV = 12.0, 0.3          # position-servo stiffness (N*m/rad) and damping
FOOT_RADIUS = 0.008         # m

# Legs in CAD order. +x = forward, +y = left.
LEG_NAMES = {1: "RR", 2: "FR", 3: "FL", 4: "RL"}


# ---------------------------------------------------------------- STEP helpers
def load():
    doc = TDocStd_Document(TCollection_ExtendedString("doc"))
    r = STEPCAFControl_Reader()
    r.SetNameMode(True)
    r.ReadFile(str(STEP))
    r.Transfer(doc)
    return doc, XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())  # keep doc alive


def name(lab):
    a = TDataStd_Name()
    return a.Get().ToExtString() if lab.FindAttribute(TDataStd_Name.GetID_s(), a) else ""


def children(st, lab, loc):
    """Yield (name, referred label, world location) for each component of an assembly label."""
    seq = Sequence_TDF_Label()
    st.GetComponents_s(lab, seq)
    for i in range(1, seq.Length() + 1):
        c = seq.Value(i)
        ref = TDF_Label()
        st.GetReferredShape_s(c, ref)
        yield name(c), ref, loc.Multiplied(st.GetLocation_s(c))


def world_shape(st, ref, loc):
    return st.GetShape_s(ref).Moved(loc)


def centre(shape):
    b = Bnd_Box()
    BRepBndLib.Add_s(shape, b)
    p0, p1 = b.CornerMin(), b.CornerMax()
    return np.array([(p0.X() + p1.X()) / 2, (p0.Y() + p1.Y()) / 2, (p0.Z() + p1.Z()) / 2])


def volume_props(shape):
    """Volume (mm^3) and centre of mass (mm) of a solid shape."""
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, g)
    c = g.CentreOfMass()
    return abs(g.Mass()), np.array([c.X(), c.Y(), c.Z()])


def tessellate(shape, deflection=0.4):
    BRepMesh_IncrementalMesh(shape, deflection, False, 0.5, True)
    verts, faces = [], []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            trsf = loc.Transformation()
            base = len(verts)
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i).Transformed(trsf)
                verts.append((p.X(), p.Y(), p.Z()))
            flip = face.Orientation() == TopAbs_REVERSED
            for i in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(i).Get()
                faces.append((base + a - 1, base + c - 1, base + b - 1) if flip else
                             (base + a - 1, base + b - 1, base + c - 1))
        exp.Next()
    return trimesh.Trimesh(np.array(verts), np.array(faces), process=True)


def servo_axis(st, ref, loc):
    """Return (point on axis, unit axis direction) in mm from horn and idler disc centres."""
    horn = idler = None
    for _, sref, sloc in children(st, ref, loc):          # SCS15_ASM wrapper
        for pname, pref, ploc in children(st, sref, sloc):
            if pname.startswith("FK-AP-019"):
                horn = centre(world_shape(st, pref, ploc))
            elif pname.startswith("WJ-DP00"):
                idler = centre(world_shape(st, pref, ploc))
    assert horn is not None and idler is not None, "servo horn/idler not found"
    d = horn - idler
    return (horn + idler) / 2, d / np.linalg.norm(d)


# ---------------------------------------------------------------- body building
class Part:
    def __init__(self):
        self.meshes, self.mass, self.mass_x_com = [], 0.0, np.zeros(3)

    def add(self, shape, mass=None):
        vol, com = volume_props(shape)
        m = mass if mass is not None else vol * PRINT_DENSITY
        self.meshes.append(tessellate(shape))
        self.mass += m
        self.mass_x_com += m * com

    def add_point_mass(self, m, pos):
        self.mass += m
        self.mass_x_com += m * pos

    @property
    def com(self):
        return self.mass_x_com / self.mass

    def inertia(self):
        """Approximate diagonal inertia (kg m^2) about the COM from the mesh hull's box."""
        lo, hi = trimesh.util.concatenate(self.meshes).bounds
        ext = (hi - lo) / 1000.0
        m = self.mass
        return m / 12 * np.array([ext[1] ** 2 + ext[2] ** 2,
                                  ext[0] ** 2 + ext[2] ** 2,
                                  ext[0] ** 2 + ext[1] ** 2])


def main():
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    doc, st = load()
    roots = Sequence_TDF_Label()
    st.GetFreeShapes(roots)
    top = roots.Value(1)

    torso = Part()
    legs = []
    for cname, ref, loc in children(st, top, TopLoc_Location()):
        if cname.startswith("CLAWER-PARTS-example_BODY"):
            torso.add(world_shape(st, ref, loc))
        elif cname.startswith("CLAWER LEG"):
            parts = {n: (r, l) for n, r, l in children(st, ref, loc)}
            legs.append(parts)

    leg_data = []
    for i, parts in enumerate(legs, start=1):
        get = lambda key: next(v for k, v in parts.items() if k.startswith(key))
        s1, s2, s3 = (get(f"ST3020 Servo v6:{k}") for k in (1, 2, 3))
        hip, thigh, shin = Part(), Part(), Part()

        torso.add(world_shape(st, *s1), SERVO_MASS)
        hip.add(world_shape(st, *get("CLAWER-PARTS-example_BASE")))
        hip.add(world_shape(st, *s2), SERVO_MASS)
        thigh.add(world_shape(st, *get("CLAWER-PARTS-example_MID v1:1")))
        thigh.add(world_shape(st, *get("CLAWER-PARTS-example_MID v1:2")))
        thigh.add(world_shape(st, *s3), SERVO_MASS)
        shin.add(world_shape(st, *get("CLAWER-PARTS-example_LEG")))

        p1, a1 = servo_axis(st, *s1)
        p2, a2 = servo_axis(st, *s2)
        p3, a3 = servo_axis(st, *s3)
        # Horizontal outward direction of this leg, from body centre towards the knee.
        out = np.array([p3[0], p3[1], 0.0])
        out /= np.linalg.norm(out)
        lift = np.cross(out, [0, 0, 1])                  # positive rotation raises the leg
        for a, want in ((a1, [0, 0, 1.0]), (a2, lift), (a3, lift)):
            assert abs(abs(np.dot(a, want)) - 1) < 1e-3, f"leg {i}: unexpected servo axis {a}"
        a1, a2, a3 = np.array([0, 0, 1.0]), lift, lift

        shin_mesh = shin.meshes[0]
        foot = shin_mesh.vertices[np.argmin(shin_mesh.vertices[:, 2])].copy()
        foot[2] += FOOT_RADIUS * 1000                    # sphere sits on the tip

        leg_data.append(dict(name=LEG_NAMES[i], hip=hip, thigh=thigh, shin=shin,
                             p1=p1, p2=p2, p3=p3, a1=a1, a2=a2, a3=a3, foot=foot))

    torso.add_point_mass(TORSO_PAYLOAD, np.array([0, 0, -23.0]))

    # ---- write meshes, each in its own body frame (metres)
    def save(part, fname, origin):
        m = trimesh.util.concatenate(part.meshes)
        m.apply_translation(-origin)
        m.apply_scale(0.001)
        m.export(MESH_DIR / fname)

    save(torso, "torso.stl", np.zeros(3))
    for L in leg_data:
        save(L["hip"], f"{L['name']}_hip.stl", L["p1"])
        save(L["thigh"], f"{L['name']}_thigh.stl", L["p2"])
        save(L["shin"], f"{L['name']}_shin.stl", L["p3"])

    # ---- MJCF
    f = lambda v: " ".join(f"{x:.5g}" for x in v)
    mm = lambda v: f(np.asarray(v) / 1000.0)

    def inertial(part, origin):
        return (f'<inertial pos="{mm(part.com - origin)}" mass="{part.mass:.4f}" '
                f'diaginertia="{f(part.inertia())}"/>')

    foot_z = min(L["foot"][2] for L in leg_data) - FOOT_RADIUS * 1000
    spawn_z = -foot_z / 1000.0 + 0.005

    legs_xml = []
    for L in leg_data:
        n = L["name"]
        knee_rel = L["p3"] - L["p2"]
        shin_tip = L["foot"] - L["p3"]
        thigh_mid = knee_rel / 2
        legs_xml.append(f"""
      <body name="{n}_hip" pos="{mm(L['p1'])}">
        {inertial(L['hip'], L['p1'])}
        <joint name="{n}_hip_yaw" axis="{f(L['a1'])}" range="-1.0 1.0"/>
        <geom class="visual" mesh="{n}_hip"/>
        <body name="{n}_thigh" pos="{mm(L['p2'] - L['p1'])}">
          {inertial(L['thigh'], L['p2'])}
          <joint name="{n}_hip_pitch" axis="{f(L['a2'])}" range="-1.4 1.4"/>
          <geom class="visual" mesh="{n}_thigh"/>
          <geom class="collision" type="capsule" size="0.012" fromto="0 0 0 {mm(knee_rel)}"/>
          <body name="{n}_shin" pos="{mm(knee_rel)}">
            {inertial(L['shin'], L['p3'])}
            <joint name="{n}_knee" axis="{f(L['a3'])}" range="-1.6 1.6"/>
            <geom class="visual" mesh="{n}_shin"/>
            <geom class="collision" type="capsule" size="0.009" fromto="0 0 0 {mm(shin_tip * 0.8)}"/>
            <geom name="{n}_foot" class="foot" pos="{mm(shin_tip)}"/>
            <site name="{n}_foot_site" pos="{mm(shin_tip)}" size="0.005"/>
          </body>
        </body>
      </body>""")

    actuators = "\n".join(
        f'    <position name="{L["name"]}_{j}" joint="{L["name"]}_{j}" class="st3020"/>'
        for L in leg_data for j in ("hip_yaw", "hip_pitch", "knee"))

    meshes = "\n".join(f'    <mesh name="{p.stem}" file="{p.name}"/>' for p in sorted(MESH_DIR.glob("*.stl")))
    total_mass = torso.mass + sum(L[k].mass for L in leg_data for k in ("hip", "thigh", "shin"))

    xml = f"""<!-- Generated by tools/step_to_mjcf.py from the CLAWER-4 STEP file. Do not hand-edit; rerun the script.
     Total mass {total_mass:.3f} kg (guessed densities + payload - weigh the real robot and update).
     ST3020: stall {SERVO_TORQUE} N*m, no-load {SERVO_SPEED} rad/s, max commanded speed {SERVO_CMD_SPEED} rad/s
     (the RL env must keep joint speeds under this). Joint angle 0 = CAD pose; servo steps = 2047 + angle*4096/(2*pi)
     only if the servo's middle position was calibrated at the CAD pose. -->
<mujoco model="clawer4">
  <compiler angle="radian" meshdir="meshes" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast"/>

  <default>
    <joint damping="{KV}" armature="0.01" frictionloss="0.02"/>
    <position kp="{KP}" forcerange="-{SERVO_TORQUE} {SERVO_TORQUE}" inheritrange="1"/>
    <default class="st3020"/>
    <default class="visual">
      <geom type="mesh" contype="0" conaffinity="0" group="2" rgba="0.75 0.75 0.78 1"/>
    </default>
    <default class="collision">
      <geom group="3" rgba="0.9 0.3 0.3 0.4" friction="0.8 0.02 0.01"/>
      <default class="foot">
        <geom type="sphere" size="{FOOT_RADIUS}" priority="1" friction="1.0 0.02 0.01"
              solref="0.01 1" rgba="0.1 0.1 0.1 1" group="2"/>
      </default>
    </default>
  </default>

  <asset>
{meshes}
    <texture name="grid" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 .15 .2" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".1"/>
  </asset>

  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid" contype="1" conaffinity="1"/>
    <body name="torso" pos="0 0 {spawn_z:.4f}">
      <freejoint name="root"/>
      <camera name="track" mode="trackcom" pos="0.6 -0.6 0.45" xyaxes="1 1 0 -0.4 0.4 1"/>
      {inertial(torso, np.zeros(3))}
      <geom class="visual" mesh="torso"/>
      <geom class="collision" type="box" size="0.045 0.045 0.025" pos="0 0 -0.023"/>
      <site name="imu" pos="0 0 -0.023"/>
{"".join(legs_xml)}
    </body>
  </worldbody>

  <actuator>
{actuators}
  </actuator>

  <sensor>
    <framequat name="imu_quat" objtype="site" objname="imu"/>
    <gyro name="imu_gyro" site="imu"/>
    <accelerometer name="imu_acc" site="imu"/>
  </sensor>

  <keyframe>
    <key name="home" qpos="0 0 {spawn_z:.4f} 1 0 0 0 {' '.join(['0'] * 12)}" ctrl="{' '.join(['0'] * 12)}"/>
  </keyframe>
</mujoco>
"""
    (OUT / "clawer4.xml").write_text(xml, encoding="utf-8")

    print(f"total mass {total_mass:.3f} kg, spawn height {spawn_z:.3f} m")
    for L in leg_data:
        print(L["name"], "hip_yaw", np.round(L["p1"], 1), "pitch", np.round(L["p2"], 1),
              "knee", np.round(L["p3"], 1), "foot", np.round(L["foot"], 1),
              "masses", round(L["hip"].mass, 3), round(L["thigh"].mass, 3), round(L["shin"].mass, 3))


if __name__ == "__main__":
    main()
