"""Load the model, hold the home pose for a few seconds, report stability and render snapshots."""
from pathlib import Path
import sys

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
m = mujoco.MjModel.from_xml_path(str(ROOT / "model" / "clawer4.xml"))
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
print(f"nq={m.nq} nu={m.nu} total mass={m.body_subtreemass[1]:.3f} kg")

frames = []
r = mujoco.Renderer(m, 480, 640)
opt = mujoco.MjvOption()
for step in range(int(3.0 / m.opt.timestep)):
    mujoco.mj_step(m, d)
    if step % 500 == 0:
        print(f"t={d.time:.2f}s torso z={d.qpos[2]:.4f} m  max |torque|={np.abs(d.actuator_force).max():.2f} N*m")
r.update_scene(d, camera="track", scene_option=opt)
Image.fromarray(r.render()).save(ROOT / "model" / "standing.png")
opt.geomgroup[3] = 1; opt.geomgroup[2] = 0
r.update_scene(d, camera="track", scene_option=opt)
Image.fromarray(r.render()).save(ROOT / "model" / "collision.png")
up = d.xmat[1].reshape(3, 3)[:, 2]
print(f"final torso z={d.qpos[2]:.4f} m, tilt={np.degrees(np.arccos(up[2])):.2f} deg, "
      f"foot contacts={sum(1 for c in d.contact[:d.ncon] if 'foot' in (m.geom(c.geom1).name + m.geom(c.geom2).name))}")
