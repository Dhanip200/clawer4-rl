"""Check the rough-ground model:
  1. the heightfield is the right way round (ray-cast the ground and compare with terrain.py)
  2. the robot stands for 2 s on the start patch of every tile (30 tiles)
  3. MJX (GPU physics) can step on it
  4. saves model/terrain_overview.png and model/terrain_tiles.png
"""
from pathlib import Path
import sys

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl"))
import terrain  # noqa: E402

m = mujoco.MjModel.from_xml_path(str(ROOT / "model" / "clawer4_terrain.xml"))
d = mujoco.MjData(m)
starts = np.load(ROOT / "model" / "terrain_starts.npy")
H, _, (rx, ry) = terrain.build(0)


def ground_z(x, y):
    """Height of the ground at (x, y), found by shooting a ray straight down."""
    geomid = np.zeros(1, np.int32)
    dist = mujoco.mj_ray(m, d, np.array([x, y, 2.0]), np.array([0, 0, -1.0]), None, 1, -1, geomid)
    return 2.0 - dist


# 1. orientation: compare ray heights with the height map at random points
mujoco.mj_forward(m, d)
rng = np.random.default_rng(1)
errs = []
for _ in range(300):
    i, j = rng.integers(0, H.shape[0]), rng.integers(0, H.shape[1])
    # MuJoCo puts heightfield samples on a grid from -rx to +rx (first/last sample on the edges)
    x = -rx + j * 2 * rx / (H.shape[1] - 1)
    y = -ry + i * 2 * ry / (H.shape[0] - 1)
    errs.append(abs(ground_z(x, y) - H[i, j]))
print(f"1. heightfield check: mean error {np.mean(errs) * 1000:.1f} mm, max {np.max(errs) * 1000:.1f} mm")

# 2. stand on every tile
home = m.keyframe("home").qpos.copy()
stand_h = home[2]                                    # torso height above flat ground in the home pose
print("2. standing 2 s on each tile (torso height above the ground under it, tilt):")
print("   level " + " ".join(f"{k:>15s}" for k in terrain.KINDS))
for lv in range(terrain.N_LEVELS):
    row = []
    for k in range(len(terrain.KINDS)):
        x, y, z = starts[lv, k]
        mujoco.mj_resetData(m, d)
        d.qpos[:] = home
        d.qpos[0:3] = x, y, z + stand_h + 0.005
        d.ctrl[:] = home[7:]
        for _ in range(1000):
            mujoco.mj_step(m, d)
        up = d.xmat[1].reshape(3, 3)[2, 2]
        ok = up > 0.95 and d.qpos[2] - z > 0.07
        row.append(f"{d.qpos[2] - z:.3f}m {np.degrees(np.arccos(min(up, 1))):4.1f}d {'ok ' if ok else 'BAD'}")
    print(f"   {lv:5d} " + " ".join(f"{s:>15s}" for s in row))

# 3. MJX
try:
    import jax
    from mujoco import mjx
    mx = mjx.put_model(m)
    dx = mjx.make_data(mx).replace(qpos=d.qpos, ctrl=d.ctrl)
    step = jax.jit(mjx.step)
    for _ in range(50):
        dx = step(mx, dx)
    print(f"3. MJX: ok, 50 steps, torso z {float(dx.qpos[2]):.3f} m (MuJoCo said {d.qpos[2]:.3f} m), "
          f"NaN: {bool(np.isnan(np.asarray(dx.qpos)).any())}")
except Exception as e:  # noqa: BLE001
    print(f"3. MJX: FAILED: {type(e).__name__}: {e}")

# 4. pictures
img = (H / H.max() * 255).astype(np.uint8)[::-1]          # flip so +y is up in the picture
Image.fromarray(img).resize((H.shape[1], H.shape[0])).save(ROOT / "model" / "terrain_overview.png")
r = mujoco.Renderer(m, 480, 640)
cam = mujoco.MjvCamera()
cam.lookat[:] = 0, 0, 0
cam.distance, cam.azimuth, cam.elevation = 17, 90, -55
mujoco.mj_resetData(m, d)
mujoco.mj_forward(m, d)
r.update_scene(d, camera=cam)
Image.fromarray(r.render()).save(ROOT / "model" / "terrain_tiles.png")
r.close()
print("4. saved model/terrain_overview.png and model/terrain_tiles.png")
