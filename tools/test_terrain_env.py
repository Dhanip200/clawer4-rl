"""Quick checks of the rough-ground env on the CPU (no training, no brax needed).

It copies what Brax's training wrapper does when an episode ends (puts back
info["first_pipeline_state"] and info["first_obs"]), so the curriculum can be tested here.

  1. Stand still on the start patch of each of the 5 kinds (level 0) for 2 s: stays up, feet on the ground.
  2. Ground height check: env._ground() matches a ray cast straight down in MuJoCo.
  3. Curriculum: a robot that "walked well" goes up a level and is moved to the new tile;
     circling on the flat patch or standing as told keeps its level; too slow or falling goes down.
  4. 16 robots, random actions, 4 s: no NaNs, shapes right.

Usage:  python tools/test_terrain_env.py
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl"))
from clawer4_env import Clawer4Walk  # noqa: E402

env = Clawer4Walk(terrain=True)


def wrapped_reset(rng):
    s = env.reset(rng)
    return s.replace(info={**s.info, "first_pipeline_state": s.pipeline_state, "first_obs": s.obs,
                           "steps": jp.zeros(())})


def wrapped_step(s, a):
    steps = jp.where(s.done > 0, 0.0, s.info["steps"])
    s = env.step(s, a)
    steps = steps + 1                                       # Brax's time limit: done after 1000 steps
    s = s.replace(done=jp.where(steps >= 1000, 1.0, s.done), info={**s.info, "steps": steps})
    pick = lambda x, y: jp.where(s.done > 0, x, y)  # noqa: E731
    ps = jax.tree.map(pick, s.info["first_pipeline_state"], s.pipeline_state)
    return s.replace(pipeline_state=ps, obs=pick(s.info["first_obs"], s.obs))


reset, step = jax.jit(wrapped_reset), jax.jit(wrapped_step)

# 1. stand still on each kind
print("[1] stand still 2 s at level 0:")
for k in range(env.n_kinds):
    s = reset(jax.random.PRNGKey(k))
    q = env._start_qpos(jax.random.PRNGKey(k), jp.int32(0), jp.int32(k))
    ps = s.pipeline_state.replace(qpos=q, ctrl=q[7:])
    s = s.replace(pipeline_state=ps, info={**s.info, "kind": jp.int32(k), "origin": q[:2],
                                           "command": jp.zeros(3)})
    fell = False
    for _ in range(100):
        s = step(s, jp.zeros(12))
        fell |= bool(s.done)
    d = s.pipeline_state
    body = float(d.qpos[2] - env._ground(d.qpos[:2]))
    print(f"    kind {k}: body {body:.3f} m above ground, fell {fell}, feet down {s.info['last_contact'].tolist()}")

# 2. ground height vs ray cast
m = env.mj_model
md = mujoco.MjData(m)
mujoco.mj_forward(m, md)
rng = np.random.default_rng(0)
errs = []
for _ in range(300):
    x, y = rng.uniform(-8.9, 8.9), rng.uniform(-7.4, 7.4)
    gid = np.zeros(1, np.int32)
    dist = mujoco.mj_ray(m, md, np.array([x, y, 2.0]), np.array([0, 0, -1.0]), None, 1, -1, gid)
    errs.append(abs((2.0 - dist) - float(env._ground(jp.array([x, y])))))
print(f"[2] ground height vs ray: mean error {np.mean(errs) * 1000:.2f} mm, max {np.max(errs) * 1000:.2f} mm")

# 3. curriculum
s = reset(jax.random.PRNGKey(5))
q = s.pipeline_state.qpos
good = {**s.info, "ep_step": jp.int32(999), "steps": jp.float32(999), "asked_dist": jp.float32(1.0),
        "progress": jp.float32(1.0), "farthest": jp.float32(1.0)}   # pretend it walked 1 m of the 1 m asked
s2 = step(s.replace(info=good), jp.zeros(12))
moved = np.asarray(s2.pipeline_state.qpos[:2])
lv, kd = int(s2.info["level"]), int(s2.info["kind"])
tile = np.asarray(env.starts[lv, kd, :2])
print(f"[3] walked well: level 0 -> {lv}, now at {moved.round(2)} (tile [{lv},{kd}] centre {tile.round(2)}), "
      f"done {bool(s2.done)}")
assert lv == 1 and np.allclose(moved, tile, atol=0.11), "walked well should go up and move to the new tile"
assert float(s2.info["progress"]) == 0 and float(s2.info["farthest"]) == 0, "bookkeeping must restart"
for name, ask, prog, far, want_lv in [("circled on the flat patch", 1.0, 1.0, 0.3, 2), ("told to stand", 0.0, 0.0, 0.0, 2),
                                      ("half the distance", 1.0, 0.6, 1.0, 2), ("too slow", 1.0, 0.2, 1.0, 1)]:
    inf = {**good, "level": jp.int32(2), "command": jp.zeros(3), "asked_dist": jp.float32(ask),
           "progress": jp.float32(prog), "farthest": jp.float32(far)}
    got = int(step(s.replace(info=inf), jp.zeros(12)).info["level"])
    print(f"    {name}: level 2 -> {got}")
    assert got == want_lv, name
# the progress counter: push the robot along world x and compare with the same sum done in numpy
raw, c = jax.jit(env.step), jp.array([0.2, 0.05, 0.5])
st, prog = reset(jax.random.PRNGKey(7)), 0.0
for i in range(60):
    st = st.replace(info={**st.info, "command": c})
    prev = np.asarray(st.pipeline_state.qpos[:2])
    st = st.replace(pipeline_state=st.pipeline_state.replace(qvel=st.pipeline_state.qvel.at[0].set(0.3)))
    st = raw(st, jp.zeros(12))
    R = np.asarray(st.pipeline_state.xmat[env.torso_id])
    f = R[:2, 0] / np.linalg.norm(R[:2, 0])
    w = (0.2 * f + 0.05 * np.array([-f[1], f[0]])) / np.hypot(0.2, 0.05)
    prog += float(np.dot(np.asarray(st.pipeline_state.qpos[:2]) - prev, w))
print(f"    progress counter {float(st.info['progress']):+.4f} m, worked out by hand {prog:+.4f} m")
assert abs(float(st.info["progress"]) - prog) < 1e-4
s3 = s2.replace(info={**s2.info, "level": jp.int32(3)})
tipped = s3.pipeline_state.replace(qpos=s3.pipeline_state.qpos.at[3:7].set(jp.array([0.0, 1.0, 0.0, 0.0])))
s4 = step(s3.replace(pipeline_state=tipped), jp.zeros(12))   # upside down -> fell
print(f"    fell at level 3: done {bool(s4.done)}, level -> {int(s4.info['level'])}")
assert bool(s4.done) and int(s4.info["level"]) == 2

# 4. 16 robots, random actions
N = 16
vreset, vstep = jax.jit(jax.vmap(wrapped_reset)), jax.jit(jax.vmap(wrapped_step))
s = vreset(jax.random.split(jax.random.PRNGKey(1), N))
key, falls = jax.random.PRNGKey(2), 0
for _ in range(200):
    key, k = jax.random.split(key)
    s = vstep(s, jax.random.uniform(k, (N, 12), minval=-1, maxval=1))
    falls += int((s.done > 0).sum())
print(f"[4] random x{N}: obs {s.obs.shape}, NaNs {bool(jp.isnan(s.obs).any() | jp.isnan(s.reward).any())}, "
      f"falls {falls}, levels {s.info['level'].tolist()}, kinds {s.info['kind'].tolist()}")
