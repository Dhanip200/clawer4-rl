"""Checks rl/evaluate.py on the PC (CPU, a few minutes). Needs clawer4_p1.params.

  1. Option parsing and the success-rate range.
  2. The brain reader: P1 on the same tile, start and command as its Kaggle video (level 2 bumps) should walk
     about as far as the video did (+1.88 m forward in 10 s). CPU and GPU never match exactly.
  3. A do-nothing brain must be "stuck" everywhere, never "success".

    .venv/Scripts/python tools/test_evaluate.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl"))
import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx

import evaluate as ev
from clawer4_env import Clawer4Walk

# 1
assert ev.parse_brain("P1=a.params") == ("P1", "a.params", {})
assert ev.parse_brain(r"P1=C:\x\a.params") == ("P1", r"C:\x\a.params", {})
assert ev.parse_brain("P3=/k/a.params:load,history=10") == ("P3", "/k/a.params", {"load": True, "history": 10})
lo, hi = ev.wilson(10, 10)
assert 0.70 < lo < 0.75 and hi > 0.999 and abs(ev.wilson(50, 100)[0] - 0.404) < 0.002
print("[1] parsing and ranges OK")

# 2: the same start as make_video in train.py
env = Clawer4Walk(terrain=True)
policy, n_in = ev.load_brain(ROOT / "clawer4_p1.params")
assert n_in == 45
step, cmd = jax.jit(env.step), jp.array([0.2, 0.0, 0.0])
s = jax.jit(env.reset)(jax.random.PRNGKey(0))
q = env._start_qpos(jax.random.PRNGKey(0), jp.int32(2), jp.int32(0))
q = q.at[0:2].set(env.starts[2, 0, :2]).at[3:7].set(jp.array([1.0, 0, 0, 0]))
d = mjx.forward(env.mjx_model, s.pipeline_state.replace(qpos=q, ctrl=q[7:]))
s = s.replace(pipeline_state=d, info={**s.info, "level": jp.int32(2), "kind": jp.int32(0), "origin": q[:2], "command": cmd})
fell = False
for _ in range(500):
    s = step(s, policy(s.obs))
    s = s.replace(info={**s.info, "command": cmd})
    fell |= bool(s.done)
dx, dy = np.asarray(s.pipeline_state.qpos[:2] - q[:2])
print(f"[2] P1 level 2 bumps, 10 s: x {dx:+.2f} m, y {dy:+.2f} m, fell {fell} (Kaggle video: +1.88, -0.64, no fall)")
assert not fell and 1.5 < dx < 2.2, "brain reader does not behave like the trained brain"

# 3
lv, kd = jp.array([0, 0, 5, 5]), jp.array([0, 2, 1, 4])
keys = jax.random.split(jax.random.PRNGKey(1), 4)
r = ev.run(env, lambda o: jp.zeros(12), lv, kd, keys, cmd, 2.0)
print(f"[3] do-nothing brain: success {r['success'].tolist()}, fell {r['fell'].tolist()}, speed {r['speed'].round(3).tolist()}")
assert not r["success"].any() and not r["fell"].any() and np.abs(r["speed"]).max() < 0.05
print("OK")
