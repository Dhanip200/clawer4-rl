"""Quick checks of the RL environment on the CPU (no training).

  1. Stand still (action 0, command 0) for 2 s: it should stay up and score well.
  2. 8 robots at once doing random actions for 2 s: shapes right, no NaNs.

Usage:  python tools/test_env.py
"""
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl"))
from clawer4_env import Clawer4Walk  # noqa: E402

env = Clawer4Walk()
reset, step = jax.jit(env.reset), jax.jit(env.step)
STEPS = 100  # 100 x 0.02 s = 2 s

# 1. stand still
t0 = time.time()
s = reset(jax.random.PRNGKey(0))
s = s.replace(info={**s.info, "command": jp.zeros(3)})
total = 0.0
for _ in range(STEPS):
    s = step(s, jp.zeros(12))
    total += float(s.reward)
print(f"[1] stand still: obs {s.obs.shape}, torso z {float(s.pipeline_state.qpos[2]):.3f} m, "
      f"fell {bool(s.done)}, total reward {total:.3f} ({time.time() - t0:.1f} s incl. compile)")
print("    reward parts:", {k.split('/')[1]: round(float(v), 4) for k, v in s.metrics.items()})

# 2. 8 robots, random actions
N = 8
vreset, vstep = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
s = vreset(jax.random.split(jax.random.PRNGKey(1), N))
key = jax.random.PRNGKey(2)
fell_any = jp.zeros(N, dtype=bool)
for _ in range(STEPS):
    key, k = jax.random.split(key)
    s = vstep(s, jax.random.uniform(k, (N, 12), minval=-1, maxval=1))
    fell_any |= s.done > 0
print(f"[2] random x{N}: obs {s.obs.shape}, reward {s.reward.shape}, "
      f"NaNs {bool(jp.isnan(s.obs).any() | jp.isnan(s.reward).any())}, "
      f"fell {int(fell_any.sum())}/{N}, commands[0] {s.info['command'][0].round(2).tolist()}")
