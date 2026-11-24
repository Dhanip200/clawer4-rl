"""Quick check of the history observation (policy P3). Runs on the PC (CPU), ~1-2 min.

    .venv/Scripts/python tools/test_history_env.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl"))
import jax
import jax.numpy as jp
import numpy as np

from clawer4_env import Clawer4Walk

env = Clawer4Walk(terrain=True, load=True, history=10)
assert env.observation_size == 570
reset, step = jax.jit(env.reset), jax.jit(env.step)
state = reset(jax.random.PRNGKey(0))
h = np.array(state.obs).reshape(10, 57)
assert state.obs.shape == (570,) and np.allclose(h, h[0]), "start = first reading repeated"

prev = h
for i in range(20):
    state = step(state, jp.zeros(12))
    h = np.array(state.obs).reshape(10, 57)
    assert np.allclose(h[1:], prev[:-1]), f"step {i}: old readings did not move back one slot"
    prev = h
print("history shifts by one slot each step: OK")

# make it fall (wild actions) and check the restart history is a single reading repeated
key = jax.random.PRNGKey(3)
for i in range(1000):
    key, k = jax.random.split(key)
    state = step(state, jax.random.uniform(k, (12,), minval=-1, maxval=1) * 3)
    if state.done:
        fh = np.array(state.info["first_obs"]).reshape(10, 57)
        nh = np.array(state.info["hist"])
        assert np.allclose(fh, fh[0]) and np.allclose(nh, fh), "restart history wrong"
        print(f"fell at step {i}; restart history = start reading repeated: OK")
        break
else:
    print("never fell (restart check skipped)")
print("old envs:", Clawer4Walk().observation_size, Clawer4Walk(load=True).observation_size)
print("OK")
