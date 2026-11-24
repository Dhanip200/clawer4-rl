"""Quick check of the servo-load observation (policy P2). Runs on the PC (CPU), ~1-2 min.

    .venv/Scripts/python tools/test_load_env.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl"))
import jax
import jax.numpy as jp
import numpy as np

from clawer4_env import Clawer4Walk

for terrain in (False, True):
    env = Clawer4Walk(terrain=terrain, load=True)
    assert env.observation_size == 57
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    state = reset(jax.random.PRNGKey(0))
    assert state.obs.shape == (57,), state.obs.shape
    for _ in range(50):                                   # stand still for 1 s
        state = step(state, jp.zeros(12))
    load = np.array(state.obs[45:])
    clean = np.array(state.pipeline_state.actuator_force / 2.45)
    assert np.all(np.isfinite(load)) and np.all(np.abs(load) < 2)
    print(f"terrain={terrain}: obs {state.obs.shape}, fell={bool(state.done)}")
    print("  load (noisy):", np.round(load, 2))
    print("  load (true): ", np.round(clean, 2))
    print("  noise std   :", round(float(np.std(load - clean)), 3))

print("old 45-number env still works:", Clawer4Walk().observation_size == 45)
print("OK")
