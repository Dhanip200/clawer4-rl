"""Train the CLAWER-4 walking policy with Brax PPO (run on a GPU, e.g. Colab T4).

In a Colab cell:
    import sys; sys.path.insert(0, 'clawer4/rl')
    from train import train_policy, make_video
    make_policy, params = train_policy()                   # ~50M steps, shows a live score graph
    make_video(make_policy, params, command=(0.2, 0, 0))   # watch it walk forward at 0.2 m/s

Rough ground (policy P1):
    make_policy, params = train_policy(terrain=True, out="clawer4_p1.params")
    make_video(make_policy, params, terrain=True, level=3, kind=2, out="p1_stairs.mp4")

The brain is saved to clawer4_policy.params after training. During training a backup copy
(clawer4_policy_backup.params) is saved at every score check, so a Colab disconnect only loses
the last few minutes.

Keeping the files: if Google Drive is mounted first (from google.colab import drive;
drive.mount('/content/drive')), every backup, the final brain and the videos are copied to
MyDrive/clawer4_runs/<run name>/ (run name = out without ".params"). No browser download is
needed, so it works even if the tab disconnects. Without Drive it falls back to browser downloads.
"""
import functools
import inspect
import os
import shutil
import time

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.io import model as brax_model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco import mjx

from clawer4_env import Clawer4Walk

# JAX 0.11 removed these two helpers, but Brax 0.14 still calls them. Add them back.
if not hasattr(jax, "device_put_replicated"):
    def _device_put_replicated(x, devices):
        return jax.tree.map(lambda a: jp.stack([jp.asarray(a)] * len(devices)), x)
    jax.device_put_replicated = _device_put_replicated
if not hasattr(jax, "device_put_sharded"):
    def _device_put_sharded(shards, devices):
        return jax.tree.map(lambda *a: jp.stack(a), *shards)
    jax.device_put_sharded = _device_put_sharded

# PPO settings, based on MuJoCo Playground's quadruped (Go1) setup, sized for a T4 GPU
PPO_SETTINGS = dict(
    num_timesteps=50_000_000,   # total practice steps across all robots (~11.5 days of robot time)
    num_envs=2048,              # robots practising at the same time
    episode_length=1000,        # 1000 steps x 0.02 s = 20 s per episode
    num_evals=20,               # how many times to check the score (= points on the graph)
    unroll_length=20,           # steps each robot takes before a learning update
    batch_size=256,
    num_minibatches=32,
    num_updates_per_batch=4,
    discounting=0.97,           # how much it cares about the future vs right now
    learning_rate=3e-4,
    entropy_cost=1e-2,          # keeps it trying new things early on
    clipping_epsilon=0.2,       # PPO's "only change a little at a time" limit
    max_grad_norm=1.0,
    reward_scaling=1.0,
    normalize_observations=True,
    action_repeat=1,
)
POLICY_LAYERS = (128, 128, 128, 128)       # the brain that goes on the robot
VALUE_LAYERS = (256, 256, 256, 256, 256)   # the critic, only used during training
DOWNLOAD_EVERY = 4                         # without Drive: download the backup every 4 score checks (~10 min)
DRIVE_ROOT = "/content/drive/MyDrive/clawer4_runs"
_run = ["videos"]                          # Drive folder name, set by train_policy


def train_policy(out="clawer4_policy.params", seed=0, terrain=False, **overrides):
    env = Clawer4Walk(terrain=terrain)
    settings = {**PPO_SETTINGS, **overrides}
    _run[0] = os.path.basename(out).replace(".params", "")
    if _drive_ok():
        print(f"saving files to Google Drive: {os.path.join(DRIVE_ROOT, _run[0])}", flush=True)
    else:
        print("Google Drive not mounted: files will be sent as browser downloads instead", flush=True)
    # on rough ground, also report the training robots' average curriculum level (if this Brax can)
    if terrain and "log_training_metrics" in inspect.signature(ppo.train).parameters:
        settings.setdefault("log_training_metrics", True)
    steps, scores, t0 = [], [], time.time()
    level = [None]

    def progress(num_steps, metrics):
        if "eval/episode_reward" not in metrics:      # a training-metrics report, not a score check
            lv = [v for k, v in metrics.items() if k.endswith("curriculum/level")]
            ln = [v for k, v in metrics.items() if k.endswith("episode/length")]
            if lv:
                # Brax adds the level up over every step of the episode, so divide by its length
                n = float(ln[0]) if ln and float(ln[0]) > 0 else settings["episode_length"]
                level[0] = float(lv[0]) / n
                print(f"{num_steps / 1e6:6.1f}M steps | average level {level[0]:.2f} / 5", flush=True)
            return
        steps.append(num_steps)
        scores.append(float(metrics["eval/episode_reward"]))
        print(f"{num_steps / 1e6:6.1f}M steps | score {scores[-1]:7.2f} | "
              f"{(time.time() - t0) / 60:5.1f} min", flush=True)
        _plot(steps, scores, settings["num_timesteps"])
        if level[0] is not None:                       # the graph clears the screen, so repeat it
            print(f"latest average level {level[0]:.2f} / 5", flush=True)

    backup = out.replace(".params", "_backup.params")
    n_saved = [0]

    def save_backup(num_steps, make_policy, params):
        brax_model.save_params(backup, params)
        n_saved[0] += 1
        if _drive_ok():
            _keep(backup)                              # quick copy, every score check
        elif n_saved[0] % DOWNLOAD_EVERY == 0:
            print(f"backup brain at {num_steps / 1e6:.1f}M steps -> downloading {backup}", flush=True)
            _keep(backup)

    make_policy, params, _ = ppo.train(
        environment=env,
        network_factory=functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=POLICY_LAYERS,
            value_hidden_layer_sizes=VALUE_LAYERS),
        progress_fn=progress,
        policy_params_fn=save_backup,
        seed=seed,
        **settings,
    )
    brax_model.save_params(out, params)
    print(f"done in {(time.time() - t0) / 60:.1f} min, brain saved to {out}")
    _keep(out)  # Colab wipes its files when the session ends, so keep a copy now
    return make_policy, params


def _drive_ok():
    return os.path.isdir("/content/drive/MyDrive")


def _keep(path):
    """Copy a file to Google Drive if it is mounted, otherwise send it as a browser download."""
    if _drive_ok():
        try:
            folder = os.path.join(DRIVE_ROOT, _run[0])
            os.makedirs(folder, exist_ok=True)
            shutil.copy(path, os.path.join(folder, os.path.basename(path)))
            print(f"saved {path} to Drive: {folder}", flush=True)
            return
        except Exception as e:
            print(f"Drive copy failed ({e}), downloading instead", flush=True)
    _download(path)


def _download(path):
    """On Colab, send a file to the PC's Downloads folder. Does nothing elsewhere."""
    try:
        from google.colab import files
        files.download(path)
    except Exception:  # not on Colab, or the browser blocked it
        pass


def _plot(steps, scores, total):
    try:
        import matplotlib.pyplot as plt
        from IPython.display import clear_output, display
    except ImportError:
        return
    clear_output(wait=True)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(np.array(steps) / 1e6, scores, "o-")
    ax.set(xlim=(0, total / 1e6), xlabel="training steps (millions)", ylabel="score per episode",
           title="CLAWER-4 learning to walk")
    ax.grid(alpha=0.3)
    display(fig)
    plt.close(fig)
    for s, r in zip(steps, scores):
        print(f"{s / 1e6:6.1f}M steps | score {r:7.2f}")


def make_video(make_policy, params, command=(0.2, 0.0, 0.0), seconds=10, out="walk.mp4",
               terrain=False, level=0, kind=0):
    """Run the trained brain with a fixed command, save a video and print how it did.
    terrain=True: start on tile [level, kind] (kinds: 0 bumps, 1 slope, 2 stairs up, 3 stairs down, 4 blocks)."""
    import mediapy

    env = Clawer4Walk(terrain=terrain)
    policy = jax.jit(make_policy(params, deterministic=True))
    step = jax.jit(env.step)
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    if terrain:  # put it on the chosen tile, facing +x
        q = env._start_qpos(jax.random.PRNGKey(0), jp.int32(level), jp.int32(kind))
        q = q.at[0:2].set(env.starts[level, kind, :2]).at[3:7].set(jp.array([1.0, 0, 0, 0]))
        data = mjx.forward(env.mjx_model, state.pipeline_state.replace(qpos=q, ctrl=q[7:]))
        state = state.replace(pipeline_state=data, info={**state.info, "level": jp.int32(level),
                                                         "kind": jp.int32(kind), "origin": q[:2]})
    x0, y0 = float(state.pipeline_state.qpos[0]), float(state.pipeline_state.qpos[1])
    state = state.replace(info={**state.info, "command": jp.array(command, dtype=jp.float32)})

    renderer = mujoco.Renderer(env.mj_model, 360, 480)
    d = mujoco.MjData(env.mj_model)
    frames, key = [], jax.random.PRNGKey(1)
    n = int(seconds / 0.02)
    for i in range(n):
        key, k = jax.random.split(key)
        action, _ = policy(state.obs, k)
        state = step(state, action)
        state = state.replace(info={**state.info, "command": jp.array(command, dtype=jp.float32)})
        if i % 2 == 0:  # 25 frames per second
            mjx.get_data_into(d, env.mj_model, state.pipeline_state)
            renderer.update_scene(d, camera="track")
            frames.append(renderer.render())
        if state.done:
            print(f"fell over after {i * 0.02:.1f} s")
            break
    x = float(state.pipeline_state.qpos[0]) - x0
    y = float(state.pipeline_state.qpos[1]) - y0
    t = (i + 1) * 0.02
    print(f"command {command} -> moved x {x:+.2f} m, y {y:+.2f} m in {t:.1f} s "
          f"({x / t:+.3f} m/s forward)")
    mediapy.write_video(out, frames, fps=25)
    mediapy.show_video(frames, fps=25)
    _keep(out)
