"""CLAWER-4 walking environment for RL (MuJoCo MJX, runs on the GPU).

The policy runs at 50 Hz. Every 0.02 s it:
  SEES  45 numbers: which way is down (3), body spin (3), joint angles (12),
        joint speeds (12), its last action (12), the command (3)
  DOES  12 numbers in [-1, 1] -> servo targets = stance pose + 0.3 rad * action
  GETS  points for matching the commanded speed, minus penalties for wobbling,
        jerky moves, using lots of torque and going over the servo speed limit.
The episode ends early if the robot tips over more than 60 deg or the body
drops below 5 cm.

Command = [forward m/s, sideways m/s, turn rad/s], picked at random each episode.
Body linear speed is NOT in what the robot sees, because the real robot can't
measure it. Down-direction and spin come from an IMU, so the real robot needs one.

Servo load (load=True, policy P2): the robot also SEES 12 more numbers, how hard each servo is
pushing (actuator force / max torque, -1..1, like the real ST3020's ReadLoad / 1000), plus
random noise, because the real reading is coarse. 57 numbers in total.

History (history=N, policy P3): the robot SEES its last N readings (newest first) instead of only
the latest one, so it can notice things that take time, like a leg stopped by a step.
N=10 with load = 570 numbers. At the start of an episode the history is the first reading repeated.

Rough ground (terrain=True): the floor is the heightfield from rl/terrain.py (5 kinds x 6
levels of tiles). Each robot starts on the flat patch of one tile. Curriculum: when an episode ends,
a robot that walked at least 80% of the distance it was told to (measured along the commanded
direction, so turning is fine) and got off the flat start patch moves one level harder; one that
fell or walked less than 40% of it moves one level easier. Robots that beat level 5 go to a random level.

Works with Brax PPO when brax is installed (Colab); falls back to a tiny
stand-in State so the env can still be tested on a PC without brax.
"""
from dataclasses import dataclass, field, replace
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

try:
    from brax.envs.base import Env, State
except ImportError:  # no brax on this PC: same fields, enough to test the env
    @jax.tree_util.register_dataclass
    @dataclass
    class State:
        pipeline_state: mjx.Data
        obs: jax.Array
        reward: jax.Array
        done: jax.Array
        metrics: dict = field(default_factory=dict)
        info: dict = field(default_factory=dict)

        def replace(self, **kw):
            return replace(self, **kw)

    class Env:
        pass

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
XML = MODEL_DIR / "clawer4.xml"
TERRAIN_XML = MODEL_DIR / "clawer4_terrain.xml"
TERRAIN_STARTS = MODEL_DIR / "terrain_starts.npy"
FEET = ["FR_foot_site", "FL_foot_site", "RR_foot_site", "RL_foot_site"]

CTRL_DT = 0.02           # policy runs at 50 Hz
ACTION_SCALE = 0.3       # rad: action of 1.0 moves a joint 0.3 rad from stance
MAX_JOINT_SPEED = 4.71   # rad/s, fastest the ST3020 can be commanded
FOOT_DOWN_Z = 0.012      # m: foot site closer than this to the ground under it = touching it
MIN_BODY_Z = 0.05        # m: body lower than this above the ground under it = fell
EPISODE_STEPS = 1000     # must match episode_length in train.py (used by the curriculum)
START_JITTER = 0.1       # m: random shift of the start point inside the flat patch (terrain only)
LEVEL_UP, LEVEL_DOWN = 0.8, 0.4   # curriculum: fraction of the commanded distance walked
MIN_ASKED = 0.5          # m: told to walk less than this = too little to judge, level stays the same
LEFT_PATCH = 0.6         # m from the start: off the flat patch (0.3 m half-width + jitter), on real terrain
SOLVER_ITERS = 2         # physics solver iterations (XML says 100: far too slow on the GPU)
SOLVER_LS_ITERS = 6
MAX_TORQUE = 2.45        # N*m, servo forcerange in the XML (load 1.0 = this much)
LOAD_NOISE = 0.05        # std of the noise added to each load reading (guess: measure on the real robot)

# Command ranges, picked at random each episode (the real robot is slow: ~0.17 m/s trot)
CMD_LO = jp.array([-0.1, -0.1, -0.6])
CMD_HI = jp.array([0.3, 0.1, 0.6])
RESAMPLE_STEPS = 500     # new random command every 500 steps (10 s)

# Reward weights. Positive = good thing to do more of, negative = penalty.
REWARD_WEIGHTS = {
    "track_lin_vel": 1.5,    # walk at the commanded forward/sideways speed
    "track_ang_vel": 0.8,    # turn at the commanded rate
    "feet_air_time": 1.0,    # take real steps instead of shuffling
    "lin_vel_z": -2.0,       # don't bounce up and down
    "ang_vel_xy": -0.05,     # don't roll or pitch quickly
    "orientation": -5.0,     # keep the body level
    "torque": -1e-3,         # don't waste servo effort
    "action_rate": -0.01,    # smooth moves, no jitter
    "joint_speed": -1.0,     # stay under the servo speed limit
    "termination": -1.0,     # falling over is bad
}


class Clawer4Walk(Env):
    def __init__(self, terrain=False, xml_path=None, load=False, history=1):
        self.terrain = terrain
        self.load = load
        self.history = history
        xml_path = xml_path or (TERRAIN_XML if terrain else XML)
        self.mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
        # Speed-ups for GPU training (same idea as MuJoCo Playground's Go1):
        # a few solver iterations are enough at 0.002 s steps...
        self.mj_model.opt.iterations = SOLVER_ITERS
        self.mj_model.opt.ls_iterations = SOLVER_LS_ITERS
        # ...and only check robot-vs-floor contacts, not leg-vs-leg (robot: type 2, floor: type 1)
        floor = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        for g in range(self.mj_model.ngeom):
            if g != floor and self.mj_model.geom_contype[g]:
                self.mj_model.geom_contype[g], self.mj_model.geom_conaffinity[g] = 2, 1
        self.mj_model.geom_contype[floor], self.mj_model.geom_conaffinity[floor] = 1, 2
        self.mjx_model = mjx.put_model(self.mj_model)
        self.n_substeps = round(CTRL_DT / self.mj_model.opt.timestep)
        self.default_qpos = jp.array(self.mj_model.keyframe("home").qpos)
        self.default_joints = self.default_qpos[7:]          # 12 joint angles of the stance pose
        self.torso_id = self.mj_model.body("torso").id
        self.feet_ids = jp.array([self.mj_model.site(f).id for f in FEET])

        # Ground height map, so "how high above the ground" works on bumps and steps too
        m = self.mj_model
        if m.nhfield:
            nr, nc = m.hfield_nrow[0], m.hfield_ncol[0]
            rx, ry, top, _ = m.hfield_size[0]
            self.ground_h = jp.array(m.hfield_data[:nr * nc].reshape(nr, nc) * top)
            self.ground_half = jp.array([rx, ry])
        else:
            self.ground_h = None
        self.n_kinds = 1
        if terrain:
            self.starts = jp.array(np.load(TERRAIN_STARTS))     # [level, kind] -> x, y, ground z
            self.n_levels, self.n_kinds = self.starts.shape[:2]

    # Brax wants these three
    @property
    def observation_size(self):
        return (57 if self.load else 45) * self.history

    @property
    def action_size(self):
        return self.mj_model.nu

    @property
    def backend(self):
        return "mjx"

    def reset(self, rng):
        rng, k_q, k_cmd, k_zero, k_tile, k_noise = jax.random.split(rng, 6)
        level = jp.zeros((), dtype=jp.int32)                 # everyone starts on the easiest level
        kind = jax.random.randint(k_tile, (), 0, self.n_kinds)
        qpos = self._start_qpos(k_q, level, kind)
        data = mjx.make_data(self.mjx_model).replace(qpos=qpos, ctrl=qpos[7:])
        data = mjx.forward(self.mjx_model, data)

        info = {
            "rng": rng,
            "step": jp.zeros((), dtype=jp.int32),
            "command": self._sample_command(k_cmd, k_zero),
            "last_act": jp.zeros(12),
            "feet_air_time": jp.zeros(4),
            "last_contact": jp.ones(4, dtype=bool),
            # curriculum bookkeeping
            "level": level,
            "kind": kind,
            "origin": qpos[0:2],                             # where this episode started
            "asked_dist": jp.zeros(()),                      # how far the commands asked it to go
            "progress": jp.zeros(()),                        # how far it went in the commanded direction
            "farthest": jp.zeros(()),                        # furthest it got from the start point
            "ep_step": jp.zeros((), dtype=jp.int32),
        }
        metrics = {f"reward/{k}": jp.zeros(()) for k in REWARD_WEIGHTS}
        metrics["curriculum/level"] = level.astype(jp.float32)
        info["hist"] = self._fresh_history(self._observe(data, info, k_noise))
        obs = info["hist"].ravel()
        return State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state, action):
        action = jp.clip(action, -1.0, 1.0)
        ctrl = self.default_joints + ACTION_SCALE * action

        # run 10 physics steps (0.002 s each) with the servos holding these targets
        def physics(_, d):
            return mjx.step(self.mjx_model, d.replace(ctrl=ctrl))
        data = jax.lax.fori_loop(0, self.n_substeps, physics, state.pipeline_state)

        # did it fall? (heights are measured from the ground under the body / foot)
        up_z = data.xmat[self.torso_id][2, 2]                # 1 = level, 0 = on its side
        body_z = data.qpos[2] - self._ground(data.qpos[0:2])
        fell = (up_z < 0.5) | (body_z < MIN_BODY_Z)

        # foot contacts and how long each foot was in the air
        feet = data.site_xpos[self.feet_ids]
        contact = feet[:, 2] - self._ground(feet[:, :2]) < FOOT_DOWN_Z
        contact_filt = contact | state.info["last_contact"]  # ignore one-step flicker
        first_contact = (state.info["feet_air_time"] > 0) & contact_filt
        air_time = state.info["feet_air_time"] + CTRL_DT

        terms = self._reward_terms(data, action, state.info, air_time, first_contact, fell)
        reward = sum(REWARD_WEIGHTS[k] * v for k, v in terms.items()) * CTRL_DT

        # The trainer restarts robots without calling reset(): when an episode ends it puts back
        # info["first_pipeline_state"] / info["first_obs"]. So everything that should change between
        # episodes is done here: new command, curriculum level, and a new start point.
        old = state.info
        rng, k_cmd, k_zero, k_q, k_tile, k_rand, k_n1, k_n2 = jax.random.split(old["rng"], 8)
        ep_step = old["ep_step"] + 1
        ended = fell | (ep_step >= EPISODE_STEPS)            # fell, or ran out of time
        cmd_xy = old["command"][:2]
        cmd_speed = jp.linalg.norm(cmd_xy)
        asked = old["asked_dist"] + cmd_speed * CTRL_DT
        # Distance moved this step along the commanded direction (which turns with the robot). A straight
        # line from the start would be unfair: a robot told to turn walks in a curve.
        fwd = data.xmat[self.torso_id][:2, 0]
        fwd = fwd / jp.maximum(jp.linalg.norm(fwd), 1e-6)
        left = jp.array([-fwd[1], fwd[0]])
        want = (cmd_xy[0] * fwd + cmd_xy[1] * left) / jp.maximum(cmd_speed, 1e-6)
        progress = old["progress"] + jp.dot(data.qpos[0:2] - state.pipeline_state.qpos[0:2], want)
        farthest = jp.maximum(old["farthest"], jp.linalg.norm(data.qpos[0:2] - old["origin"]))

        level, kind = old["level"], old["kind"]
        if self.terrain:
            judged = asked > MIN_ASKED
            up = ~fell & judged & (progress > LEVEL_UP * asked) & (farthest > LEFT_PATCH)
            down = fell | (judged & (progress < LEVEL_DOWN * asked))
            new_level = jp.clip(level + up.astype(jp.int32) - down.astype(jp.int32), 0, self.n_levels - 1)
            beat_top = up & (level == self.n_levels - 1)     # beat the hardest level: random level
            new_level = jp.where(beat_top, jax.random.randint(k_rand, (), 0, self.n_levels), new_level)
            level = jp.where(ended, new_level, level)
            kind = jp.where(ended, jax.random.randint(k_tile, (), 0, self.n_kinds), kind)
        next_qpos = self._start_qpos(k_q, level, kind)        # used only if this episode ended

        new_cmd = (old["step"] >= RESAMPLE_STEPS) | ended
        info = dict(old)
        info["rng"] = rng
        info["step"] = jp.where(new_cmd, 0, old["step"] + 1)
        info["command"] = jp.where(new_cmd, self._sample_command(k_cmd, k_zero), old["command"])
        info["last_act"] = jp.where(ended, 0.0, action)
        info["feet_air_time"] = air_time * ~contact_filt * ~ended
        info["last_contact"] = jp.where(ended, True, contact)
        info["level"], info["kind"] = level, kind
        info["origin"] = jp.where(ended, next_qpos[0:2], old["origin"])
        info["asked_dist"] = jp.where(ended, 0.0, asked)
        info["progress"] = jp.where(ended, 0.0, progress)
        info["farthest"] = jp.where(ended, 0.0, farthest)
        info["ep_step"] = jp.where(ended, 0, ep_step)
        # the start state the trainer puts back when this episode ends
        template = old.get("first_pipeline_state", data)
        start = template.replace(qpos=next_qpos, qvel=jp.zeros_like(data.qvel), ctrl=next_qpos[7:])
        info["first_pipeline_state"] = start
        start_hist = self._fresh_history(self._observe(start, info, k_n1))
        info["first_obs"] = start_hist.ravel()
        # keep any metrics the Brax wrappers added (e.g. "reward"), just update ours
        metrics = dict(state.metrics)
        metrics.update({f"reward/{k}": v for k, v in terms.items()})
        metrics["curriculum/level"] = level.astype(jp.float32)
        # newest reading first, drop the oldest; a robot that is restarting gets the start history
        hist = jp.concatenate([self._observe(data, info, k_n2)[None], old["hist"][:-1]])
        obs = hist.ravel()
        info["hist"] = jp.where(ended, start_hist, hist)
        return state.replace(pipeline_state=data, obs=obs, reward=reward,
                             done=fell.astype(jp.float32), metrics=metrics, info=info)

    # ---------- helpers ----------

    def _ground(self, xy):
        """Ground height (m) under points xy [..., 2], read from the heightfield (0 on flat ground)."""
        if self.ground_h is None:
            return jp.zeros(xy.shape[:-1])
        nr, nc = self.ground_h.shape
        # MuJoCo spreads the samples evenly from -half to +half (first/last sample on the edges)
        u = jp.clip((xy[..., 0] + self.ground_half[0]) / (2 * self.ground_half[0]) * (nc - 1), 0, nc - 1.001)
        v = jp.clip((xy[..., 1] + self.ground_half[1]) / (2 * self.ground_half[1]) * (nr - 1), 0, nr - 1.001)
        j, i = jp.floor(u).astype(jp.int32), jp.floor(v).astype(jp.int32)
        fu, fv = u - j, v - i
        H = self.ground_h
        return (H[i, j] * (1 - fu) * (1 - fv) + H[i, j + 1] * fu * (1 - fv)
                + H[i + 1, j] * (1 - fu) * fv + H[i + 1, j + 1] * fu * fv)

    def _start_qpos(self, key, level, kind):
        """Stance pose with the joints nudged a little (so it learns to recover).
        On rough ground: placed on the flat patch of tile [level, kind], facing a random way."""
        k_q, k_xy, k_yaw = jax.random.split(key, 3)
        qpos = self.default_qpos.at[7:].add(jax.random.uniform(k_q, (12,), minval=-0.05, maxval=0.05))
        if self.terrain:
            x, y, z = self.starts[level, kind]
            xy = jp.array([x, y]) + jax.random.uniform(k_xy, (2,), minval=-START_JITTER, maxval=START_JITTER)
            yaw = jax.random.uniform(k_yaw, (), minval=-jp.pi, maxval=jp.pi)
            qpos = qpos.at[0:3].set(jp.array([xy[0], xy[1], z + self.default_qpos[2] + 0.005]))
            qpos = qpos.at[3:7].set(jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)]))
        return qpos

    def _sample_command(self, k_cmd, k_zero):
        cmd = jax.random.uniform(k_cmd, (3,), minval=CMD_LO, maxval=CMD_HI)
        stand_still = jax.random.uniform(k_zero) < 0.1        # 10% of episodes: just stand
        return jp.where(stand_still, jp.zeros(3), cmd)

    def _body_frame(self, data):
        rot = data.xmat[self.torso_id]                       # body -> world rotation
        lin_vel = rot.T @ data.qvel[0:3]                     # body-frame velocity (m/s)
        ang_vel = data.qvel[3:6]                             # free joint spin is already body-frame
        gravity = rot.T @ jp.array([0.0, 0.0, -1.0])         # which way is down, seen from the body
        return lin_vel, ang_vel, gravity

    def _fresh_history(self, one):
        return jp.tile(one[None], (self.history, 1))         # [history, numbers per reading]

    def _observe(self, data, info, key):
        _, ang_vel, gravity = self._body_frame(data)
        parts = [
            gravity,                                         # 3
            ang_vel * 0.25,                                  # 3  (scaled to roughly -1..1)
            data.qpos[7:] - self.default_joints,             # 12
            data.qvel[6:] * 0.05,                            # 12
            info["last_act"],                                # 12
            info["command"],                                 # 3
        ]
        if self.load:                                        # 12 servo loads, -1..1, noisy
            load = data.actuator_force / MAX_TORQUE
            parts.append(load + LOAD_NOISE * jax.random.normal(key, (12,)))
        return jp.concatenate(parts)

    def _reward_terms(self, data, action, info, air_time, first_contact, fell):
        lin_vel, ang_vel, gravity = self._body_frame(data)
        cmd = info["command"]
        moving = jp.linalg.norm(cmd) > 0.05
        lin_err = jp.sum((cmd[:2] - lin_vel[:2]) ** 2)
        over_speed = jp.maximum(jp.abs(data.qvel[6:]) - MAX_JOINT_SPEED, 0.0)
        return {
            # 1.0 when spot on, ~0.37 at 0.14 m/s off. Standing still when told to walk 0.17 m/s
            # scores only ~0.23, so walking is worth more than standing.
            "track_lin_vel": jp.exp(-lin_err / 0.02),
            "track_ang_vel": jp.exp(-(cmd[2] - ang_vel[2]) ** 2 / 0.25),
            "feet_air_time": jp.sum((air_time - 0.1) * first_contact) * moving,
            "lin_vel_z": lin_vel[2] ** 2,
            "ang_vel_xy": jp.sum(ang_vel[:2] ** 2),
            "orientation": jp.sum(gravity[:2] ** 2),
            "torque": jp.sum(data.actuator_force ** 2),
            "action_rate": jp.sum((action - info["last_act"]) ** 2),
            "joint_speed": jp.sum(over_speed ** 2),
            "termination": fell.astype(jp.float32),
        }
