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

Works with Brax PPO when brax is installed (Colab); falls back to a tiny
stand-in State so the env can still be tested on a PC without brax.
"""
from dataclasses import dataclass, field, replace
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
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

XML = Path(__file__).resolve().parents[1] / "model" / "clawer4.xml"
FEET = ["FR_foot_site", "FL_foot_site", "RR_foot_site", "RL_foot_site"]

CTRL_DT = 0.02           # policy runs at 50 Hz
ACTION_SCALE = 0.3       # rad: action of 1.0 moves a joint 0.3 rad from stance
MAX_JOINT_SPEED = 4.71   # rad/s, fastest the ST3020 can be commanded
FOOT_DOWN_Z = 0.012      # m: foot site lower than this = touching the ground
SOLVER_ITERS = 2         # physics solver iterations (XML says 100: far too slow on the GPU)
SOLVER_LS_ITERS = 6

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
    def __init__(self, xml_path=XML):
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

    # Brax wants these three
    @property
    def observation_size(self):
        return 45

    @property
    def action_size(self):
        return self.mj_model.nu

    @property
    def backend(self):
        return "mjx"

    def reset(self, rng):
        rng, k_q, k_cmd, k_zero = jax.random.split(rng, 4)
        # start from the stance pose with the joints nudged a little, so it learns to recover
        qpos = self.default_qpos.at[7:].add(jax.random.uniform(k_q, (12,), minval=-0.05, maxval=0.05))
        data = mjx.make_data(self.mjx_model).replace(qpos=qpos, ctrl=qpos[7:])
        data = mjx.forward(self.mjx_model, data)

        info = {
            "rng": rng,
            "step": jp.zeros((), dtype=jp.int32),
            "command": self._sample_command(k_cmd, k_zero),
            "last_act": jp.zeros(12),
            "feet_air_time": jp.zeros(4),
            "last_contact": jp.ones(4, dtype=bool),
        }
        metrics = {f"reward/{k}": jp.zeros(()) for k in REWARD_WEIGHTS}
        obs = self._observe(data, info)
        return State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state, action):
        action = jp.clip(action, -1.0, 1.0)
        ctrl = self.default_joints + ACTION_SCALE * action

        # run 10 physics steps (0.002 s each) with the servos holding these targets
        def physics(_, d):
            return mjx.step(self.mjx_model, d.replace(ctrl=ctrl))
        data = jax.lax.fori_loop(0, self.n_substeps, physics, state.pipeline_state)

        # did it fall?
        up_z = data.xmat[self.torso_id][2, 2]                # 1 = level, 0 = on its side
        fell = (up_z < 0.5) | (data.qpos[2] < 0.05)

        # foot contacts and how long each foot was in the air
        contact = data.site_xpos[self.feet_ids][:, 2] < FOOT_DOWN_Z
        contact_filt = contact | state.info["last_contact"]  # ignore one-step flicker
        first_contact = (state.info["feet_air_time"] > 0) & contact_filt
        air_time = state.info["feet_air_time"] + CTRL_DT

        terms = self._reward_terms(data, action, state.info, air_time, first_contact, fell)
        reward = sum(REWARD_WEIGHTS[k] * v for k, v in terms.items()) * CTRL_DT

        # The trainer restarts fallen robots without calling reset(), so pick a fresh command
        # here every 10 s and after a fall, and forget the last action after a fall.
        rng, k_cmd, k_zero = jax.random.split(state.info["rng"], 3)
        new_cmd = (state.info["step"] >= RESAMPLE_STEPS) | fell
        info = dict(state.info)
        info["rng"] = rng
        info["step"] = jp.where(new_cmd, 0, state.info["step"] + 1)
        info["command"] = jp.where(new_cmd, self._sample_command(k_cmd, k_zero), state.info["command"])
        info["last_act"] = jp.where(fell, 0.0, action)
        info["feet_air_time"] = air_time * ~contact_filt * ~fell
        info["last_contact"] = contact
        # keep any metrics the Brax wrappers added (e.g. "reward"), just update ours
        metrics = dict(state.metrics)
        metrics.update({f"reward/{k}": v for k, v in terms.items()})
        obs = self._observe(data, info)
        return state.replace(pipeline_state=data, obs=obs, reward=reward,
                             done=fell.astype(jp.float32), metrics=metrics, info=info)

    # ---------- helpers ----------

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

    def _observe(self, data, info):
        _, ang_vel, gravity = self._body_frame(data)
        return jp.concatenate([
            gravity,                                         # 3
            ang_vel * 0.25,                                  # 3  (scaled to roughly -1..1)
            data.qpos[7:] - self.default_joints,             # 12
            data.qvel[6:] * 0.05,                            # 12
            info["last_act"],                                # 12
            info["command"],                                 # 3
        ])

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
