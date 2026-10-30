"""Hand-made sine-wave trot (no RL). Diagonal pairs FR+RL and FL+RR move together.

Each leg: hip_yaw sweeps the foot forward during swing and back during stance,
hip_pitch lifts the foot during swing. Joint 0 = CAD pose.

Usage:
  python tools/sine_gait.py            # headless: print distance walked, save gait.gif
  python tools/sine_gait.py --view     # watch it in the MuJoCo viewer
"""
from pathlib import Path
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEGS = ["FR", "FL", "RR", "RL"]
PHASE = {"FR": 0.0, "RL": 0.0, "FL": np.pi, "RR": np.pi}   # trot: diagonal pairs
X_SIGN = {"FR": 1, "RR": 1, "FL": -1, "RL": -1}            # +yaw moves right feet +x, left feet -x


def gait_targets(t, freq, stride, lift):
    """Joint targets (rad) at time t, in actuator order."""
    q = {}
    for leg in LEGS:
        ph = 2 * np.pi * freq * t + PHASE[leg]
        # swing (0..pi): foot travels back->front while lifted; stance (pi..2pi): front->back on ground
        q[f"{leg}_hip_yaw"] = X_SIGN[leg] * -stride * np.cos(ph)
        q[f"{leg}_hip_pitch"] = lift * max(0.0, np.sin(ph))
        q[f"{leg}_knee"] = 0.0
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--freq", type=float, default=1.5, help="steps per second per leg pair (Hz)")
    ap.add_argument("--stride", type=float, default=0.3, help="hip-yaw sweep amplitude (rad)")
    ap.add_argument("--lift", type=float, default=0.4, help="hip-pitch lift during swing (rad)")
    ap.add_argument("--seconds", type=float, default=10.0)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(str(ROOT / "model" / "clawer4.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    act = [m.actuator(i).name for i in range(m.nu)]

    def step():
        q = gait_targets(d.time, args.freq, args.stride, args.lift)
        d.ctrl[:] = [q[a] for a in act]
        mujoco.mj_step(m, d)

    if args.view:
        with mujoco.viewer.launch_passive(m, d) as v:
            while v.is_running():
                t0 = time.time()
                step()
                v.sync()
                time.sleep(max(0.0, m.opt.timestep - (time.time() - t0)))
        return

    from PIL import Image
    r = mujoco.Renderer(m, 360, 480)
    frames, max_tilt, max_qvel = [], 0.0, 0.0
    while d.time < args.seconds:
        step()
        up = d.xmat[1].reshape(3, 3)[:, 2]
        max_tilt = max(max_tilt, np.degrees(np.arccos(np.clip(up[2], -1, 1))))
        max_qvel = max(max_qvel, np.abs(d.qvel[6:]).max())
        if len(frames) < d.time * 25:
            r.update_scene(d, camera="track")
            frames.append(Image.fromarray(r.render()))
    frames[0].save(ROOT / "model" / "gait.gif", save_all=True, append_images=frames[1:], duration=40, loop=0)
    x, y = d.qpos[0], d.qpos[1]
    print(f"walked x={x:+.3f} m, y={y:+.3f} m in {args.seconds:.0f} s -> {x / args.seconds:+.3f} m/s; "
          f"max tilt {max_tilt:.1f} deg; max joint speed {max_qvel:.2f} rad/s (servo cap 4.71); "
          f"torso z {d.qpos[2]:.3f} m")


if __name__ == "__main__":
    main()
