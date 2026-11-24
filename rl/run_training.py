"""Run a training from the command line (used by the Kaggle notebook, where it runs in the background).

    python rl/run_training.py --terrain --out clawer4_p1.params
    python rl/run_training.py --terrain --load --out clawer4_p2.params   # P2: + servo load
    python rl/run_training.py --out practice.params --practice     # quick 7-minute check

Saves the brain (plus a backup at every score check) and videos in the current folder.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")

from train import make_video, train_policy  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="clawer4_policy.params")
ap.add_argument("--terrain", action="store_true", help="rough ground + levels (P1)")
ap.add_argument("--load", action="store_true", help="also feel servo load (P2)")
ap.add_argument("--practice", action="store_true", help="tiny run to check everything works")
args = ap.parse_args()

extra = dict(num_timesteps=500_000, num_envs=256, batch_size=64, num_evals=3) if args.practice else {}
make_policy, params = train_policy(terrain=args.terrain, load=args.load, out=args.out, **extra)

name = args.out.replace(".params", "")
videos = ([dict(level=2, kind=0, out=f"{name}_bumps.mp4"), dict(level=2, kind=2, out=f"{name}_stairs.mp4")]
          if args.terrain else [dict(out=f"{name}_walk.mp4")])
for v in videos:
    try:  # the brain is already saved, so a video problem must not crash the run
        make_video(make_policy, params, terrain=args.terrain, load=args.load, **v)
    except Exception as e:
        print(f"video {v['out']} failed: {e}", flush=True)
