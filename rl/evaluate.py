"""Test trained brains on every kind of rough ground and every level (research plan step 6).

    python rl/evaluate.py --brain B1=clawer4_policy.params --brain P1=clawer4_p1.params
    python rl/evaluate.py --brain P2=clawer4_p2.params:load --brain P3=clawer4_p3.params:load,history=10
    python rl/evaluate.py --brain P1=clawer4_p1.params --n 2 --levels 0 2 --kinds 0 2     # quick check on a PC

Every robot starts on the flat patch in the middle of a tile, facing a random way, and is told to walk
straight ahead. Each brain gets exactly the same start positions. One try ends as:
  success  it got CROSS_R (1.2 m, past the last stair step) from the tile centre without falling
  fell     it tipped over or its body dropped too low before that
  stuck    neither happened before the time ran out

Plan: 1,000 tries per level = 200 robots x 5 kinds of ground. Writes a .csv with one row per
brain + kind + level, a second one (<out>_robots.csv) with one row per robot, and prints tables.
Brax is not needed (the brain file is read directly).
"""
import argparse
import csv
import math
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx

from clawer4_env import CTRL_DT, MAX_TORQUE, Clawer4Walk
from terrain import KINDS, MAX_STEP

# m: square distance from the tile centre. The last stair step is at 1.1 m. Not further out: some neighbouring
# tiles start higher (a wall at 1.5 m), and a robot's front feet reach it when its middle is at about 1.28 m.
CROSS_R = 1.2
CHUNK = 50           # control steps per jitted block (1 s)


# ---------- brains ----------

class _Plain:
    """Stands in for Brax classes inside the saved file, so Brax is not needed to read it."""
    def __init__(self, *args, **kw):
        self.__dict__.update(kw)

    def __setstate__(self, state):
        self.__dict__.update(state)


class _Reader(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("brax"):
            return type(name, (_Plain,), {})
        return super().find_class(module, name)


def load_brain(path):
    """Returns policy(obs) -> action in [-1, 1] (no random exploration) and the input size.
    Only open files you trained yourself: pickle files can run code."""
    with open(path, "rb") as f:
        saved = _Reader(f).load()
    norm, layers = saved[0], saved[1]["params"]
    mean, std = jp.asarray(norm.mean), jp.asarray(norm.std)
    names = sorted(layers, key=lambda n: int(n.split("_")[-1]))
    weights = [(jp.asarray(layers[n]["kernel"]), jp.asarray(layers[n]["bias"])) for n in names]

    def policy(obs):
        x = (obs - mean) / std
        for i, (w, b) in enumerate(weights):
            x = x @ w + b
            if i < len(weights) - 1:
                x = jax.nn.swish(x)
        return jp.tanh(x[..., : x.shape[-1] // 2])       # first half = mean, second half = spread

    return policy, mean.shape[-1]


def parse_brain(text):
    """'P3=file.params:load,history=10' -> name, path, env options"""
    name, _, rest = text.partition("=")
    path, colon, opts = rest.rpartition(":")
    if not colon or opts.startswith(("\\", "/")):        # no options (a Windows path has a ':' too)
        path, opts = rest, ""
    env_args = {}
    for o in filter(None, opts.split(",")):
        key, _, val = o.partition("=")
        if key == "load":
            env_args["load"] = True
        elif key == "history":
            env_args["history"] = int(val)
        else:
            raise SystemExit(f"unknown option '{o}' in --brain {text}")
    return name, path, env_args


# ---------- one brain on many robots ----------

def start_state(env, key, level, kind, command):
    """Like env.reset, but on a chosen tile and with a fixed command."""
    k_reset, k_start, k_noise = jax.random.split(key, 3)
    s = env.reset(k_reset)
    q = env._start_qpos(k_start, level, kind)
    data = mjx.make_data(env.mjx_model).replace(qpos=q, ctrl=q[7:])
    data = mjx.forward(env.mjx_model, data)
    info = {**s.info, "level": level, "kind": kind, "origin": q[:2], "command": command}
    info["hist"] = env._fresh_history(env._observe(data, info, k_noise))
    # env.step adds these two (Brax's wrapper normally does): put them in now so the state keeps one shape
    info["first_pipeline_state"], info["first_obs"] = data, info["hist"].ravel()
    return s.replace(pipeline_state=data, obs=info["hist"].ravel(), info=info)


def heading(env, data):
    f = data.xmat[env.torso_id][:2, 0]
    return f / jp.maximum(jp.linalg.norm(f), 1e-6)


def run(env, policy, levels, kinds, keys, command, seconds):
    """levels, kinds, keys: one entry per robot. Returns a dict of numpy arrays, one value per robot."""
    states = jax.jit(jax.vmap(lambda k, lv, kd: start_state(env, k, lv, kd, command)))(keys, levels, kinds)
    centres = env.starts[levels, kinds, :2]
    n = len(levels)
    rec = {"time": jp.zeros(n), "forward": jp.zeros(n), "effort": jp.zeros(n),
           "success": jp.zeros(n, bool), "fell": jp.zeros(n, bool), "bad": jp.zeros(n, bool),
           "far": jp.zeros(n), "x0": states.pipeline_state.qpos[:, :2], "x1": states.pipeline_state.qpos[:, :2],
           "h0": jax.vmap(lambda d: heading(env, d))(states.pipeline_state)}

    def one(s, r, c):
        active = ~(r["success"] | r["fell"])
        before = s.pipeline_state.qpos[:2]
        # keep the command fixed: env.step would pick a new random one after 10 s
        s = s.replace(info={**s.info, "command": command, "step": jp.zeros_like(s.info["step"])})
        s = env.step(s, policy(s.obs))
        d = s.pipeline_state
        bad = ~jp.isfinite(d.qpos).all()                 # physics blew up: counts as a fall, reported
        fell = (s.done > 0) | bad
        dist = jp.max(jp.abs(d.qpos[:2] - c))
        crossed = dist >= CROSS_R
        upd = lambda new, old: jp.where(active, new, old)  # noqa: E731
        return s, {
            "time": upd(r["time"] + CTRL_DT, r["time"]),
            "forward": upd(r["forward"] + jp.dot(d.qpos[:2] - before, heading(env, d)), r["forward"]),
            "effort": upd(r["effort"] + jp.mean(jp.abs(d.actuator_force)) / MAX_TORQUE, r["effort"]),
            "fell": r["fell"] | (active & fell),
            "success": r["success"] | (active & crossed & ~fell),
            "bad": r["bad"] | (active & bad),
            "far": upd(jp.maximum(r["far"], dist), r["far"]),
            "x0": r["x0"], "x1": upd(d.qpos[:2], r["x1"]), "h0": r["h0"]}

    @jax.jit
    def chunk(s, r):
        def body(carry, _):
            return jax.vmap(one)(*carry, centres), None
        return jax.lax.scan(body, (s, r), None, length=CHUNK)[0]

    for _ in range(math.ceil(seconds / CTRL_DT / CHUNK)):
        states, rec = chunk(states, rec)
        if bool((rec["success"] | rec["fell"]).all()):
            break
    r = {k: np.asarray(v) for k, v in rec.items()}
    steps = np.maximum(r["time"] / CTRL_DT, 1)
    move = r["x1"] - r["x0"]
    side = r["h0"][:, 0] * move[:, 1] - r["h0"][:, 1] * move[:, 0]     # left of the starting direction
    return {"success": r["success"], "fell": r["fell"], "bad": r["bad"], "time": r["time"],
            "speed": r["forward"] / np.maximum(r["time"], CTRL_DT),
            "drift": np.abs(side), "effort": r["effort"] / steps * 100, "far": r["far"],
            "start_heading_deg": np.degrees(np.arctan2(r["h0"][:, 1], r["h0"][:, 0]))}


# ---------- tables ----------

def wilson(k, n, z=1.96):
    """95% range for a success rate of k out of n (works for small n and for 0% / 100% too)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    mid = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, mid - half), min(1.0, mid + half)


def summarise(name, kind, level, r, pick):
    n, k = int(pick.sum()), int(r["success"][pick].sum())
    lo, hi = wilson(k, n)
    fell = int(r["fell"][pick].sum())
    # averages skip robots whose physics blew up (they would turn the average into nan)
    avg = lambda v: round(float(np.mean(v[pick & ~r["bad"]])), 4) if (pick & ~r["bad"]).any() else ""  # noqa: E731
    return {"brain": name, "kind": KINDS[kind], "level": level, "tries": n, "success": k, "fell": fell,
            "stuck": n - k - fell, "success_rate": round(k / n, 4), "low95": round(lo, 4), "high95": round(hi, 4),
            "speed": avg(r["speed"]), "drift_m": avg(r["drift"]), "effort_pct": avg(r["effort"]),
            "time_s": avg(r["time"]), "physics_errors": int(r["bad"][pick].sum())}


def print_tables(rows, kinds, levels):
    for name in dict.fromkeys(r["brain"] for r in rows):
        mine = {(r["kind"], r["level"]): r for r in rows if r["brain"] == name}
        print(f"\n{name}: success rate (fell / stuck), forward speed in m/s")
        print("  " + "kind".ljust(11) + "".join(f"level {lv}".center(24) for lv in levels))
        for k in kinds:
            cells = []
            for lv in levels:
                r = mine[(KINDS[k], lv)]
                speed = "  -  " if r["speed"] == "" else f"{r['speed']:.3f}"
                cells.append(f"{r['success_rate']:4.0%} ({r['fell']}/{r['stuck']}) {speed}".center(24))
            print("  " + KINDS[k].ljust(11) + "".join(cells))
        for kind in ("stairs_up", "stairs_dn"):
            got = [lv for lv in levels if (kind, lv) in mine and mine[(kind, lv)]["success_rate"] >= 0.8]
            if any((kind, lv) in mine for lv in levels):
                best = f"{(0.01 + (MAX_STEP - 0.01) * max(got) / 5) * 100:.0f} cm" if got else "none of the tested steps"
                print(f"  highest {kind} step with >= 80% success: {best}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", action="append", required=True, help="NAME=file.params[:load,history=10]")
    ap.add_argument("--n", type=int, default=200, help="robots per kind and level (200 x 5 kinds = 1,000 per level)")
    ap.add_argument("--kinds", type=int, nargs="+", default=list(range(len(KINDS))))
    ap.add_argument("--levels", type=int, nargs="+", default=list(range(6)))
    ap.add_argument("--speed", type=float, default=0.2, help="commanded forward speed, m/s")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--batch", type=int, default=2000, help="robots simulated at the same time")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()
    if args.seconds >= 20:
        raise SystemExit("--seconds must be under 20: env episodes end at 20 s")

    command = jp.array([args.speed, 0.0, 0.0])
    grid = [(lv, k, i) for lv in args.levels for k in args.kinds for i in range(args.n)]
    levels = np.array([g[0] for g in grid], np.int32)
    kinds = np.array([g[1] for g in grid], np.int32)
    # each robot's key depends only on (seed, level, kind, robot number): the same for every brain and batch size
    ids = jp.array([lv * 1_000_000 + k * 100_000 + i for lv, k, i in grid], jp.uint32)
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(args.seed), i))(ids)

    rows, robots = [], []
    for text in args.brain:
        name, path, env_args = parse_brain(text)
        policy, n_in = load_brain(path)
        env = Clawer4Walk(terrain=True, **env_args)
        if env.observation_size != n_in:
            raise SystemExit(f"{name}: the brain takes {n_in} numbers but the env gives {env.observation_size} "
                             f"(missing :load or history=?)")
        t0, parts = time.time(), []
        for a in range(0, len(grid), args.batch):
            b = slice(a, a + args.batch)
            parts.append(run(env, policy, jp.array(levels[b]), jp.array(kinds[b]), keys[b], command, args.seconds))
            print(f"{name}: {min(a + args.batch, len(grid))}/{len(grid)} robots, {(time.time() - t0) / 60:.1f} min", flush=True)
        r = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        for k in args.kinds:
            for lv in args.levels:
                rows.append(summarise(name, k, lv, r, (kinds == k) & (levels == lv)))
        robots += [{"brain": name, "kind": KINDS[k], "level": lv, "robot": i,
                    **{m: (int(r[m][j]) if r[m].dtype == bool else round(float(r[m][j]), 4)) for m in r}}
                   for j, (lv, k, i) in enumerate(grid)]
        with open(str(Path(args.out).with_suffix("")) + "_robots.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(robots[0]))
            w.writeheader()
            w.writerows(robots)
        with open(args.out, "w", newline="") as f:       # saved after every brain, so a crash loses little
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print_tables(rows, args.kinds, args.levels)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
