"""Rough-ground terrain for CLAWER-4 (research plan step 4).

The ground is one big MuJoCo heightfield split into square tiles:
  5 columns = 5 kinds of ground, 6 rows = 6 difficulty levels (0 = almost flat, 5 = hardest).

  kind 0 "bumps"      random lumps
  kind 1 "slope"      a pit with sloped sides: walk uphill out of it
  kind 2 "stairs_up"  a pit with steps around it: climb out
  kind 3 "stairs_dn"  a hill with steps: walk down off it
  kind 4 "blocks"     random small boxes to step over

Each robot starts on the flat patch in the middle of a tile. The curriculum (added to the env next)
moves a robot one row harder when it walks well and one row easier when it falls.

Heights are kept small because the robot is small (body ~10 cm off the ground, foot lift ~4 cm).
"""
import numpy as np

KINDS = ["bumps", "slope", "stairs_up", "stairs_dn", "blocks"]
N_LEVELS = 6
TILE = 3.0            # m, side of one tile
RES = 0.03            # m, size of one heightfield cell
CENTER_FLAT = 0.3     # m, flat start patch (half-width) in the middle of every tile
STEP_WIDTH = 0.20     # m, depth of one stair step

# hardest level (level 5) values; level 0 is close to flat
MAX_BUMP = 0.03       # m, lump height
MAX_SLOPE_DEG = 20.0
MAX_STEP = 0.06       # m, stair height (the plan's 2-6 cm target)
MAX_BLOCK = 0.05      # m, box height

CELLS = round(TILE / RES)          # cells per tile side (100)


def tile_heights(kind, level, rng):
    """Height map (m) of one tile, CELLS x CELLS, indexed [y, x]."""
    f = level / (N_LEVELS - 1)                         # 0 .. 1 difficulty
    c = (np.arange(CELLS) + 0.5) * RES - TILE / 2      # cell centres, -1.5 .. 1.5 m
    x, y = np.meshgrid(c, c)                           # [y, x]
    r = np.maximum(np.abs(x), np.abs(y))               # square "radius" from the tile centre
    out = np.clip(r - CENTER_FLAT, 0.0, None)          # distance outside the start patch
    name = KINDS[kind]

    if name == "bumps":
        block = 2                                       # lumps of 2x2 cells = 6 cm
        n = CELLS // block
        lumps = rng.uniform(0, 0.004 + (MAX_BUMP - 0.004) * f, (n, n))
        h = np.kron(lumps, np.ones((block, block)))
        h[r < CENTER_FLAT] = 0.0
    elif name == "slope":
        h = np.tan(np.radians(2 + (MAX_SLOPE_DEG - 2) * f)) * np.minimum(out, 1.0)
    elif name in ("stairs_up", "stairs_dn"):
        step = 0.01 + (MAX_STEP - 0.01) * f
        n_steps = np.minimum(np.ceil(out / STEP_WIDTH), 5)    # 5 steps, then a flat top
        h = step * n_steps
        if name == "stairs_dn":
            h = h.max() - h                              # start on top, walk down
    elif name == "blocks":
        h = np.zeros((CELLS, CELLS))
        top = 0.01 + (MAX_BLOCK - 0.01) * f
        for _ in range(40):
            w, d = rng.uniform(0.08, 0.25, 2)
            cx, cy = rng.uniform(-TILE / 2, TILE / 2, 2)
            if max(abs(cx), abs(cy)) < CENTER_FLAT + 0.15:
                continue                                 # keep the start patch clear
            box = (np.abs(x - cx) < w / 2) & (np.abs(y - cy) < d / 2)
            h[box] = rng.uniform(0.3, 1.0) * top
    return h


def build(seed=0):
    """Whole ground: heights [y, x] in m (all >= 0), plus the start point of every tile."""
    rng = np.random.default_rng(seed)
    H = np.zeros((len(KINDS) * CELLS, N_LEVELS * CELLS))  # rows = y (kinds), cols = x (levels)
    starts = np.zeros((N_LEVELS, len(KINDS), 3))          # [level, kind] -> x, y, ground z
    size_x, size_y = N_LEVELS * TILE, len(KINDS) * TILE
    for lv in range(N_LEVELS):
        for k in range(len(KINDS)):
            h = tile_heights(k, lv, rng)
            H[k * CELLS:(k + 1) * CELLS, lv * CELLS:(lv + 1) * CELLS] = h
            cx = (lv + 0.5) * TILE - size_x / 2
            cy = (k + 0.5) * TILE - size_y / 2
            starts[lv, k] = cx, cy, h[CELLS // 2, CELLS // 2]
    return H, starts, (size_x / 2, size_y / 2)


def write_hfield(path, H):
    """MuJoCo's own heightfield file: int32 rows, int32 cols, then float32 heights (0..1).
    MuJoCo stores row 0 at the lowest y, which is how H is laid out."""
    top = H.max()
    with open(path, "wb") as fh:
        np.array(H.shape, dtype=np.int32).tofile(fh)
        (H / top).astype(np.float32).tofile(fh)
    return top
