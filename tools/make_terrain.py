"""Build model/clawer4_terrain.xml: the same robot, but on rough ground instead of a flat floor.

Run:  .venv\\Scripts\\python tools\\make_terrain.py
Look: .venv\\Scripts\\python -m mujoco.viewer --mjcf=model\\clawer4_terrain.xml
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl"))
import terrain  # noqa: E402

FLAT_FLOOR = '<geom name="floor" type="plane" size="0 0 0.05" material="grid" contype="1" conaffinity="1"/>'


def main(seed=0):
    H, starts, (rx, ry) = terrain.build(seed)
    top = terrain.write_hfield(ROOT / "model" / "terrain.bin", H)
    np.save(ROOT / "model" / "terrain_starts.npy", starts)

    xml = (ROOT / "model" / "clawer4.xml").read_text()
    assert FLAT_FLOOR in xml, "floor line in clawer4.xml changed: update FLAT_FLOOR"
    hfield = (f'<hfield name="terrain" file="../terrain.bin" size="{rx} {ry} {top:.4f} 0.05"/>\n'
              '    <texture name="grid"')
    xml = xml.replace('<texture name="grid"', hfield, 1)
    xml = xml.replace(FLAT_FLOOR, '<geom name="floor" type="hfield" hfield="terrain" material="grid" '
                                  'contype="1" conaffinity="1"/>')
    xml = xml.replace('<mujoco model="clawer4">',
                      '<!-- Made by tools/make_terrain.py: clawer4.xml on rough ground. -->\n'
                      '<mujoco model="clawer4_terrain">', 1)
    (ROOT / "model" / "clawer4_terrain.xml").write_text(xml)
    print(f"terrain {H.shape[1]}x{H.shape[0]} cells, {2 * rx:.0f} x {2 * ry:.0f} m, highest point {top:.3f} m")
    print("wrote model/clawer4_terrain.xml, model/terrain.bin, model/terrain_starts.npy")


if __name__ == "__main__":
    main()
