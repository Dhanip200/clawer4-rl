"""Draw the graphs for the results of rl/evaluate.py (research plan step 6).

    .venv/Scripts/python tools/plot_results.py results_b1_p1.csv --out docs/results_b1_p1

Writes <out>_success.png (success rate with its 95% range) and <out>_speed.png (average forward speed).
One small graph per kind of ground; the exact numbers are in the .csv.
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# A brain keeps its colour in every graph. B1 is the baseline, so it is grey; P1-P3 are colours checked
# for colour-blind readers. Marker shapes differ too, so colour is never the only clue.
STYLE = {"B1": ("#8a8984", "s"), "P1": ("#2a78d6", "o"), "P2": ("#eb6834", "^"), "P3": ("#1baf7a", "D")}
TITLES = {"bumps": "Bumps", "slope": "Slope", "stairs_up": "Stairs up", "stairs_dn": "Stairs down", "blocks": "Blocks"}
# what "level" means on each kind of ground (level 0 -> level 5, from rl/terrain.py)
MEANING = {"bumps": "lumps up to 0.4-3 cm", "slope": "2-20 degrees", "stairs_up": "steps 1-6 cm",
           "stairs_dn": "steps 1-6 cm", "blocks": "boxes up to 1-5 cm"}


def read(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["level"], r["tries"] = int(r["level"]), int(r["tries"])
        for k in ("success_rate", "low95", "high95", "speed"):
            r[k] = float(r[k]) if r[k] != "" else float("nan")
    return rows


def draw(rows, what, out):
    kinds = [k for k in TITLES if any(r["kind"] == k for r in rows)]
    brains = list(dict.fromkeys(r["brain"] for r in rows))
    unknown = [b for b in brains if b not in STYLE]
    if unknown:
        raise SystemExit(f"add a colour and marker for {unknown} to STYLE")
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": INK, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, len(kinds), figsize=(2.5 * len(kinds) + 0.6, 3.4), sharey=True, squeeze=False,
                             facecolor=SURFACE)
    tries = sorted({r["tries"] for r in rows})
    for ax, kind in zip(axes[0], kinds):
        ax.set_facecolor(SURFACE)
        for b in brains:
            mine = sorted((r for r in rows if r["brain"] == b and r["kind"] == kind), key=lambda r: r["level"])
            x = [r["level"] for r in mine]
            color, marker = STYLE[b]
            if what == "success":
                y = [100 * r["success_rate"] for r in mine]
                err = [[100 * (r["success_rate"] - r["low95"]) for r in mine],
                       [100 * (r["high95"] - r["success_rate"]) for r in mine]]
                ax.errorbar(x, y, yerr=err, fmt="none", ecolor=color, elinewidth=1.2, capsize=0, alpha=0.6, zorder=2)
            else:
                y = [r["speed"] for r in mine]
            ax.plot(x, y, color=color, lw=2, solid_capstyle="round", solid_joinstyle="round", marker=marker, ms=7.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=b, zorder=3, clip_on=False)
        ax.set_title(TITLES[kind], loc="left", fontsize=10, fontweight="bold", pad=18)
        ax.text(0, 1.03, MEANING[kind], transform=ax.transAxes, fontsize=8, color=MUTED)
        ax.set_xlabel("level")
        ax.set_xticks(sorted({r["level"] for r in rows}))
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(length=0)
        ax.tick_params(axis="x", pad=9)
    first = axes[0][0]
    if what == "success":
        first.set_ylim(0, 102)
        first.set_yticks([0, 20, 40, 60, 80, 100])
        first.set_ylabel("robots that got across (%)")
        title = "Success rate by ground type and level"
        note = f"{', '.join(map(str, tries))} tries per point; thin bars = 95% range"
    else:
        first.set_ylim(bottom=min(0, first.get_ylim()[0]))
        first.set_ylabel("average forward speed (m/s)")
        title = "Forward speed by ground type and level"
        note = f"{', '.join(map(str, tries))} tries per point, including robots that fell or got stuck"
    fig.suptitle(title, x=0.01, y=0.985, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, 0.895, note, fontsize=8.5, color=MUTED)
    handles, labels = first.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=len(labels), frameon=False, bbox_to_anchor=(0.995, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f"{out}_{what}.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {out}_{what}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    data = read(a.csv)
    draw(data, "success", a.out)
    draw(data, "speed", a.out)
