"""
run_benchmarks.py — verification and benchmark runs for topopt.py
=================================================================
1. Reproduce Andreassen et al. (2011), Fig. 3: MBB beam 60x20, volfrac 0.5, penal 3,
   rmin 2.4 -> c = 216.81 (sensitivity filter) and 233.71 (density filter)
2. Mesh refinement: MBB at 60x20, 120x40, 240x80 with rmin = 0.04 * nelx
3. Cantilever beam, 160x80
4. Animation of the optimization history (GIF)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import topopt
from style import INK, INK2, BLUE, ORANGE, AQUA, grid

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT, exist_ok=True)
PUBLISHED = {"sensitivity": 216.81, "density": 233.71}


def full_mbb(x):
    """Mirror the half-MBB design to show the whole beam."""
    return np.hstack([x[:, ::-1], x])


def show(ax, x, title):
    ax.imshow(x, cmap="gray_r", vmin=0, vmax=1, origin="lower", interpolation="none")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("#c8c7c2")


def main():
    summary = {}

    # ---------------------------------------------------------- 1. published benchmark
    print("1. Published benchmark (Andreassen et al. 2011, Fig. 3)")
    bench = {}
    for ft in ("sensitivity", "density"):
        r = topopt.top(60, 20, 0.5, 3.0, 2.4, ft=ft)
        bench[ft] = r
        err = (r["c"] - PUBLISHED[ft]) / PUBLISHED[ft] * 100
        print(f"   {ft:12s} c = {r['c']:.2f}  published {PUBLISHED[ft]:.2f}  "
              f"diff {err:+.3f}%  iterations {r['iters']}")
        summary[f"mbb60_{ft}"] = dict(c=round(r["c"], 4), published=PUBLISHED[ft], iters=r["iters"])
    fig, axs = plt.subplots(2, 1, figsize=(7, 3.5))
    for ax, ft in zip(axs, ("sensitivity", "density")):
        r = bench[ft]
        show(ax, full_mbb(r["x"]),
             f"{ft.capitalize()} filter:  c = {r['c']:.2f}   (published {PUBLISHED[ft]:.2f})")
    fig.suptitle("MBB beam 60×20 (half), volfrac 0.5, p = 3, rmin = 2.4", x=0.02, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_mbb_benchmark.png"), dpi=170); plt.close(fig)

    # ---------------------------------------------------------- 2. mesh refinement
    print("\n2. Mesh refinement, density filter, rmin = 0.04 * nelx")
    fig, axs = plt.subplots(3, 1, figsize=(7, 4.9))
    for ax, (nx, ny) in zip(axs, [(60, 20), (120, 40), (240, 80)]):
        r = topopt.top(nx, ny, 0.5, 3.0, 0.04 * nx, ft="density", ctol=1e-4)
        print(f"   {nx:3d} x {ny:2d}  c = {r['c']:.2f}  iterations {r['iters']}  "
              f"stop: {r['stop']}")
        summary[f"mbb_refine_{nx}x{ny}"] = dict(c=round(r["c"], 4), iters=r["iters"])
        show(ax, full_mbb(r["x"]), f"{nx}×{ny} elements (half beam):  c = {r['c']:.2f}")
    fig.suptitle("Same physical filter size on three meshes: the layout is mesh-independent",
                 x=0.02, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_mbb_mesh_refinement.png"), dpi=170)
    plt.close(fig)

    # ---------------------------------------------------------- 3. cantilever
    print("\n3. Cantilever 160x80, volfrac 0.4, density filter")
    rc = topopt.top(160, 80, 0.4, 3.0, 4.0, ft="density", problem="cantilever", ctol=1e-4)
    print(f"   c = {rc['c']:.2f}  iterations {rc['iters']}  stop: {rc['stop']}")
    summary["cantilever_160x80"] = dict(c=round(rc["c"], 4), iters=rc["iters"])
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    show(ax, rc["x"], f"Cantilever 160×80, volfrac 0.4:  c = {rc['c']:.2f}")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_cantilever.png"), dpi=170); plt.close(fig)

    # ---------------------------------------------------------- 4. animation + history
    print("\n4. Animation run: MBB 150x50, density filter")
    ra = topopt.top(150, 50, 0.5, 3.0, 6.0, ft="density", record_every=2, ctol=1e-4)
    print(f"   c = {ra['c']:.2f}  iterations {ra['iters']}  stop: {ra['stop']}  frames {len(ra['snaps'])}")
    frames = ra["snaps"]
    fig, ax = plt.subplots(figsize=(7, 1.9))
    fig.subplots_adjust(0.01, 0.02, 0.99, 0.85)
    im = ax.imshow(full_mbb(frames[0]), cmap="gray_r", vmin=0, vmax=1, origin="lower",
                   interpolation="none")
    ax.set_xticks([]); ax.set_yticks([])
    ttl = ax.set_title("", fontsize=10)

    def update(k):
        im.set_data(full_mbb(frames[k]))
        it = min(1 + 2 * k, ra["iters"]) if k < len(frames) - 1 else ra["iters"]
        ttl.set_text(f"Iteration {it}   compliance {ra['hist'][it - 1]:.1f}")
        return im, ttl

    seq = list(range(len(frames))) + [len(frames) - 1] * 15        # hold the final frame
    anim = FuncAnimation(fig, update, frames=seq, blit=False)
    anim.save(os.path.join(OUT, "mbb_optimization.gif"), writer=PillowWriter(fps=12), dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(np.arange(1, len(ra["hist"]) + 1), ra["hist"], color=BLUE, lw=2)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Compliance c (log scale)")
    ax.set_title("Convergence history, MBB 150×50")
    grid(ax)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_convergence_history.png"), dpi=170)
    plt.close(fig)
    summary["mbb_150x50_animation"] = dict(c=round(ra["c"], 4), iters=ra["iters"])

    with open(os.path.join(OUT, "benchmark_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved figures, GIF and benchmark_summary.json to results/")


if __name__ == "__main__":
    main()
