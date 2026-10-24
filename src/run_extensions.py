"""
run_extensions.py — engineering extensions beyond the textbook benchmarks
=========================================================================
A. Bridge design: half-span bridge with a solid deck under uniform load.
   Pinned vs roller abutment -> the optimizer discovers a half-through arch vs a tied arch.
B. Stress check: the compliance-optimal MBB design is thresholded to black/white,
   refined 4x4, re-analysed without void elements, and its von Mises stresses are
   compared with the full solid beam.
C. Material trade-off: compliance vs volume fraction, compared with spreading the
   same material uniformly (a thinner solid plate).
"""

import os
import json
import functools
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import fem
import topopt
from style import INK, INK2, BLUE, ORANGE, AQUA, SURF, grid

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT, exist_ok=True)
NU = 0.3
KE = fem.q4_stiffness(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float), fem.plane_stress_D(1.0, NU))


def show(ax, x, title):
    ax.imshow(x, cmap="gray_r", vmin=0, vmax=1, origin="lower", interpolation="none")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("#c8c7c2")


def spec_of(problem, nelx, nely):
    s = topopt.PROBLEMS[problem](nelx, nely) if isinstance(problem, str) else problem(nelx, nely)
    return s if isinstance(s, dict) else dict(F=s[0], fixed=s[1])


# ======================================================================= analysis helpers
def analyse(solid, problem, Emin=None):
    """
    Linear analysis of a design on a structured unit-square grid.
    solid: (nely, nelx) array of element stiffness factors in [0, 1] (1 = solid).
    If Emin is None, elements with factor 0 are removed from the model; otherwise
    they keep stiffness Emin.
    Returns compliance, displacement field, node coordinates, element list and the
    mask of elements kept in the model.
    """
    nely, nelx = solid.shape
    nodes, elems = fem.rect_mesh(nelx, nely, nelx, nely)
    spec = spec_of(problem, nelx, nely)
    F, fixed = spec["F"], spec["fixed"]
    fac = solid.ravel().astype(float)
    if Emin is None:
        keep = fac > 0
    else:
        keep = np.ones_like(fac, bool)
        fac = Emin + fac * (1 - Emin)
    el = elems[keep]
    used = np.unique(el)
    edof = fem.element_dofs(el)
    ndof = 2 * len(nodes)
    iK = np.repeat(edof, 8, axis=1).ravel()
    jK = np.tile(edof, (1, 8)).ravel()
    sK = (KE.ravel()[:, None] * fac[keep][None, :]).T.ravel()
    K = sp.csc_matrix((sK, (iK, jK)), shape=(ndof, ndof))
    active = np.zeros(ndof, bool)
    active[np.r_[2 * used, 2 * used + 1]] = True
    loaded = np.nonzero(F)[0]
    if not active[loaded].all():
        raise RuntimeError("A loaded node is not connected to the structure.")
    active[fixed] = False
    free = np.nonzero(active)[0]
    U = np.zeros(ndof)
    U[free] = spla.spsolve(K[free][:, free], F[free], permc_spec="MMD_AT_PLUS_A")
    return float(F @ U), U, nodes, elems, keep


def element_von_mises(nodes, elems, keep, U, h):
    """Element-average von Mises stress (E = 1) for square elements of size h."""
    D = fem.plane_stress_D(1.0, NU)
    sq = np.array([[0, 0], [h, 0], [h, h], [0, h]], float)
    Bg = np.array([fem.q4_B(sq, xi, eta)[0] for xi, eta in fem.GAUSS])      # (4, 3, 8)
    ue = U[fem.element_dofs(elems[keep])]                                     # (n, 8)
    sig = np.einsum("ij,gjk,nk->ngi", D, Bg, ue)                             # (n, 4, 3)
    vm = fem.von_mises(sig).mean(axis=1)
    out = np.full(len(elems), np.nan)
    out[keep] = vm
    return out


def refine(x, r):
    return np.kron(x, np.ones((r, r)))


def to_black_white(x, volfrac):
    """Volume-preserving threshold: keep the volfrac fraction of elements with the highest density."""
    thr = np.quantile(x, 1.0 - volfrac)
    return (x >= thr).astype(float)


def mbb_pad(nelx, nely, pad=2.0 / 150):
    """
    MBB half beam with the load and the roller each spread over a short pad
    (pad * nelx long) instead of a single node, to avoid point-load stress singularities.
    """
    ndof = 2 * (nelx + 1) * (nely + 1)
    n = max(1, int(round(pad * nelx)))
    F = np.zeros(ndof)
    w = np.ones(n + 1); w[-1] = 0.5; w[0] = 0.5          # trapezoidal tributary weights
    w /= w.sum()
    for i in range(n + 1):
        F[2 * topopt.node_id(i, nely, nelx) + 1] = -w[i]
    left = np.array([topopt.node_id(0, j, nelx) for j in range(nely + 1)])
    pad_nodes = np.array([topopt.node_id(nelx - i, 0, nelx) for i in range(n + 1)])
    return F, np.r_[2 * left, 2 * pad_nodes + 1]


# ======================================================================= A. bridge
def bridge_study(summary):
    print("A. Bridge: pinned vs roller abutment (half span 120x80, volfrac 0.25)")
    res = {}
    for sup in ("pin", "roller"):
        p = functools.partial(topopt.bridge, support=sup)
        r = topopt.top(120, 80, 0.25, 3.0, 3.0, ft="density", problem=p, ctol=1e-4)
        res[sup] = r
        print(f"   {sup:6s}  c = {r['c']:.3f}  iterations {r['iters']}  stop: {r['stop']}")
        summary[f"bridge_{sup}"] = dict(c=round(r["c"], 4), iters=r["iters"])
    ratio = res["roller"]["c"] / res["pin"]["c"]
    print(f"   roller / pin compliance ratio = {ratio:.2f}")
    summary["bridge_compliance_ratio_roller_over_pin"] = round(ratio, 3)

    fig, axs = plt.subplots(2, 1, figsize=(7.5, 5.6))
    labels = {"pin": "Pinned abutments (take thrust): half-through arch",
              "roller": "Roller abutments (no thrust): a tie appears, tied arch"}
    for ax, sup in zip(axs, ("pin", "roller")):
        x = res[sup]["x"]
        show(ax, np.hstack([x, x[:, ::-1]]), f"{labels[sup]}   c = {res[sup]['c']:.2f}")
        ny, nx2 = x.shape[0], 2 * x.shape[1]
        jt = int(round(0.3 * ny)) + 3
        for xa in np.linspace(0, nx2 - 1, 25):                               # deck load arrows
            ax.annotate("", xy=(xa, jt), xytext=(xa, jt + 7),
                        arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=0.8, mutation_scale=6))
        for xs in (0, nx2 - 1):                                              # supports
            ax.plot(xs, -1.5, marker="^" if sup == "pin" else "o", color=AQUA if sup == "pin" else BLUE,
                    ms=9, clip_on=False)
        ax.set_ylim(-4, ny)
    fig.suptitle("Same deck, same load, same material: the support condition decides the structural system",
                 x=0.02, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_bridge.png"), dpi=170); plt.close(fig)


# ======================================================================= B. stress check
def stress_study(summary, nelx=150, nely=50, r=4, R_sing=10.0):
    print("\nB. Stress check of the MBB design (150x50, volfrac 0.5)")
    opt = topopt.top(nelx, nely, 0.5, 3.0, 6.0, ft="density", ctol=1e-4)
    x = opt["x"]
    bw = to_black_white(x, 0.5)

    c_gray, *_ = analyse(x ** 3.0, "mbb", Emin=1e-9)                # SIMP-penalized gray design
    c_bw, *_ = analyse(bw, "mbb", Emin=1e-9)                        # thresholded, same mesh
    # refined analyses use the padded load/support (same total load) to avoid point singularities
    c_fine, U, nodes, elems, keep = analyse(refine(bw, r), mbb_pad)   # thresholded, refined, void removed
    c_solid, Us, nodes_s, elems_s, keep_s = analyse(np.ones((nely * r, nelx * r)), mbb_pad)
    vol_bw = bw.mean()
    print(f"   point load, design mesh: optimizer (gray, penalized) c = {c_gray:.2f}, "
          f"black/white c = {c_bw:.2f}")
    print(f"   padded load, refined x{r}: black/white c = {c_fine:.2f}, full solid c = {c_solid:.2f}")
    print(f"   black/white volume fraction {vol_bw:.3f}")

    h = 1.0 / r
    vm = element_von_mises(nodes * h, elems, keep, U, h)
    vm_s = element_von_mises(nodes_s * h, elems_s, keep_s, Us, h)
    # element centres in coarse units
    ny_f, nx_f = nely * r, nelx * r
    cx = (np.arange(nx_f) + 0.5) * h
    cy = (np.arange(ny_f) + 0.5) * h
    CX, CY = np.meshgrid(cx, cy)
    sing_pts = [(1.0, float(nely)), (float(nelx) - 1.0, 0.0)]       # load pad, support pad
    far = np.ones_like(CX, bool)
    for px, py in sing_pts:
        far &= np.hypot(CX - px, CY - py) > R_sing
    far = far.ravel()

    def peak(v):
        vv = np.where(far & np.isfinite(v), v, -np.inf)
        k = int(np.argmax(vv))
        return vv[k], (CX.ravel()[k], CY.ravel()[k])

    pk, loc = peak(vm)
    pk_s, loc_s = peak(vm_s)
    mean_vm = np.nanmean(np.where(far, vm, np.nan))
    print(f"   peak von Mises outside singular zones: optimized {pk:.3f} at ({loc[0]:.1f}, {loc[1]:.1f}), "
          f"full solid {pk_s:.3f} at ({loc_s[0]:.1f}, {loc_s[1]:.1f})")
    print(f"   optimized design: peak / mean stress in material = {pk / mean_vm:.2f}")
    print(f"   peak stress, optimized / full solid = {pk / pk_s:.2f}  (material used {vol_bw:.1%})")
    summary["stress_check"] = dict(
        c_gray=round(c_gray, 3), c_blackwhite=round(c_bw, 3), c_blackwhite_refined=round(c_fine, 3),
        c_full_solid=round(c_solid, 3), volume_blackwhite=round(vol_bw, 4),
        peak_vm_optimized=round(pk, 4), peak_vm_full_solid=round(pk_s, 4),
        peak_location_optimized=[round(loc[0], 2), round(loc[1], 2)],
        peak_over_mean_optimized=round(pk / mean_vm, 3))

    vmax = np.nanpercentile(np.where(far, vm, np.nan), 99.5)
    img = vm.reshape(ny_f, nx_f)
    fig, ax = plt.subplots(figsize=(8, 3.3))
    cmap = plt.get_cmap("Blues").copy(); cmap.set_bad(SURF)
    im = ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax, origin="lower", interpolation="none",
                   extent=(0, nelx, 0, nely))
    for px, py in sing_pts:
        ax.add_patch(Circle((px, py), R_sing, fill=False, ls="--", lw=1, ec=INK2))
    ax.plot(*loc, marker="o", mfc="none", mec=ORANGE, mew=2, ms=12)
    ax.annotate(f"peak = {pk / mean_vm:.1f}× mean stress", xy=loc, xytext=(nelx * 0.42, nely * 0.42),
                color=INK, fontsize=9, arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1))
    ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    ax.set_title("Von Mises stress, black/white design refined ×4 (dashed: load-introduction zones, excluded)",
                 fontsize=10)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
    cb.set_label("von Mises (relative)", color=INK2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_stress_check.png"), dpi=170); plt.close(fig)


# ======================================================================= C. trade-off
def tradeoff_study(summary, nelx=120, nely=40):
    print("\nC. Material trade-off (MBB 120x40)")
    c_full, *_ = analyse(np.ones((nely, nelx)), "mbb")
    vfs = [0.3, 0.4, 0.5, 0.6, 0.7]   # below ~0.3 members become 1-2 elements thick on this mesh
    rows = []
    designs = []
    for v in vfs:
        r = topopt.top(nelx, nely, v, 3.0, 0.04 * nelx, ft="density", ctol=1e-4)
        bw = to_black_white(r["x"], v)              # the part that would actually be made
        vb = bw.mean()
        c_bw, *_ = analyse(bw, "mbb", Emin=1e-9)
        c_uniform = c_full / vb                     # same amount of material as a thinner solid plate
        rows.append((vb, c_bw, c_uniform, c_uniform / c_bw))
        designs.append(bw)
        print(f"   target {v:.1f}  black/white volume {vb:.3f}  optimized c = {c_bw:8.2f}   "
              f"uniform plate c = {c_uniform:8.2f}   stiffness gain {c_uniform / c_bw:.2f}x")
    summary["tradeoff"] = [dict(volume=round(v, 4), c_opt_blackwhite=round(co, 3), c_uniform=round(cu, 3),
                                gain=round(g, 3)) for v, co, cu, g in rows]

    fig = plt.figure(figsize=(9, 4.6))
    gs = fig.add_gridspec(2, len(vfs), height_ratios=[0.55, 3.2], hspace=0.12)
    for k, (v, x) in enumerate(zip(vfs, designs)):
        ax = fig.add_subplot(gs[0, k])
        show(ax, np.hstack([x[:, ::-1], x]), f"{rows[k][0] * 100:.0f}% material")
    ax = fig.add_subplot(gs[1, :])
    v = np.array([r[0] for r in rows]); co = np.array([r[1] for r in rows]); cu = np.array([r[2] for r in rows])
    ax.plot(v * 100, cu, "o-", color=INK2, lw=2, ms=6, label="Same material, uniform thinner plate")
    ax.plot(v * 100, co, "o-", color=BLUE, lw=2, ms=6, label="Topology optimized (black/white)")
    for vi, ci, g in zip(v, co, cu / co):
        ax.annotate(f"{g:.2f}× stiffer", xy=(vi * 100, ci), xytext=(0, 9), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_yscale("log")
    ax.set_xlabel("Material used (% of design domain)")
    ax.set_ylabel("Compliance (log scale, lower = stiffer)")
    ax.legend(frameon=False, fontsize=9)
    grid(ax)
    fig.suptitle("How much stiffness topology optimization buys at each material budget", x=0.02, ha="left",
                 fontsize=11)
    fig.savefig(os.path.join(OUT, "fig_tradeoff.png"), dpi=170, bbox_inches="tight"); plt.close(fig)


def main():
    summary = {}
    bridge_study(summary)
    stress_study(summary)
    tradeoff_study(summary)
    with open(os.path.join(OUT, "extensions_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved fig_bridge.png, fig_stress_check.png, fig_tradeoff.png, extensions_summary.json")


if __name__ == "__main__":
    main()
