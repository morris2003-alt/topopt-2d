"""
lbracket.py — L-bracket benchmark for stress-based topology optimization
=========================================================================
Step 1 of the stress-constrained extension: define the L-bracket problem and run the
ordinary minimum-compliance optimizer on it, to show where the stress concentrates
when stress is ignored.

Geometry (square design domain of side L = nelx = nely, y pointing up):
    - vertical arm   0 <= x <= 0.4 L,  0 <= y <= L
    - horizontal arm 0 <= x <= L,      0 <= y <= 0.4 L
    - the upper-right block (x > 0.4 L and y > 0.4 L) is forced to be void
    - the top edge of the vertical arm is clamped
    - a downward load of total magnitude 1 is spread over the upper part of the
      right end of the horizontal arm (so the load itself is not a point singularity),
      with a small passive solid pad under it
The re-entrant corner of the L sits at (0.4 L, 0.4 L). With sharp corners, linear
elasticity predicts a stress singularity there, which is why the L-bracket is the
standard test problem for stress-constrained topology optimization
(e.g. Le et al. 2010; Giraldo-Londono & Paulino 2021).
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import fem
import topopt
from style import INK, ORANGE, SURF

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT, exist_ok=True)
NU = 0.3
ARM = 0.4            # arm width as a fraction of L
LOAD_SPAN = 0.1      # load spread over the top 10% of L on the right end of the horizontal arm


def node_id(i, j, nelx):
    return j * (nelx + 1) + i


def lbracket(nelx, nely):
    assert nelx == nely, "the L-bracket uses a square design domain"
    L = nelx
    a = int(round(ARM * L))
    ndof = 2 * (nelx + 1) * (nely + 1)

    # clamp: top edge of the vertical arm
    top = np.array([node_id(i, nely, nelx) for i in range(a + 1)])
    fixed = np.r_[2 * top, 2 * top + 1]

    # load: downward, total 1, spread over right-edge nodes with a - span <= y <= a
    span = max(1, int(round(LOAD_SPAN * L)))
    js = np.arange(a - span, a + 1)
    w = np.ones(len(js)); w[0] = w[-1] = 0.5
    w /= w.sum()
    F = np.zeros(ndof)
    for jj, wj in zip(js, w):
        F[2 * node_id(nelx, jj, nelx) + 1] = -wj

    # void: upper-right block
    I, J = np.meshgrid(np.arange(nelx), np.arange(nely))
    void = ((I >= a) & (J >= a)).ravel()
    # a small solid pad under the load, so the loaded nodes always stay attached to the part
    pad = max(2, int(round(0.03 * L)))
    solid = ((I >= nelx - pad) & (J >= a - span) & (J < a)).ravel()
    return dict(F=F, fixed=fixed, void=void, solid=solid, arm=a, load_nodes=js)


# ----------------------------------------------------------------- stress evaluation
def centroid_von_mises(x, nelx, nely, U, penal_stress=0.0):
    """
    Von Mises stress at element centroids (E0 = 1).
    penal_stress = 0 gives the stress of the solid material (what matters for a made part);
    the stress-constrained steps will use a relaxed version of this.
    """
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    B0, _ = fem.q4_B(sq, 0.0, 0.0)
    D = fem.plane_stress_D(1.0, NU)
    _, elems = fem.rect_mesh(nelx, nely, nelx, nely)
    ue = U[fem.element_dofs(elems)]
    sig = ue @ (D @ B0).T                      # (n_el, 3)
    return fem.von_mises(sig) * x.ravel() ** penal_stress


def analyse(x, nelx, nely, spec, Emin=1e-9, penal=3.0):
    nodes, elems = fem.rect_mesh(nelx, nely, nelx, nely)
    KE = fem.q4_stiffness(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float), fem.plane_stress_D(1.0, NU))
    edof = fem.element_dofs(elems)
    iK = np.repeat(edof, 8, axis=1).ravel()
    jK = np.tile(edof, (1, 8)).ravel()
    ndof = 2 * len(nodes)
    Ee = Emin + x.ravel() ** penal * (1 - Emin)
    K = sp.csc_matrix(((KE.ravel()[:, None] * Ee[None, :]).T.ravel(), (iK, jK)), shape=(ndof, ndof))
    free = np.setdiff1d(np.arange(ndof), spec["fixed"])
    U = np.zeros(ndof)
    U[free] = spla.spsolve(K[free][:, free], spec["F"][free], permc_spec="MMD_AT_PLUS_A")
    return float(spec["F"] @ U), U


def main(nelx=100, vol_L=0.4, rmin=3.0):
    """vol_L: material volume as a fraction of the L-shaped area (not of the full square)."""
    nely = nelx
    spec = lbracket(nelx, nely)
    a = spec["arm"]
    volfrac = vol_L * (1 - spec["void"].mean())          # the optimizer counts the full square
    print(f"L-bracket {nelx}x{nely}, arm {a}, {vol_L:.0%} of the L area, density filter rmin {rmin}")
    r = topopt.top(nelx, nely, volfrac, 3.0, rmin, ft="density", problem=lbracket, ctol=1e-4)
    x = r["x"]
    print(f"   compliance-only design: c = {r['c']:.3f}  iterations {r['iters']}  stop: {r['stop']}")

    # stress of the black/white part that would actually be made
    thr = np.quantile(x, 1 - volfrac)
    bw = (x >= thr).astype(float)
    c_bw, U = analyse(bw, nelx, nely, spec)
    vm = centroid_von_mises(bw, nelx, nely, U, penal_stress=1.0).reshape(nely, nelx)

    # exclude the load-introduction zone and the two ends of the clamped edge
    cx, cy = np.meshgrid(np.arange(nelx) + 0.5, np.arange(nely) + 0.5)
    far = np.ones_like(cx, bool)
    for px, py in [(nelx, a - 0.5 * LOAD_SPAN * nelx), (0, nely), (a, nely)]:
        far &= np.hypot(cx - px, cy - py) > 0.08 * nelx
    vm_far = np.where(far & (bw > 0), vm, -np.inf)
    k = np.unravel_index(np.argmax(vm_far), vm.shape)
    peak, loc = vm[k], (cx[k], cy[k])
    mean = vm[(bw > 0) & far].mean()
    d_corner = np.hypot(loc[0] - a, loc[1] - a)
    print(f"   black/white part: c = {c_bw:.3f}, volume {bw.mean():.3f}")
    print(f"   peak von Mises {peak:.3f} at ({loc[0]:.1f}, {loc[1]:.1f}); "
          f"distance to re-entrant corner {d_corner:.1f} elements; peak/mean = {peak / mean:.2f}")

    # figure
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.4))
    axs[0].imshow(x, cmap="gray_r", vmin=0, vmax=1, origin="lower", interpolation="none")
    axs[0].set_title(f"Compliance-only design, c = {r['c']:.2f}", fontsize=10)
    im = axs[1].imshow(np.where(bw > 0, vm, np.nan), cmap="Blues", origin="lower", interpolation="none",
                       vmin=0, vmax=np.nanpercentile(np.where(bw > 0, vm, np.nan), 99.5))
    axs[1].plot(*loc, "o", mfc="none", mec=ORANGE, mew=2, ms=12)
    axs[1].annotate(f"peak = {peak / mean:.1f}× mean", xy=loc, xytext=(0.55 * nelx, 0.75 * nely),
                    color=INK, fontsize=9, arrowprops=dict(arrowstyle="-", color=ORANGE))
    axs[1].set_title("Von Mises stress (black/white part)", fontsize=10)
    fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_lbracket_compliance.png"), dpi=170)
    plt.close(fig)

    summary = dict(nelx=nelx, vol_L=vol_L, rmin=rmin, c_opt=round(r["c"], 4), iters=r["iters"],
                   c_blackwhite=round(c_bw, 4), peak_vm=round(float(peak), 4),
                   peak_location=[round(float(loc[0]), 1), round(float(loc[1]), 1)],
                   distance_to_corner=round(float(d_corner), 1), peak_over_mean=round(float(peak / mean), 3))
    with open(os.path.join(OUT, "lbracket_compliance_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved fig_lbracket_compliance.png and lbracket_compliance_summary.json to results/")
    return r, bw, vm


if __name__ == "__main__":
    main()
