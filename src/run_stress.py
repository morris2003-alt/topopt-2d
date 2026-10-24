"""
run_stress.py — stress-constrained topology optimization of the L-bracket with MMA
==================================================================================
Step 4 of the stress-constrained extension.

    min_x   c(x) / c_ref
    s.t.    V(x) / V* - 1        <= 0          (volume)
            sigma_PN(x) / s_lim - 1 <= 0       (p-norm stress, optional)
            0 <= x <= 1

The optimality-criteria update used in topopt.py handles a single volume constraint only,
so this script uses the Method of Moving Asymptotes (Svanberg 1987) through the mmapy
package (Python port of Svanberg's MATLAB code, https://github.com/arjendeetman/GCMMA-MMA-Python).

Densities are filtered with the density filter of topopt.py; stresses and sensitivities come
from stress.py (checked in verify_stress.py).
"""

import os
import json
import numpy as np
from mmapy import mmasub
import topopt
import lbracket
from stress import StressModel

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
R_LOAD = 0.06      # stresses within 0.06 L of the load are left out of the stress measure (load-introduction zone)
os.makedirs(OUT, exist_ok=True)


def optimize(nelx=100, vol_L=0.4, rmin=3.0, s_lim=None, penal=3.0, P=8.0, q=0.5,
             move=0.2, max_iter=400, tol=0.01, ctol=2e-4, verbose=False):
    nely = nelx
    spec = lbracket.lbracket(nelx, nely)
    model = StressModel(nelx, nely, spec, penal=penal, q=q, P=P)
    H, Hs = topopt.filter_matrix(nelx, nely, rmin)
    void, solid = spec["void"], spec["solid"]
    act = ~(void | solid)
    nel = nelx * nely
    Vstar = vol_L * (1 - void.mean()) * nel             # volume budget in element units

    # stress measure: all non-void elements except a small zone around the load pad
    a = spec["arm"]
    cx, cy = np.meshgrid(np.arange(nelx) + 0.5, np.arange(nely) + 0.5)
    load_y = spec["load_nodes"].mean()
    mask = (~void) & (np.hypot(cx - nelx, cy - load_y) > R_LOAD * nelx).ravel()

    def physical(x):
        xp = (H @ x) / Hs
        xp[void] = 0.0
        xp[solid] = 1.0
        return xp

    def back(g):                                        # chain rule through the filter
        return H.T @ ((g * act) / Hs)

    m = 1 if s_lim is None else 2
    ia = np.nonzero(act)[0]                             # MMA only sees the design (non-passive) elements
    n = len(ia)
    x = np.where(void, 0.0, np.where(solid, 1.0, Vstar / act.sum()))
    xa = x[ia].copy()
    xold1, xold2 = xa.copy(), xa.copy()
    low, upp = np.zeros((n, 1)), np.ones((n, 1))
    a0, am, cm, dm = 1.0, np.zeros((m, 1)), 1000.0 * np.ones((m, 1)), np.ones((m, 1))

    c_ref = None
    hist = []
    for it in range(1, max_iter + 1):
        x[ia] = xa
        xp = physical(x)
        U, lu = model.solve(xp)
        c, dc = model.compliance(xp, U)
        if c_ref is None:
            c_ref = c
        f0, df0 = c / c_ref * 10.0, back(dc)[ia] / c_ref * 10.0
        fval = [xp.sum() / Vstar - 1.0]
        dfdx = [back(np.ones(nel))[ia] / Vstar]
        sPN, dPN, vm = model.pnorm(xp, U, lu, mask=mask)
        if s_lim is not None:
            fval.append(sPN / s_lim - 1.0)
            dfdx.append(back(dPN)[ia] / s_lim)
        fval = np.array(fval)[:, None]
        dfdx = np.vstack(dfdx)

        lo = np.maximum(0.0, xa - move); hi = np.minimum(1.0, xa + move)
        xmma, *_rest, low, upp = mmasub(m, n, it, xa[:, None], lo[:, None], hi[:, None],
                                        xold1[:, None], xold2[:, None], f0, df0[:, None], fval, dfdx,
                                        low, upp, a0, am, cm, dm, move=move)
        xnew = xmma.ravel()
        change = np.abs(xnew - xa).max()
        xold2, xold1, xa = xold1, xa, xnew
        hist.append(dict(it=it, c=c, vol=xp.sum() / Vstar * vol_L, sPN=sPN, change=change))
        if verbose and (it % 10 == 0 or it == 1):
            print(f"  it {it:4d}  c {c:9.3f}  vol {hist[-1]['vol']:.3f}  sPN {sPN:.4f}  change {change:.3f}")
        feasible = np.all(fval <= 0.005)
        plateau = it > 30 and abs(c - hist[-11]["c"]) / c < ctol
        if it > 20 and feasible and (change < tol or plateau):
            break

    x[ia] = xa
    xp = physical(x)
    U, lu = model.solve(xp)
    c, _ = model.compliance(xp, U)
    sPN, _, vm = model.pnorm(xp, U, lu, mask=mask)
    return dict(x=xp.reshape(nely, nelx), c=c, sPN=sPN, vm=vm.reshape(nely, nelx),
                vmax=float(vm[mask].max()), vol=float(xp.sum() / Vstar * vol_L), iters=it,
                hist=hist, mask=mask.reshape(nely, nelx), spec=spec)


# ===================================================================== step 5: study
def evaluate_part(r, vol_L=0.4):
    """
    Evaluate the part that would actually be made: threshold the design to black/white at the
    same volume, re-analyse it, and report the solid-material von Mises stress.
    - corner: highest stress within 0.05 L of the re-entrant corner
    - peak:   highest stress outside the load-introduction zone and the ends of the clamped edge
    Both use the same mesh as the optimization. The corner stress of a sharp re-entrant corner
    is mesh-dependent (it is singular in linear elasticity), so values are only compared on the
    same mesh.
    """
    spec, x = r["spec"], r["x"]
    nely, nelx = x.shape
    a = spec["arm"]
    volfrac = vol_L * (1 - spec["void"].mean())
    bw = (x >= np.quantile(x, 1 - volfrac)).astype(float)
    c_bw, U = lbracket.analyse(bw, nelx, nely, spec)
    vm = lbracket.centroid_von_mises(bw, nelx, nely, U, penal_stress=1.0).reshape(nely, nelx)
    cx, cy = np.meshgrid(np.arange(nelx) + 0.5, np.arange(nely) + 0.5)
    load_y = spec["load_nodes"].mean()
    far = (np.hypot(cx - nelx, cy - load_y) > 0.08 * nelx) & (np.hypot(cx, cy - nely) > 0.08 * nelx) \
        & (np.hypot(cx - a, cy - nely) > 0.08 * nelx)
    solid = bw > 0
    near_corner = np.hypot(cx - a, cy - a) < 0.05 * nelx
    v = np.where(far & solid, vm, -np.inf)
    k = np.unravel_index(np.argmax(v), vm.shape)
    corner = float(np.max(np.where(near_corner & solid, vm, 0.0)))
    return dict(c_bw=c_bw, peak=float(vm[k]), peak_xy=(float(cx[k]), float(cy[k])),
                peak_to_corner=float(np.hypot(cx[k] - a, cy[k] - a)), corner=corner,
                mean=float(vm[far & solid].mean()), bw=bw, vm=vm)


def study(nelx=100, ratios=(0.9, 0.8, 0.78), max_iter=700, tag=""):
    """Unconstrained design first, then stress limits set as fractions of its sigma_PN."""
    rows, designs = [], []
    base = optimize(nelx=nelx, rmin=0.03 * nelx, max_iter=max_iter)
    for ratio in (None,) + tuple(ratios):
        r = base if ratio is None else optimize(nelx=nelx, rmin=0.03 * nelx, s_lim=ratio * base["sPN"],
                                                max_iter=max_iter)
        e = evaluate_part(r)
        feasible = r["vol"] <= 0.4 * 1.005 and (ratio is None or r["sPN"] <= ratio * base["sPN"] * 1.005)
        row = dict(ratio=ratio, s_lim=None if ratio is None else ratio * base["sPN"], c=r["c"], vol=r["vol"],
                   sPN=r["sPN"], iters=r["iters"], feasible=bool(feasible), c_bw=e["c_bw"], peak=e["peak"],
                   peak_xy=e["peak_xy"], peak_to_corner=e["peak_to_corner"], corner=e["corner"],
                   peak_over_mean=e["peak"] / e["mean"])
        rows.append(row); designs.append((r, e))
        lab = "no stress limit" if ratio is None else f"limit {ratio:.2f} x unconstrained"
        print(f"   {lab:<30s} c {r['c']:8.2f}  vol {r['vol']:.3f}  sPN {r['sPN']:.4f}  it {r['iters']:4d}  "
              f"{'feasible' if feasible else 'NOT feasible'} | part: corner stress {e['corner']:.3f}  "
              f"peak {e['peak']:.3f} at ({e['peak_xy'][0]:.1f}, {e['peak_xy'][1]:.1f})", flush=True)
    b = rows[0]
    for row in rows[1:]:
        row["stiffness_loss_pct"] = 100 * (row["c"] / b["c"] - 1)
        row["corner_reduction_pct"] = 100 * (1 - row["corner"] / b["corner"])
        row["peak_reduction_pct"] = 100 * (1 - row["peak"] / b["peak"])
    with open(os.path.join(OUT, f"stress_study{tag}_summary.json"), "w") as f:
        json.dump([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()} for row in rows],
                  f, indent=2, default=lambda o: [round(t, 2) for t in o])
    return rows, designs


def figure(rows, designs, name="fig_stress_constrained.png"):
    import matplotlib.pyplot as plt
    from style import INK, ORANGE
    n = len(rows)
    fig, axs = plt.subplots(2, n, figsize=(3.3 * n, 6.6))
    for k, (row, (r, e)) in enumerate(zip(rows, designs)):
        x = r["x"]; nely, nelx = x.shape; a = r["spec"]["arm"]
        title = "No stress limit" if row["ratio"] is None else f"Limit {row['ratio']:.2f} × unconstrained"
        axs[0, k].imshow(x, cmap="gray_r", vmin=0, vmax=1, origin="lower", interpolation="none")
        axs[0, k].set_title(f"{title}\nc = {row['c']:.1f}" +
                            ("" if row["ratio"] is None else f"  (+{row['stiffness_loss_pct']:.1f}%)"), fontsize=9)
        w = int(0.2 * nelx)
        sub = np.where(e["bw"] > 0, e["vm"], np.nan)[a - w:a + w, a - w:a + w]
        vmax = np.nanmax(np.where(designs[0][1]["bw"] > 0, designs[0][1]["vm"], np.nan)[a - w:a + w, a - w:a + w])
        axs[1, k].imshow(sub, cmap="Blues", vmin=0, vmax=vmax, origin="lower", interpolation="none",
                         extent=[a - w, a + w, a - w, a + w])
        axs[1, k].plot([a], [a], "+", color=ORANGE, ms=14, mew=2)
        txt = f"corner stress {row['corner']:.3f}"
        if row["ratio"] is not None:
            txt += f"\n({-row['corner_reduction_pct']:+.0f}%)"
        axs[1, k].set_title(txt, fontsize=9)
        for ax in axs[:, k]:
            ax.set_xticks([]); ax.set_yticks([])
    axs[1, 0].set_ylabel("stress near the re-entrant corner\n(black/white part, + = corner)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    nelx = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    print(f"Stress-constrained L-bracket study, {nelx}x{nelx}, 40% of the L area")
    rows, designs = study(nelx=nelx, tag="" if nelx == 100 else f"_{nelx}")
    figure(rows, designs, name="fig_stress_constrained.png" if nelx == 100 else f"fig_stress_constrained_{nelx}.png")
    print("Saved figure and summary JSON to results/")
