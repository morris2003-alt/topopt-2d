"""
topopt.py — 2D minimum-compliance topology optimization (SIMP)
==============================================================
Problem
    min_x  c(x) = F^T U(x)
    s.t.   K(x) U = F,   sum(x_e) / N = volfrac,   0 <= x_e <= 1

Method
    Modified SIMP      E_e = E_min + x_e^p (E_0 - E_min)
    Sensitivities      dc/dx_e = -p x_e^(p-1) (E_0 - E_min) u_e^T K_0 u_e
    Filters            sensitivity filter (Sigmund 2001) or density filter (Bruns & Tortorelli 2001)
    Update             optimality criteria (OC) with bisection on the volume multiplier

The element stiffness K_0 comes from fem.q4_stiffness on a unit square, i.e. the same
Q4 element that is verified in verify_fem.py.

Mesh: nelx x nely unit-square elements on [0,nelx] x [0,nely], y pointing up.
Element e = j*nelx + i has its centre at (i + 0.5, j + 0.5).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import fem


# ----------------------------------------------------------------- problem definitions
def node_id(i, j, nelx):
    return j * (nelx + 1) + i


def mbb(nelx, nely):
    """Half MBB beam: symmetry on the left edge, roller at bottom-right, unit load at top-left."""
    ndof = 2 * (nelx + 1) * (nely + 1)
    F = np.zeros(ndof)
    F[2 * node_id(0, nely, nelx) + 1] = -1.0
    left = [node_id(0, j, nelx) for j in range(nely + 1)]
    fixed = np.r_[2 * np.array(left), 2 * node_id(nelx, 0, nelx) + 1]
    return F, fixed


def cantilever(nelx, nely):
    """Cantilever: left edge clamped, unit load at the middle of the right edge."""
    ndof = 2 * (nelx + 1) * (nely + 1)
    F = np.zeros(ndof)
    F[2 * node_id(nelx, nely // 2, nelx) + 1] = -1.0
    left = np.array([node_id(0, j, nelx) for j in range(nely + 1)])
    fixed = np.r_[2 * left, 2 * left + 1]
    return F, fixed


def bridge(nelx, nely, deck_y=0.3, deck_rows=3, support="pin"):
    """
    Half of a symmetric bridge (midspan on the right edge).
    - abutment at the bottom-left corner: support="pin" (u = v = 0, can take thrust)
      or support="roller" (v = 0 only, no horizontal reaction)
    - symmetry at midspan: u = 0 on the right edge
    - a solid deck (passive elements) at height deck_y * nely, carrying a uniform load
      of total magnitude 1 on its top surface
    The rest of the domain, above and below the deck, is free for the optimizer.
    """
    ndof = 2 * (nelx + 1) * (nely + 1)
    j0 = int(round(deck_y * nely))
    jt = j0 + deck_rows                                   # node row on top of the deck
    F = np.zeros(ndof)
    w = np.ones(nelx + 1); w[0] = w[-1] = 0.5             # tributary lengths
    w /= w.sum()
    for i in range(nelx + 1):
        F[2 * node_id(i, jt, nelx) + 1] = -w[i]
    right = np.array([node_id(nelx, j, nelx) for j in range(nely + 1)])
    pin = node_id(0, 0, nelx)
    fixed = np.r_[2 * pin + 1, 2 * right] if support == "roller" else np.r_[2 * pin, 2 * pin + 1, 2 * right]
    I, J = np.meshgrid(np.arange(nelx), np.arange(nely))
    solid = ((J >= j0) & (J < jt)).ravel()
    return dict(F=F, fixed=fixed, solid=solid)


PROBLEMS = {"mbb": mbb, "cantilever": cantilever, "bridge": bridge}


# ----------------------------------------------------------------- filter
def filter_matrix(nelx, nely, rmin):
    """Linear 'hat' weights H_ef = max(0, rmin - dist(e, f)), and row sums Hs."""
    r = int(np.ceil(rmin)) - 1
    ii, jj, vv = [], [], []
    I, J = np.meshgrid(np.arange(nelx), np.arange(nely))
    I, J = I.ravel(), J.ravel()
    e = J * nelx + I
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            w = rmin - np.hypot(di, dj)
            if w <= 0:
                continue
            I2, J2 = I + di, J + dj
            ok = (I2 >= 0) & (I2 < nelx) & (J2 >= 0) & (J2 < nely)
            ii.append(e[ok]); jj.append(J2[ok] * nelx + I2[ok]); vv.append(np.full(ok.sum(), w))
    H = sp.csr_matrix((np.concatenate(vv), (np.concatenate(ii), np.concatenate(jj))),
                      shape=(nelx * nely, nelx * nely))
    Hs = np.asarray(H.sum(axis=1)).ravel()
    return H, Hs


# ----------------------------------------------------------------- optimizer
def top(nelx, nely, volfrac=0.5, penal=3.0, rmin=1.5, ft="sensitivity", problem="mbb",
        E0=1.0, Emin=1e-9, nu=0.3, move=0.2, tol=0.01, max_iter=500, record_every=0,
        ctol=None, verbose=False):
    """
    Returns a dict with the final physical densities, compliance history and
    (optionally) snapshots of the design for animation.

    Stopping: max design change < tol (the criterion of Andreassen et al. 2011).
    If ctol is given, the run also stops once the compliance has changed by less than
    ctol (relative) over the last 10 iterations; useful for large meshes where the
    density filter makes the design change oscillate just above tol.
    """
    nodes, elems = fem.rect_mesh(nelx, nely, nelx, nely)
    KE = fem.q4_stiffness(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float),
                          fem.plane_stress_D(1.0, nu))
    edof = fem.element_dofs(elems)
    iK = np.repeat(edof, 8, axis=1).ravel()     # row index of KE[a, b] -> edof[a]
    jK = np.tile(edof, (1, 8)).ravel()          # column index          -> edof[b]
    ndof = 2 * len(nodes)

    spec = PROBLEMS[problem](nelx, nely) if isinstance(problem, str) else problem(nelx, nely)
    if isinstance(spec, dict):
        F, fixed = spec["F"], spec["fixed"]
        solid, void = spec.get("solid"), spec.get("void")
    else:
        (F, fixed), solid, void = spec, None, None
    free = np.setdiff1d(np.arange(ndof), fixed)

    def passive(v):
        """Force passive elements: solid = 1, void = 0."""
        if solid is not None:
            v[solid] = 1.0
        if void is not None:
            v[void] = 0.0
        return v
    H, Hs = filter_matrix(nelx, nely, rmin)

    nel = nelx * nely
    x = passive(np.full(nel, volfrac))
    xPhys = x.copy()
    hist, snaps = [], []
    U = np.zeros(ndof)

    for it in range(1, max_iter + 1):
        # --- FE analysis
        Ee = Emin + xPhys**penal * (E0 - Emin)
        sK = (KE.ravel()[:, None] * Ee[None, :]).T.ravel()
        K = sp.csc_matrix((sK, (iK, jK)), shape=(ndof, ndof))
        K = (K + K.T) / 2
        U[:] = 0
        U[free] = spla.spsolve(K[free][:, free], F[free], permc_spec="MMD_AT_PLUS_A")

        # --- objective and sensitivities
        ue = U[edof]
        ce = np.einsum("ij,jk,ik->i", ue, KE, ue)
        c = float((Ee * ce).sum())
        dc = -penal * xPhys ** (penal - 1) * (E0 - Emin) * ce
        dv = np.ones(nel)

        # --- filtering
        if ft == "sensitivity":
            dc = H @ (x * dc) / Hs / np.maximum(1e-3, x)
        elif ft == "density":
            dc = H @ (dc / Hs)
            dv = H @ (dv / Hs)
        else:
            raise ValueError("ft must be 'sensitivity' or 'density'")

        # --- optimality criteria update with bisection on the Lagrange multiplier
        l1, l2 = 0.0, 1e9
        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l1 + l2)
            xnew = passive(np.clip(x * np.sqrt(np.maximum(-dc, 0) / dv / lmid),
                                   np.maximum(0.0, x - move), np.minimum(1.0, x + move)))
            xPhys_new = xnew if ft == "sensitivity" else passive((H @ xnew) / Hs)
            if xPhys_new.sum() > volfrac * nel:
                l1 = lmid
            else:
                l2 = lmid
        change = np.abs(xnew - x).max()
        x, xPhys = xnew, xPhys_new

        hist.append(c)
        if record_every and (it % record_every == 0 or it == 1):
            snaps.append(xPhys.reshape(nely, nelx).copy())
        if verbose:
            print(f"  it {it:4d}  c {c:10.4f}  vol {xPhys.mean():.3f}  change {change:.4f}")
        stop = "design change" if change < tol else None
        if stop is None and ctol and it > 20:
            ref = hist[-11]
            if abs(c - ref) / ref < ctol:
                stop = "compliance plateau"
        if stop:
            break

    if record_every:
        snaps.append(xPhys.reshape(nely, nelx).copy())
    # c is the compliance of the last analysed design, as reported by Andreassen et al. (2011);
    # the returned x includes the final OC update (the difference is below the stopping tolerance)
    return dict(x=xPhys.reshape(nely, nelx), c=hist[-1], hist=np.array(hist), iters=it,
                stop=stop or "max_iter",
                snaps=snaps, nelx=nelx, nely=nely, volfrac=volfrac, ft=ft, problem=problem)


def grayness(x):
    """Measure of non-discreteness: 0 = fully black/white, 1 = all 0.5."""
    return float(4 * np.mean(x * (1 - x)))
