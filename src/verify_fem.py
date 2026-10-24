"""
verify_fem.py — verification of the Q4 plane-stress solver
==========================================================
Case 1  Cantilever with end shear load  vs  Timoshenko beam theory
Case 2  Plate with a circular hole under uniaxial tension  vs  Kirsch solution
        (quarter model; exact Kirsch tractions applied on the outer edges so the
         finite domain reproduces the infinite-plate solution exactly)
Both cases include a mesh-convergence study.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import fem
from style import INK, INK2, BLUE, ORANGE, AQUA, grid

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT, exist_ok=True)


# =============================================================== case 1
def cantilever(nx, ny, L=100.0, h=10.0, E=200e3, nu=0.3, P=100.0, t=1.0):
    nodes, elems = fem.rect_mesh(L, h, nx, ny)
    D = fem.plane_stress_D(E, nu)
    K = fem.assemble(nodes, elems, D, t)
    left = np.where(np.isclose(nodes[:, 0], 0))[0]
    right = np.where(np.isclose(nodes[:, 0], L))[0]
    right = right[np.argsort(nodes[right, 1])]
    # parabolic shear traction on the free end, resultant P downward
    tau = lambda x, y: (0.0, -1.5 * P / (h * t) * (1 - (2 * (y - h / 2) / h) ** 2))
    F = fem.edge_load(nodes, list(zip(right[:-1], right[1:])), tau, t)
    fixed = np.concatenate([2 * left, 2 * left + 1])
    U = fem.solve(K, F, fixed)
    mid = right[np.argmin(np.abs(nodes[right, 1] - h / 2))]
    I = t * h**3 / 12
    G = E / (2 * (1 + nu))
    ref = P * L**3 / (3 * E * I) + P * L / (5 / 6 * G * h * t)
    return -U[2 * mid + 1], ref, 2 * len(nodes)


# =============================================================== case 2
def hole_mesh(a, W, nr, nt, ratio=30.0):
    """Mapped Q4 mesh of the quarter plate [0,W]^2 minus a hole of radius a."""
    theta = np.linspace(0, np.pi / 2, nt + 1)
    inner = np.column_stack([a * np.cos(theta), a * np.sin(theta)])
    outer = np.where((theta <= np.pi / 4)[:, None],
                     np.column_stack([np.full_like(theta, W), W * np.tan(theta)]),
                     np.column_stack([W / np.tan(np.maximum(theta, 1e-12)), np.full_like(theta, W)]))
    outer[-1] = [0.0, W]
    q = ratio ** (1.0 / nr)          # same outer/inner element-size ratio on every mesh
    s = (q ** np.arange(nr + 1) - 1) / (q ** nr - 1)       # radial grading toward the hole
    nodes = np.array([(1 - si) * inner[k] + si * outer[k] for si in s for k in range(nt + 1)])
    elems = []
    for i in range(nr):
        for k in range(nt):
            n0 = i * (nt + 1) + k
            elems.append([n0, n0 + 1, n0 + 1 + nt + 1, n0 + nt + 1])
    # node order: radial i, angular k -> make counter-clockwise in (x, y)
    elems = np.array(elems)[:, [0, 3, 2, 1]]
    return nodes, elems


def kirsch(x, y, a, s0=1.0):
    r = np.hypot(x, y); th = np.arctan2(y, x)
    c2, s2 = np.cos(2 * th), np.sin(2 * th)
    q2, q4 = (a / r) ** 2, (a / r) ** 4
    srr = s0 / 2 * (1 - q2) + s0 / 2 * (1 - 4 * q2 + 3 * q4) * c2
    stt = s0 / 2 * (1 + q2) - s0 / 2 * (1 + 3 * q4) * c2
    srt = -s0 / 2 * (1 + 2 * q2 - 3 * q4) * s2
    c, s = np.cos(th), np.sin(th)
    sxx = srr * c * c + stt * s * s - 2 * srt * s * c
    syy = srr * s * s + stt * c * c + 2 * srt * s * c
    sxy = (srr - stt) * s * c + srt * (c * c - s * s)
    return sxx, syy, sxy


def plate_with_hole(nr, nt, a=1.0, W=10.0, E=200e3, nu=0.3, s0=1.0):
    nodes, elems = hole_mesh(a, W, nr, nt)
    D = fem.plane_stress_D(E, nu)
    K = fem.assemble(nodes, elems, D)
    ring = np.arange(nr * (nt + 1), (nr + 1) * (nt + 1))          # outer boundary nodes
    segs_x, segs_y = [], []
    for p, q in zip(ring[:-1], ring[1:]):
        mid = (nodes[p] + nodes[q]) / 2
        (segs_x if np.isclose(mid[0], W) else segs_y).append((p, q))

    def t_right(x, y):                       # normal (1, 0)
        sxx, syy, sxy = kirsch(x, y, a, s0); return sxx, sxy

    def t_top(x, y):                         # normal (0, 1)
        sxx, syy, sxy = kirsch(x, y, a, s0); return sxy, syy

    F = fem.edge_load(nodes, segs_x, t_right) + fem.edge_load(nodes, segs_y, t_top)
    on_x = np.where(np.isclose(nodes[:, 1], 0))[0]                # y = 0 : v = 0
    on_y = np.where(np.isclose(nodes[:, 0], 0, atol=1e-9))[0]     # x = 0 : u = 0
    U = fem.solve(K, F, np.concatenate([2 * on_x + 1, 2 * on_y]))
    S = fem.nodal_stresses(nodes, elems, D, U)
    top_of_hole = np.argmin(np.hypot(nodes[:, 0], nodes[:, 1] - a))
    return nodes, elems, S, S[top_of_hole, 0] / s0, 2 * len(nodes)


# =============================================================== run
def main():
    # ---- case 1
    print("Case 1: cantilever (L/h = 10), tip deflection vs Timoshenko beam theory")
    c1 = []
    for nx, ny in [(10, 1), (20, 2), (40, 4), (80, 8), (160, 16), (320, 32)]:
        d, ref, ndof = cantilever(nx, ny)
        c1.append((ndof, d, ref))
        print(f"  {nx:4d} x {ny:2d}  dof {ndof:6d}  FEM {d:.5f}  theory {ref:.5f}  error {100*(d-ref)/ref:+6.2f}%")

    # ---- case 2
    print("\nCase 2: plate with hole (W/a = 10), stress concentration at hole edge vs Kirsch (Kt = 3)")
    c2 = []
    for n in [8, 16, 32, 64, 128]:
        nodes, elems, S, kt, ndof = plate_with_hole(n, 2 * n)
        c2.append((ndof, kt))
        print(f"  {n:3d} x {2*n:3d}  dof {ndof:6d}  Kt = {kt:.4f}  error {100*(kt-3)/3:+6.2f}%")
    fine = (nodes, elems, S)

    # ---- figure 1: convergence of both cases
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    nd, d, ref = np.array(c1).T
    axs[0].loglog(nd, np.abs(d - ref) / ref * 100, "o-", color=BLUE, lw=2, ms=6)
    axs[0].set_xlabel("Degrees of freedom"); axs[0].set_ylabel("|Error| in tip deflection (%)")
    axs[0].set_title("Cantilever vs Timoshenko beam theory"); grid(axs[0])
    nd2, kt = np.array(c2).T
    axs[1].loglog(nd2, np.abs(kt - 3) / 3 * 100, "o-", color=BLUE, lw=2, ms=6)
    axs[1].set_xlabel("Degrees of freedom"); axs[1].set_ylabel("|Error| in Kt (%)")
    axs[1].set_title("Plate with hole vs Kirsch (Kt = 3)"); grid(axs[1])
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_verify_convergence.png"), dpi=170)
    plt.close(fig)

    # ---- figure 2: plate with hole, stress field and profile
    nodes, elems, S = fine
    tri = Triangulation(nodes[:, 0], nodes[:, 1],
                        np.vstack([elems[:, [0, 1, 2]], elems[:, [0, 2, 3]]]))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
    cs = axs[0].tripcolor(tri, S[:, 0], shading="gouraud", cmap="Blues", vmin=0, vmax=3)
    axs[0].set_aspect("equal"); axs[0].set_xlim(0, 4); axs[0].set_ylim(0, 4)
    axs[0].set_title("σxx / σ0 near the hole (FEM)")
    axs[0].set_xlabel("x / a"); axs[0].set_ylabel("y / a")
    fig.colorbar(cs, ax=axs[0], shrink=0.85)
    on_axis = np.where(np.isclose(nodes[:, 0], 0, atol=1e-9))[0]
    on_axis = on_axis[np.argsort(nodes[on_axis, 1])]
    yy = np.linspace(1, 10, 300)
    sel = on_axis[::6]
    axs[1].plot(nodes[sel, 1], S[sel, 0], "o", mfc="none", mec=BLUE, mew=1.5, ms=7,
                label="FEM (every 6th node)")
    axs[1].plot(yy, kirsch(np.zeros_like(yy), yy, 1.0)[0], color=ORANGE, lw=2,
                label="Kirsch (exact)", zorder=3)
    axs[1].set_xlabel("Distance from hole centre along x = 0, y / a")
    axs[1].set_ylabel("σxx / σ0"); axs[1].set_xlim(1, 6)
    axs[1].set_title("Stress along the section through the hole")
    axs[1].legend(frameon=False); grid(axs[1])
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_verify_plate_hole.png"), dpi=170)
    plt.close(fig)
    print("\nSaved: results/fig_verify_convergence.png, results/fig_verify_plate_hole.png")


if __name__ == "__main__":
    main()
