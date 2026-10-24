"""
verify_stress.py — checks for stress.py (steps 2 and 3 of the stress-constrained extension)
============================================================================================
Step 2 (the stress itself) is checked against exact answers; step 3 (the adjoint
sensitivity) is checked against finite differences. A finite-difference check alone is not
enough: it only shows that the derivative matches the function as coded, so a wrong stress
formula would still pass it.

1-3  constant-stress patch tests with exact solutions (uniaxial, pure shear, equibiaxial)
4    centroid stress equals the average of the four Gauss-point stresses (independent code path)
5    von Mises formula equals fem.von_mises (the form used in the verified Kirsch study)
6    stress relaxation: factor x^q, so x = 1 gives the solid stress and x = 0.25 gives 0.5x
7    p-norm properties: sigma_PN >= max stress, and sigma_PN -> max as P grows
8    adjoint sensitivities of sigma_PN and compliance vs central finite differences
"""

import numpy as np
import fem
import stress
import lbracket
from stress import StressModel

NU = 0.3


def main():
    nelx, nely = 10, 5
    nodes, elems = fem.rect_mesh(nelx, nely, nelx, nely)
    X, Y = nodes[:, 0], nodes[:, 1]
    m = StressModel(nelx, nely, dict(F=np.zeros(2 * len(nodes)), fixed=np.array([0, 1])))
    x1 = np.ones(nelx * nely)

    def field(u, v):
        U = np.zeros(2 * len(nodes)); U[0::2] = u; U[1::2] = v
        return U

    ok = True

    def report(name, value, exact, tol=1e-10):
        nonlocal ok
        err = np.max(np.abs(np.asarray(value) - exact))
        ok &= err < tol
        print(f"   {name:<48s} {np.round(value, 6)!s:>22}  exact {exact!s:<9}  err {err:.1e}")

    print("Step 2: stress")
    report("1. uniaxial tension, von Mises", m.von_mises(x1, field(X, -NU * Y)).max(), 1.0)
    report("2. pure shear, von Mises", m.von_mises(x1, field(2 * (1 + NU) * Y, 0 * Y)).max(), round(np.sqrt(3), 12), 1e-9)
    report("3. equibiaxial tension, von Mises", m.von_mises(x1, field((1 - NU) * X, (1 - NU) * Y)).max(), 1.0)

    U = np.random.default_rng(0).normal(size=2 * len(nodes))
    D = fem.plane_stress_D(1.0, NU)
    Bg = np.array([fem.q4_B(stress.SQ, xi, eta)[0] for xi, eta in fem.GAUSS])
    sg = np.einsum("ij,gjk,nk->ngi", D, Bg, U[fem.element_dofs(elems)]).mean(axis=1)
    report("4. centroid - Gauss-average stress (max diff)", np.abs(sg - m.solid_stress(U)).max(), 0.0)
    report("5. von Mises - fem.von_mises (max diff)",
           np.abs(m.von_mises(x1, U, relaxed=False) - fem.von_mises(m.solid_stress(U))).max(), 0.0)
    report("6. relaxation factor at x = 0.25", (m.von_mises(0.25 * x1, U) / m.von_mises(x1, U, relaxed=False)).max(), 0.5)

    spec = lbracket.lbracket(40, 40)
    mm = StressModel(40, 40, spec)
    x = np.where(spec["void"], 0.0, 1.0)
    U, lu = mm.solve(x)
    vmax = mm.von_mises(x, U).max()
    print("   7. p-norm / max stress (solid L-bracket, 40x40):", end="")
    prev = np.inf
    for P in [2, 8, 32, 128]:
        mm.P = P
        s, _, _ = mm.pnorm(x, U, lu)
        ok &= (s >= vmax * (1 - 1e-12)) and (s <= prev)
        prev = s
        print(f"  P={P}: {s / vmax:.4f}", end="")
    print()

    print("Step 3: adjoint sensitivities vs central finite differences (30x30, random gray design)")
    for seed in [0, 1, 2]:
        _, rel_s, rel_c, _, _ = stress.fd_check(seed=seed, n_check=20)
        ok &= rel_s.max() < 1e-3 and rel_c.max() < 1e-3
        print(f"   seed {seed}: max relative error  p-norm {rel_s.max():.1e}   compliance {rel_c.max():.1e}")

    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    main()
