"""
stress.py — relaxed element stresses, the p-norm stress measure and its adjoint sensitivity
============================================================================================
Step 2 and 3 of the stress-constrained extension.

Stiffness interpolation (modified SIMP, as in topopt.py)
    E_e(x) = E_min + x_e^p (1 - E_min)

Relaxed element stress (qp-relaxation; Bruggi 2008, Le et al. 2010)
    sigma_e = x_e^q  D B0 u_e          with q < p (here q = 0.5)
    vm_e    = sqrt(sigma_e^T V sigma_e) von Mises stress, V = [[1,-1/2,0],[-1/2,1,0],[0,0,3]]
Without relaxation (q = p), elements with x_e -> 0 still show a finite "solid-material"
stress, and the optimizer cannot remove them: the stress singularity phenomenon
(Kirsch 1990; Cheng & Guo 1997; Duysinx & Bendsoe 1998). Multiplying by x_e^q with q < p
makes the stress of a vanishing element go to zero.

Global stress measure (p-norm aggregation of the local constraints)
    sigma_PN = ( sum_e vm_e^P )^(1/P)
sigma_PN -> max_e vm_e as P -> infinity, and sigma_PN >= max_e vm_e for any finite P
over all elements; large P makes the problem ill-conditioned, so P = 8 is a common choice.

Adjoint sensitivity (one extra linear solve per stress measure)
    d sigma_PN / d x_e = explicit term (x_e^q factor) - lambda_e^T (dK_e/dx_e) u_e
    with K lambda = d sigma_PN / d u
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import fem

NU = 0.3
V_VM = np.array([[1.0, -0.5, 0.0], [-0.5, 1.0, 0.0], [0.0, 0.0, 3.0]])
SQ = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)


class StressModel:
    """FE model on a structured unit-square grid with stiffness, stress and p-norm routines."""

    def __init__(self, nelx, nely, spec, penal=3.0, q=0.5, P=8.0, Emin=1e-9):
        self.nelx, self.nely = nelx, nely
        self.penal, self.q, self.P, self.Emin = penal, q, P, Emin
        self.F, self.fixed = spec["F"], spec["fixed"]
        nodes, elems = fem.rect_mesh(nelx, nely, nelx, nely)
        self.edof = fem.element_dofs(elems)
        self.ndof = 2 * len(nodes)
        D = fem.plane_stress_D(1.0, NU)
        self.KE = fem.q4_stiffness(SQ, D)
        B0, _ = fem.q4_B(SQ, 0.0, 0.0)                       # strain at the element centroid
        self.S = D @ B0                                      # (3, 8): solid-material stress per unit u_e
        self.iK = np.repeat(self.edof, 8, axis=1).ravel()
        self.jK = np.tile(self.edof, (1, 8)).ravel()
        self.free = np.setdiff1d(np.arange(self.ndof), self.fixed)

    # ---------------------------------------------------------------- analysis
    def solve(self, xPhys):
        Ee = self.Emin + xPhys ** self.penal * (1 - self.Emin)
        sK = (self.KE.ravel()[:, None] * Ee[None, :]).T.ravel()
        K = sp.csc_matrix((sK, (self.iK, self.jK)), shape=(self.ndof, self.ndof))
        K = ((K + K.T) / 2)[self.free][:, self.free].tocsc()
        lu = spla.splu(K, permc_spec="MMD_AT_PLUS_A")       # factorize once, reuse for the adjoint
        U = np.zeros(self.ndof)
        U[self.free] = lu.solve(self.F[self.free])
        return U, lu

    def compliance(self, xPhys, U):
        ue = U[self.edof]
        ce = np.einsum("ij,jk,ik->i", ue, self.KE, ue)
        Ee = self.Emin + xPhys ** self.penal * (1 - self.Emin)
        c = float((Ee * ce).sum())
        dc = -self.penal * xPhys ** (self.penal - 1) * (1 - self.Emin) * ce
        return c, dc

    def solid_stress(self, U):
        """Stress of solid material, sigma0_e = D B0 u_e, (n_el, 3)."""
        return U[self.edof] @ self.S.T

    def von_mises(self, xPhys, U, relaxed=True):
        s0 = self.solid_stress(U)
        vm0 = np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", s0, V_VM, s0), 1e-30))
        return vm0 * xPhys ** self.q if relaxed else vm0

    # ---------------------------------------------------------------- p-norm and sensitivity
    def pnorm(self, xPhys, U, lu, mask=None):
        """
        p-norm of the relaxed von Mises stress over the elements in mask (default: all).
        Returns sigma_PN, d sigma_PN / d xPhys, and the relaxed vm field.
        """
        P, q, p = self.P, self.q, self.penal
        m = np.ones_like(xPhys, bool) if mask is None else mask
        s0 = self.solid_stress(U)
        vm0 = np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", s0, V_VM, s0), 1e-30))
        xq = xPhys ** q
        vm = vm0 * xq

        sPN = float((vm[m] ** P).sum() ** (1.0 / P))
        dPN_dvm = np.zeros_like(vm)
        dPN_dvm[m] = sPN ** (1 - P) * vm[m] ** (P - 1)

        # explicit part: vm_e = x_e^q vm0_e  ->  d vm_e / d x_e = q x_e^(q-1) vm0_e
        # (written as q * vm_e / x_e so that elements with x_e = 0, e.g. passive voids, give 0, not 0 * inf)
        pos = xPhys > 0
        d_explicit = np.zeros_like(vm)
        d_explicit[pos] = dPN_dvm[pos] * q * vm[pos] / xPhys[pos]

        # implicit part through u: d vm_e / d u_e = x_e^q (S^T V sigma0_e) / vm0_e
        g_e = (dPN_dvm * xq / vm0)[:, None] * (s0 @ V_VM) @ self.S      # (n_el, 8)
        rhs = np.zeros(self.ndof)
        np.add.at(rhs, self.edof, g_e)
        lam = np.zeros(self.ndof)
        lam[self.free] = lu.solve(rhs[self.free])        # K is symmetric, so K^T lam = K lam
        le, ue = lam[self.edof], U[self.edof]
        dKdx = p * xPhys ** (p - 1) * (1 - self.Emin)
        d_implicit = -dKdx * np.einsum("ij,jk,ik->i", le, self.KE, ue)
        return sPN, d_explicit + d_implicit, vm


# -------------------------------------------------------------------- finite-difference check
def fd_check(nelx=30, seed=0, n_check=12, h=1e-4, rmin=1.5):
    """
    Compare the adjoint sensitivities of sigma_PN and compliance with central finite
    differences, through the density filter, at a random gray design.
    Step h = 1e-4: smaller steps make the finite difference itself less accurate
    (round-off grows like eps / h), larger steps add truncation error (grows like h^2).
    """
    import topopt
    import lbracket
    rng = np.random.default_rng(seed)
    nely = nelx
    spec = lbracket.lbracket(nelx, nely)
    model = StressModel(nelx, nely, spec)
    H, Hs = topopt.filter_matrix(nelx, nely, rmin)

    def physical(x):
        xp = (H @ x) / Hs
        xp[spec["void"]] = 0.0
        xp[spec["solid"]] = 1.0
        return xp

    def evaluate(x):
        xp = physical(x)
        U, lu = model.solve(xp)
        sPN, dPN, _ = model.pnorm(xp, U, lu)
        c, dc = model.compliance(xp, U)
        return sPN, dPN, c, dc

    x = rng.uniform(0.3, 0.9, nelx * nely)
    sPN, dPN, c, dc = evaluate(x)
    # chain rule through the density filter (passive elements get zero sensitivity)
    act = ~(spec["void"] | spec["solid"])
    dPN_x = H.T @ ((dPN * act) / Hs)
    dc_x = H.T @ ((dc * act) / Hs)

    design = np.nonzero(act)[0]
    idx = rng.choice(design, n_check, replace=False)
    rows = []
    for e in idx:
        xp_, xm_ = x.copy(), x.copy()
        xp_[e] += h; xm_[e] -= h
        sp_, _, cp_, _ = evaluate(xp_)
        sm_, _, cm_, _ = evaluate(xm_)
        rows.append((e, dPN_x[e], (sp_ - sm_) / (2 * h), dc_x[e], (cp_ - cm_) / (2 * h)))
    rows = np.array(rows)
    rel_s = np.abs(rows[:, 1] - rows[:, 2]) / np.maximum(np.abs(rows[:, 2]), 1e-12)
    rel_c = np.abs(rows[:, 3] - rows[:, 4]) / np.maximum(np.abs(rows[:, 4]), 1e-12)
    return rows, rel_s, rel_c, sPN, c


if __name__ == "__main__":
    rows, rel_s, rel_c, sPN, c = fd_check()
    print(f"Finite-difference check on a 30x30 L-bracket, random gray design (sigma_PN = {sPN:.4f}, c = {c:.4f})")
    print(f"{'element':>8} {'dPN adjoint':>14} {'dPN FD':>14} {'rel err':>9} {'dc adjoint':>14} {'dc FD':>14} {'rel err':>9}")
    for r, es, ec in zip(rows, rel_s, rel_c):
        print(f"{int(r[0]):8d} {r[1]:14.6e} {r[2]:14.6e} {es:9.1e} {r[3]:14.6e} {r[4]:14.6e} {ec:9.1e}")
    print(f"max relative error: p-norm stress {rel_s.max():.1e}, compliance {rel_c.max():.1e}")
