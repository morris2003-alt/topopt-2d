"""
fem.py — 2D linear-elastic finite element solver (plane stress)
================================================================
Element : 4-node isoparametric quadrilateral (Q4), 2x2 Gauss quadrature
DOFs    : [u1, v1, u2, v2, ...]  (node k -> 2k, 2k+1)
Units   : consistent (e.g. N, mm, MPa)

Main pieces
  plane_stress_D       constitutive matrix
  q4_stiffness         element stiffness for an arbitrary (distorted) quad
  rect_mesh            structured mesh on a rectangle
  assemble             global sparse stiffness
  solve                apply Dirichlet BCs and solve K u = f
  nodal_stresses       Gauss-point stresses extrapolated to nodes and averaged
  edge_load            consistent nodal forces from a traction t(x, y) on boundary segments
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Gauss points ordered like the element nodes (counter-clockwise from (-1,-1))
_G = 1.0 / np.sqrt(3.0)
GAUSS = np.array([[-_G, -_G], [_G, -_G], [_G, _G], [-_G, _G]])
NAT = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])   # natural node coords


# ----------------------------------------------------------------- material
def plane_stress_D(E, nu):
    return E / (1.0 - nu**2) * np.array([[1.0, nu, 0.0],
                                         [nu, 1.0, 0.0],
                                         [0.0, 0.0, (1.0 - nu) / 2.0]])


# ----------------------------------------------------------------- element
def shape(xi, eta):
    """Bilinear shape functions N (4,) and derivatives dN/d(xi,eta) (2,4)."""
    N = 0.25 * (1 + NAT[:, 0] * xi) * (1 + NAT[:, 1] * eta)
    dN = 0.25 * np.vstack([NAT[:, 0] * (1 + NAT[:, 1] * eta),
                           NAT[:, 1] * (1 + NAT[:, 0] * xi)])
    return N, dN


def q4_B(xy, xi, eta):
    """Strain-displacement matrix B (3x8) and det(J) at (xi, eta)."""
    _, dN = shape(xi, eta)
    J = dN @ xy
    detJ = np.linalg.det(J)
    if detJ <= 0:
        raise ValueError("Element with non-positive Jacobian (check node ordering).")
    dNdx = np.linalg.solve(J, dN)
    B = np.zeros((3, 8))
    B[0, 0::2] = dNdx[0]
    B[1, 1::2] = dNdx[1]
    B[2, 0::2] = dNdx[1]
    B[2, 1::2] = dNdx[0]
    return B, detJ


def q4_stiffness(xy, D, t=1.0):
    """8x8 element stiffness of a Q4 with nodal coordinates xy (4x2)."""
    Ke = np.zeros((8, 8))
    for xi, eta in GAUSS:
        B, detJ = q4_B(xy, xi, eta)
        Ke += B.T @ D @ B * detJ * t
    return Ke


def element_dofs(elems):
    """(n_el, 8) array of global DOF indices."""
    return np.column_stack([2 * elems[:, k] + d for k in range(4) for d in (0, 1)])


# ----------------------------------------------------------------- meshes
def rect_mesh(Lx, Ly, nx, ny):
    """Structured mesh on [0,Lx]x[0,Ly]. Node id = j*(nx+1) + i."""
    x = np.linspace(0, Lx, nx + 1)
    y = np.linspace(0, Ly, ny + 1)
    X, Y = np.meshgrid(x, y)
    nodes = np.column_stack([X.ravel(), Y.ravel()])
    i, j = np.meshgrid(np.arange(nx), np.arange(ny))
    n0 = (j * (nx + 1) + i).ravel()
    elems = np.column_stack([n0, n0 + 1, n0 + nx + 2, n0 + nx + 1])
    return nodes, elems


# ----------------------------------------------------------------- global
def assemble(nodes, elems, D, t=1.0):
    edofs = element_dofs(elems)
    Ke_all = np.array([q4_stiffness(nodes[e], D, t) for e in elems])
    rows = np.repeat(edofs, 8, axis=1).ravel()
    cols = np.tile(edofs, (1, 8)).ravel()
    n = 2 * len(nodes)
    return sp.csr_matrix((Ke_all.ravel(), (rows, cols)), shape=(n, n))


def solve(K, F, fixed_dofs, fixed_vals=None):
    n = K.shape[0]
    fixed_dofs = np.unique(np.asarray(fixed_dofs, int))
    U = np.zeros(n)
    if fixed_vals is not None:
        U[fixed_dofs] = fixed_vals
    free = np.setdiff1d(np.arange(n), fixed_dofs)
    rhs = F[free] - K[free][:, fixed_dofs] @ U[fixed_dofs]
    U[free] = spla.spsolve(K[free][:, free].tocsc(), rhs)
    return U


# ----------------------------------------------------------------- stresses
def nodal_stresses(nodes, elems, D, U):
    """
    Stresses [sxx, syy, sxy] at Gauss points, extrapolated to element nodes
    with the inverse bilinear map, then averaged over elements sharing a node.
    """
    # extrapolation matrix: node value = sum_k N_k(+-sqrt3) * gauss value_k
    s3 = np.sqrt(3.0)
    Ext = np.array([shape(s3 * a, s3 * b)[0] for a, b in NAT])
    edofs = element_dofs(elems)
    acc = np.zeros((len(nodes), 3))
    cnt = np.zeros(len(nodes))
    for e, el in enumerate(elems):
        xy = nodes[el]
        ue = U[edofs[e]]
        sg = np.array([D @ (q4_B(xy, xi, eta)[0] @ ue) for xi, eta in GAUSS])
        sn = Ext @ sg
        acc[el] += sn
        cnt[el] += 1
    return acc / cnt[:, None]


def von_mises(s):
    sx, sy, txy = s[..., 0], s[..., 1], s[..., 2]
    return np.sqrt(sx**2 - sx * sy + sy**2 + 3 * txy**2)


# ----------------------------------------------------------------- loads
def edge_load(nodes, segments, traction, t=1.0):
    """
    Consistent nodal forces for traction(x, y) -> (tx, ty) applied on boundary
    segments given as node pairs (a, b). 3-point Gauss along each segment.
    """
    F = np.zeros(2 * len(nodes))
    gp = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])
    gw = np.array([5, 8, 5]) / 9.0
    for a, b in segments:
        pa, pb = nodes[a], nodes[b]
        L = np.linalg.norm(pb - pa)
        for s, w in zip(gp, gw):
            Na, Nb = (1 - s) / 2, (1 + s) / 2
            x, y = Na * pa + Nb * pb
            tx, ty = traction(x, y)
            f = w * L / 2 * t
            F[2 * a] += f * Na * tx; F[2 * a + 1] += f * Na * ty
            F[2 * b] += f * Nb * tx; F[2 * b + 1] += f * Nb * ty
    return F
