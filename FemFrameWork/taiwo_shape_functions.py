import numpy as np
import sympy as sp
def _validate_inputs_constraints(Constraints: np.ndarray,
                                 dofs_per_node: int,
                                 NumNodes: int) -> None:
    """
    Lightweight input validation with friendly errors.

    Parameters
    ----------
    Constraints : np.ndarray
        Expected shape (NCons, 3). Each row: [node (1-based), dof (1-based), value].
        'value' is not used for numbering; only node/dof indices matter.
    dofs_per_node : int
        Positive integer, number of DOFs per node.
    NumNodes : int
        Nonnegative integer, total number of nodes.
    """
    if not isinstance(Constraints, np.ndarray):
        raise TypeError("Oops! 'Constraints' must be a NumPy array of shape (NCons, 3).")
    if Constraints.ndim != 2 or Constraints.shape[1] != 3:
        raise ValueError("Expected 'Constraints' of shape (NCons, 3): [node, dof, value].")

    # Accept both Python ints and NumPy integer scalars
    if not isinstance(dofs_per_node, (int, np.integer)) or dofs_per_node <= 0:
        raise ValueError("'dofs_per_node' must be a positive integer.")
    if not isinstance(NumNodes, (int, np.integer)) or NumNodes < 0:
        raise ValueError("'NumNodes' must be a nonnegative integer.")
def Create_ID_Matrix(Constraints: np.ndarray,
                     dofs_per_node: int,
                     NumNodes: int) -> tuple[np.ndarray, int]:
    """
    Assign equation IDs to free DOFs and constraint IDs to fixed DOFs (vectorized).

    Semantics (matches the standard classroom loop):
      • Each constraint row (in order) is assigned an ID of -1, -2, ..., -NCons at its (node, dof).
      • All remaining free DOFs are assigned positive equation numbers 1..NEqns
        in row-major order (i=rows → j=cols).
      • If a (node, dof) appears multiple times in Constraints, the last row wins.

    Parameters
    ----------
    Constraints : np.ndarray
        Array of shape (NCons, 3): [node (1-based), dof (1-based), value].
        'value' is ignored for numbering.
    dofs_per_node : int
        Number of DOFs per node (must be > 0).
    NumNodes : int
        Total number of nodes (must be >= 0).

    Returns
    -------
    GlobalID : np.ndarray
        Shape (NumNodes, dofs_per_node), dtype=int. Constrained DOFs carry negative IDs;
        free DOFs carry positive contiguous IDs (starting at 1).
    NEqns : int
        Number of free DOFs (i.e., count of positive IDs).

    Example
    -------
    >>> import numpy as np
    >>> # 2 nodes, 3 dofs per node; constrain (node=1,dof=2) and (node=2,dof=1)
    >>> C = np.array([[1, 2, 0.0],
    ...               [2, 1, 5.0]])
    >>> G, n = Create_ID_Matrix(C, dofs_per_node=3, NumNodes=2)
    >>> G
    array([[ 1, -1,  2],
           [-2,  3,  4]])
    >>> n
    4
    """
    _validate_inputs_constraints(Constraints, dofs_per_node, NumNodes)

    # Normalize numpy integers to built-in ints for downstream consistency
    dofs_per_node = int(dofs_per_node)
    NumNodes = int(NumNodes)

    # 1) Start with zeros: 0 means "free & unnumbered (yet)".
    GlobalID = np.zeros((NumNodes, dofs_per_node), dtype=int)

    # 2) Short-circuit when there are no constraints.
    NCons = Constraints.shape[0]
    if NCons == 0:
        total_free = GlobalID.size
        GlobalID.flat[:] = np.arange(1, total_free + 1, dtype=int)
        return GlobalID, total_free

    # 3) Convert 1-based node/dof indices to 0-based for NumPy indexing.
    nodes0 = Constraints[:, 0].astype(int) - 1
    dofs0  = Constraints[:, 1].astype(int) - 1

    # 4) Range checks (clear, early failure if out-of-bounds).
    if (nodes0 < 0).any() or (nodes0 >= NumNodes).any():
        raise IndexError("A constraint has a node index outside [1, NumNodes].")
    if (dofs0 < 0).any() or (dofs0 >= dofs_per_node).any():
        raise IndexError("A constraint has a dof index outside [1, dofs_per_node].")

    # 5) Constrained IDs: -1, -2, ..., -NCons (order is exactly as provided).
    neg_ids = -np.arange(1, NCons + 1, dtype=int)

    # 6) Vectorized scatter: put the negative IDs into the (node,dof) slots in one shot.
    GlobalID[nodes0, dofs0] = neg_ids

    # 7) Number remaining free DOFs (zeros) in row-major order, matching nested for-loops.
    GlobalID_flat = GlobalID.ravel(order='C')            # row-major view (no copy)
    free_positions = np.flatnonzero(GlobalID_flat == 0)  # indices of free slots in row-major
    NEqns = free_positions.size
    GlobalID_flat[free_positions] = np.arange(1, NEqns + 1, dtype=int)

    return GlobalID, NEqns
# =========================
# Pretty printing helpers (readable, grader-friendly)
# =========================
def _preview_matrix(A: np.ndarray, max_entries: int = 12) -> str:
    """
    Build a concise preview string for matrix A.
    Prints full matrix if m*n <= max_entries; otherwise prints the first k rows
    such that rows*dcols <= max_entries and annotates that the rest is omitted.
    """
    m, n = A.shape
    total = m * n
    if total <= max_entries or m == 0 or n == 0:
        return f"{A}"
    # compute how many rows we can show without exceeding max_entries
    max_rows = max(1, min(m, max_entries // n))
    shown = A[:max_rows, :]
    omitted_rows = m - max_rows
    lines = [
        f"{shown}",
        f"... (omitting {omitted_rows} more row(s) for readability; total shape = {A.shape})"
    ]
    return "\n".join(lines)


def _preview_constraints(C: np.ndarray, max_rows: int = 6) -> str:
    """
    Compact preview for the Constraints array (NCons x 3).
    Shows up to 'max_rows' rows, with an omission note if longer.
    """
    if C.size == 0:
        return "Constraints: [] (no constraints)"
    rows = C.shape[0]
    if rows <= max_rows:
        return f"Constraints (showing all {rows}):\n{C}"
    shown = C[:max_rows, :]
    omitted = rows - max_rows
    return f"Constraints (showing first {max_rows} of {rows}):\n{shown}\n... (omitting {omitted} more row(s))"


def _summarize_id_matrix(G: np.ndarray) -> str:
    """
    Return a one-line summary of counts of constrained vs free entries.
    """
    constrained = int((G < 0).sum())
    free = int((G > 0).sum())
    return f"[free: {free}, constrained: {constrained}, shape: {G.shape}]"
# =========================
# Tests (robust but simple)
# =========================
def test_Create_ID_Matrix() -> None:
    """
    Run sanity tests (valid & invalid paths) with readable output.
    Shows compact previews to help the grader verify inputs and outputs at a glance.
    """
    print("\n===== Problem 1: Create_ID_Matrix — VALID CASES =====")

    # Case A: No constraints (simple, all free)
    NumNodes, dofs_per_node = 2, 3
    Constraints = np.empty((0, 3))
    print("PASS - Case A (no constraints):")
    print(f"  NumNodes = {NumNodes}, dofs_per_node = {dofs_per_node}")
    print("  " + _preview_constraints(Constraints))
    G, n = Create_ID_Matrix(Constraints, dofs_per_node, NumNodes)
    assert n == NumNodes * dofs_per_node, "All DOFs should be free."
    assert (G > 0).all() and np.array_equal(G.ravel(order='C'), np.arange(1, G.size + 1))
    print("  Output Summary:", _summarize_id_matrix(G))
    print(_preview_matrix(G, max_entries=12))

    # Case B: A few constraints (exact layout check)
    NumNodes, dofs_per_node = 2, 3
    Constraints = np.array([
        [1, 2, 0.0],  # -> (0,1) = -1
        [2, 1, 5.0],  # -> (1,0) = -2
        [2, 3, 7.0],  # -> (1,2) = -3
    ])
    print("\nPASS - Case B (few constraints):")
    print(f"  NumNodes = {NumNodes}, dofs_per_node = {dofs_per_node}")
    print("  " + _preview_constraints(Constraints))
    G, n = Create_ID_Matrix(Constraints, dofs_per_node, NumNodes)
    G_expected = np.array([[ 1, -1,  2],
                           [-2,  3, -3]])
    assert n == 3  # 6 total - 3 constrained = 3 free
    assert np.array_equal(G, G_expected)
    print("  Output Summary:", _summarize_id_matrix(G))
    print(_preview_matrix(G, max_entries=12))

    # Case C: Duplicate constraint (same node,dof) → last wins (same as loop semantics)
    NumNodes, dofs_per_node = 1, 3
    Constraints = np.array([
        [1, 1, 10.0],
        [1, 1, 20.0],  # later row overwrites earlier (ID becomes -2)
        [1, 3, 0.0],
    ])
    print("\nPASS - Case C (duplicate → last wins):")
    print(f"  NumNodes = {NumNodes}, dofs_per_node = {dofs_per_node}")
    print("  " + _preview_constraints(Constraints))
    G, n = Create_ID_Matrix(Constraints, dofs_per_node, NumNodes)
    assert n == 1
    assert G[0, 0] == -2 and G[0, 2] == -3 and G[0, 1] == 1
    print("  Output Summary:", _summarize_id_matrix(G))
    print(_preview_matrix(G, max_entries=12))

    # Case D: Larger display demo (concise preview; avoids printing the full matrix)
    NumNodes, dofs_per_node = 10, 4  # 40 DOFs total → preview kicks in
    Constraints = np.array([
        [1, 1, 0.0],
        [5, 4, 0.0],
        [10, 2, 0.0],
        [7, 3, 0.0],
        [3, 1, 0.0],
    ])
    print("\nPASS - Case D (larger preview demo):")
    print(f"  NumNodes = {NumNodes}, dofs_per_node = {dofs_per_node}")
    print("  " + _preview_constraints(Constraints))
    G, n = Create_ID_Matrix(Constraints, dofs_per_node, NumNodes)
    assert G.shape == (NumNodes, dofs_per_node)
    assert n == NumNodes * dofs_per_node - Constraints.shape[0]
    assert (G < 0).sum() == Constraints.shape[0]
    print("  Output Summary:", _summarize_id_matrix(G))
    print(_preview_matrix(G, max_entries=12))

    print("\n===== Problem 1: Create_ID_Matrix — INVALID CASES =====")

    # Invalid 1: wrong Constraints shape
    try:
        print("\n--> Invalid 1: bad Constraints shape")
        print("  NumNodes = 2, dofs_per_node = 2")
        bad = np.array([1, 2, 3])
        print("  Constraints (bad):", bad)
        Create_ID_Matrix(bad, 2, 2)
        print("FAIL - Invalid 1: did not raise for bad shape")
    except Exception as e:
        print(f"  Raised: {type(e).__name__}: {e}")
        if isinstance(e, ValueError):
            print("PASS - Invalid 1 (bad Constraints shape): raised as expected.")
        else:
            raise

    # Invalid 2: out-of-range node index
    try:
        print("\n--> Invalid 2: node index out-of-bounds")
        print("  NumNodes = 2, dofs_per_node = 2")
        bad = np.array([[3, 1, 0.0]])  # node=3 but NumNodes=2
        print("  " + _preview_constraints(bad))
        Create_ID_Matrix(bad, 2, 2)
        print("FAIL - Invalid 2: did not raise for out-of-range node")
    except Exception as e:
        print(f"  Raised: {type(e).__name__}: {e}")
        if isinstance(e, IndexError):
            print("PASS - Invalid 2 (node index OOB): raised as expected.")
        else:
            raise

    # Invalid 3: out-of-range dof index
    try:
        print("\n--> Invalid 3: dof index out-of-bounds")
        print("  NumNodes = 2, dofs_per_node = 2")
        bad = np.array([[1, 3, 0.0]])  # dof=3 but dofs_per_node=2
        print("  " + _preview_constraints(bad))
        Create_ID_Matrix(bad, 2, 2)
        print("FAIL - Invalid 3: did not raise for out-of-range dof")
    except Exception as e:
        print(f"  Raised: {type(e).__name__}: {e}")
        if isinstance(e, IndexError):
            print("PASS - Invalid 3 (dof index OOB): raised as expected.")
        else:
            raise

    print("\nAll Create_ID_Matrix tests completed.\n")

# =========================
# Entry point
# =========================
if __name__ == "__main__":
    test_Create_ID_Matrix()

# Valid element set (single-level dispatcher)
_VALID_TYPES = {
    'L2', 'L3',
    'Q4', 'Q8', 'Q9',
    'T3', 'T6',
    'B8', 'B27', 'TET4', 'W6'
}

# Dimensionality per element type (validation only)
DIMENSION_OF = {
    'L2': 1, 'L3': 1,
    'Q4': 2, 'Q8': 2, 'Q9': 2, 'T3': 2, 'T6': 2,
    'B8': 3, 'B27': 3, 'TET4': 3, 'W6': 3,
}
ELEMENT_NODES = {
    # 1D (L3 order aligned with slide: [-1], [+1], [0])
    'L2': [np.array([-1.0]), np.array([+1.0])],
    'L3': [np.array([-1.0]), np.array([+1.0]), np.array([0.0])],

    # 2D triangles (reference right triangle)
    'T3': [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])],
    'T6': [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]),
           np.array([0.5, 0.0]), np.array([0.5, 0.5]), np.array([0.0, 0.5])],

    # 2D quads on [-1,1]^2, corners CCW from (-1,-1)
    'Q4': [np.array([-1.0, -1.0]), np.array([+1.0, -1.0]),
           np.array([+1.0, +1.0]), np.array([-1.0, +1.0])],
    'Q8': [np.array([-1.0, -1.0]), np.array([+1.0, -1.0]),
           np.array([+1.0, +1.0]), np.array([-1.0, +1.0]),
           np.array([0.0, -1.0]),  np.array([+1.0, 0.0]),
           np.array([0.0, +1.0]),  np.array([-1.0, 0.0])],
    'Q9': [np.array([-1.0, -1.0]), np.array([+1.0, -1.0]),
           np.array([+1.0, +1.0]), np.array([-1.0, +1.0]),
           np.array([0.0, -1.0]),  np.array([+1.0, 0.0]),
           np.array([0.0, +1.0]),  np.array([-1.0, 0.0]),
           np.array([0.0,  0.0])],

    # 3D bricks on [-1,1]^3 (standard hexahedral order)
    # 1:(-1,-1,-1), 2:(+1,-1,-1), 3:(+1,+1,-1), 4:(-1,+1,-1),
    # 5:(-1,-1,+1), 6:(+1,-1,+1), 7:(+1,+1,+1), 8:(-1,+1,+1)
    'B8': [np.array([-1.0, -1.0, -1.0]), np.array([+1.0, -1.0, -1.0]),
           np.array([+1.0, +1.0, -1.0]), np.array([-1.0, +1.0, -1.0]),
           np.array([-1.0, -1.0, +1.0]), np.array([+1.0, -1.0, +1.0]),
           np.array([+1.0, +1.0, +1.0]), np.array([-1.0, +1.0, +1.0])],

    # 3D tetra (reference right tetra)
    # 1:(0,0,0), 2:(1,0,0), 3:(0,1,0), 4:(0,0,1)
    'TET4': [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
             np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])],

    # 3D wedge (prism): bottom triangle zeta_3=-1 -> nodes 1..3, top triangle zeta_3=+1 -> nodes 4..6
    # bottom: (0,0,-1), (1,0,-1), (0,1,-1); top: (0,0,+1), (1,0,+1), (0,1,+1)
    'W6': [np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, -1.0]), np.array([0.0, 1.0, -1.0]),
           np.array([0.0, 0.0, +1.0]), np.array([1.0, 0.0, +1.0]), np.array([0.0, 1.0, +1.0])],
}

# ---- add B27 nodes in tensor-product order: z1 in {-1,0,1} outer, z2 mid, z3 fastest ----
def _grid_coords(vals):
    out = []
    for v1 in vals:
        for v2 in vals:
            for v3 in vals:
                out.append(np.array([v1, v2, v3], dtype=float))
    return out

ELEMENT_NODES['B27'] = _grid_coords([-1.0, 0.0, 1.0])

def _normalize_type(EleType: str) -> str:
    if not isinstance(EleType, str):
        raise TypeError("Oops! 'EleType' must be a string like 'Q4', 'T3', etc.")
    t = EleType.strip().upper()
    if t not in _VALID_TYPES:
        raise ValueError(f"Invalid element type '{EleType}'. Valid options: {sorted(_VALID_TYPES)}.")
    return t


def _validate_zeta_dim(E: str, zeta: np.ndarray) -> np.ndarray:
    z = np.asarray(zeta, dtype=float).reshape(-1)
    need = DIMENSION_OF[E]
    if z.size != need:
        raise ValueError(f"{E} expects {need} reference coord(s); got {z.size}.")
    return z


def _make_symbols(n_dim: int):
    names = ["zeta_1", "zeta_2", "zeta_3"]
    return [sp.Symbol(names[i]) for i in range(n_dim)]
def ShapeFunctions(EleType: str, zeta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return shape functions N and their reference-space derivatives DN at a point `zeta`
    for the requested finite element type.

    Contract & Invariants
    ---------------------
    - Single entry point (no dimension-based sub-dispatch outside this function).
    - Element-specific definitions with symbolic differentiation (via SymPy) internally.
    - Invariants at any admissible `zeta`:
        * Partition of unity: sum_i N_i = 1
        * Column-wise derivative sum: for each axis k, sum_i dN_i/d(zeta_k) = 0

    Parameters
    ----------
    EleType : str
        Element type selector. One of:
        {'L2','L3','Q4','Q8','Q9','T3','T6','B8','B27','TET4','W6'}.
    zeta : np.ndarray
        Reference coordinate of the evaluation point, shaped by element dimension:
          - 1D: [zeta_1]
          - 2D: [zeta_1, zeta_2]
          - 3D: [zeta_1, zeta_2, zeta_3]

    Returns
    -------
    N : np.ndarray
        Shape-function values at `zeta`. Shape: (n_nodes,).
    DN : np.ndarray
        Reference-space derivatives at `zeta`. Shape: (n_nodes, n_dim),
        where `n_dim` is the element's reference dimension (1, 2, or 3).

    Raises
    ------
    TypeError
        If `EleType` is not a string, or `zeta` is not array-like / numeric.
    ValueError
        If `EleType` is not a supported element type, or if the length of `zeta`
        does not match the element’s reference dimension.
    RuntimeError
        If the computed derivative array `DN` does not have the expected shape
        (n_nodes, n_dim) after evaluation.

    Notes
    -----
    - This function validates `EleType` and the dimensionality of `zeta` before dispatch.
    - Numerical outputs are returned as float64 arrays.

    Examples
    --------
    Q4 at the square center (zeta_1=0, zeta_2=0):
    >>> N, DN = ShapeFunctions('Q4', np.array([0.0, 0.0]))
    >>> N
    array([0.25, 0.25, 0.25, 0.25])
    >>> DN
    array([[-0.25, -0.25],
            [ 0.25, -0.25],
            [ 0.25,  0.25],
            [-0.25,  0.25]])

    T3 at the triangle centroid (zeta_1=1/3, zeta_2=1/3):
    >>> N, DN = ShapeFunctions('T3', np.array([1/3, 1/3]))
    >>> N
    array([0.33333333, 0.33333333, 0.33333333])
    >>> DN
    array([[-1., -1.],
            [ 1.,  0.],
            [ 0.,  1.]])
    """

    E = _normalize_type(EleType)
    z = _validate_zeta_dim(E, zeta)

    dispatch = {
        'L2': _sf_L2, 'L3': _sf_L3,
        'Q4': _sf_Q4, 'Q8': _sf_Q8, 'Q9': _sf_Q9,
        'T3': _sf_T3, 'T6': _sf_T6,
        'B8': _sf_B8, 'B27': _sf_B27, 'TET4': _sf_TET4, 'W6': _sf_W6,
    }

    N, DN = dispatch[E](z)
    N = np.asarray(N, dtype=float).reshape(-1)
    DN = np.asarray(DN, dtype=float)
    n_dim = DIMENSION_OF[E]
    if DN.shape != (N.size, n_dim):
        raise RuntimeError(f"{E} DN shape mismatch. Expected {(N.size, n_dim)}, got {DN.shape}.")
    return N, DN
# ---- 1D ----
def _sf_L2(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    L2 (2-node line) on reference interval zeta_1 ∈ [-1, 1].

    Parameters
    ----------
    z : np.ndarray
        Reference coordinate [zeta_1].

    Returns
    -------
    N : np.ndarray
        (2,) shape-function values at z.
    DN : np.ndarray
        (2,1) derivatives dN/d(zeta_1) at z.
    """
    (zeta_1,) = _make_symbols(1)
    N_expr = [(1 - zeta_1)/2, (1 + zeta_1)/2]
    dN_dz1 = [sp.diff(Ni, zeta_1) for Ni in N_expr]
    z1 = float(z[0])
    N  = np.array([float(Ni.subs(zeta_1, z1)) for Ni in N_expr], dtype=float)
    DN = np.array([float(d.subs(zeta_1, z1)) for d in dN_dz1], dtype=float).reshape(-1, 1)
    return N, DN


def _sf_L3(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    L3 (3-node quadratic line) on reference interval zeta_1 ∈ [-1, 1].

    Parameters
    ----------
    z : np.ndarray
        Reference coordinate [zeta_1].

    Returns
    -------
    N : np.ndarray
        (3,) shape-function values at z.
    DN : np.ndarray
        (3,1) derivatives dN/d(zeta_1) at z.
    """
    (zeta_1,) = _make_symbols(1)
    N_expr = [
        0.5*zeta_1*(zeta_1 - 1),
        0.5*zeta_1*(zeta_1 + 1),
        1 - zeta_1**2,
    ]
    dN_dz1 = [sp.diff(Ni, zeta_1) for Ni in N_expr]
    z1 = float(z[0])
    N  = np.array([float(Ni.subs(zeta_1, z1)) for Ni in N_expr], dtype=float)
    DN = np.array([float(d.subs(zeta_1, z1)) for d in dN_dz1], dtype=float).reshape(-1, 1)
    return N, DN


# ---- 2D triangles ----
def _sf_T3(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    T3 (3-node linear triangle) on reference simplex:
      zeta_1 ≥ 0, zeta_2 ≥ 0, zeta_1 + zeta_2 ≤ 1.

    Parameters
    ----------
    z : np.ndarray
        Reference coordinates [zeta_1, zeta_2].

    Returns
    -------
    N : np.ndarray
        (3,) shape-function values at z.
    DN : np.ndarray
        (3,2) derivatives w.r.t. [zeta_1, zeta_2] at z.
    """
    z1, z2 = map(float, z)
    zeta_1, zeta_2 = _make_symbols(2)
    N_expr = [1 - zeta_1 - zeta_2, zeta_1, zeta_2]
    dN_dz1 = [sp.diff(Ni, zeta_1) for Ni in N_expr]
    dN_dz2 = [sp.diff(Ni, zeta_2) for Ni in N_expr]
    subs = {zeta_1: z1, zeta_2: z2}
    N  = np.array([float(Ni.subs(subs)) for Ni in N_expr], dtype=float)
    DN = np.vstack([
        np.array([float(d.subs(subs)) for d in dN_dz1], dtype=float),
        np.array([float(d.subs(subs)) for d in dN_dz2], dtype=float),
    ]).T
    return N, DN


def _sf_T6(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    T6 (6-node quadratic triangle) on reference simplex:
      zeta_1 ≥ 0, zeta_2 ≥ 0, zeta_1 + zeta_2 ≤ 1.

    Parameters
    ----------
    z : np.ndarray
        Reference coordinates [zeta_1, zeta_2].

    Returns
    -------
    N : np.ndarray
        (6,) shape-function values at z.
    DN : np.ndarray
        (6,2) derivatives w.r.t. [zeta_1, zeta_2] at z.
    """
    z1, z2 = map(float, z)
    zeta_1, zeta_2 = _make_symbols(2)
    N_expr = [
        (1 - zeta_1 - zeta_2)*(1 - 2*zeta_1 - 2*zeta_2),
        zeta_1*(2*zeta_1 - 1),
        zeta_2*(2*zeta_2 - 1),
        4*zeta_1*(1 - zeta_1 - zeta_2),
        4*zeta_1*zeta_2,
        4*zeta_2*(1 - zeta_1 - zeta_2),
    ]
    dN_dz1 = [sp.diff(Ni, zeta_1) for Ni in N_expr]
    dN_dz2 = [sp.diff(Ni, zeta_2) for Ni in N_expr]
    subs = {zeta_1: z1, zeta_2: z2}
    N  = np.array([float(Ni.subs(subs)) for Ni in N_expr], dtype=float)
    DN = np.vstack([
        np.array([float(d.subs(subs)) for d in dN_dz1], dtype=float),
        np.array([float(d.subs(subs)) for d in dN_dz2], dtype=float),
    ]).T
    return N, DN


# ---- 2D quads ----
def _sf_Q4(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Q4 (4-node bilinear quad) on reference square [-1, 1]^2.

    Parameters
    ----------
    z : np.ndarray
        Reference coordinates [zeta_1, zeta_2].

    Returns
    -------
    N : np.ndarray
        (4,) shape-function values at z.
    DN : np.ndarray
        (4,2) derivatives w.r.t. [zeta_1, zeta_2] at z.
    """
    z1, z2 = map(float, z)
    zeta_1, zeta_2 = _make_symbols(2)
    N_expr = [
        0.25*(1 - zeta_1)*(1 - zeta_2),
        0.25*(1 + zeta_1)*(1 - zeta_2),
        0.25*(1 + zeta_1)*(1 + zeta_2),
        0.25*(1 - zeta_1)*(1 + zeta_2),
    ]
    dN_dz1 = [sp.diff(Ni, zeta_1) for Ni in N_expr]
    dN_dz2 = [sp.diff(Ni, zeta_2) for Ni in N_expr]
    subs = {zeta_1: z1, zeta_2: z2}
    N  = np.array([float(Ni.subs(subs)) for Ni in N_expr], dtype=float)
    DN = np.vstack([
        np.array([float(d.subs(subs)) for d in dN_dz1], dtype=float),
        np.array([float(d.subs(subs)) for d in dN_dz2], dtype=float),
    ]).T
    return N, DN


def _sf_Q8(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Q8 (8-node serendipity quad) on reference square [-1, 1]^2.

    Parameters
    ----------
    z : np.ndarray
        Reference coordinates [zeta_1, zeta_2].

    Returns
    -------
    N : np.ndarray
        (8,) shape-function values at z.
    DN : np.ndarray
        (8,2) derivatives w.r.t. [zeta_1, zeta_2] at z.
    """
    z1, z2 = map(float, z)
    zeta_1, zeta_2 = _make_symbols(2)
    N_expr = [
        0.25*(1 - zeta_1)*(1 - zeta_2)*(-zeta_1 - zeta_2 - 1),
        0.25*(1 + zeta_1)*(1 - zeta_2)*( zeta_1 - zeta_2 - 1),
        0.25*(1 + zeta_1)*(1 + zeta_2)*( zeta_1 + zeta_2 - 1),
        0.25*(1 - zeta_1)*(1 + zeta_2)*(-zeta_1 + zeta_2 - 1),
        0.5*(1 - zeta_1**2)*(1 - zeta_2),
        0.5*(1 + zeta_1)*(1 - zeta_2**2),
        0.5*(1 - zeta_1**2)*(1 + zeta_2),
        0.5*(1 - zeta_1)*(1 - zeta_2**2),
    ]
    dN_dz1 = [sp.diff(Ni, zeta_1) for Ni in N_expr]
    dN_dz2 = [sp.diff(Ni, zeta_2) for Ni in N_expr]
    subs = {zeta_1: z1, zeta_2: z2}
    N  = np.array([float(Ni.subs(subs)) for Ni in N_expr], dtype=float)
    DN = np.vstack([
        np.array([float(d.subs(subs)) for d in dN_dz1], dtype=float),
        np.array([float(d.subs(subs)) for d in dN_dz2], dtype=float),
    ]).T
    return N, DN


def _sf_Q9(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Q9 (9-node quadratic quad) on reference square [-1, 1]^2.

    Parameters
    ----------
    z : np.ndarray
        Reference coordinates [zeta_1, zeta_2].

    Returns
    -------
    N : np.ndarray
        (9,) shape-function values at z.
    DN : np.ndarray
        (9,2) derivatives w.r.t. [zeta_1, zeta_2] at z.
    """
    z1, z2 = map(float, z)
    zeta_1, zeta_2 = _make_symbols(2)
    N_expr = [
        +0.25*zeta_1*zeta_2*(1 - zeta_1)*(1 - zeta_2),
        -0.25*zeta_1*zeta_2*(1 + zeta_1)*(1 - zeta_2),
        +0.25*zeta_1*zeta_2*(1 + zeta_1)*(1 + zeta_2),
        -0.25*zeta_1*zeta_2*(1 - zeta_1)*(1 + zeta_2),
        -0.5*(1 - zeta_1**2)*zeta_2*(1 - zeta_2),
        +0.5*zeta_1*(1 + zeta_1)*(1 - zeta_2**2),
        +0.5*(1 - zeta_1**2)*zeta_2*(1 + zeta_2),
        -0.5*zeta_1*(1 - zeta_1)*(1 - zeta_2**2),
        (1 - zeta_1**2)*(1 - zeta_2**2),
    ]
    dN_dz1 = [sp.diff(Ni, zeta_1) for Ni in N_expr]
    dN_dz2 = [sp.diff(Ni, zeta_2) for Ni in N_expr]
    subs = {zeta_1: z1, zeta_2: z2}
    N  = np.array([float(Ni.subs(subs)) for Ni in N_expr], dtype=float)
    DN = np.vstack([
        np.array([float(d.subs(subs)) for d in dN_dz1], dtype=float),
        np.array([float(d.subs(subs)) for d in dN_dz2], dtype=float),
    ]).T
    return N, DN


# ---- 3D elements ----
def _sf_B8(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    B8 (8-node brick) on reference cube [-1, 1]^3.

    Parameters
    ----------
    z : np.ndarray
        Reference coordinates [zeta_1, zeta_2, zeta_3].

    Returns
    -------
    N : np.ndarray
        (8,) shape-function values at z.
    DN : np.ndarray
        (8,3) derivatives w.r.t. [zeta_1, zeta_2, zeta_3] at z.
    """
    z1, z2, z3 = map(float, z)
    zeta_1, zeta_2, zeta_3 = _make_symbols(3)
    N_expr = [
        sp.Rational(1, 8)*(1 - zeta_1)*(1 - zeta_2)*(1 - zeta_3),
        sp.Rational(1, 8)*(1 + zeta_1)*(1 - zeta_2)*(1 - zeta_3),
        sp.Rational(1, 8)*(1 + zeta_1)*(1 + zeta_2)*(1 - zeta_3),
        sp.Rational(1, 8)*(1 - zeta_1)*(1 + zeta_2)*(1 - zeta_3),
        sp.Rational(1, 8)*(1 - zeta_1)*(1 - zeta_2)*(1 + zeta_3),
        sp.Rational(1, 8)*(1 + zeta_1)*(1 - zeta_2)*(1 + zeta_3),
        sp.Rational(1, 8)*(1 + zeta_1)*(1 + zeta_2)*(1 + zeta_3),
        sp.Rational(1, 8)*(1 - zeta_1)*(1 + zeta_2)*(1 + zeta_3),
    ]
    dN_dz1 = [sp.diff(Ni, zeta_1) for Ni in N_expr]
    dN_dz2 = [sp.diff(Ni, zeta_2) for Ni in N_expr]
    dN_dz3 = [sp.diff(Ni, zeta_3) for Ni in N_expr]
    subs = {zeta_1: z1, zeta_2: z2, zeta_3: z3}
    N  = np.array([float(Ni.subs(subs)) for Ni in N_expr], dtype=float)
    DN = np.vstack([
        np.array([float(d.subs(subs)) for d in dN_dz1], dtype=float),
        np.array([float(d.subs(subs)) for d in dN_dz2], dtype=float),
        np.array([float(d.subs(subs)) for d in dN_dz3], dtype=float),
    ]).T
    return N, DN


def _sf_TET4(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    TET4 (4-node tetrahedron) on reference simplex:
      zeta_1 ≥ 0, zeta_2 ≥ 0, zeta_3 ≥ 0, zeta_1 + zeta_2 + zeta_3 ≤ 1.

    Parameters
    ----------
    z : np.ndarray
        Reference coordinates [zeta_1, zeta_2, zeta_3].

    Returns
    -------
    N : np.ndarray
        (4,) shape-function values at z.
    DN : np.ndarray
        (4,3) derivatives w.r.t. [zeta_1, zeta_2, zeta_3] at z.
    """
    z1, z2, z3 = map(float, z)
    zeta_1, zeta_2, zeta_3 = _make_symbols(3)
    N_expr = [
        1 - zeta_1 - zeta_2 - zeta_3,
        zeta_1,
        zeta_2,
        zeta_3,
    ]
    dN_dz1 = [sp.diff(Ni, zeta_1) for Ni in N_expr]
    dN_dz2 = [sp.diff(Ni, zeta_2) for Ni in N_expr]
    dN_dz3 = [sp.diff(Ni, zeta_3) for Ni in N_expr]
    subs = {zeta_1: z1, zeta_2: z2, zeta_3: z3}
    N  = np.array([float(Ni.subs(subs)) for Ni in N_expr], dtype=float)
    DN = np.vstack([
        np.array([float(d.subs(subs)) for d in dN_dz1], dtype=float),
        np.array([float(d.subs(subs)) for d in dN_dz2], dtype=float),
        np.array([float(d.subs(subs)) for d in dN_dz3], dtype=float),
    ]).T
    return N, DN


def _sf_W6(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    W6 (6-node wedge/prism):
      triangular base (zeta_1, zeta_2) with zeta_1 ≥ 0, zeta_2 ≥ 0, zeta_1 + zeta_2 ≤ 1,
      and zeta_3 ∈ [-1, 1].

    Parameters
    ----------
    z : np.ndarray
        Reference coordinates [zeta_1, zeta_2, zeta_3].

    Returns
    -------
    N : np.ndarray
        (6,) shape-function values at z.
    DN : np.ndarray
        (6,3) derivatives w.r.t. [zeta_1, zeta_2, zeta_3] at z.
    """
    z1, z2, z3 = map(float, z)
    zeta_1, zeta_2, zeta_3 = _make_symbols(3)

    N_expr = [
        0.5*(1 - zeta_3)*(1 - zeta_1 - zeta_2),
        0.5*(1 - zeta_3)*zeta_1,
        0.5*(1 - zeta_3)*zeta_2,
        0.5*(1 + zeta_3)*(1 - zeta_1 - zeta_2),
        0.5*(1 + zeta_3)*zeta_1,
        0.5*(1 + zeta_3)*zeta_2,
    ]
    dN_dz1 = [sp.diff(Ni, zeta_1) for Ni in N_expr]
    dN_dz2 = [sp.diff(Ni, zeta_2) for Ni in N_expr]
    dN_dz3 = [sp.diff(Ni, zeta_3) for Ni in N_expr]

    subs = {zeta_1: z1, zeta_2: z2, zeta_3: z3}
    N  = np.array([float(Ni.subs(subs)) for Ni in N_expr], dtype=float)
    DN = np.vstack([
        np.array([float(d.subs(subs)) for d in dN_dz1], dtype=float),
        np.array([float(d.subs(subs)) for d in dN_dz2], dtype=float),
        np.array([float(d.subs(subs)) for d in dN_dz3], dtype=float),
    ]).T
    return N, DN


def _sf_B27(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    B27 (27-node quadratic brick) on reference cube [-1, 1]^3.
    Constructed via tensor product of 1D quadratic Lagrange polynomials at z = {-1, 0, +1}.

    Parameters
    ----------
    z : np.ndarray
        Reference coordinates [zeta_1, zeta_2, zeta_3].

    Returns
    -------
    N : np.ndarray
        (27,) shape-function values at z.
    DN : np.ndarray
        (27,3) derivatives w.r.t. [zeta_1, zeta_2, zeta_3] at z.
    """
    z1, z2, z3 = map(float, z)
    zeta_1, zeta_2, zeta_3 = _make_symbols(3)

    def L3(zsym):
        return [
            0.5*zsym*(zsym - 1),   # value 1 at z=-1
            1 - zsym**2,           # value 1 at z= 0
            0.5*zsym*(zsym + 1),   # value 1 at z=+1
        ]

    L1 = L3(zeta_1)
    L2 = L3(zeta_2)
    L3z = L3(zeta_3)

    # Tensor product basis (z1 outer, z2 mid, z3 fastest)
    N_expr = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                N_expr.append(L1[i] * L2[j] * L3z[k])

    # Derivatives
    dN_dz1_expr = [sp.diff(Ni, zeta_1) for Ni in N_expr]
    dN_dz2_expr = [sp.diff(Ni, zeta_2) for Ni in N_expr]
    dN_dz3_expr = [sp.diff(Ni, zeta_3) for Ni in N_expr]

    subs = {zeta_1: z1, zeta_2: z2, zeta_3: z3}
    N  = np.array([float(Ni.subs(subs)) for Ni in N_expr], dtype=float)
    d1 = np.array([float(di.subs(subs)) for di in dN_dz1_expr], dtype=float)
    d2 = np.array([float(di.subs(subs)) for di in dN_dz2_expr], dtype=float)
    d3 = np.array([float(di.subs(subs)) for di in dN_dz3_expr], dtype=float)
    DN = np.vstack([d1, d2, d3]).T
    return N, DN

# =========================
# Pretty printing & tests
# =========================

# --- controls (only B27 is compact) ---
_B27_MAX_NODE_LINES = 8       # show first 8 nodes; rest summarized
_B27_SHOW_N_VECTOR  = False   # do NOT print the big N=[...] for B27
_OTHERS_SHOW_N_VECTOR = True  # keep N=[e_j] for all other elements

def _preview_ndarray(A: np.ndarray, max_entries: int = 12) -> str:
    A = np.asarray(A)
    total = A.size
    if total <= max_entries or A.ndim == 0:
        return f"{A}"
    if A.ndim == 1:
        shown = A[:max_entries]
        omitted = total - max_entries
        return f"{shown}\n... (omitting {omitted} entries; shape={A.shape})"
    elif A.ndim == 2:
        m, n = A.shape
        max_rows = max(1, min(m, max_entries // max(1, n)))
        shown = A[:max_rows, :]
        omitted_rows = m - max_rows
        suffix = "" if omitted_rows <= 0 else f"\n... (omitting {omitted_rows} more row(s); shape={A.shape})"
        return f"{shown}{suffix}"
    else:
        return f"{A.shape} (omit large tensor preview)"

def _label_point(z: np.ndarray) -> str:
    z = np.asarray(z).reshape(-1)
    names = ["zeta_1", "zeta_2", "zeta_3"]
    return "[" + ", ".join(f"{names[i]}={z[i]:.6g}" for i in range(z.size)) + "]"

PASS = "PASS"
FAIL = "FAIL"

def _element_centroid(E: str) -> np.ndarray:
    """Return a representative interior point for each element."""
    if E in ('L2','L3','Q4','Q8','Q9','B8','B27'):
        return np.zeros(DIMENSION_OF[E])                 # center of [-1,1]^d
    if E in ('T3','T6'):
        return np.array([1/3, 1/3])                      # triangle centroid
    if E == 'TET4':
        return np.array([1/3, 1/3, 1/3])                 # tetra centroid
    if E == 'W6':
        return np.array([1/3, 1/3, 0.0])                 # wedge: mid-plane
    return np.zeros(DIMENSION_OF[E])

def _rand_points_in_element(E: str, k: int, seed: int = 123) -> list[np.ndarray]:
    """Generate k random points inside the reference domain of element E."""
    rng = np.random.default_rng(seed)

    pts = []
    if E in ('L2','L3'):
        for _ in range(k):
            pts.append(np.array([rng.uniform(-1.0, 1.0)]))
        return pts

    if E in ('Q4','Q8','Q9'):
        for _ in range(k):
            pts.append(rng.uniform(-1.0, 1.0, size=2))
        return pts

    if E in ('B8','B27'):
        for _ in range(k):
            pts.append(rng.uniform(-1.0, 1.0, size=3))
        return pts

    if E in ('T3','T6'):
        # Sample barycentric (u,v) with u>=0, v>=0, u+v<=1
        for _ in range(k):
            a, b = rng.random(2)
            u = 1 - np.sqrt(a)
            v = b * np.sqrt(a)
            pts.append(np.array([u, v]))
        return pts

    if E == 'TET4':
        # Sample barycentric (4 weights), convert to (z1,z2,z3)
        for _ in range(k):
            w = rng.random(4)
            w = w / w.sum()
            z1, z2, z3 = w[1], w[2], w[3]
            pts.append(np.array([z1, z2, z3]))
        return pts

    if E == 'W6':
        # Triangle (z1,z2) using barycentric on base, and z3 in [-1,1]
        for _ in range(k):
            a, b = rng.random(2)
            u = 1 - np.sqrt(a)
            v = b * np.sqrt(a)
            z3 = rng.uniform(-1.0, 1.0)
            pts.append(np.array([u, v, z3]))
        return pts

    return [np.zeros(DIMENSION_OF[E]) for _ in range(k)]

def _kronecker_check(E: str) -> None:
    if E not in ELEMENT_NODES:
        print("  [A] Kronecker delta at nodes")
        print("    (skip) Nodes not defined for this element.")
        return

    nodes = ELEMENT_NODES[E]
    n_nodes = len(nodes)
    print("  [A] Kronecker delta at nodes")

    # --- B27: compact printing ---
    if E == 'B27':
        limit = min(_B27_MAX_NODE_LINES, n_nodes)
        failures = 0

        for j in range(limit):
            zj = nodes[j]
            N, _ = ShapeFunctions(E, zj)
            expected = np.zeros_like(N); expected[j] = 1.0
            ok = np.allclose(N, expected, atol=1e-10)
            status = "PASS" if ok else "FAIL"
            failures += 0 if ok else 1
            if _B27_SHOW_N_VECTOR:
                print(f"    Node {j+1} at {_label_point(zj)} -> {status}  N={N}")
            else:
                print(f"    Node {j+1} at {_label_point(zj)} -> {status}")

        if n_nodes > limit:
            rem_ok = 0
            for j in range(limit, n_nodes):
                zj = nodes[j]
                N, _ = ShapeFunctions(E, zj)
                expected = np.zeros_like(N); expected[j] = 1.0
                if np.allclose(N, expected, atol=1e-10):
                    rem_ok += 1
            rem_fail = (n_nodes - limit) - rem_ok
            failures += rem_fail
            print(f"    ... {n_nodes - limit} more nodes checked: {rem_ok} PASS, {rem_fail} FAIL")

        if failures == 0:
            print("    => All nodes satisfy Kronecker delta.")
        else:
            print(f"    => {failures} node(s) FAILED Kronecker check.")
        return

    # --- others: original verbose behavior with N vector (diagonal form) ---
    for j, zj in enumerate(nodes):
        N, _ = ShapeFunctions(E, zj)
        expected = np.zeros_like(N); expected[j] = 1.0
        ok = np.allclose(N, expected, atol=1e-10)
        status = "PASS" if ok else "FAIL"
        if _OTHERS_SHOW_N_VECTOR:
            print(f"    Node {j+1} at {_label_point(zj)} -> {status}  N={N}")
        else:
            print(f"    Node {j+1} at {_label_point(zj)} -> {status}")

def _partition_and_derivative_sum_checks(E: str, test_points: list[np.ndarray]) -> None:
    print("  [B] Partition of unity (sum N = 1) and derivative-sum (sum by column = 0)")
    for z in test_points:
        z = np.asarray(z, dtype=float).reshape(-1)
        try:
            N, DN = ShapeFunctions(E, z)
            sumN = float(np.sum(N))
            col_sums = np.sum(DN, axis=0)
            ok_unity = np.isclose(sumN, 1.0, atol=1e-10)
            ok_deriv = np.allclose(col_sums, 0.0, atol=1e-10)
            status = PASS if (ok_unity and ok_deriv) else FAIL
            print(f"    z = {_label_point(z)} -> sum(N)={sumN:.12f}, sum(DN)={col_sums}  {status}")
        except Exception as e:
            print(f"    z = {_label_point(z)} -> ERROR: {e}")
def test_ShapeFunctions(elements_to_test=None) -> None:
    if elements_to_test is None:
        elements_to_test = ['L2', 'L3', 'T3', 'T6', 'Q4', 'Q8', 'Q9', 'B8', 'B27', 'TET4', 'W6']

    banner = "=" * 68
    subbar = "-" * 68

    print("\n" + banner)
    print("== PROBLEM 2: SHAPE FUNCTIONS - CONCEPTUAL TESTS ==")
    print(banner)

    for E in elements_to_test:
        print("\n" + subbar)
        print(f"[ELEMENT] {E}")
        print(subbar)

        dim = DIMENSION_OF[E]

        # (1) Always show the element's “center”/centroid coordinates
        center = _element_centroid(E)
        print(f"  [center] {_label_point(center)}")

        # (2) Build a small pool of valid points to test
        pts = []
        pts.append(center.copy())  # include center first

        if dim == 1:
            pts.extend([np.array([-1.0]), np.array([0.0]), np.array([+1.0])])
        elif E in ('Q4','Q8','Q9','B8','B27'):
            pts.extend([ -np.ones(dim), +np.ones(dim) ])
        elif E in ('T3','T6'):
            pts.extend([np.array([0.30, 0.20]), np.array([0.20, 0.70])])
        elif E == 'TET4':
            pts.extend([np.array([0.20, 0.20, 0.20])])
        elif E == 'W6':
            pts.extend([np.array([0.20, 0.30, 0.80]), np.array([0.20, 0.30, -0.80])])

        # add two random valid points (seeded)
        pts.extend(_rand_points_in_element(E, k=2, seed=2025))

        # (3) Run conceptual checks
        _kronecker_check(E)
        print("  " + "-" * 64)
        _partition_and_derivative_sum_checks(E, pts)

        # (4) Compact sample at centroid
        try:
            N_c, DN_c = ShapeFunctions(E, center)
            print("  " + "-" * 64)
            print("  [C] Sample evaluation at center (values trimmed if large)")
            print("    N :", _preview_ndarray(N_c))
            print("    DN:", _preview_ndarray(DN_c))
        except Exception as e:
            print("  " + "-" * 64)
            print(f"  [C] Center evaluation -> skipped: {e}")
# =========================
# Entry point
# =========================
if __name__ == "__main__":
    test_ShapeFunctions(['L2','L3','T3','T6','Q4','Q8','Q9','B8','B27','TET4','W6'])

    banner = "=" * 68
    print("\n" + banner)
    print("== PROBLEM 2: SHAPE FUNCTIONS - ERROR HANDLING ==")
    print(banner)
    try:
        ShapeFunctions("L99", np.array([0.0]))
    except Exception as e:
        print(f"  Raised (as expected): {type(e).__name__}: {e}")
    try:
        ShapeFunctions("L2", np.array([0.0, 0.1]))
    except Exception as e:
        print(f"  Raised (as expected): {type(e).__name__}: {e}")
    try:
        ShapeFunctions(123, np.array([0.0]))
    except Exception as e:
        print(f"  Raised (as expected): {type(e).__name__}: {e}")
# help(ShapeFunctions)
def Vector(A: np.ndarray) -> np.ndarray:
    """
    Stack the columns of a matrix into a single COLUMN vector (the "vec" operator).

    Given A of shape (m, n), returns vecA of shape (m*n, 1), containing
    A[:, 0], then A[:, 1], ..., A[:, n-1] stacked top-to-bottom.

    Parameters
    ----------
    A : np.ndarray
        Real/complex numeric 2D array (matrix). Need not be square. Empty matrices allowed.

    Returns
    -------
    vecA : np.ndarray
        2D array of shape (m*n, 1) with columns of A stacked in order.

    Raises
    ------
    TypeError
        If A is not array-like numeric.
    ValueError
        If A is not 2-dimensional.

    Notes
    -----
    - Equivalent to `A.ravel(order='F').reshape(-1, 1)` (Fortran/column-major).
    - We return a 2D column vector per the assignment requirement.

    Examples
    --------
    >>> A = np.array([[1, 4],
    ...               [2, 5],
    ...               [3, 6]])
    >>> Vector(A)
    array([[1],
           [2],
           [3],
           [4],
           [5],
           [6]])
    """
    A = np.asarray(A)
    if not np.issubdtype(A.dtype, np.number):
        raise TypeError("Vector: A must be numeric (real/complex).")
    if A.ndim != 2:
        raise ValueError(f"Vector: A must be 2D (got {A.ndim}D).")

    return A.ravel(order="F").reshape(-1, 1)
PASS = "PASS"
FAIL = "FAIL"


def _preview_ndarray(A: np.ndarray, max_entries: int = 12) -> str:
    """Compact, readable preview for arrays (used in prints)."""
    A = np.asarray(A)
    total = A.size
    if total <= max_entries or A.ndim == 0:
        return f"{A}"
    if A.ndim == 1:
        shown = A[:max_entries]
        omitted = total - max_entries
        return f"{shown}\n... (omitting {omitted} entries; shape={A.shape})"
    elif A.ndim == 2:
        m, n = A.shape
        max_rows = max(1, min(m, max_entries // max(1, n)))
        shown = A[:max_rows, :]
        omitted_rows = m - max_rows
        suffix = "" if omitted_rows <= 0 else f"\n... (omitting {omitted_rows} more row(s); shape={A.shape})"
        return f"{shown}{suffix}"
    else:
        return f"{A.shape} (omit large tensor preview)"
def test_Vector() -> None:
    """
    Deterministic tests + clean printout for Vector(A).

    Notes
    -----
    - Tests with `expected_error` are expected to raise that error.
      If they do, the test is reported as PASS (expected error observed).
    """
    banner = "=" * 58
    subbar = "-" * 58

    print("\n" + banner)
    print("== Problem 3: Vector(A) — demo & deterministic tests")
    print(banner + "\n")

    # Demo
    A_demo = np.array([[1, 4],
                       [2, 5],
                       [3, 6]])
    print("[DEMO] A =")
    print(A_demo)
    v_demo = Vector(A_demo)
    print("[DEMO] Vector(A) (column) =")
    print(v_demo)
    print(f"[DEMO] shape: {v_demo.shape}\n")

    # Deterministic test table
    cases = [
        {"name": "2x2 ints",
         "A": np.array([[1, 2], [3, 4]])},

        {"name": "3x5 fixed",
         "A": np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                        [1.1, 1.2, 1.3, 1.4, 1.5],
                        [2.1, 2.2, 2.3, 2.4, 2.5]])},

        {"name": "empty 0x0",
         "A": np.empty((0, 0))},

        {"name": "row vector 1x7",
         "A": np.arange(1, 8, dtype=float).reshape(1, -1)},

        {"name": "col vector 7x1",
         "A": np.arange(1, 8, dtype=float).reshape(-1, 1)},

        {"name": "complex 2x2",
         "A": np.array([[1+5j, 2+6j], [3+7j, 4+8j]])},

        # Expected to raise TypeError (per our numeric-only spec)
        {"name": "logical/bool 2x3  (expected TypeError)",
         "A": np.array([[True, False, True],
                        [False, True, False]], dtype=bool),
         "expected_error": TypeError},

        {"name": "magic-like 4x4",
         "A": np.array([[16,  2,  3, 13],
                        [ 5, 11, 10,  8],
                        [ 9,  7,  6, 12],
                        [ 4, 14, 15,  1]])},
    ]

    passed = 0
    failed = 0

    print(subbar)
    print("[A] Checks vs. NumPy column-major ravel (as column vector)")
    print(subbar)

    for case in cases:
        name = case["name"]
        A = case["A"]
        expected_error = case.get("expected_error", None)

        try:
            v = Vector(A)
            if expected_error is not None:
                failed += 1
                print(f"[FAIL] {name:40s} -> expected {expected_error.__name__}, but no error was raised")
                continue

            ref = np.ravel(np.asarray(A), order="F").reshape(-1, 1)
            if np.array_equal(v, ref):
                passed += 1
                print(f"[PASS] {name:40s}  size(A)={A.shape}  -> vec shape={v.shape}")
            else:
                failed += 1
                print(f"[FAIL] {name:40s}  (Vector(A) != A.ravel('F').reshape(-1,1))")

        except Exception as e:
            if expected_error is not None and isinstance(e, expected_error):
                passed += 1
                print(f"[PASS] {name:40s} -> raised expected {expected_error.__name__}")
            else:
                failed += 1
                print(f"[FAIL] {name:40s} -> unexpected error: {type(e).__name__}: {e}")

    # Summary
    print("\n" + subbar)
    print(f"Summary: {passed} PASS / {failed} FAIL")
    if failed == 0:
        print("All tests passed.")
    else:
        print("Some tests failed.")
    print(subbar + "\n")

    # Extra: compact preview
    B = np.arange(1, 13).reshape(4, 3, order="F")
    print("[EXTRA] B =")
    print(B)
    print("[EXTRA] Vector(B) preview (column):")
    print(_preview_ndarray(Vector(B)))
    print(f"[EXTRA] shape: {Vector(B).shape}")
    print()


if __name__ == "__main__":
    test_Vector()
if __name__ == "__main__":
    test_Vector()
"""
============================================================
W18 SHAPE FUNCTIONS (SYMBOLIC) — STYLE-CONFORMING SCRIPT
============================================================

Overview
--------
This script constructs the 18-node wedge/prism (W18) shape functions
as a tensor product of:
  - T6 (six-node triangle) shape functions in (zet1, zet2), and
  - L3 (three-node line) shape functions in zet3.

It then verifies three conceptual properties symbolically:
  1) Partition of Unity  →  sum(N_i) = 1
  2) Zero Derivative Sum →  sum_i dN_i/d(zet_k) = 0  for k ∈ {1,2,3}
  3) Kronecker-Delta     →  each nodal basis peaks at its own node

Notes
-----
- ASCII-only variable names are used (zet1, zet2, zet3).
- Uses SymPy for exact symbolic algebra.
- Prints compact, grader-friendly summaries (no large dumps).

Example
-------
>>> # Run this file directly to see the checks printed:
>>> # Partition of Unity (should be 1): 1
>>> # Column sums of DN (should be [0, 0, 0]): [0, 0, 0]
>>> # Kronecker-Delta Property (...):
"""

