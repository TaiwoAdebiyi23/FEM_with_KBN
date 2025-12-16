"""
======================================================================
CIVE 7336 - Finite Element Methods
Machine Problem #4
Instructor: Professor K. B. Nakshatrala
Prepared by: Taiwo Adebiyi
Last Modified: 2025-11-12
======================================================================
"""

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

# Ensure the FEM framework modules within this MP4 directory are importable.
CURRENT_DIR = Path(__file__).resolve().parent
FEM_DIR = CURRENT_DIR.parent.parent / "FemFrameWork"
if str(FEM_DIR) not in sys.path:
    sys.path.append(str(FEM_DIR))

# Use my MP1/MP2 implementations housed inside the MP4 FemFrameWork folder.
from taiwo_shape_functions import ShapeFunctions, Create_ID_Matrix, Vector
from taiwo_gauss_points import GaussPoints


# =====================================================================
# (1) Calculate Local Matrices
# =====================================================================
def CalculateLocalMatrices(
    medium_set: dict,
    dofs_per_node: int,
    EleNodes: np.ndarray,
    EleType: str,
    load_type: dict,
    r: np.ndarray,
    w: np.ndarray,
    xCap: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the local stiffness matrix and load vector for quasi-static linear elasticity.

    This routine mirrors the lecturer-provided Gauss-loop MATLAB implementation but uses
    the existing MP framework utilities (shape functions, ``Vector`` helper, etc.).
    Parameters
    ----------
    medium_set : dict
        Material description (e.g., Lame parameters) consumed by :func:`Get_Elastic_Moduli`.
    dofs_per_node : int
        Degrees of freedom per node (typically equals spatial dimension).
    EleNodes : np.ndarray
        1-based element connectivity row.
    EleType : str
        Element mnemonic for :func:`ShapeFunctions`.
    load_type : dict
        Body-force descriptor passed to :func:`Get_Rhob`.
    r, w : np.ndarray
        Gauss points and weights for the element.
    xCap : np.ndarray
        Physical nodal coordinates for the element (``NodesPerEle x dim``).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(Klocal, rlocal)`` each of size ``(NodesPerEle*dofs_per_node, ..)``.
    """
    num_nodes = EleNodes.size
    dim = xCap.shape[1]
    ndofs_e = num_nodes * dofs_per_node

    Klocal = np.zeros((ndofs_e, ndofs_e))
    rlocal = np.zeros((ndofs_e, 1))

    reference_points = np.asarray(r, dtype=float)
    if reference_points.ndim == 1:
        reference_points = reference_points[:, None]
    weights = np.asarray(w, dtype=float).reshape(-1)

    dim_ref = reference_points.shape[1]
    if dim_ref != dim:
        raise ValueError(f"Reference dimension {dim_ref} inconsistent with physical dim {dim}.")
    if reference_points.shape[0] != weights.size:
        raise ValueError("Mismatch between Gauss points and weights.")

    xCap_T = xCap.T
    TMat = TMatrix(dim, dim)
    eye_dim = np.eye(dim)

    # ==============================================================
    # Loop over Gauss points
    # ==============================================================
    for gpt in range(weights.size):
        zeta = reference_points[gpt, :]
        N, DN = ShapeFunctions(EleType, zeta)

        # Geometric data
        J = xCap_T @ DN
        detJ = np.linalg.det(J)
        invJ = np.linalg.inv(J)
        B = DN @ invJ
        x = xCap_T @ N # -------------------------check this out ------------------------>

        # Material properties
        lambda_, mu_ = Get_Elastic_Moduli(medium_set, x)

        # === Klocal contributions due to linear elasticity (using derived three-terms) ===
        Klocal += weights[gpt] * lambda_ * (Vector(B.T) @ Vector(B.T).T) * detJ
        Klocal += weights[gpt] * mu_ * (np.kron(B, eye_dim) @ np.kron(B.T, eye_dim)) * detJ
        Klocal += weights[gpt] * mu_ * (np.kron(B, eye_dim) @ TMat @ np.kron(B.T, eye_dim)) * detJ

        # === rlocal contribution ===
        rlocal += weights[gpt] * (
            np.kron(N[:, None], eye_dim) @ Get_Rhob(load_type, x)
        ) * detJ

    return Klocal, rlocal


# =====================================================================
# Elastic material
# =====================================================================
def Get_Elastic_Moduli(medium_set: dict, x: np.ndarray | None = None) -> tuple[float, float]:
    """
    Return the Lamé parameters (lambda, mu) for linear elasticity.

    Currently supports the ``"Lame_params"`` case where the dictionary provides
    the two parameters explicitly. The coordinate ``x`` is accepted for future
    spatially varying media but is unused.
    """
    medium_type = medium_set.get("type")
    if medium_type == "Lame_params":
        lam = float(medium_set["lambda"])
        mu = float(medium_set["mu"])
        return lam, mu
    raise ValueError(f"Unknown medium_set type '{medium_type}'. Expected 'Lame_params'.")

# =====================================================================
# Body force
# =====================================================================
def Get_Rhob(load_type: dict, x: np.ndarray | None = None) -> np.ndarray:
    """
    Return the body-force vector f(x) for the requested case.

    For this project:
    - '1D_gravity' : 1D rod, x measured **downwards**, so f = [ +rhob ].
    - '2D_gravity' : 2D body, y upwards, so f = [0, -rhob]^T.
    - '3D_gravity' : 3D body, z upwards, so f = [0, 0, -rhob]^T.
    """
    case = load_type.get("case")
    if "rhob" not in load_type:
        raise ValueError("load_type must supply 'rhob' (effective weight density).")
    rhob = float(load_type["rhob"])

    if case == "1D_gravity":
        f = np.zeros((1, 1))
        f[0, 0] = rhob   # positive downward body force (x is downward)
        return f

    if case == "2D_gravity":
        f = np.zeros((2, 1))
        f[1, 0] = -rhob
        return f

    if case == "3D_gravity":
        f = np.zeros((3, 1))
        f[2, 0] = -rhob
        return f

    raise ValueError(
        f"Unknown load_type case '{case}'. "
        "Supported cases: '1D_gravity', '2D_gravity', '3D_gravity'."
    )

# =====================================================================
# T-matrix utility (direct MATLAB transcription)
# =====================================================================
def TMatrix(m: int, n: int) -> np.ndarray:
    """
    Build the transposer matrix used in class linear elasticity derivations.

    Matches the MATLAB helper distributed for MP3/MP4.
    """
    Im = np.eye(m)
    In = np.eye(n)
    T = np.zeros((m * n, m * n))

    for i in range(1, n + 1):
        T[(i - 1) * m + np.arange(0, m), :] = np.kron(Im, In[i - 1, :])

    return T


# =====================================================================
# (2) Assemble
# =====================================================================
def Assemble(
    dofs_per_node: int,
    EleNodes: np.ndarray,
    GlobalId: np.ndarray,
    Klocal: np.ndarray,
    rlocal: np.ndarray,
    K_FF: np.ndarray,
    K_FP: np.ndarray,
    R_F: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scatter a single element contribution into the global partitions.

    Parameters
    ----------
    dofs_per_node : int
        Number of degrees of freedom per node.
    EleNodes : np.ndarray
        1-based node numbers for the current element (shape ``(NodesPerEle,)``).
    GlobalId : np.ndarray
        Global ID map (``>0`` for free DOFs, ``<0`` for prescribed DOFs).
    Klocal : np.ndarray
        Local stiffness matrix of shape ``(NodesPerEle*dofs_per_node, NodesPerEle*dofs_per_node)``.
    rlocal : np.ndarray
        Local load vector of shape ``(NodesPerEle*dofs_per_node, 1)``.
    K_FF : np.ndarray
        Global free–free block to be updated in place.
    K_FP : np.ndarray
        Global free–prescribed block to be updated in place.
    R_F : np.ndarray
        Global right-hand-side for free DOFs, updated in place.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Updated ``(K_FF, K_FP, R_F)``.
    """
    nodes_on_element = EleNodes.size
    dofs_per_element = nodes_on_element * dofs_per_node

    # --------------------------------------------------------------
    # Step 1: Build flattened DOF map (local ordering -> global IDs)
    # --------------------------------------------------------------
    zero_based_nodes = EleNodes.astype(int) - 1
    dof_lookup = GlobalId[zero_based_nodes, :].reshape(dofs_per_element)

    # --------------------------------------------------------------
    # Step 2: Prepare masks for FF/FP partitions (vectorized scatter)
    # --------------------------------------------------------------
    row_targets = np.repeat(dof_lookup[:, None], dofs_per_element, axis=1)
    col_targets = row_targets.T

    mask_ff = (row_targets > 0) & (col_targets > 0)
    mask_fp = (row_targets > 0) & (col_targets < 0)

    ff_rows = row_targets[mask_ff].astype(int) - 1
    ff_cols = col_targets[mask_ff].astype(int) - 1
    np.add.at(K_FF, (ff_rows, ff_cols), Klocal[mask_ff])

    fp_rows = row_targets[mask_fp].astype(int) - 1
    fp_cols = np.abs(col_targets[mask_fp]).astype(int) - 1
    np.add.at(K_FP, (fp_rows, fp_cols), Klocal[mask_fp])

    # --------------------------------------------------------------
    # Step 3: Assemble the load vector for free DOFs
    # --------------------------------------------------------------
    mask_rhs = dof_lookup > 0
    rhs_indices = dof_lookup[mask_rhs].astype(int) - 1
    np.add.at(R_F[:, 0], rhs_indices, rlocal[mask_rhs, 0])

    return K_FF, K_FP, R_F


# =====================================================================
# (3) Calculate Global Matrices
# =====================================================================
def CalculateGlobalMatrices(
    connectivity: np.ndarray,
    coord: np.ndarray,
    medium_set: dict,
    dim: int,
    dofs_per_node: int,
    EleType: str,
    GlobalId: np.ndarray,
    load_type: dict,
    NCons: int,
    Nele: int,
    NEqns: int,
    NGPTS: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate partitioned global matrices (K_FF, K_FP) and R_F.

    Parameters
    ----------
    connectivity : np.ndarray
        (Nele, nnode) connectivity with 1-based node numbers.
    coord : np.ndarray
        (NumNodes, dim) nodal coordinates.
    medium_set : dict
        Elastic material description passed to :func:`Get_Elastic_Moduli`.
    dim : int
        Spatial dimension (1,2,3).
    dofs_per_node : int
        DOFs per node.
    EleType : str
        Element type string for ShapeFunctions.
    GlobalId : np.ndarray
        Mapping matrix (>0 free, <0 prescribed).
    load_type : dict
        Load description dictionary, e.g.
          * ``{'type': 'homogeneous', 'value': float}``
          * ``{'type': 'inhomogeneous', 'value': array_like}``
          * ``{'type': 'spatial', 'fun': callable}``
    NCons, Nele, NEqns, NGPTS : int
        Sizes per class instruction (NEqns/NCons may be validated from GlobalId).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (K_FF, K_FP, R_F).
    """
    NEqns_auto = int(np.max(GlobalId[GlobalId > 0])) if np.any(GlobalId > 0) else 0
    NCons_auto = int(np.max(np.abs(GlobalId[GlobalId < 0]))) if np.any(GlobalId < 0) else 0
    # --------------------------------------------------------------
    # Step 0: Determine actual counts of equations and constraints
    # --------------------------------------------------------------
    if NEqns == 0:
        NEqns = NEqns_auto
    if NCons == 0:
        NCons = NCons_auto

    K_FF = np.zeros((NEqns, NEqns))
    K_FP = np.zeros((NEqns, NCons))
    R_F = np.zeros((NEqns, 1))

    # --------------------------------------------------------------
    # Step 1: Obtain Gauss rule shared by all elements
    # --------------------------------------------------------------
    gauss_points, gauss_weights = GaussPoints(dim, EleType, NGPTS)

    # --------------------------------------------------------------
    # Step 2: Loop over elements and assemble
    # --------------------------------------------------------------
    for ele in range(Nele):
        element_nodes = connectivity[ele, :]
        element_coords = coord[element_nodes.astype(int) - 1, :]

        local_stiffness, local_load_vector = CalculateLocalMatrices(
            medium_set,
            dofs_per_node,
            element_nodes,
            EleType,
            load_type,
            gauss_points,
            gauss_weights,
            element_coords,
        )

        K_FF, K_FP, R_F = Assemble(
            dofs_per_node, element_nodes, GlobalId, local_stiffness, local_load_vector, K_FF, K_FP, R_F
        )

    return K_FF, K_FP, R_F


# =====================================================================
# (4) Create Constraints Vector
# =====================================================================
def Create_ConstraintsVector(Constraints: np.ndarray, GlobalID: np.ndarray) -> np.ndarray:
    """
    Build prescribed values vector U_P ordered by constraint ID (vectorized).

    Parameters
    ----------
    Constraints : np.ndarray
        (NCons, 3) rows: [node, dof, value] (1-based indices).
    GlobalID : np.ndarray
        (NumNodes, dofs_per_node) mapping (>0 free, <0 prescribed).

    Returns
    -------
    np.ndarray
        (NConsTotal, 1) vector, where index k corresponds to constraint ID k+1.
    """
    NCons = Constraints.shape[0]
    if NCons == 0:
        return np.zeros((0, 1))

    # --------------------------------------------------------------
    # Step 1: Translate node/DOF pairs to zero-based indices
    # --------------------------------------------------------------
    nodes = Constraints[:, 0].astype(int) - 1
    dofs = Constraints[:, 1].astype(int) - 1
    vals = Constraints[:, 2].astype(float)
    ids = GlobalID[nodes, dofs]
    if np.any(ids > 0):
        raise ValueError("One or more constraints map to free DOFs (got positive IDs, please check GlobalID codes).")

    # --------------------------------------------------------------
    # Step 2: Populate U_P according to the constraint numbering (-1 -> 0, etc.)
    # --------------------------------------------------------------
    NConsTotal = int(np.max(np.abs(GlobalID[GlobalID < 0]))) if np.any(GlobalID < 0) else 0
    U_P = np.zeros((NConsTotal, 1))
    idx = np.abs(ids).astype(int) - 1
    U_P[idx, 0] = vals
    return U_P


# =====================================================================
# (5) Get DMat
# =====================================================================
def Get_DMat(diffusivity_function: dict, x: np.ndarray) -> np.ndarray:
    """
    Evaluate the diffusivity tensor D(x) at a spatial location.

    Parameters
    ----------
    diffusivity_function : dict
        Material description. Supported forms:
          - {'type': 'constant', 'value': float}
            Isotropic, constant diffusivity D*I.
          - {'type': 'rotation', 'angle': float, 'd1': float, 'd2': float}
            2D principal values rotated by 'angle' radians.
    x : np.ndarray
        Spatial location (used for dimension inference).

    Returns
    -------
    np.ndarray
        Diffusivity tensor of shape (dim, dim).

    Raises
    ------
    ValueError
        If the 'type' field is unsupported.

    Examples
    --------
    >>> Get_DMat({'type':'constant','value':2.0}, np.array([0.0]))
    array([[2.]])
    >>> Get_DMat({'type':'rotation','angle':0.0,'d1':2.0,'d2':1.0}, np.zeros(2))
    array([[2., 0.],
           [0., 1.]])
    """
    ftype = str(diffusivity_function.get('type', 'constant')).lower()

    if ftype == 'constant':
        val = float(diffusivity_function['value'])
        dim = x.size if x.ndim == 1 else x.shape[-1]
        return val * np.eye(dim)

    if ftype == 'rotation':
        theta = float(diffusivity_function['angle'])
        d1 = float(diffusivity_function['d1'])
        d2 = float(diffusivity_function['d2'])
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return R @ np.diag([d1, d2]) @ R.T

    raise ValueError("Unsupported diffusivity_function 'type'. Use 'constant' or 'rotation'.")


# =====================================================================
# (6) Get Volumetric Source
# =====================================================================
def Get_VolumetricSource(load_type: dict, x: np.ndarray) -> np.ndarray:
    """
    Evaluate the volumetric source f(x) using class naming conventions.

    Parameters
    ----------
    load_type : dict
        Supported forms:
          - {'type': 'homogeneous', 'value': float}
            Constant scalar source term.
          - {'type': 'inhomogeneous', 'value': array_like (dim,)}
            Constant vector source term.
          - {'type': 'spatial', 'fun': callable}
            Function of position; may return scalar or (dim,).
    x : np.ndarray
        Spatial location.

    Returns
    -------
    np.ndarray
        (1,) for scalar sources or (dim,) for vector sources.

    Raises
    ------
    ValueError
        If dimensions are inconsistent or type is unsupported.

    Examples
    --------
    >>> Get_VolumetricSource({'type':'homogeneous','value':5.0}, np.array([0.0]))
    array([5.])
    """
    if not isinstance(x, np.ndarray):
        x = np.asarray(x, dtype=float)
    dim = x.size if x.ndim == 1 else x.shape[-1]

    ftype = str(load_type.get('type', 'homogeneous')).lower()

    if ftype == 'homogeneous':
        return np.array([float(load_type['value'])])

    if ftype == 'inhomogeneous':
        v = np.asarray(load_type['value'], dtype=float)
        if v.size != dim:
            raise ValueError(f"Inhomogeneous vector source must match spatial dimension {dim}.")
        return v

    if ftype == 'spatial':
        fun = load_type['fun']
        out = np.asarray(fun(x), dtype=float)
        if out.ndim == 0:
            return np.array([float(out)])
        if out.size == dim:
            return out
        raise ValueError("Spatial output must be scalar or match spatial dimension.")

    raise ValueError("Unsupported load_type 'type'. Use 'homogeneous', 'inhomogeneous', or 'spatial'.")

# =====================================================================
# (7) Post-Processing
# =====================================================================
def PostProcessing(GlobalID: np.ndarray, U_F: np.ndarray, U_P: np.ndarray) -> np.ndarray:
    """
    Reconstruct the full nodal field ``U`` from the free and prescribed components.

    Parameters
    ----------
    GlobalID : np.ndarray
        Global ID table produced by ``Create_ID_Matrix`` (``>0`` free, ``<0`` prescribed).
    U_F : np.ndarray
        Solution vector at free DOFs (shape ``(NEqns, 1)``).
    U_P : np.ndarray
        Prescribed values vector (shape ``(NCons, 1)``).

    Returns
    -------
    np.ndarray
        Matrix of nodal values with shape ``(NumNodes, dofs_per_node)``.
    """
    NumNodes, dofs_per_node = GlobalID.shape
    U = np.zeros((NumNodes, dofs_per_node))

    # --------------------------------------------------------------
    # Step 1: Scatter free DOFs
    # --------------------------------------------------------------
    mask_F = GlobalID > 0
    idx_F = GlobalID[mask_F].astype(int) - 1
    U[mask_F] = U_F[idx_F, 0]

    # --------------------------------------------------------------
    # Step 2: Scatter prescribed DOFs
    # --------------------------------------------------------------
    mask_P = GlobalID < 0
    idx_P = -GlobalID[mask_P].astype(int) - 1
    U[mask_P] = U_P[idx_P, 0]

    return U


# =====================================================================
# (8) Driver Steady Diffusion
# =====================================================================
def Driver_LE(
    Connectivity: np.ndarray,
    Constraints: np.ndarray,
    Coord: np.ndarray,
    medium_set: dict,
    dim: int,
    dofs_per_node: int,
    EleType: str,
    load_type: dict,
    NCons: int,
    Nele: int,
    NGPTS: int,
    point_load: np.ndarray | None = None,
) -> np.ndarray:
    """
    Driver that wires the FEM framework steps together for quasistatic linear elasticity simulation.

    Parameters
    ----------
    Connectivity : np.ndarray
        Element connectivity (shape ``(Nele, NodesPerEle)``) using 1-based indices.
    Constraints : np.ndarray
        ``[node, dof, value]`` Dirichlet specifications.
    Coord : np.ndarray
        Nodal coordinates (``NumNodes x dim``).
    medium_set : dict
        Material description passed to :func:`Get_Elastic_Moduli`.
    dim : int
        Spatial dimension of the problem.
    dofs_per_node : int
        Degrees of freedom per node.
    EleType : str
        Element mnemonic understood by ``ShapeFunctions``/``GaussPoints``.
    load_type : dict
        Volumetric source description passed to ``Get_VolumetricSource``.
    NCons : int
        Number of prescribed DOFs (per assignment signature). Validated internally.
    Nele : int
        Number of elements (per assignment signature). Validated internally.
    NGPTS : int
        Number of Gauss points per direction.
    point_load : np.ndarray | None, optional
        Optional nodal point loads of shape ``(nloads, 3)`` with rows
        ``[node, dof, value]`` using 1-based node numbering and 1-based local
        dof numbering (e.g., 1=x, 2=y in 2D). If ``None``, no point loads are applied.

    Returns
    -------
    np.ndarray
        Nodal solution matrix shaped ``(NumNodes, dofs_per_node)``.
    """
    header = "=" * 56
    print(header)
    print("       Quasi-static Linear Elasticity Simulation Status Report               ")
    print(header)

    # Step 1: Create the Global ID table.
    print("       Step 1: Creating Global ID Matrix...             ")
    NumNodes = Coord.shape[0]
    GlobalID, NEqns = Create_ID_Matrix(Constraints, dofs_per_node, NumNodes)

    # Step 2: Constraints vector.
    print("       Step 2: Creating Constraints Vector...           ")
    U_P = Create_ConstraintsVector(Constraints, GlobalID)

    # Validate NCons/Nele if provided by the caller.
    computed_NCons = int(np.max(np.abs(GlobalID[GlobalID < 0]))) if np.any(GlobalID < 0) else 0
    computed_Nele = Connectivity.shape[0]
    if NCons != computed_NCons:
        NCons = computed_NCons
    if Nele != computed_Nele:
        Nele = computed_Nele

    # Step 3: Assemble global matrices.
    print("       Step 3: Calculating Global Matrices...           ")
    K_FF, K_FP, R_F = CalculateGlobalMatrices(
        Connectivity,
        Coord,
        medium_set,
        dim,
        dofs_per_node,
        EleType,
        GlobalID,
        load_type,
        NCons,
        Nele,
        NEqns,
        NGPTS,
    )

    # Step 3.5: Apply nodal point loads (if provided).
    if point_load is not None:
        print("       Step 3.5: Applying Nodal Point Loads...          ")

        point_load = np.asarray(point_load, dtype=float)
        if point_load.ndim != 2 or point_load.shape[1] != 3:
            raise ValueError(
                "point_load must be a 2D array with shape (nloads, 3): [node, dof, value]."
            )

        for i in range(point_load.shape[0]):
            node = int(point_load[i, 0])       # 1-based node number
            dof_local = int(point_load[i, 1])  # 1-based dof (1=x, 2=y in 2D)
            val = float(point_load[i, 2])      # load magnitude

            # Basic validation (helps catch silent mistakes early)
            if node < 1 or node > NumNodes:
                raise ValueError(f"point_load row {i}: node={node} out of range [1, {NumNodes}].")
            if dof_local < 1 or dof_local > dofs_per_node:
                raise ValueError(
                    f"point_load row {i}: dof={dof_local} out of range [1, {dofs_per_node}]."
                )

            # Convert (node, dof) -> equation number using GlobalID.
            eqn = GlobalID[node - 1, dof_local - 1]

            if eqn > 0:
                # Free DOF: add into reduced RHS vector R_F (shape: (NEqns, 1))
                R_F[eqn - 1, 0] += val
            else:
                # Prescribed DOF: keep going but warn (load is "reacted" at the support)
                print(
                    f"           Caution: point load applied on prescribed DOF "
                    f"(node={node}, dof={dof_local})."
                )

        print("       Point loads applied successfully.                ")

    # Step 4: Solve the reduced system.
    print("       Step 4: Solving Linear Equations...              ")
    rhs = R_F - K_FP @ U_P
    U_F = np.linalg.solve(K_FF, rhs)

    # Step 5: Post-process to reconstruct nodal values.
    print("       Step 5: Post-Processing...                       ")
    U = PostProcessing(GlobalID, U_F, U_P)

    print(header)
    print("               Simulation Completed                     ")
    print(header)

    return U


# =====================================================================
# End of module; tests and 1D driver are in separate files.
# =====================================================================
