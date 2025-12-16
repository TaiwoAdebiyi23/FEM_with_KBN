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
from taiwo_shape_functions import ShapeFunctions, Create_ID_Matrix
from taiwo_gauss_points import GaussPoints


# =====================================================================
# (1) Calculate Local Matrices
# =====================================================================
def CalculateLocalMatrices(
    diffusivity_function: dict,
    dofs_per_node: int,
    EleNodes: np.ndarray,
    EleType: str,
    load_type: dict,
    r: np.ndarray,
    w: np.ndarray,
    xCap: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the element stiffness matrix ``Klocal`` and load vector ``rlocal``.

    Parameters
    ----------
    diffusivity_function : dict
        Admissible definitions:
          * ``{'type': 'constant', 'value': float}`` — isotropic constant diffusivity
            (scalar multiplied by the identity tensor).
          * ``{'type': 'rotation', 'angle': float, 'd1': float, 'd2': float}`` — orthotropic
            material in the local x/y directions, rotated by ``angle`` radians.
    dofs_per_node : int
        Number of degrees of freedom attached to each node.
    EleNodes : np.ndarray
        1-based node indices for this element (shape ``(NodesPerEle,)``).
    EleType : str
        Element mnemonic understood by ``ShapeFunctions`` (for example ``'L2'``, ``'Q4'``, ``'T3'``).
    load_type : dict
        Admissible load descriptions:
          * ``{'type': 'homogeneous', 'value': float}`` — scalar volumetric source.
          * ``{'type': 'inhomogeneous', 'value': array_like}`` — constant vector source of length ``dim``.
          * ``{'type': 'spatial', 'fun': callable}`` — callable ``fun(x)`` returning either a scalar
            or a vector of length ``dim`` evaluated at the physical coordinate ``x``.
    r : np.ndarray
        Reference Gauss points; either ``(NGPTS,)`` or ``(NGPTS, dim_ref)``.
    w : np.ndarray
        Gauss weights; array of length ``NGPTS``.
    xCap : np.ndarray
        Physical coordinates of the element nodes (shape ``(NodesPerEle, dim)``).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``Klocal`` of shape ``(NodesPerEle*dofs_per_node, NodesPerEle*dofs_per_node)`` and ``rlocal`` of
        shape ``(NodesPerEle*dofs_per_node, 1)``.

    Raises
    ------
    ValueError
        If the reference dimension implied by the Gauss points differs from the physical dimension,
        or if the volumetric source returns an incompatible number of components.
    """
    num_element_nodes = EleNodes.size
    physical_dim = xCap.shape[1]

    reference_points = np.asarray(r)
    if reference_points.ndim == 1:
        reference_points = reference_points[:, None]
    weights = np.asarray(w, dtype=float).reshape(-1)

    num_gauss, reference_dim = reference_points.shape
    if physical_dim != reference_dim:
        raise ValueError(f"Dim mismatch: xCap dim={physical_dim}, reference dim={reference_dim}")

    dofs_per_element = num_element_nodes * dofs_per_node
    Klocal = np.zeros((dofs_per_element, dofs_per_element))
    rlocal = np.zeros((dofs_per_element, 1))

    # --------------------------------------------------------------
    # Step 1: Shape functions and reference gradients at each Gauss point
    # --------------------------------------------------------------
    shape_values_per_gp: list[np.ndarray] = []
    ref_gradients_per_gp: list[np.ndarray] = []
    for ref_point in reference_points:
        shape_vals, ref_grads = ShapeFunctions(EleType, ref_point)
        shape_values_per_gp.append(shape_vals)
        ref_gradients_per_gp.append(ref_grads)

    shape_values = np.vstack(shape_values_per_gp)                     # (NGPTS, NodesPerEle)
    ref_gradients = np.stack(ref_gradients_per_gp, axis=0)            # (NGPTS, NodesPerEle, dim)

    # --------------------------------------------------------------
    # Step 2: Map gradients to physical space (Jacobian, B matrix) and points x(r)
    # --------------------------------------------------------------
    jacobians = np.einsum('ij,gjk->gik', xCap.T, ref_gradients)       # (NGPTS, dim, dim)
    jacobian_dets = np.linalg.det(jacobians)
    inv_jacobians = np.linalg.inv(jacobians)
    physical_gradients = np.einsum('gni,gij->gnj', ref_gradients, inv_jacobians)
    mapped_coords = np.einsum('gn,nd->gd', shape_values, xCap)

    # --------------------------------------------------------------
    # Step 3: Evaluate material tensor D(x) at each Gauss point
    # --------------------------------------------------------------
    diffusion_tensors = np.stack(
        [Get_DMat(diffusivity_function, mapped_coords[g]) for g in range(num_gauss)]
    )  # (NGPTS, dim, dim)

    # --------------------------------------------------------------
    # Step 4: Accumulate stiffness using Einstein summation
    # --------------------------------------------------------------
    Klocal = np.einsum('gnd,gde,gme,g->nm', physical_gradients, diffusion_tensors,
                       physical_gradients, jacobian_dets * weights)

    # --------------------------------------------------------------
    # Step 5: Assemble the consistent load vector (scalar or vector sources)
    # --------------------------------------------------------------
    source_samples = [
        np.asarray(Get_VolumetricSource(load_type, mapped_coords[g]), dtype=float).reshape(-1)
        for g in range(num_gauss)
    ]
    first_sample = source_samples[0]

    if first_sample.size == 1:
        if dofs_per_node == 1:
            scalar_values = np.array([sample[0] for sample in source_samples])
            contributions = np.einsum('gn,g,g->n', shape_values, scalar_values, jacobian_dets * weights)
            rlocal = contributions.reshape(-1, 1)
        else:
            assembled_load = np.zeros((dofs_per_element,))
            identity_block = np.eye(dofs_per_node)
            ones_vector = np.ones(dofs_per_node)
            for g in range(num_gauss):
                shape_block = np.kron(shape_values[g], identity_block)
                assembled_load += (jacobian_dets[g] * weights[g]) * (shape_block @ (ones_vector * source_samples[g][0]))
            rlocal = assembled_load.reshape(-1, 1)
    else:
        assembled_load = np.zeros((dofs_per_element,))
        identity_block = np.eye(dofs_per_node)
        for g in range(num_gauss):
            shape_block = np.kron(shape_values[g], identity_block)
            sample = source_samples[g]
            if sample.size == dofs_per_node:
                assembled_load += (jacobian_dets[g] * weights[g]) * (shape_block @ sample)
            elif sample.size == physical_dim and dofs_per_node == physical_dim:
                assembled_load += (jacobian_dets[g] * weights[g]) * (shape_block @ sample)
            else:
                raise ValueError(
                    f"Volumetric source size {sample.size} incompatible with dofs_per_node="
                    f"{dofs_per_node} or physical dimension {physical_dim}."
                )
        rlocal = assembled_load.reshape(-1, 1)

    return Klocal, rlocal


def CalculateLocalMatrices_v2(
    diffusivity_function: dict,
    dofs_per_node: int,
    EleNodes: np.ndarray,
    EleType: str,
    load_type: dict,
    r: np.ndarray,
    w: np.ndarray,
    xCap: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Loop-based element routine that mirrors the lecturer's MATLAB reference.

    This version follows the traditional Gauss-point loop to accumulate the local stiffness
    matrix ``Klocal`` and load vector ``rlocal`` using only basic linear algebra operations.
    The signature matches ``CalculateLocalMatrices`` so it can be dropped into existing
    MP4 drivers (e.g., the notebook ``taiwo_mp4_2.ipynb``) without additional wiring.

    Parameters
    ----------
    diffusivity_function : dict
        Material description passed to :func:`Get_DMat`.
    dofs_per_node : int
        Degrees of freedom carried by each node.
    EleNodes : np.ndarray
        1-based node indices for the element (shape ``(NodesPerEle,)``).
    EleType : str
        Element mnemonic understood by :func:`ShapeFunctions`.
    load_type : dict
        Volumetric load description consumed by :func:`Get_VolumetricSource`.
    r, w : np.ndarray
        Gauss points and weights shared by all elements.
    xCap : np.ndarray
        Physical coordinates of the element nodes (``NodesPerEle x dim``).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``Klocal`` and ``rlocal`` sized ``(NodesPerEle*dofs_per_node, NodesPerEle*dofs_per_node)``
        and ``(NodesPerEle*dofs_per_node, 1)``, respectively.
    """
    num_element_nodes = EleNodes.size
    dim = xCap.shape[1]
    dofs_per_element = num_element_nodes * dofs_per_node

    reference_points = np.asarray(r, dtype=float)
    if reference_points.ndim == 1:
        reference_points = reference_points[:, None]
    weights = np.asarray(w, dtype=float).reshape(-1)

    if reference_points.shape[0] != weights.size:
        raise ValueError("Mismatch between Gauss points and weights.")
    if reference_points.shape[1] != dim:
        raise ValueError(
            f"Dim mismatch: element coordinates dim={dim}, reference dim={reference_points.shape[1]}"
        )

    Klocal = np.zeros((dofs_per_element, dofs_per_element))
    rlocal = np.zeros((dofs_per_element, 1))
    identity_block = np.eye(dofs_per_node)

    # --------------------------------------------------------------
    # Loop over Gauss points 
    # --------------------------------------------------------------
    for gpt in range(weights.size):
        zeta = reference_points[gpt]
        N, DN = ShapeFunctions(EleType, zeta)

        # Jacobian, determinant, and inverse
        J = xCap.T @ DN
        detJ = np.linalg.det(J)
        invJ = np.linalg.inv(J)

        # B matrix (physical gradients) and mapped coordinate x(zeta)
        B = DN @ invJ
        x_loc = (xCap.T @ N).reshape(-1)

        # Material tensor at the Gauss point
        Dmat = Get_DMat(diffusivity_function, x_loc)

        # Stiffness accumulation
        stiffness_increment = B @ Dmat @ B.T
        if dofs_per_node == 1:
            Klocal += weights[gpt] * stiffness_increment * detJ
        else:
            block = np.kron(stiffness_increment, identity_block)
            Klocal += weights[gpt] * block * detJ

        # Load vector accumulation
        source_vec = np.asarray(Get_VolumetricSource(load_type, x_loc), dtype=float).reshape(-1)
        if source_vec.size == 0:
            continue

        if source_vec.size == 1:
            scalar_val = source_vec[0]
            if dofs_per_node == 1:
                rlocal[:, 0] += weights[gpt] * detJ * (N * scalar_val)
            else:
                shape_block = np.kron(N, identity_block)
                ones_vec = np.ones(dofs_per_node) * scalar_val
                rlocal[:, 0] += weights[gpt] * detJ * (shape_block @ ones_vec)
        elif source_vec.size == dofs_per_node:
            shape_block = np.kron(N, identity_block)
            rlocal[:, 0] += weights[gpt] * detJ * (shape_block @ source_vec)
        else:
            raise ValueError(
                f"Volumetric source size {source_vec.size} incompatible with dofs_per_node={dofs_per_node}."
            )

    return Klocal, rlocal


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
    diffusivity_function: dict,
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
    diffusivity_function : dict
        Material description dictionary, e.g.
          * ``{'type': 'constant', 'value': float}``
          * ``{'type': 'rotation', 'angle': float, 'd1': float, 'd2': float}``
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

        local_stiffness, local_load_vector = CalculateLocalMatrices_v2(
            diffusivity_function,
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
def Driver_Steady_Diffusion(
    Connectivity: np.ndarray,
    Constraints: np.ndarray,
    Coord: np.ndarray,
    diffusivity_function: dict,
    dim: int,
    dofs_per_node: int,
    EleType: str,
    load_type: dict,
    NCons: int,
    Nele: int,
    NGPTS: int,
) -> np.ndarray:
    """
    MP4 driver that wires the FEM framework steps together for steady diffusion.

    Parameters
    ----------
    Connectivity : np.ndarray
        Element connectivity (shape ``(Nele, NodesPerEle)``) using 1-based indices.
    Constraints : np.ndarray
        ``[node, dof, value]`` Dirichlet specifications.
    Coord : np.ndarray
        Nodal coordinates (``NumNodes x dim``).
    diffusivity_function : dict
        Material description passed to ``Get_DMat``.
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

    Returns
    -------
    np.ndarray
        Nodal solution matrix shaped ``(NumNodes, dofs_per_node)``.
    """
    header = "=" * 56
    print(header)
    print("       Diffusion Simulation Status Report               ")
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
        diffusivity_function,
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
