# ======================================================================
#  Stress_Recovery (1D bar, L2 elements)
#  Author : Taiwo (Python version based on KBN's MATLAB code)
# ----------------------------------------------------------------------
#  Calculates stress field using several methods:
#       - Gauss points        ("GPT")
#       - At element nodes    ("EleNodes")
#       - Averaged at nodes   ("AvgNodes")
# ======================================================================

import numpy as np
from taiwo_gauss_points import GaussPoints
from taiwo_shape_functions import ShapeFunctions
from Physics_model.Kernel_LE.taiwo_linear_elasticity import Get_Elastic_Moduli


def _normalize_U_1d(U: np.ndarray, num_nodes: int) -> np.ndarray:
    """
    Ensure U has shape (num_nodes, 1) for the 1D bar case.
    """
    U_arr = np.asarray(U)

    if U_arr.ndim == 1:
        if U_arr.size != num_nodes:
            raise ValueError(
                "For 1D U array, size must equal number of nodes in Coord."
            )
        Umat = U_arr.reshape(num_nodes, 1)
    elif U_arr.ndim == 2:
        if U_arr.shape[0] != num_nodes:
            raise ValueError(
                "For 2D U, U.shape[0] must equal number of nodes in Coord."
            )
        Umat = U_arr
    else:
        raise ValueError("U must be a 1D or 2D array.")

    if Umat.shape[1] != 1:
        raise ValueError(
            "Stress_Recovery is currently implemented for 1D bar with "
            "dofs_per_node = 1."
        )

    return Umat


def Stress_Recovery(
    Connectivity: np.ndarray,
    Coord: np.ndarray,
    EleType: str,
    medium_set: dict,
    NGPTS: int,
    recovery_type: str,
    U: np.ndarray,
) -> np.ndarray:
    """
    Compute the stress field using different recovery strategies.

    *********************************
    Parameters
    *********************************
    Connectivity : (Nele x NodesPerEle) int array
        Element connectivity (1-based node numbering).
    Coord : (NumNodes x dim) float array
        Nodal coordinates. For the 1D bar, dim must be 1.
    EleType : str
        Element type identifier (e.g., "L2").
    medium_set : dict
        Material properties dictionary (Lamé parameters, etc.).
    NGPTS : int
        Number of Gauss points per element (used in GPT recovery).
    recovery_type : str
        One of {"GPT", "EleNodes", "AvgNodes"} (case-insensitive).
    U : array_like
        Global nodal displacement from Driver_LE.
        Accepted shapes:
            - (NumNodes,) or (NumNodes, 1) for 1D bar.

    *********************************
    Returns
    *********************************
    sigma : ndarray
        Stress field in [x, sigma_xx] format. Shape depends on recovery_type:
            - "GPT"      : (Nele * NGPTS, 2)
            - "EleNodes" : (Nele * NodesPerEle, 2)
            - "AvgNodes" : (NumNodes, 2)
    """

    rt = str(recovery_type).lower()

    if rt == "gpt":
        return Stress_Recovery_GPT(
            Connectivity, Coord, EleType, medium_set, NGPTS, U
        )

    if rt == "elenodes":
        return Stress_Recovery_EleNodes(
            Connectivity, Coord, EleType, medium_set, U
        )

    if rt == "avgnodes":
        return Stress_Recovery_AvgNodes(
            Connectivity, Coord, EleType, medium_set, U
        )

    raise ValueError(
        f"Unknown recovery_type '{recovery_type}'. "
        "Use 'GPT', 'EleNodes', or 'AvgNodes'."
    )


# =================================================
#  Subfunction: Calculate stress at Gauss points
# -------------------------------------------------
def Stress_Recovery_GPT(
    Connectivity: np.ndarray,
    Coord: np.ndarray,
    EleType: str,
    medium_set: dict,
    NGPTS: int,
    U: np.ndarray,
) -> np.ndarray:
    """
    Calculate stress at Gauss points (GPT recovery).

    *********************************
    Returns
    *********************************
    stress_GPT : (Nele * NGPTS x 2) array
        [x, sigma_xx] at each Gauss point.
    """
    Coord = np.asarray(Coord, dtype=float)
    Connectivity = np.asarray(Connectivity, dtype=int)

    dim = Coord.shape[1]
    if dim != 1:
        raise ValueError(
            "Stress_Recovery_GPT is currently implemented for dim = 1 only."
        )

    Nele = Connectivity.shape[0]
    NumNodes = Coord.shape[0]
    Umat = _normalize_U_1d(U, NumNodes)  # (NumNodes, 1)

    # Get Gauss points and weights
    r, w = GaussPoints(dim, EleType, NGPTS)
    r = np.asarray(r).reshape(-1)  # (NGPTS,)

    stress_GPT = np.zeros((Nele * NGPTS, 2), dtype=float)

    for ele in range(Nele):
        EleNodes = Connectivity[ele, :].astype(int)
        node_ids = EleNodes - 1  # convert 1-based -> 0-based

        xCap = Coord[node_ids, :]         # (NodesPerEle, 1)
        uCap = Umat[node_ids, :]          # (NodesPerEle, 1)
        xCap_T = xCap.T                   # (1, NodesPerEle)

        for gpt in range(NGPTS):
            zeta = r[gpt]

            # Shape functions and derivatives on reference element
            N, DN = ShapeFunctions(EleType, np.array([zeta]))

            # Spatial coordinate (scalar)
            x_val = float(xCap_T @ N)

            # Jacobian J and B (1D: J is scalar, but we keep matrix form)
            J = float(xCap_T @ DN)      # (1x1) -> scalar
            if J == 0.0:
                raise ZeroDivisionError(
                    f"Jacobian is zero in element {ele+1}, Gauss point {gpt+1}."
                )
            B = DN / J                  # (NodesPerEle, 1)

            # Gradient of displacement at this Gauss point
            gradu = float(B.T @ uCap)   # scalar

            # Strain in 1D: epsilon = gradu
            strain_gpt = 0.5 * (gradu + gradu)  # scalar

            # Young's modulus from Lamé parameters at this x
            lambda_, mu_ = Get_Elastic_Moduli(medium_set, np.array([x_val]))
            E = mu_ * (3.0 * lambda_ + 2.0 * mu_) / (lambda_ + mu_)

            sigma_val = E * strain_gpt

            row = NGPTS * ele + gpt
            stress_GPT[row, 0] = x_val
            stress_GPT[row, 1] = sigma_val

    return stress_GPT


# =============================================
#  Subfunction: Calculate stress at EleNodes
# ---------------------------------------------
def Stress_Recovery_EleNodes(
    Connectivity: np.ndarray,
    Coord: np.ndarray,
    EleType: str,
    medium_set: dict,
    U: np.ndarray,
) -> np.ndarray:
    """
    Calculate stress at element nodes (EleNodes recovery).

    *********************************
    Returns
    *********************************
    stress_EleNodes : (Nele * NodesPerEle x 2) array
        [x, sigma_xx] at each element node.
    """
    Coord = np.asarray(Coord, dtype=float)
    Connectivity = np.asarray(Connectivity, dtype=int)

    dim = Coord.shape[1]
    if dim != 1:
        raise ValueError(
            "Stress_Recovery_EleNodes is currently implemented for dim = 1 only."
        )

    Nele, NodesPerEle = Connectivity.shape
    NumNodes = Coord.shape[0]
    Umat = _normalize_U_1d(U, NumNodes)

    # Node locations in reference L2 element: zeta = -1, +1
    r_ref = np.array([-1.0, 1.0], dtype=float)
    if NodesPerEle != r_ref.size:
        raise ValueError(
            f"Expected 2-node L2 elements. Got NodesPerEle = {NodesPerEle}."
        )

    stress_EleNodes = np.zeros((Nele * NodesPerEle, 2), dtype=float)

    for ele in range(Nele):
        EleNodes = Connectivity[ele, :].astype(int)
        node_ids = EleNodes - 1

        xCap = Coord[node_ids, :]        # (2, 1)
        uCap = Umat[node_ids, :]        # (2, 1)
        xCap_T = xCap.T                 # (1, 2)

        for gpt in range(NodesPerEle):
            zeta = r_ref[gpt]

            N, DN = ShapeFunctions(EleType, np.array([zeta]))

            # Spatial coordinate (scalar)
            x_val = float(xCap_T @ N)

            # Jacobian and B
            J = float(xCap_T @ DN)
            if J == 0.0:
                raise ZeroDivisionError(
                    f"Jacobian is zero in element {ele+1}, local node {gpt+1}."
                )
            B = DN / J

            gradu = float(B.T @ uCap)   # scalar

            strain_node = 0.5 * (gradu + gradu)

            lambda_, mu_ = Get_Elastic_Moduli(medium_set, np.array([x_val]))
            E = mu_ * (3.0 * lambda_ + 2.0 * mu_) / (lambda_ + mu_)

            sigma_val = E * strain_node

            row = NodesPerEle * ele + gpt
            stress_EleNodes[row, 0] = x_val
            stress_EleNodes[row, 1] = sigma_val

    return stress_EleNodes


# ==========================================
#  Subfunction: Averaging stress at nodes
# ------------------------------------------
def Stress_Recovery_AvgNodes(
    Connectivity: np.ndarray,
    Coord: np.ndarray,
    EleType: str,
    medium_set: dict,
    U: np.ndarray,
) -> np.ndarray:
    """
    Calculate nodal stresses by averaging stresses at element nodes.

    *********************************
    Assumptions
    *********************************
    - 1D bar with L2 elements.
    - Structured mesh where element e connects global nodes e and e+1.
      (This is consistent with build_bar_mesh and the MATLAB code.)

    *********************************
    Returns
    *********************************
    stress_Nodes : (NumNodes x 2) array
        [x, sigma_avg] at each global node.
    """
    Coord = np.asarray(Coord, dtype=float)
    Connectivity = np.asarray(Connectivity, dtype=int)

    dim = Coord.shape[1]
    if dim != 1:
        raise ValueError(
            "Stress_Recovery_AvgNodes is currently implemented for dim = 1 only."
        )

    NumNodes = Coord.shape[0]
    Nele = Connectivity.shape[0]

    # First compute stresses at element nodes
    stress_EleNodes = Stress_Recovery_EleNodes(
        Connectivity, Coord, EleType, medium_set, U
    )

    # Allocate [x, sigma] at nodes
    stress_Nodes = np.zeros((NumNodes, 2), dtype=float)

    # Coordinates: Coord is (NumNodes, 1)
    stress_Nodes[:, 0] = Coord[:, 0]

    # Accumulate contributions from each element
    # Matches the MATLAB logic:
    #   stress_Nodes(ele,2)   += stress_EleNodes(2*ele-1,2)
    #   stress_Nodes(ele+1,2) += stress_EleNodes(2*ele,2)
    for ele in range(Nele):
        left_node = ele        # global node index ele   (0-based)
        right_node = ele + 1   # global node index ele+1 (0-based)

        row_left = 2 * ele
        row_right = 2 * ele + 1

        stress_Nodes[left_node, 1] += stress_EleNodes[row_left, 1]
        stress_Nodes[right_node, 1] += stress_EleNodes[row_right, 1]

    # Average interior nodes: indices 1 .. NumNodes-2 (0-based),
    # corresponding to 2..NumNodes-1 in MATLAB.
    if NumNodes > 2:
        stress_Nodes[1:NumNodes - 1, 1] *= 0.5

    return stress_Nodes
