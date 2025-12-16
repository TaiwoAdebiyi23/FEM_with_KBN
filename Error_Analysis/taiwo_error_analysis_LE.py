import numpy as np
from taiwo_gauss_points import GaussPoints
from taiwo_shape_functions import ShapeFunctions


def Calculate_Error(
    Connectivity,
    Coord,
    EleType,
    NGPTS,
    U,
    get_exact_solution,
    verbose=True,
):
    """
    Compute L2 and H1-seminorm errors for scalar or vector-valued FEM solutions.

    STRICT VERSION (no broadcasting, no fallback reshapes):

    - Scalar case (e.g. diffusion):
        dofs_per_node = 1
        get_exact_solution(x) must return:
            u_exact    : scalar or length-1 array
            grad_exact : 1D array of length dim

    - Vector case (e.g. linear elasticity):
        dofs_per_node > 1
        get_exact_solution(x) must return:
            u_exact    : 1D array of length dofs_per_node
            grad_exact : 2D array of shape (dim, dofs_per_node)
                        with grad_exact[k, j] = ∂u_j/∂x_k
    """

    border = '=' * 65
    if verbose:
        print(border)
        print('       Error Metrics Report')
        print(border)

    # ----------------------------------
    # Basic mesh and problem dimensions
    # ----------------------------------
    Nele = Connectivity.shape[0]    # number of elements
    nnode = Connectivity.shape[1]   # nodes per element
    dim = Coord.shape[1]            # spatial dimension
    NumNodes = Coord.shape[0]

    # ----------------------------------
    # Normalize U into (NumNodes, dofs_per_node)
    # ----------------------------------
    U_arr = np.asarray(U)
    if U_arr.ndim == 1:
        if U_arr.size != NumNodes:
            raise ValueError(
                "For 1D U, size must equal number of nodes in Coord."
            )
        dofs_per_node = 1
        Umat = U_arr.reshape(NumNodes, 1)
    elif U_arr.ndim == 2:
        if U_arr.shape[0] != NumNodes:
            raise ValueError(
                "For 2D U, U.shape[0] must equal number of nodes in Coord."
            )
        NumNodes_U, dofs_per_node = U_arr.shape
        Umat = U_arr
    else:
        raise ValueError("U must be a 1D or 2D array.")

    # ----------------------------------
    # Gauss points and weights
    # ----------------------------------
    r, w = GaussPoints(dim, EleType, NGPTS)
    r = np.asarray(r)
    if r.ndim == 1:
        r = r[:, None]  # (ngauss, 1)
    ngauss = r.shape[0]

    if verbose:
        print(
            f"       dim={dim}, nnode={nnode}, Nele={Nele}, "
            f"NGPTS={NGPTS}, ngauss={ngauss}, dofs_per_node={dofs_per_node}"
        )

    # ----------------------------------
    # Accumulators for squared errors
    # ----------------------------------
    err_L2_sq = 0.0
    err_H1_sq = 0.0

    # ----------------------------------
    # Element loop
    # ----------------------------------
    for ele in range(Nele):
        EleNodes = Connectivity[ele, :].astype(int)
        xCap = Coord[EleNodes - 1, :]      # (nnode, dim)
        uCap = Umat[EleNodes - 1, :]       # (nnode, dofs_per_node)

        # ------------------------------
        # Gauss integration loop
        # ------------------------------
        for g in range(ngauss):
            zeta = r[g]

            # Shape functions and derivatives on reference element
            N, DN = ShapeFunctions(EleType, zeta)  # N: (nnode,), DN: (nnode, dim)

            # Jacobian and its determinant/inverse
            J = xCap.T @ DN              # (dim, dim)
            detJ = np.linalg.det(J)
            if detJ <= 0:
                raise ValueError(f"detJ <= 0 in element {ele + 1}.")
            invJ = np.linalg.inv(J)
            Bgrad = DN @ invJ            # (nnode, dim) = ∂N_i/∂x_k

            # Physical coordinates at this Gauss point
            x = N @ xCap                 # (dim,)

            # FE solution at x (vector of dofs_per_node)
            uh_vec = N @ uCap            # (dofs_per_node,)
            uh_vec = np.asarray(uh_vec).reshape(-1)

            # Gradient of FE solution at x: (dim, dofs_per_node)
            grad_uh_full = Bgrad.T @ uCap
            grad_uh_vec = grad_uh_full.reshape(-1)  # flattened gradient

            # Exact solution and gradient
            u_exact, grad_exact = get_exact_solution(np.asarray(x))

            # --------------------------
            # Strict shape checks: u_exact
            # --------------------------
            u_exact_arr = np.asarray(u_exact)

            if dofs_per_node == 1:
                # Scalar case: expect scalar or length-1
                u_exact_flat = u_exact_arr.reshape(-1)
                if u_exact_flat.size != 1:
                    raise ValueError(
                        "Scalar case: u_exact must be scalar or length-1 array."
                    )
                u_exact_vec = u_exact_flat  # length 1
            else:
                # Vector case: expect length = dofs_per_node
                u_exact_flat = u_exact_arr.reshape(-1)
                if u_exact_flat.size != dofs_per_node:
                    raise ValueError(
                        "Vector case: u_exact must have length dofs_per_node "
                        f"= {dofs_per_node}, got {u_exact_flat.size}."
                    )
                u_exact_vec = u_exact_flat  # length = dofs_per_node

            # --------------------------
            # Strict shape checks: grad_exact
            # --------------------------
            grad_exact_arr = np.asarray(grad_exact)

            if dofs_per_node == 1:
                # Scalar case: gradient is (dim,)
                if not (grad_exact_arr.ndim == 1 and grad_exact_arr.size == dim):
                    raise ValueError(
                        "Scalar case: grad_exact must be 1D of length dim "
                        f"= {dim}, got shape {grad_exact_arr.shape}."
                    )
                grad_exact_vec = grad_exact_arr.reshape(-1)  # length dim
            else:
                # Vector case: gradient is (dim, dofs_per_node)
                if grad_exact_arr.shape != (dim, dofs_per_node):
                    raise ValueError(
                        "Vector case: grad_exact must have shape "
                        f"(dim, dofs_per_node) = ({dim}, {dofs_per_node}), "
                        f"got shape {grad_exact_arr.shape}."
                    )
                grad_exact_vec = grad_exact_arr.reshape(-1)  # length dim*dofs_per_node

            # Local quadrature weight
            weight = w[g] * detJ

            # L2 contribution: ||uh - u_exact||^2
            diff_u = uh_vec - u_exact_vec
            err_L2_sq += weight * np.dot(diff_u, diff_u)

            # H1 seminorm contribution: Frobenius norm squared of gradient error
            diff_grad = grad_uh_vec - grad_exact_vec
            err_H1_sq += weight * np.dot(diff_grad, diff_grad)

    # ----------------------------------
    # Take square roots
    # ----------------------------------
    err_L2 = float(np.sqrt(err_L2_sq))
    err_H1 = float(np.sqrt(err_H1_sq))

    if verbose:
        print(f"       L2_error           = {err_L2:.6e}")
        print(f"       H1_error_seminorm  = {err_H1:.6e}")
        print(border)

    return err_L2, err_H1
