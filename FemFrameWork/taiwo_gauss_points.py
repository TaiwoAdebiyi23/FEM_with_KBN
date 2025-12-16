import numpy as np
import re
def Gauss_1D(NGPTS: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns Gauss-Legendre quadrature points and weights for 1D integration.

    This function computes the Gauss quadrature points and weights for numerical
    integration using Legendre polynomials. For small values of NGPTS (1, 2, or 3),
    hardcoded values are used for simplicity. For larger values of NGPTS, the function
    uses numpy's `leggauss` method to generate the points and weights.

    Parameters:
        NGPTS (int): The number of Gauss points (must be >= 1).

    Returns:
        tuple:
            r (np.ndarray): Quadrature points (ξ values) of shape (NGPTS,).
            w (np.ndarray): Quadrature weights of shape (NGPTS,).

    Raises:
        ValueError: If NGPTS < 1.

    Example:
        >>> r, w = Gauss_1D(3)
        >>> print(r)
        [-0.77459667  0.          0.77459667]
        >>> print(w)
        [0.55555556 0.88888889 0.55555556]
    """
    if NGPTS < 1:
        raise ValueError(f"Number of Gauss points must be >= 1, got {NGPTS}.")

    if NGPTS == 1:
        # 1-point rule
        r = np.array([0.0])
        w = np.array([2.0])

    elif NGPTS == 2:
        # 2-point rule
        r = np.array([-1.0/np.sqrt(3), 1.0/np.sqrt(3)])
        w = np.array([1.0, 1.0])

    elif NGPTS == 3:
        # 3-point rule
        r = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])
        w = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])

    else:
        # Use numpy's leggauss for NGPTS > 3
        r, w = np.polynomial.legendre.leggauss(NGPTS)

    return r, w

def test_Gauss_1D():
    """Test Gauss_1D with different number of points."""
    for ngpts in [1, 2, 3, 5]:
        r, w = Gauss_1D(ngpts)
        print(f"Gauss_1D points for {ngpts} points:", r)
        print(f"Gauss_1D weights for {ngpts} points:", w)

        # Check the sum of the weights for validity (should be 2 for 1D)
        if not np.isclose(np.sum(w), 2.0, atol=1e-6):
            print(f"Warning: Sum of weights for {ngpts} points is not 2. Got sum = {np.sum(w)}.")
        else:
            print(f"Sum of weights is valid for {ngpts} points.")

if __name__ == "__main__":
    # Testing Gauss_1D for various points
    test_Gauss_1D()
    help(Gauss_1D)

def Gauss_Triangle(NGPTS: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns Gauss-Legendre quadrature points and weights for a triangular domain.

    This function calculates the Gauss quadrature points and weights for numerical
    integration over a triangle. The implementation uses provided points and weights
    for specific values of NGPTS (1, 3, 4, 7, 9, 12, 13).

    Parameters:
        NGPTS (int): Number of Gauss points (valid values are 1, 3, 4, 7, 9, 12, 13).

    Returns:
        tuple:
            r (np.ndarray): Gauss points coordinates (xi, eta) for the triangle.
            w (np.ndarray): Weights associated with each Gauss point.

    Raises:
        ValueError: If NGPTS is invalid.

    Example:
        >>> r, w = Gauss_Triangle(3)
        >>> print(r)
        [[0.5 0.5]
         [0.5 0. ]
         [0.  0.5]]
        >>> print(w)
        [0.16666667 0.16666667 0.16666667]
    """
    if NGPTS == 1:
        r = np.array([[1/3, 1/3]])
        w = np.array([0.5])

    elif NGPTS == 3:
        # Three-point rule (use points and weights from the provided data)
        r = 0.5 * np.array([
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        w = np.array([1/6, 1/6, 1/6])

    elif NGPTS == 4:
        # Four-point rule (weights and points from provided data)
        r = np.array([
            [1/3, 1/3],
            [0.6, 0.2],
            [0.2, 0.6],
            [0.2, 0.2]
        ])
        w = 0.5 * np.array([-0.5625, 0.520833, 0.520833, 0.520833])

    elif NGPTS == 7:
        # Seven-point rule (weights and points from provided data)
        r = np.array([
            [1/3, 1/3],
            [0.797426985353087, 0.101286507323456],
            [0.101286507323456, 0.797426985353087],
            [0.101286507323456, 0.101286507323456],
            [0.470142064105115, 0.059715871789770],
            [0.059715871789770, 0.470142064105115],
            [0.470142064105115, 0.470142064105115]
        ])
        w = 0.5 * np.array([0.225000000000000, 0.125939180544827, 0.125939180544827, 0.125939180544827,
                            0.132394152788506, 0.132394152788506, 0.132394152788506])

    elif NGPTS == 9:
        # Nine-point rule (weights and points from provided data)
        r = np.array([
            [0.124949503233232, 0.437525248383384],
            [0.437525248383384, 0.124949503233232],
            [0.437525248383384, 0.437525248383384],
            [0.797112651860071, 0.165409927389841],
            [0.797112651860071, 0.037477420750088],
            [0.165409927389841, 0.797112651860071],
            [0.165409927389841, 0.037477420750088],
            [0.037477420750088, 0.797112651860071],
            [0.037477420750088, 0.165409927389841]
        ])
        w = 0.5 * np.array([0.205950504760887, 0.205950504760887, 0.205950504760887, 0.063691414286223,
                            0.063691414286223, 0.063691414286223, 0.063691414286223, 0.063691414286223,
                            0.063691414286223])

    elif NGPTS == 12:
        # Twelve-point rule (weights and points from provided data)
        r = np.array([
            [0.873821971016996, 0.063089014491502],
            [0.063089014491502, 0.873821971016996],
            [0.063089014491502, 0.063089014491502],
            [0.501426509658179, 0.249286745170910],
            [0.249286745170910, 0.501426509658179],
            [0.249286745170910, 0.249286745170910],
            [0.636502499121399, 0.310352451033785],
            [0.636502499121399, 0.053145049844816],
            [0.310352451033785, 0.636502499121399],
            [0.310352451033785, 0.053145049844816],
            [0.053145049844816, 0.636502499121399],
            [0.053145049844816, 0.310352451033785]
        ])
        w = 0.5 * np.array([0.050844906370207, 0.050844906370207, 0.050844906370207, 0.116786275726379,
                            0.116786275726379, 0.116786275726379, 0.082851075618374, 0.082851075618374,
                            0.082851075618374, 0.082851075618374, 0.082851075618374, 0.082851075618374])

    elif NGPTS == 13:
        # Thirteen-point rule (weights and points from provided data)
        r = np.array([
            [0.333333333333333, 0.333333333333333],
            [0.479308067841923, 0.260345966079038],
            [0.260345966079038, 0.479308067841923],
            [0.260345966079038, 0.260345966079038],
            [0.869739794195568, 0.065130102902216],
            [0.065130102902216, 0.869739794195568],
            [0.065130102902216, 0.065130102902216],
            [0.638444188569809, 0.312865496004875],
            [0.638444188569809, 0.086903154253160],
            [0.312865496004875, 0.638444188569809],
            [0.312865496004875, 0.086903154253160],
            [0.086903154253160, 0.638444188569809],
            [0.086903154253160, 0.312865496004875]
        ])
        w = 0.5 * np.array([-0.149570044467670, 0.175615257433204, 0.175615257433204, 0.175615257433204,
                            0.053347235608839, 0.053347235608839, 0.053347235608839, 0.077113760890257,
                            0.077113760890257, 0.077113760890257, 0.077113760890257, 0.077113760890257,
                            0.077113760890257])

    else:
        raise ValueError(f"Unsupported NGPTS value {NGPTS}. Supported values: 1, 3, 4, 7, 9, 12, 13.")

    return r, w

def Gauss_Quad(NGPTS: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns Gauss-Legendre quadrature points and weights for a quadrilateral domain using Kronecker product.

    This function calculates the Gauss quadrature points and weights for numerical
    integration over a quadrilateral. The quadrilateral is formed by combining 1D Gauss points
    using the Kronecker product.

    Parameters:
        NGPTS (int): The number of Gauss points in each dimension (positive integer).

    Returns:
        tuple:
            r (np.ndarray): 2D Gauss points (x, y) for the quadrilateral.
            w (np.ndarray): Corresponding weights for the 2D Gauss points.

    Example:
        >>> r, w = Gauss_Quad(2)
        >>> print(r)
        [[-0.57735027  0.57735027]
         [-0.57735027 -0.57735027]
         [ 0.57735027  0.57735027]
         [ 0.57735027 -0.57735027]]
        >>> print(w)
        [1. 1. 1. 1.]
    """
    # Get 1D Gauss points and weights
    r1D, w1D = Gauss_1D(NGPTS)

    # Construct 2D Gauss points using Kronecker product
    r = np.column_stack([
        np.kron(r1D, np.ones(NGPTS)),  # x-coordinates
        np.kron(np.ones(NGPTS), r1D)   # y-coordinates
    ])

    # Construct 2D weights (element-wise product of 1D weights)
    w = np.kron(w1D, w1D)

    return r, w

def validate_element_type(EleType: str) -> str:
    """
    Validates the element type and suggests corrections if necessary.

    Parameters:
        EleType (str): Element type (e.g., 'T3', 'Q4', 'Q8').

    Returns:
        str: Valid element type after correction if necessary.

    Raises:
        ValueError: If the element type is invalid.
        TypeError: If EleType is not a string.
    """
    if not isinstance(EleType, str):
        raise TypeError(
            "Oops! Looks like 'EleType' is not a string. Please provide a valid "
            "string for element type (e.g., 'T3', 'Q4', or 'Q8')."
        )

    EleType = EleType.upper()

    # Now includes Q8
    valid_elements = ['T3', 'Q4', 'Q8']

    if EleType not in valid_elements:
        corrected_elements = [
            el for el in valid_elements
            if re.fullmatch(r'.*' + re.escape(EleType) + r'.*', el, re.IGNORECASE)
        ]
        if corrected_elements:
            print(
                f"Did you mean '{corrected_elements[0]}'? "
                "Please confirm or provide the correct element type."
            )
        else:
            raise ValueError(
                f"Invalid element type '{EleType}'. I can't work with that. "
                f"The valid types are {valid_elements}. Please double-check your input and try again!"
            )

    return EleType


def validate_ngpts_triangle(NGPTS: int) -> None:
    """
    Triangle-only NGPTS validator using the exact supported rules from the file.
    """
    if not isinstance(NGPTS, int):
        raise TypeError("Uh-oh! 'NGPTS' is not an integer. We need an integer here, like 3 or 5.")
    if NGPTS < 1:
        raise ValueError(f"NGPTS must be >= 1. You gave {NGPTS}.")
    if NGPTS not in [1, 3, 4, 7, 9, 12, 13]:
        raise ValueError(
            f"NGPTS = {NGPTS} is not supported for T3. "
            "Valid values are: 1, 3, 4, 7, 9, 12, 13."
        )
    # Friendly ping:
    print(f"All good! NGPTS = {NGPTS} is valid for T3. Let's move on!")

def validate_ngpts_q4(NGPTS: int) -> None:
    """
    Q4 validator: any positive integer is allowed (tensor-product Gauss-Legendre).
    """
    if not isinstance(NGPTS, int):
        raise TypeError("For Q4, 'NGPTS' must be an integer: 1, 2, 3, ...")
    if NGPTS < 1:
        raise ValueError(f"For Q4, NGPTS must be >= 1. You gave {NGPTS}.")


# --- dispatcher --------------------------------------------------------------

def Gauss_2D(EleType: str, NGPTS: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Dispatcher for 2D Gauss quadrature.
    - T3 : uses tabulated triangle rules (strict NGPTS set).
    - Q4 : tensor-product Gauss–Legendre (any NGPTS >= 1).
    - Q8 : same quadrature as Q4 (still a quadrilateral on [-1,1] x [-1,1]),
           but usually you choose a higher NGPTS (e.g., 3x3) for accuracy.
    """
    EleType = validate_element_type(EleType)

    if EleType == 'T3':
        validate_ngpts_triangle(NGPTS)
        return Gauss_Triangle(NGPTS)

    if EleType in ('Q4', 'Q8'):
        # Reuse the same validator and rule for both Q4 and Q8
        validate_ngpts_q4(NGPTS)
        return Gauss_Quad(NGPTS)

    # Should never reach here because validate_element_type filters invalid types
    raise ValueError(f"Unsupported element type '{EleType}'.")


# --- tests ------------------------------------------------------------------

def test_invalid_inputs():
    """
    Intentionally exercise invalid inputs to show the error messages are helpful.
    """
    print("\n===== INTENTIONAL INVALID-INPUT TESTS =====")

    # 1) Bad element type (spelling)
    try:
        print("\n[Case] EleType='T4' (invalid for dispatcher)")
        Gauss_2D('T4', 3)
    except Exception as e:
        print("-> Caught:", e)

    # 2) Bad NGPTS type for T3
    try:
        print("\n[Case] EleType='T3', NGPTS='two' (type error)")
        Gauss_2D('T3', 'two')  # will fail in validate_ngpts_triangle
    except Exception as e:
        print("-> Caught:", e)

    # 3) Bad NGPTS value for T3 (not in table)
    try:
        print("\n[Case] EleType='T3', NGPTS=5 (unsupported rule count for T3)")
        Gauss_2D('T3', 5)
    except Exception as e:
        print("-> Caught:", e)

    # 4) Bad NGPTS for Q4 (non-integer)
    try:
        print("\n[Case] EleType='Q4', NGPTS=3.5 (type error)")
        Gauss_2D('Q4', 3.5)
    except Exception as e:
        print("-> Caught:", e)

    # 5) Bad NGPTS for Q4 (non-positive)
    try:
        print("\n[Case] EleType='Q4', NGPTS=0 (must be >=1)")
        Gauss_2D('Q4', 0)
    except Exception as e:
        print("-> Caught:", e)

def test_Gauss_2D():
    """
    Valid-path tests + area checks, with tidy printing.
    """
    print("\n===== VALID-INPUT TESTS =====")
    for ele_type in ['T3', 'Q4', 'Q8']:   # <-- Q8 added here
        # Choose NGPTS that are valid for each
        if ele_type == 'T3':
            valid_ngpts = [1, 3, 7, 9, 12]
        else:  # Q4 and Q8
            valid_ngpts = [1, 2, 3, 4, 5, 6]
        for ngpts in valid_ngpts:
            try:
                r, w = Gauss_2D(ele_type, ngpts)
                print("=============================================")
                print(f"Testing Gauss_2D for {ele_type} with {ngpts} points:")
                print("=============================================")

                if ngpts <= 3:
                    print("Gauss points:\n", r)
                    print("Weights:\n", w)
                else:
                    print(f"(Not printing {len(w)} points to keep logs readable.)")

                # Area check
                target_area = 0.5 if ele_type == 'T3' else 4.0
                if np.isclose(np.sum(w), target_area, atol=1e-12):
                    print(f"✓ Sum of weights OK (= {np.sum(w):.12g}) matches area {target_area}\n")
                else:
                    print(f"⚠ Sum of weights {np.sum(w):.12g} != {target_area}\n")
            except Exception as e:
                print(f"Unexpected error in valid-path test for {ele_type}, NGPTS={ngpts}: {e}\n")

    # Run the invalid-input tests at the end so readers see both paths in one go
    test_invalid_inputs()

if __name__ == "__main__":
    # Test Gauss_2D for various elements and Gauss points
  test_Gauss_2D()
def Gauss_B8(NGPTS: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns Gauss-Legendre quadrature points and weights for a brick element in 3D.

    This function calculates the Gauss quadrature points and weights for numerical
    integration over a brick element using a tensor-product approach (Kronecker product)
    across three dimensions.

    Parameters:
        NGPTS (int): Number of Gauss points in each dimension (positive integer).

    Returns:
        tuple:
            r (np.ndarray): 3D Gauss points (x, y, z) for the brick element.
            w (np.ndarray): Corresponding weights for the 3D Gauss points.

    Raises:
        ValueError: If NGPTS is invalid.

    Example:
        >>> r, w = Gauss_B8(2)
        >>> print(r)
        [[-0.57735027 -0.57735027 -0.57735027]
         [-0.57735027 -0.57735027  0.57735027]
         [-0.57735027  0.57735027 -0.57735027]
         [-0.57735027  0.57735027  0.57735027]
         [ 0.57735027 -0.57735027 -0.57735027]
         [ 0.57735027 -0.57735027  0.57735027]
         [ 0.57735027  0.57735027 -0.57735027]
         [ 0.57735027  0.57735027  0.57735027]]
        >>> print(w)
        [1. 1. 1. 1. 1. 1. 1. 1.]
    """

    # Check if NGPTS is a positive integer
    if not isinstance(NGPTS, int) or NGPTS < 1:
        raise ValueError(f"NGPTS should be a positive integer. You provided {NGPTS}. Please check your input.")

    # Get 1D Gauss points and weights
    r1D, w1D = Gauss_1D(NGPTS)

    # Construct 3D Gauss points using Kronecker product (Tensor product in 3D)
    r_x = np.kron(np.ones(NGPTS), np.kron(np.ones(NGPTS), r1D))  # x-coordinates
    r_y = np.kron(np.ones(NGPTS), np.kron(r1D, np.ones(NGPTS)))  # y-coordinates
    r_z = np.kron(r1D, np.kron(np.ones(NGPTS), np.ones(NGPTS)))  # z-coordinates

    # Stack to form the 3D Gauss points
    r = np.column_stack([r_x, r_y, r_z])

    # Construct 3D weights by taking the element-wise product of the 1D weights
    w = np.kron(np.kron(w1D, w1D), w1D)

    return r, w

def test_Gauss_B8():
    """
    Test the Gauss_Brick function for various valid and invalid NGPTS values.
    """
    print("\n===== TESTING Gauss_Brick Function =====")

    # Test valid NGPTS (e.g., NGPTS = 2, NGPTS = 3)
    for NGPTS in [2, 3]:
        try:
            print(f"\nTesting with NGPTS = {NGPTS}")
            r, w = Gauss_B8(NGPTS)

            # Display Gauss points and weights for small NGPTS values
            print("Gauss points:\n", r)
            print("Weights:\n", w)

            # Check if the sum of weights is correct (for a brick element with volume 8)
            expected_sum = 8  # Volume of a unit brick is typically 8
            weight_sum = np.sum(w)
            print(f"Sum of weights: {weight_sum:.6f}")

            # Validate the sum of weights
            if np.isclose(weight_sum, expected_sum, atol=1e-6):
                print(f"✓ Sum of weights matches expected volume: {expected_sum}")
            else:
                print(f"⚠ Sum of weights does not match expected volume. Got: {weight_sum:.6f}")

        except Exception as e:
            print(f"Error for NGPTS = {NGPTS}: {e}")

    # Test invalid NGPTS values (non-integer and negative)
    print("\n===== TESTING INVALID INPUTS =====")

    # Test with invalid NGPTS values
    invalid_ngpts = ['three', -5, 0]
    for NGPTS in invalid_ngpts:
        try:
            print(f"\nTesting with invalid NGPTS = {NGPTS}")
            r, w = Gauss_B8(NGPTS)  # This should raise an error
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_Gauss_B8()
def Gauss_TET4(NGPTS: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns Gauss-Legendre quadrature points and weights for a tetrahedral element (TET4).

    This function calculates the Gauss quadrature points and weights for numerical
    integration over a tetrahedral element using provided points and weights
    for specific NGPTS values (1, 4, 5, 11, 15).

    Parameters:
        NGPTS (int): Number of Gauss points (valid values are 1, 4, 5, 11, 15).

    Returns:
        tuple:
            r (np.ndarray): Gauss points coordinates (xi, eta, zeta) for the tetrahedron.
            w (np.ndarray): Weights associated with each Gauss point.

    Raises:
        ValueError: If NGPTS is invalid.

    Example:
        >>> r, w = Gauss_TET4(4)
        >>> print(r)
        [[0.5854102 0.1381966 0.1381966]
        [0.1381966 0.5854102 0.1381966]
        [0.1381966 0.1381966 0.5854102]
        [0.5854102 0.5854102 0.5854102]]
        >>> print(w)
         [0.04166667 0.04166667 0.04166667 0.04166667]
    """
    if NGPTS == 1:
        # One-point rule
        r = np.array([[1/4, 1/4, 1/4]])  # p1 = [1/4, 1/4, 1/4]
        w = np.array([1/6])  # w1 = 1/6

    elif NGPTS == 4:
        # Four-point rule
        p1 = 0.5854101966249638
        p2 = 0.1381966011250105
        r = np.array([
            [p1, p2, p2],
            [p2, p1, p2],
            [p2, p2, p1],
            [p1, p1, p1]
        ])
        w = (1 / (6 * 4)) * np.ones(4)

    elif NGPTS == 5:
        # Five-point rule
        r = np.array([
            [1/4, 1/4, 1/4],
            [1/2, 1/6, 1/6],
            [1/6, 1/2, 1/6],
            [1/6, 1/6, 1/2],
            [1/6, 1/6, 1/6]
        ])
        w = 1/6 * np.array([-4/5, 9/20, 9/20, 9/20, 9/20])

    elif NGPTS == 11:
        # Eleven-point rule
        p1, p2, p3, p4, p5 = 0.25, 0.785714285714286, 0.071428571428571, 0.399403576166799, 0.100596423833201
        q1, q2, q3 = -0.013155555555556, 0.007622222222222, 0.024888888888889
        r = np.array([
            [p1, p1, p1],
            [p2, p3, p3],
            [p3, p2, p3],
            [p3, p3, p2],
            [p3, p3, p3],
            [p4, p5, p5],
            [p5, p4, p5],
            [p5, p5, p4],
            [p5, p4, p4],
            [p4, p5, p4],
            [p4, p4, p5]
        ])
        w = np.array([q1, q2, q2, q2, q2, q3, q3, q3, q3, q3, q3])  # weights for each point

    elif NGPTS == 15:
        # Fifteen-point rule
        p1, p2, p3, p4, p5, p6, p7 = 0.25, 0.0, 0.333333333333333, 0.727272727272727, 0.090909090909091, 0.066550153573664, 0.433449846426336
        q1, q2, q3, q4 = 0.030283678097089, 0.006026785714286, 0.011645249086029, 0.010949141561386
        r = np.array([
            [p1, p1, p1],
            [p2, p3, p3],
            [p3, p2, p3],
            [p3, p3, p2],
            [p3, p3, p3],
            [p4, p5, p5],
            [p5, p4, p5],
            [p5, p5, p4],
            [p5, p5, p5],
            [p6, p7, p7],
            [p7, p6, p7],
            [p7, p7, p6],
            [p7, p6, p6],
            [p6, p7, p6],
            [p6, p6, p7]
        ])
        w = np.array([q1, q2, q2, q2, q2, q3, q3, q3, q3, q4, q4, q4, q4, q4, q4])  # weights for each point

    else:
        raise ValueError(f"Unsupported NGPTS value {NGPTS}. Supported values: 1, 4, 5, 11, 15.")

    return r, w

def test_Gauss_TET4():
    """
    Test the Gauss_TET4 function for various valid and invalid NGPTS values.
    """
    print("\n===== TESTING Gauss_TET4 Function =====")

    # Test valid NGPTS (e.g., NGPTS = 1, 4, 5)
    for NGPTS in [1, 4, 5]:
        try:
            print(f"\nTesting with NGPTS = {NGPTS}")
            r, w = Gauss_TET4(NGPTS)

            # Display Gauss points and weights for small NGPTS values
            print("Gauss points:\n", r)
            print("Weights:\n", w)

            # Check if the sum of weights is correct (for a unit tetrahedron, the volume is 1)
            expected_sum = 1/6  # Volume of a unit tetrahedron is typically 1/6
            weight_sum = np.sum(w)
            print(f"Sum of weights: {weight_sum:.6f}")

            # Validate the sum of weights
            if np.isclose(weight_sum, expected_sum, atol=1e-6):
                print(f"✓ Sum of weights matches expected volume: {expected_sum}")
            else:
                print(f"⚠ Sum of weights does not match expected volume. Got: {weight_sum:.6f}")

        except Exception as e:
            print(f"Error for NGPTS = {NGPTS}: {e}")

    # Test invalid NGPTS values (non-integer and negative)
    print("\n===== TESTING INVALID INPUTS =====")

    # Test with invalid NGPTS values
    invalid_ngpts = ['three', -5, 0]
    for NGPTS in invalid_ngpts:
        try:
            print(f"\nTesting with invalid NGPTS = {NGPTS}")
            r, w = Gauss_TET4(NGPTS)  # This should raise an error
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_Gauss_TET4()

def Gauss_W6(NGPTS: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes Gauss-Legendre quadrature points and weights for a wedge-shaped element.
    This function combines a 2D triangular base (T3) quadrature and a 1D quadrature for extrusion along the z-axis.

    Parameters:
        NGPTS (int): The number of Gauss points (valid values: 1, 3, 4, 7, 9, 12, 13).

    Returns:
        tuple:
            r (np.ndarray): Gauss points for the wedge in 3D (shape: (num_1D_pts * num_2D_pts, 3)).
            w (np.ndarray): Weights associated with each Gauss point for the wedge (shape: (num_1D_pts * num_2D_pts,)).

    Raises:
        ValueError: If NGPTS is invalid for the T3 rule.
        TypeError: If NGPTS is not an integer.

    Example:
        >>> r, w = Gauss_W6(3)
        >>> print(r)
        [[ 0.5        -0.77459667  0.5       ]
        [ 0.5         0.          0.5       ]
        [ 0.5         0.77459667  0.5       ]
        [ 0.5        -0.77459667  0.        ]
        [ 0.5         0.          0.        ]
        [ 0.5         0.77459667  0.        ]
        [ 0.         -0.77459667  0.5       ]
        [ 0.          0.          0.5       ]
        [ 0.          0.77459667  0.5       ]]
        >>> print(w)
        [0.09259259 0.14814815 0.09259259 0.09259259 0.14814815 0.09259259
            0.09259259 0.14814815 0.09259259]
    """

    # Error check for NGPTS (must be a positive integer)
    if not isinstance(NGPTS, int):
        raise TypeError(f"NGPTS must be an integer. You provided {type(NGPTS)}.")

    if NGPTS not in [1, 3, 4, 7, 9, 12, 13]:
        raise ValueError(f"NGPTS={NGPTS} is not supported for T3 rule. Valid values are {1, 3, 4, 7, 9, 12, 13}.")

    # Get the Gauss points and weights for the triangular base (T3)
    r2D, w2D = Gauss_2D('T3', NGPTS)  # T3 triangle points and weights

    # Get the Gauss points and weights for the 1D extrusion along the z-axis
    r1D, w1D = Gauss_1D(NGPTS)  # 1D line points and weights

    # Initializing the result arrays for Gauss points (3D points: x, y, z)
    r = np.zeros((NGPTS * NGPTS, 3))  # 3D points matrix

    # Vectorized combination of points.
    repeated_triangle_points = np.repeat(r2D, NGPTS, axis=0)  # Repeat each 2D point for each 1D point
    tiled_line_points = np.tile(r1D, NGPTS)          # Tile 1D points to match repetitions

    # Combine x, y, z coordinates
    r[:, 0] = repeated_triangle_points[:, 0]  # x-coordinate from the triangle points
    r[:, 1] = tiled_line_points      # y-coordinate from the 1D extrusion
    r[:, 2] = repeated_triangle_points[:, 1]  # z-coordinate from the triangle points

    # Combine the weights for the wedge
    repeated_triangle_weights = np.repeat(w2D, NGPTS)  # Repeat the triangle weights
    tiled_line_weights = np.tile(w1D, NGPTS)  # Tile the 1D weights
    w = repeated_triangle_weights * tiled_line_weights       # Element-wise multiplication of weights

    return r, w
# ---- Test Function ----

def test_Gauss_W6() -> None:
    """
    Test Gauss_Wedge with various valid and invalid NGPTS values, ensuring correctness.
    """
    print("\n===== VALID-INPUT TESTS: Gauss_W6 =====")

    # Test with valid NGPTS values
    cases = [1, 3, 4, 7, 9, 12, 13]  # All valid NGPTS values for testing
    for ngpts in cases:
        R, W = Gauss_W6(ngpts)
        print("------------------------------------------------")
        print(f"T3 rule: {ngpts} pts  => total {len(W)}")
        if ngpts <= 4:
            print("Points:\n", R)
            print("Weights:\n", W)
        else:
            print(f"(Not printing {len(W)} points to keep output readable.)")

        # Weight sum check: reference wedge volume = 0.5 * 2 = 1.0
        s = float(W.sum())
        if np.isclose(s, 1.0, atol=1e-12):
            print(f"✓ Sum of weights OK (= {s:.12g})")
        else:
            print(f"⚠ Sum of weights does not match expected volume. Got: {s:.12g}")

    print("\n===== INVALID-INPUT TESTS: Gauss_W6 =====")

    # Invalid NGPTS values for testing
    try:
        Gauss_W6(5)  # 5 is not a supported T3 rule in your table
    except Exception as e:
        print("Caught (bad T3 count 5):", e)

    try:
        Gauss_W6(0)  # Invalid NGPTS value
    except Exception as e:
        print("Caught (NGPTS=0):", e)

    try:
        Gauss_W6(2.5)  # Type error (non-integer NGPTS)
    except Exception as e:
        print("Caught (NGPTS not int):", e)

    # Using the single-argument style (same count for both directions)
    print("\n===== SINGLE-ARGUMENT STYLE EXAMPLE =====")
    R, W = Gauss_W6(3)  # Uses 3 for T3 and 3 for 1D
    print(f"(3,3) total points: {len(W)}, sum(weights)={W.sum():.6f}")

if __name__ == "__main__":
    test_Gauss_W6()

# Validate Element Type for 3D
def validate_element_type_3D(EleType: str) -> str:
    """
    Validates the element type for 3D (TET4, W8, B8).

    Parameters:
        EleType (str): Element type (e.g., 'TET4', 'W6', 'B8').

    Returns:
        str: Validated element type.

    Raises:
        ValueError: If the element type is invalid.
        TypeError: If EleType is not a string.
    """
    if not isinstance(EleType, str):
        raise TypeError("Oops! 'EleType' is not a string. Please provide a valid string for element type. (e.g., 'TET4', 'W6', 'B8')")

    EleType = EleType.upper()
    valid_elements = ['TET4', 'W6', 'B8']

    if EleType not in valid_elements:
        raise ValueError(f"Invalid element type '{EleType}'. Valid types are {valid_elements}.")

    return EleType

# Validate NGPTS for TET4
def validate_ngpts_tet4(NGPTS: int) -> None:
    """
    TET4-only NGPTS validator using the exact supported rules from the file.
    """
    if not isinstance(NGPTS, int):
        raise TypeError(f"NGPTS for TET4 must be an integer. You provided {type(NGPTS)}.")
    if NGPTS < 1:
        raise ValueError(f"NGPTS must be >= 1. You gave {NGPTS}.")
    if NGPTS not in [1, 4, 5, 11, 15]:
        raise ValueError(f"Unsupported NGPTS value {NGPTS} for TET4. Valid values are [1, 4, 5, 11, 15].")
    print(f"All good! NGPTS = {NGPTS} is valid for TET4.")

# Validate NGPTS for Wedge
def validate_ngpts_w6(NGPTS: int) -> None:
    """
    W6-only NGPTS validator.
    """
    if not isinstance(NGPTS, int):
        raise TypeError(f"NGPTS for W6 must be an integer. You provided {type(NGPTS)}.")
    if NGPTS < 1:
        raise ValueError(f"NGPTS must be >= 1. You gave {NGPTS}.")
    if NGPTS not in [1, 3, 4, 7, 9, 12, 13]:
        raise ValueError(f"Unsupported NGPTS value {NGPTS} for W6. Valid values are [1, 3, 4, 7, 9, 12, 13].")
    print(f"All good! NGPTS = {NGPTS} is valid for Wedge.")

# Validate NGPTS for Brick
def validate_ngpts_b8(NGPTS: int) -> None:
    """
    B8-only NGPTS validator.
    """
    if not isinstance(NGPTS, int):
        raise TypeError(f"NGPTS for B8 must be an integer. You provided {type(NGPTS)}.")
    if NGPTS < 1:
        raise ValueError(f"NGPTS must be >= 1. You gave {NGPTS}.")
    print(f"All good! NGPTS = {NGPTS} is valid for B8.")



# Gauss Points Dispatcher for 3D
def Gauss_3D(EleType: str, NGPTS: int) -> tuple:
    """
    Dispatcher for 3D Gauss quadrature.
    - TET4: uses specific Gauss quadrature points for the tetrahedral element.
    - W6: uses tensor-product Gauss-Legendre quadrature for the wedge element.
    - B8: uses Kronecker product approach for the brick element.
    """
    EleType = validate_element_type_3D(EleType)  # Validate element type

    if EleType == 'TET4':
        validate_ngpts_tet4(NGPTS)  # Validate NGPTS for TET4
        return Gauss_TET4(NGPTS)

    elif EleType == 'W6':
        validate_ngpts_w6(NGPTS)  # Validate NGPTS for Wedge
        return Gauss_W6(NGPTS)

    elif EleType == 'B8':
        validate_ngpts_b8(NGPTS)  # Validate NGPTS for Brick
        return Gauss_B8(NGPTS)

    else:
        raise ValueError(f"Unsupported element type '{EleType}'. Valid options are 'TET4', 'W6', 'B8'.")

# Test function for valid and invalid inputs
def test_Gauss_3D():
    """
    Test the 3D Gauss quadrature dispatcher function for multiple elements (T4, W6, B8).
    """
    for element in ['TET4', 'W6', 'B8']:
        print(f"\n===== TESTING {element.capitalize()} =====")
        for NGPTS in [1, 3, 5]:  # Valid NGPTS values for testing
            print(f"\nTesting with NGPTS = {NGPTS}")
            try:
                r, w = Gauss_3D(element, NGPTS)
                print("Gauss points:\n", r)
                print("Weights:\n", w)

                # Check if the sum of weights is correct
                if element == 'TET4':
                    expected_sum = 1/6  # Volume of a unit tetrahedron
                elif element == 'B8':
                    expected_sum = 8.0  # Volume of a unit brick
                else:
                    expected_sum = 1.0  # Default assumption for others

                weight_sum = np.sum(w)
                print(f"Sum of weights: {weight_sum:.6f}")

                # Validate the sum of weights
                if np.isclose(weight_sum, expected_sum, atol=1e-6):
                    print(f"✓ Sum of weights matches expected volume: {expected_sum}")
                else:
                    print(f"⚠ Sum of weights does not match expected volume. Got: {weight_sum:.6f}")

            except Exception as e:
                print(f"Error for NGPTS = {NGPTS}: {e}")

    # Invalid input testing
    invalid_ngpts = ['three', -5, 0]
    for NGPTS in invalid_ngpts:
        try:
            print(f"\nTesting with invalid NGPTS = {NGPTS}")
            r, w = Gauss_3D('TET4', NGPTS)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_Gauss_3D()  # Run the test function to validate the dispatcher

def GaussPoints(dim: int, EleType: str, NGPTS: int) -> tuple:
    """
    Main dispatcher for Gauss quadrature points and weights based
    on the problem dimension and element type.

    Parameters
    ----------
    dim : int
        Problem dimension (1, 2, or 3).
    EleType : str
        Element type, e.g. 'L2', 'Q4', 'Q8', 'T3', 'B8', 'TET4', 'W6'.
    NGPTS : int
        Number of Gauss points (per direction for tensor-product rules).

    Returns
    -------
    (r, w) : tuple[np.ndarray, np.ndarray]
        r : Gauss point coordinates in the reference element.
        w : Associated quadrature weights.

    Raises
    ------
    TypeError
        If any of the inputs has an incorrect type.
    ValueError
        If the dimension or element type is not supported.
    """
    # ----------------------------
    # Basic type checks
    # ----------------------------
    if not isinstance(dim, int) or not isinstance(EleType, str) or not isinstance(NGPTS, int):
        raise TypeError(
            "Invalid input type. 'dim' must be int, 'EleType' must be str, "
            "and 'NGPTS' must be int."
        )

    # ----------------------------
    # Check dimension
    # ----------------------------
    if dim not in (1, 2, 3):
        raise ValueError("Invalid dimension. 'dim' must be 1, 2, or 3.")

    # ----------------------------
    # Check element type, by dimension
    # ----------------------------
    EleType = EleType.upper()

    valid_by_dim = {
        1: ['L2'],
        2: ['Q4', 'Q8', 'T3'],   # <- Q8 included here
        3: ['B8', 'TET4', 'W6'],
    }

    allowed = valid_by_dim[dim]
    if EleType not in allowed:
        raise ValueError(
            f"{EleType} is an invalid element type for dim={dim}. "
            f"Valid entries are: {allowed}"
        )

    # ----------------------------
    # Dispatch
    # ----------------------------
    if dim == 1:
        return Gauss_1D(NGPTS)

    if dim == 2:
        return Gauss_2D(EleType, NGPTS)

    if dim == 3:
        return Gauss_3D(EleType, NGPTS)

    # Should not reach here
    raise ValueError("Unsupported dimension. Only 1, 2, and 3 are supported.")



# Test function for GaussPoints dispatcher
def test_GaussPoints():
    """
    Test the GaussPoints dispatcher function for different dimensions and element types.
    """
    print("\n===== TESTING 1D, 2D, 3D GaussPoints Function =====")

    # Test for 1D with L2 element
    for NGPTS in [1, 2, 3]:
        print(f"\nTesting 1D with NGPTS = {NGPTS}")
        try:
            r, w = GaussPoints(1, 'L2', NGPTS)
            print(f"Gauss points (1D) for {NGPTS} points:\n", r)
            print(f"Gauss weights (1D) for {NGPTS} points:\n", w)
        except Exception as e:
            print(f"Error: {e}")

    # Test for 2D with T3 and Q4 elements
    for EleType in ['T3', 'Q4']:
        for NGPTS in [1, 3, 7, 9]:
            print(f"\nTesting 2D {EleType} with NGPTS = {NGPTS}")
            try:
                r, w = GaussPoints(2, EleType, NGPTS)
                if NGPTS <= 4:
                  print(f"Gauss points (2D) for {EleType} and {NGPTS} points:\n", r)
                  print(f"Gauss weights (2D) for {EleType} and {NGPTS} points:\n", w)
            except Exception as e:
                print(f"Error: {e}")

    # Test for 3D with Tetrahedron, Wedge, and Brick elements
    for EleType in ['TET4', 'W6', 'B8']:
        for NGPTS in [1, 4, 5]:
            print(f"\nTesting 3D {EleType.capitalize()} with NGPTS = {NGPTS}")
            try:
                r, w = GaussPoints(3, EleType, NGPTS)
                if NGPTS <= 4:
                  print(f"Gauss points (3D) for {EleType} with {NGPTS} points:\n", r)
                  print(f"Gauss weights (3D) for {EleType} with {NGPTS} points:\n", w)
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    test_GaussPoints()  # Run the test function to validate the dispatcher

