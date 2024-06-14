import numpy as np
import matplotlib.pyplot as plt

# github


def compute_side_length(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_vertices(A_prev, B_prev, C_prev, xk):
    A_k = (xk * A_prev + xk**2 * B_prev + C_prev) / (1 + xk + xk**2)
    B_k = (A_prev + xk * B_prev + xk**2 * C_prev) / (1 + xk + xk**2)
    C_k = (xk**2 * A_prev + B_prev + xk * C_prev) / (1 + xk + xk**2)
    return A_k, B_k, C_k


def compute_shape_function(ak, bk, ck, zeta):
    return (ak**2 + zeta**2 * bk**2 + zeta * ck**2) / (ak**2 + bk**2 + ck**2)


def main(A0, B0, C0, n, s):
    # Initialize
    A_k, B_k, C_k = np.array(A0), np.array(B0), np.array(C0)
    vertices = [(A_k, B_k, C_k)]
    sides = []
    shape_functions = []

    zeta = np.exp(2 * np.pi * 1j / 3)

    for k in range(1, n + 1):
        xk = s[k - 1] / (1 - s[k - 1])

        # Compute new vertices
        A_k, B_k, C_k = compute_vertices(
            vertices[-1][0], vertices[-1][1], vertices[-1][2], xk
        )
        vertices.append((A_k, B_k, C_k))

        # Compute sides
        ak = compute_side_length(B_k, C_k)
        bk = compute_side_length(A_k, C_k)
        ck = compute_side_length(A_k, B_k)
        sides.append((ak, bk, ck))

        # Compute shape function
        sigma_k = compute_shape_function(ak, bk, ck, zeta)
        shape_functions.append(sigma_k)

        # Output results
        print(f"Iteration {k}:")
        print(f"Vertices: A_k = {A_k}, B_k = {B_k}, C_k = {C_k}")
        print(f"Sides: ak = {ak}, bk = {bk}, ck = {ck}")
        print(f"Shape function: sigma_k = {sigma_k}")

        # Check condition
        if k > 1:
            sigma_prev = shape_functions[-2]
            condition = s[k - 1] + zeta * (s[k - 1] + zeta * sigma_prev)
            if not np.isclose(sigma_k, condition):
                print(f"Condition failed at iteration {k}. Aborting.")
                return vertices, sides, shape_functions

    return vertices, sides, shape_functions


def visualize(vertices, n):
    plt.figure()
    for i in range(min(n, len(vertices))):
        A_k, B_k, C_k = vertices[i]
        triangle = np.array([A_k, B_k, C_k, A_k])
        plt.plot(triangle[:, 0], triangle[:, 1], label=f"Iteration {i}")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Triangle Iterations")
    plt.show()


# Example input
A0 = (0, 0)
B0 = (1, 0)
C0 = (0.5, np.sqrt(3) / 2)
n = 5
s = [0.3, 0.4, 0.5, 0.6, 0.7]

vertices, sides, shape_functions = main(A0, B0, C0, n, s)
visualize(vertices, n)


"""def compute_side_length(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_vertices(A_prev, B_prev, C_prev, xk):
    A_k = (xk * A_prev + xk**2 * B_prev + C_prev) / (1 + xk + xk**2)
    B_k = (A_prev + xk * B_prev + xk**2 * C_prev) / (1 + xk + xk**2)
    C_k = (xk**2 * A_prev + B_prev + xk * C_prev) / (1 + xk + xk**2)
    return A_k, B_k, C_k


def compute_shape_function(ak, bk, ck, zeta):
    return (ak**2 + zeta**2 * bk**2 + zeta * ck**2) / (ak**2 + bk**2 + ck**2)


def main(A0, B0, C0, n, s):
    # Initialize
    A_k, B_k, C_k = np.array(A0), np.array(B0), np.array(C0)
    vertices = [(A_k, B_k, C_k)]
    sides = []
    shape_functions = []

    zeta = np.exp(2 * np.pi * 1j / 3)

    for k in range(1, n + 1):
        xk = s[k - 1] / (1 - s[k - 1])

        # Compute new vertices
        A_k, B_k, C_k = compute_vertices(
            vertices[-1][0], vertices[-1][1], vertices[-1][2], xk
        )
        vertices.append((A_k, B_k, C_k))

        # Compute sides
        ak = compute_side_length(B_k, C_k)
        bk = compute_side_length(A_k, C_k)
        ck = compute_side_length(A_k, B_k)
        sides.append((ak, bk, ck))

        # Compute shape function
        sigma_k = compute_shape_function(ak, bk, ck, zeta)
        shape_functions.append(sigma_k)

        # Check condition
        if k > 1:
            sigma_prev = shape_functions[-2]
            if not np.isclose(
                sigma_k, s[k - 1] + zeta * (s[k - 1] + zeta * sigma_prev)
            ):
                print(f"Condition failed at iteration {k}. Aborting.")
                return vertices, sides, shape_functions

    return vertices, sides, shape_functions


def visualize(vertices, n):
    plt.figure()
    for i in range(min(n, len(vertices))):
        A_k, B_k, C_k = vertices[i]
        triangle = np.array([A_k, B_k, C_k, A_k])
        plt.plot(triangle[:, 0], triangle[:, 1], label=f"Iteration {i}")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Triangle Iterations")
    plt.show()


# Example input
A0 = (0, 0)
B0 = (1, 0)
C0 = (0.5, np.sqrt(3) / 2)
n = 5
s = [0.3, 0.4, 0.5, 0.6, 0.7]

vertices, sides, shape_functions = main(A0, B0, C0, n, s)
visualize(vertices, n)
"""
