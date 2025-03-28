import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import Axes3D

###############################################################################
# 1. TARGET FUNCTION
###############################################################################
def f(x, y):
    """
    The function to approximate.
    """
    return x - 0.4 * np.float_power(np.float_power(0.75 * x, 1.0 / 3.0) - y / 1000.0, 3)

###############################################################################
# 2. GENERATE A RECTANGULAR GRID
###############################################################################
def generate_initial_grid(x_range, y_range, num_x, num_y):
    """
    Creates an initial rectangular grid covering the given domain.
    """
    x_vals = np.linspace(x_range[0], x_range[1], num_x)
    y_vals = np.linspace(y_range[0], y_range[1], num_y)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    points = np.column_stack((X.ravel(), Y.ravel()))
    return X, Y, points

###############################################################################
# 3. COMPUTE PLANE COEFFICIENTS
###############################################################################
def compute_plane_coefficients(x_vals, y_vals, f_vals):
    """
    Computes the coefficients (a0, a1, a2) of the plane equation:
    f(x, y) ≈ a0 + a1*x + a2*y
    """
    x1, x2 = x_vals
    y1, y2 = y_vals
    f11, f12, f21, f22 = f_vals

    # Solve for a0, a1, a2 using the least squares method
    A = np.array([
        [1, x1, y1],
        [1, x1, y2],
        [1, x2, y1],
        [1, x2, y2]
    ])
    b = np.array([f11, f12, f21, f22])
    
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)  # Least squares solution
    return coeffs  # (a0, a1, a2)

def plane_interpolation(a0, a1, a2, x, y):
    """
    Evaluates the plane function at (x, y).
    """
    return a0 + a1 * x + a2 * y

###############################################################################
# 4. COMPUTE ERROR
###############################################################################
def compute_errors(X, Y, f_values):
    """
    Computes the approximation error for each rectangular cell.
    """
    num_x, num_y = X.shape
    errors = np.zeros((num_x - 1, num_y - 1))

    for i in range(num_x - 1):
        for j in range(num_y - 1):
            x_vals = (X[i, j], X[i+1, j])
            y_vals = (Y[i, j], Y[i, j+1])
            f_vals = (f_values[i, j], f_values[i, j+1], f_values[i+1, j], f_values[i+1, j+1])

            # Compute plane coefficients
            a0, a1, a2 = compute_plane_coefficients(x_vals, y_vals, f_vals)

            # Compute error at the center of the rectangle
            x_mid = (x_vals[0] + x_vals[1]) / 2
            y_mid = (y_vals[0] + y_vals[1]) / 2
            f_true = f(x_mid, y_mid)
            f_approx = plane_interpolation(a0, a1, a2, x_mid, y_mid)
            errors[i, j] = abs(f_true - f_approx) / (abs(f_true) + 1e-6) * 100  # Percentage error

    return errors

###############################################################################
# 5. REFINEMENT STRATEGY
###############################################################################
def refine_grid(X, Y, errors, error_threshold):
    """
    Refines the grid by adding more points in high-error regions.
    """
    num_x, num_y = X.shape
    refine_mask = errors > error_threshold

    new_x = []
    new_y = []

    for i in range(num_x - 1):
        for j in range(num_y - 1):
            if refine_mask[i, j]:
                # Add midpoint
                x_mid = (X[i, j] + X[i+1, j]) / 2
                y_mid = (Y[i, j] + Y[i, j+1]) / 2
                new_x.append(x_mid)
                new_y.append(y_mid)

    if not new_x:
        return X, Y, False

    new_x = np.unique(np.array(new_x))
    new_y = np.unique(np.array(new_y))

    X_refined, Y_refined = np.meshgrid(np.sort(np.concatenate([X[:, 0], new_x])),
                                       np.sort(np.concatenate([Y[0, :], new_y])),
                                       indexing='ij')
    return X_refined, Y_refined, True

###############################################################################
# 6. MAIN ADAPTIVE LOOP
###############################################################################
def adaptive_rectangular_approximation(x_range, y_range, num_x, num_y, error_threshold, max_iterations):
    """
    Performs adaptive piecewise linear approximation using rectangular cells.
    """
    X, Y, points = generate_initial_grid(x_range, y_range, num_x, num_y)

    f_values = f(X, Y)
    errors = compute_errors(X, Y, f_values)
    max_err = np.max(errors)
    print(f"[Iteration 0] Max error: {max_err:.3f}%")

    for iteration in range(max_iterations):
        plt.figure(figsize=(15,10))
        for i in range(X.shape[0]-1):
            for j in range(Y.shape[1]-1):
                x_mid = (X[i, j] + X[i+1, j]) / 2
                y_mid = (Y[i, j] + Y[i, j+1]) / 2
                plt.text(x_mid, y_mid, f'{errors[i, j]:.0f}%', ha='center', va='center', fontsize=8)

        plt.plot(X, Y, color='black', linewidth=0.5)
        plt.plot(X.T, Y.T, color='black', linewidth=0.5)
        plt.title(f"Error value of each rectangle in Iteration {iteration+1}, Max error: {max_err:.3f}%")
        plt.xlabel("cwnd")
        plt.ylabel("rtt")
        plt.show()

        if max_err < error_threshold:
            print("Error below threshold; stopping refinement.")
            break

        X, Y, did_refine = refine_grid(X, Y, errors, error_threshold)
        if not did_refine:
            print("No refinement performed; stopping.")
            break

        f_values = f(X, Y)
        errors = compute_errors(X, Y, f_values)
        max_err = np.max(errors)
        print(f"[Iteration {iteration+1}] Max error: {max_err:.3f}%")

    num_rectangles = (X.shape[0] - 1) * (Y.shape[1] - 1)
    print(f"Total number of rectangles: {X.shape[0] - 1} * {Y.shape[1] - 1} = {num_rectangles}")

    return X, Y, f_values, errors

###############################################################################
# 7. EXECUTION EXAMPLE
###############################################################################
if __name__ == "__main__":
    # Domain + parameters
    parser = argparse.ArgumentParser(description='Adaptive Piecewise Linear Approximation')
    parser.add_argument('--error', type=float, default=20, help='Percentage error threshold for refinement')
    parser.add_argument('--cwnd', type=int, default=100, help='Maximum value for last_max_cwnd range')
    parser.add_argument('--rtt', type=int, default=10000, help='Maximum value for rtt range')
    parser.add_argument('--max_iterations', type=int, default=10, help='Maximum number of iterations')
    args = parser.parse_args()
    x_range = [1, args.cwnd]
    y_range = [1, args.rtt]
    error_threshold = args.error
    max_iterations = args.max_iterations
    num_x, num_y = 3, 3

    print(f"x range: {x_range}")
    print(f"y range: {y_range}")
    print(f"Error threshold: {error_threshold}%")
    print(f"max iterations: {max_iterations}")
    print(f"Initial number of rectangles: {num_x - 1} * {num_y - 1} = {(num_x-1)*(num_y-1)}")

    # Run adaptive piecewise linear approximation
    X, Y, f_values, errors = adaptive_rectangular_approximation(x_range, y_range, num_x, num_y, error_threshold, max_iterations)


    #============ Another 2D Visualization of Error ============#
    plt.figure(figsize=(15,10))
    plt.pcolormesh(X, Y, errors, shading='auto', cmap='coolwarm')
    plt.colorbar(label="Error (%)")
    plt.scatter(X, Y, color='black', s=10)  # Show grid points
    plt.title("Error color of each rectangle")
    plt.xlabel("cwnd")
    plt.ylabel("rtt")
    plt.show()




    #============ 3D Visualization of Approximation ============#
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, f_values, cmap='viridis', edgecolor='k', alpha=0.8)
    ax.set_title("Piecewise Linear Approximation with Rectangles")
    ax.set_xlabel("cwnd")
    ax.set_ylabel("rtt")
    ax.set_zlabel("f(cwnd,rtt)")
    plt.show()
