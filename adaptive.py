import numpy as np

# Constants
BICTCP_BETA_SCALE = 1024
BICTCP_HZ = 10
HZ = 1000
BITS_PER_LONG = 64

# Original final1 function
def final1(last_max_cwnd, rtt, x=10):
    return (last_max_cwnd - (410 * ((cubic_root((1 << (10 + 3 * x)) // 410 * (last_max_cwnd - (717 * last_max_cwnd // 1024))) - ((1 << 10) * rtt // 1000)) ** 3) >> (10 + 3 * x)))

# Cubic root function (from earlier)
def cubic_root(a):
    v = [
        0, 54, 54, 54, 118, 118, 118, 118, 123, 129, 134, 138, 143, 147, 151, 156,
        157, 161, 164, 168, 170, 173, 176, 179, 181, 185, 187, 190, 192, 194, 197, 199,
        200, 202, 204, 206, 209, 211, 213, 215, 217, 219, 221, 222, 224, 225, 227, 229,
        231, 232, 234, 236, 237, 239, 240, 242, 244, 245, 246, 248, 250, 251, 252, 254
    ]
    b = fls64(a)
    if b < 7:
        return (v[a] + 35) >> 6
    b = ((b * 84) >> 8) - 1
    shift = (a >> (b * 3))
    x = ((v[shift] + 10) << b) >> 6
    x = (2 * x + (a // (x * (x - 1))))
    x = (x * 341) >> 10
    return x

# Function to find the last set bit in a 64-bit integer
def fls64(x):
    if x == 0:
        return 0
    return __fls(x) + 1

def __fls(word):
    num = BITS_PER_LONG - 1
    if BITS_PER_LONG == 64:
        if not (word & (~0 << 32)):
            num -= 32
            word <<= 32
    if not (word & (~0 << (BITS_PER_LONG - 16))):
        num -= 16
        word <<= 16
    if not (word & (~0 << (BITS_PER_LONG - 8))):
        num -= 8
        word <<= 8
    if not (word & (~0 << (BITS_PER_LONG - 4))):
        num -= 4
        word <<= 4
    if not (word & (~0 << (BITS_PER_LONG - 2))):
        num -= 2
        word <<= 2
    if not (word & (~0 << (BITS_PER_LONG - 1))):
        num -= 1
    return num

# Adaptive multivariable linear piecewise approximation
def adaptive_multivariable_approximation(last_max_cwnd_min, last_max_cwnd_max, rtt_min, rtt_max, max_segments=20, error_threshold=1e-3):
    # Initialize segments
    segments = [((last_max_cwnd_min, last_max_cwnd_max), (rtt_min, rtt_max))]
    approximations = []

    while len(segments) < max_segments:
        errors = []
        new_segments = []

        for (x_start, x_end), (y_start, y_end) in segments:
            # Generate grid points for this segment
            x_values = np.linspace(x_start, x_end, 10, dtype=int)
            y_values = np.linspace(y_start, y_end, 10, dtype=int)
            X, Y = np.meshgrid(x_values, y_values)
            Z = np.array([[final1(int(x), int(y)) for x in x_values] for y in y_values])

            # Flatten the grid for regression
            X_flat = X.flatten()
            Y_flat = Y.flatten()
            Z_flat = Z.flatten()

            # Fit a plane z = a*x + b*y + c
            A = np.vstack([X_flat, Y_flat, np.ones_like(X_flat)]).T
            a, b, c = np.linalg.lstsq(A, Z_flat, rcond=None)[0]

            # Calculate the approximation error
            approx_Z = a * X + b * Y + c
            error = np.max(np.abs(Z - approx_Z))
            errors.append((error, (x_start, x_end), (y_start, y_end), a, b, c))

            # If error is too high, split the segment
            if error > error_threshold:
                x_mid = (x_start + x_end) // 2
                y_mid = (y_start + y_end) // 2
                new_segments.append(((x_start, x_mid), (y_start, y_mid)))
                new_segments.append(((x_mid, x_end), (y_start, y_mid)))
                new_segments.append(((x_start, x_mid), (y_mid, y_end)))
                new_segments.append(((x_mid, x_end), (y_mid, y_end)))
            else:
                new_segments.append(((x_start, x_end), (y_start, y_end)))

        # Find the segment with the worst error
        worst_error, worst_x_range, worst_y_range, worst_a, worst_b, worst_c = max(errors, key=lambda x: x[0])

        # If the worst error is below the threshold, stop
        if worst_error <= error_threshold:
            break

        # Update segments
        segments = new_segments

    # Collect the final approximations
    for (x_start, x_end), (y_start, y_end) in segments:
        x_values = np.linspace(x_start, x_end, 10, dtype=int)
        y_values = np.linspace(y_start, y_end, 10, dtype=int)
        X, Y = np.meshgrid(x_values, y_values)
        Z = np.array([[final1(int(x), int(y)) for x in x_values] for y in y_values])

        # Flatten the grid for regression
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = Z.flatten()

        # Fit a plane z = a*x + b*y + c
        A = np.vstack([X_flat, Y_flat, np.ones_like(X_flat)]).T
        a, b, c = np.linalg.lstsq(A, Z_flat, rcond=None)[0]
        approximations.append(((x_start, x_end), (y_start, y_end), a, b, c))

    return approximations

# Function to evaluate accuracy
def evaluate_accuracy(approximations, last_max_cwnd_min, last_max_cwnd_max, rtt_min, rtt_max):
    x_values = np.linspace(last_max_cwnd_min, last_max_cwnd_max, 100, dtype=int)
    y_values = np.linspace(rtt_min, rtt_max, 100, dtype=int)
    abs_errors = []
    percent_errors = []

    for x in x_values:
        for y in y_values:
            # Find the correct segment for (x, y)
            for (x_start, x_end), (y_start, y_end), a, b, c in approximations:
                if x_start <= x <= x_end and y_start <= y <= y_end:
                    # Compute original and approximated values
                    original = final1(x, y)
                    approx = a * x + b * y + c
                    # Compute errors
                    abs_error = abs(original - approx)
                    percent_error = (abs_error / original) * 100 if original != 0 else 0
                    # Store errors
                    abs_errors.append(abs_error)
                    percent_errors.append(percent_error)
                    break

    # Compute average errors
    avg_abs_error = np.mean(abs_errors)
    avg_percent_error = np.mean(percent_errors)

    return avg_abs_error, avg_percent_error, abs_errors, percent_errors

# Example usage
last_max_cwnd_min = 1
last_max_cwnd_max = 10000
rtt_min = 1
rtt_max = 1000
approximations = adaptive_multivariable_approximation(last_max_cwnd_min, last_max_cwnd_max, rtt_min, rtt_max)

# Evaluate accuracy
avg_abs_error, avg_percent_error, abs_errors, percent_errors = evaluate_accuracy(approximations, last_max_cwnd_min, last_max_cwnd_max, rtt_min, rtt_max)

# Print the linear approximations
for (x_start, x_end), (y_start, y_end), a, b, c in approximations:
    print(f"Range: last_max_cwnd [{x_start}, {x_end}], rtt [{y_start}, {y_end}], Approximation: z = {a:.4f}x + {b:.4f}y + {c:.4f}")

# Print accuracy metrics
print(f"\nAverage Absolute Error: {avg_abs_error:.4f}")
print(f"Average Percentage Error: {avg_percent_error:.4f}%")
print(f"Maximum Absolute Error: {max(abs_errors):.4f}")
print(f"Maximum Percentage Error: {max(percent_errors):.4f}%")