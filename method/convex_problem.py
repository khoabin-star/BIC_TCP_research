import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# ---------------------- Setup and Function Definition ----------------------
x = sp.symbols('x')
f = x**2 + 1           # True function
f_prime = sp.diff(f, x)

# Lambdify true function and derivative for numeric use
f_func = sp.lambdify(x, f, 'numpy')
f_prime_func = sp.lambdify(x, f_prime, 'numpy')

# Thresholds and globals
MAPE_threshold = 20      # in percent
min_interval_length = 5  # minimum interval size to subdivide
iteration_count = 0      # global iteration counter


def compute_errors(f_true, f_approx, a, b, num_points=100):
    """Compute worst-case MSE, MAE, and MAPE on [a,b]."""
    xs = np.linspace(a, b, num_points)
    true_vals = f_true(xs)
    approx_vals = f_approx(xs)
    sq_errors = (approx_vals - true_vals)**2
    abs_errors = np.abs(approx_vals - true_vals)
    ape = abs_errors / np.maximum(np.abs(true_vals), 1e-6) * 100
    return np.max(sq_errors), np.max(abs_errors), np.max(ape)


def find_best_tangent(a, b, num_candidates=50):
    """
    Search for the tangent line y=f(x0)+f'(x0)*(x-x0) within [a,b]
    that minimizes max MAPE over [a,b].
    Returns (expr, best_mape, best_x0).
    """
    xs = np.linspace(a, b, num_candidates)
    # print(f"Candidate x0 values ${xs} in range [{a:.2f}, {b:.2f}]")
    best_mape = np.inf
    best_expr = None
    best_x0 = None
    for x0 in xs:
        y0 = f_func(x0)
        slope = f_prime_func(x0)
        expr = y0 + slope*(x - x0)
        tangent_fn = sp.lambdify(x, expr, 'numpy')
        _, _, mape = compute_errors(f_func, tangent_fn, a, b)
        # print(f"x0={x0:.2f}, MAPE={mape:.2f}%")
        if mape < best_mape:
            best_mape = mape
            best_expr = expr
            best_x0 = x0
    # print(f"Best tangent at x0={best_x0:.2f} with MAPE={best_mape:.2f}%")
    return best_expr, best_mape, best_x0

def _ensure_array(y, xs):
    """Ensure y is an array matching shape of xs (for constant outputs)."""
    y_arr = np.asarray(y)
    if y_arr.shape != xs.shape:
        y_arr = np.full_like(xs, float(y_arr))
    return y_arr

def plot_subregion(a, b, c, f_a, f_c, f_b,
                   LL_expr, UL_expr, LR_expr, UR_expr):
    """Plot true function, best tangents (lower) and secants (upper)."""
    global iteration_count
    iteration_count += 1
    xs = np.linspace(a, b, 300)

    # lambdify
    true_y = f_func(xs)
    LL_y = _ensure_array(sp.lambdify(x, LL_expr, 'numpy')(xs), xs)
    UL_y = _ensure_array(sp.lambdify(x, UL_expr, 'numpy')(xs), xs)
    LR_y = _ensure_array(sp.lambdify(x, LR_expr, 'numpy')(xs), xs)
    UR_y = _ensure_array(sp.lambdify(x, UR_expr, 'numpy')(xs), xs)

    plt.figure(figsize=(8,5))
    plt.plot(xs, true_y, label='True f(x)=x^2')
    plt.plot(xs, LL_y, '--', label='Best Lower Tangent')
    plt.plot(xs, LR_y, '--', label='Best Lower Tangent Right')
    plt.plot(xs, UL_y, ':', label='Upper Secant Left')
    plt.plot(xs, UR_y, ':', label='Upper Secant Right')

    # key points
    for xv, yv in zip([a, c, b], [f_a, f_c, f_b]):
        plt.axvline(xv, color='gray', linewidth=0.8)
        plt.scatter([xv], [yv], color='red')

    plt.title(f"Iteration {iteration_count}: Region [{a:.2f}, {b:.2f}]")
    plt.xlabel('x'); plt.ylabel('y')
    plt.legend(); plt.grid(True)
    plt.show()


def process_subregion(a_val, b_val, threshold):
    """
    Split [a,b] at midpoint c.
    Upper bounds: secants. Lower bounds: best tangent in each half.
    Recurse if max MAPE > threshold.
    Returns list of subregion dicts with report info.
    """
    subregions = []
    c = (a_val + b_val) / 2
    # function values
    f_a = float(f.subs(x, a_val))
    f_c = float(f.subs(x, c))
    f_b = float(f.subs(x, b_val))
    # upper secants
    slope_ul = (f_c - f_a) / (c - a_val)
    UL_expr = f_a + slope_ul * (x - a_val)
    slope_ur = (f_b - f_c) / (b_val - c)
    UR_expr = f_c + slope_ur * (x - c)
    # best lower tangents with location
    LL_expr, mape_ll, x0_ll = find_best_tangent(a_val, c)
    LR_expr, mape_lr, x0_lr = find_best_tangent(c, b_val)
    # evaluate upper MAPE
    _, _, mape_ul = compute_errors(f_func, sp.lambdify(x, UL_expr, 'numpy'), a_val, c)
    _, _, mape_ur = compute_errors(f_func, sp.lambdify(x, UR_expr, 'numpy'), c, b_val)
    left_max_mape  = max(mape_ll, mape_ul)
    right_max_mape = max(mape_lr, mape_ur)
    # plot iteration
    plot_subregion(a_val, b_val, c, f_a, f_c, f_b,
                   LL_expr, UL_expr, LR_expr, UR_expr)
    # build subregion info
    left = {
        'range': (a_val, c),
        'lower_expr': LL_expr,
        'lower_x0': x0_ll,
        'upper_expr': UL_expr,
        'mape': left_max_mape
    }
    right = {
        'range': (c, b_val),
        'lower_expr': LR_expr,
        'lower_x0': x0_lr,
        'upper_expr': UR_expr,
        'mape': right_max_mape
    }
    # recurse or collect
    if left_max_mape > threshold and (c - a_val) > min_interval_length:
        subregions.extend(process_subregion(a_val, c, threshold))
    else:
        subregions.append(left)
    if right_max_mape > threshold and (b_val - c) > min_interval_length:
        subregions.extend(process_subregion(c, b_val, threshold))
    else:
        subregions.append(right)
    return subregions

# ------------------------- Run -------------------------
if __name__ == '__main__':
    final = process_subregion(-100, 100, MAPE_threshold)
    print(f"Produced {len(final)} subregions.")
    for idx, reg in enumerate(final, 1):
        a, b = reg['range']
        print(f"Subregion {idx}: Range [{a:.2f}, {b:.2f}]")
        print(f"  Max MAPE: {reg['mape']:.2f}%")
        print(f"  Lower tangent at x0 = {reg['lower_x0']:.2f}")
        print(f"    Lower bound expr: {sp.pretty(reg['lower_expr'])}")
        print(f"  Upper secant expr: {sp.pretty(reg['upper_expr'])}\n")
