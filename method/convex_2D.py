import heapq
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

# ---------------------- Setup ----------------------
x1, x2 = sp.symbols('x1 x2')
f = x1**2 + x2**2 + 1
f_true = sp.lambdify((x1, x2), f, 'numpy')

grad_f    = [sp.diff(f, x1), sp.diff(f, x2)]
grad_func = [sp.lambdify((x1, x2), g, 'numpy') for g in grad_f]

MaxAPE_threshold   = 20     # in percent
default_candidates = 10     # grid size for tangent search
min_edge_length    = 1.0    # minimum rectangle side length to split further

split_history = []
region_list   = []

# ---------------------- Core Helpers ----------------------

def compute_MaxAPE_plane(expr_fn, a1, b1, a2, b2, grid_points=20):
    xs1 = np.linspace(a1, b1, grid_points)
    xs2 = np.linspace(a2, b2, grid_points)
    X1, X2 = np.meshgrid(xs1, xs2)
    true_vals   = f_true(X1, X2)
    approx_vals = expr_fn(X1, X2)
    ape         = np.abs(approx_vals - true_vals) / np.maximum(np.abs(true_vals), 1e-6) * 100
    return np.max(ape)

def compute_MaxAE_plane(expr_fn, a1, b1, a2, b2, grid_points=20):
    xs1 = np.linspace(a1, b1, grid_points)
    xs2 = np.linspace(a2, b2, grid_points)
    X1, X2 = np.meshgrid(xs1, xs2)
    true_vals   = f_true(X1, X2)
    approx_vals = expr_fn(X1, X2)
    ae = np.abs(approx_vals - true_vals)
    return np.max(ae)


def find_best_tangent_plane(a1, b1, a2, b2, candidates=default_candidates):
    xs1 = np.linspace(a1, b1, candidates)
    xs2 = np.linspace(a2, b2, candidates)
    best_MaxAPE = np.inf
    best_expr   = None
    best_point  = None

    for x10 in xs1:
        for x20 in xs2:
            f0 = f_true(x10, x20)
            g1 = grad_func[0](x10, x20)
            g2 = grad_func[1](x10, x20)
            plane   = f0 + g1*(x1 - x10) + g2*(x2 - x20)
            plane_fn = sp.lambdify((x1, x2), plane, 'numpy')
            err = compute_MaxAPE_plane(plane_fn, a1, b1, a2, b2)
            if err < best_MaxAPE:
                best_MaxAPE, best_expr, best_point = err, plane, (x10, x20)

    return best_expr, best_MaxAPE, best_point

def find_upper_plane(a1, b1, a2, b2):
    corners = [(a1, a2), (a1, b2), (b1, a2), (b1, b2)]
    vals = [(pt, f_true(pt[0], pt[1])) for pt in corners]
    top3 = sorted(vals, key=lambda x: x[1], reverse=True)[:3]

    alpha, beta, gamma = sp.symbols('alpha beta gamma')
    eqs = [alpha*x + beta*y + gamma - z for (x, y), z in top3]
    sol = sp.solve(eqs, (alpha, beta, gamma))
    plane = sol[alpha]*x1 + sol[beta]*x2 + sol[gamma]

    plane_fn = sp.lambdify((x1, x2), plane, 'numpy')
    err = compute_MaxAPE_plane(plane_fn, a1, b1, a2, b2)
    return plane, err

# ---------------------- Error‐Variation Driven Split ----------------------

def sample_ape(expr_fn, a1, b1, a2, b2, grid_points=40):
    xs1 = np.linspace(a1, b1, grid_points)
    xs2 = np.linspace(a2, b2, grid_points)
    X1, X2 = np.meshgrid(xs1, xs2)
    true_vals   = f_true(X1, X2)
    ape         = np.abs(expr_fn(X1, X2) - true_vals) / np.maximum(np.abs(true_vals), 1e-6) * 100
    return ape

def compute_error_variation(mape_grid, axis):
    avg = mape_grid.mean(axis=axis)
    return np.abs(np.diff(avg)).sum()

def choose_split_axis(lower_plane, upper_plane, a1, b1, a2, b2):
    up_fn  = sp.lambdify((x1, x2), upper_plane, 'numpy')
    low_fn = sp.lambdify((x1, x2), lower_plane, 'numpy')
    grid_up  = sample_ape(up_fn,  a1, b1, a2, b2)
    grid_low = sample_ape(low_fn, a1, b1, a2, b2)

    var_x1 = compute_error_variation(grid_up,   axis=0) + \
             compute_error_variation(grid_low,  axis=0)
    var_x2 = compute_error_variation(grid_up,   axis=1) + \
             compute_error_variation(grid_low,  axis=1)

    return 'x1' if var_x1 >= var_x2 else 'x2'

# ---------------------- Packing MaxAPE + Planes ----------------------

def maxape_info(a1, b1, a2, b2):
    low_pl, low_err, low_pt = find_best_tangent_plane(a1, b1, a2, b2)
    up_pl,  up_err          = find_upper_plane(a1, b1, a2, b2)
    return max(low_err, up_err), low_pl, up_pl, low_pt

# ---------------------- Final 3D plot ----------------------

def plot_final_approximation(final_leaves):
    pts = 80
    xs = np.linspace(-10, 10, pts)
    ys = np.linspace(-10, 10, pts)
    X, Y = np.meshgrid(xs, ys)
    Z_true = f_true(X, Y)

    fig = plt.figure(figsize=(10,8))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, Z_true,
                      color='gray', linewidth=0.5, rcount=30, ccount=30, alpha=0.7)

    for rep in final_leaves:
        a1, b1, a2, b2 = rep['range']
        low_fn = sp.lambdify((x1, x2), rep['lower_expr'], 'numpy')
        up_fn  = sp.lambdify((x1, x2), rep['upper_expr'], 'numpy')

        pts_r = 4
        xs_r  = np.linspace(a1, b1, pts_r)
        ys_r  = np.linspace(a2, b2, pts_r)
        Xr, Yr = np.meshgrid(xs_r, ys_r)
        Zl = low_fn(Xr, Yr)
        Zu = up_fn(Xr, Yr)

        ax.plot_surface(Xr, Yr, Zl, color='blue',  alpha=0.3, linewidth=0, antialiased=False)
        ax.plot_wireframe(Xr, Yr, Zu, color='red', linewidth=1, rcount=pts_r, ccount=pts_r, alpha=0.8)

    ax.set_title("Final Piecewise Linear Bounds\n"
                 "(True f = gray wireframe, lower = blue fill, upper = red edges)")
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("f(x)")
    plt.tight_layout()
    plt.show()

# ------------------------- Run & Report -------------------------

if __name__ == '__main__':
    # Initialize best‐first heap
    heap = []
    leaves = []
    root = (-10, 10, -10, 10)
    region_list = [root]

    err, lp, up, pt = maxape_info(*root)
    heapq.heappush(heap, (-err, root, lp, up, pt))

    # Iterate splits
    while heap:
        neg_err, (a1,b1,a2,b2), low_pl, up_pl, low_pt = heapq.heappop(heap)
        err = -neg_err

        # Stop if below threshold or tiny
        if err <= MaxAPE_threshold or (b1-a1 <= min_edge_length and b2-a2 <= min_edge_length):
            leaves.append(((a1,b1,a2,b2), err, low_pl, up_pl, low_pt))
            continue

        # Choose split
        axis = choose_split_axis(low_pl, up_pl, a1, b1, a2, b2)
        mid  = (a1+b1)/2 if axis=='x1' else (a2+b2)/2
        split_history.append((axis, a1,b1,a2,b2, mid))

        # Update active regions
        region_list.remove((a1,b1,a2,b2))
        if axis=='x1':
            children = [(a1,mid,a2,b2), (mid,b1,a2,b2)]
        else:
            children = [(a1,b1,a2,mid), (a1,b1,mid,b2)]
        region_list.extend(children)

        # — 2D step visualization with MaxAPE labels —
        fig, ax = plt.subplots(figsize=(10,8))
        patches, errors = [], []

        # Build patches and collect errors
        for (u1, v1, u2, v2) in region_list:
            patches.append(Rectangle((u1, u2), v1 - u1, v2 - u2))
            c_err, *_ = maxape_info(u1, v1, u2, v2)
            errors.append(c_err)

        # Color the rectangles by error
        pc = PatchCollection(
            patches,
            cmap='coolwarm',
            edgecolor='black',
            linewidth=0.5,
            alpha=0.6
        )
        pc.set_array(np.array(errors))
        ax.add_collection(pc)

        # Annotate each rectangle with its MaxAPE
        for (u1, v1, u2, v2), c_err in zip(region_list, errors):
            cx = 0.5 * (u1 + v1)
            cy = 0.5 * (u2 + v2)
            ax.text(cx, cy, f"{c_err:.1f}%", 
                    ha='center', va='center',
                    color='white', fontsize=8, weight='bold')

        # Draw all split lines so far
        for axn, xa1, xb1, ya1, yb1, m in split_history:
            if axn == 'x1':
                ax.plot([m, m], [ya1, yb1], 'r--', linewidth=1.5)
            else:
                ax.plot([xa1, xb1], [m, m], 'r--', linewidth=1.5)

        # Finalize
        fig.colorbar(pc, ax=ax, label='MaxAPE (%)')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal', 'box')
        ax.set_title(f"Split {axis}@{mid:.2f}")
        ax.set_xlabel('x1'); ax.set_ylabel('x2')
        plt.tight_layout()
        plt.show()

        # Push children onto heap
        for child in children:
            c_err, c_lp, c_up, c_pt = maxape_info(*child)
            heapq.heappush(heap, (-c_err, child, c_lp, c_up, c_pt))

    # Build final_leaves for downstream plotting & report
    final_leaves = [{
        'range':       r,
        'MaxAPE':      e,
        'lower_point': pt,
        'lower_expr':  lp,
        'upper_expr':  up
    } for (r,e,lp,up,pt) in leaves]

    # 2D error‐map of final leaves
    fig2, ax2 = plt.subplots(figsize=(8,8))
    patches_list, errors = [], []
    for rep in final_leaves:
        a1, b1, a2, b2 = rep['range']
        patches_list.append(Rectangle((a1,a2), b1-a1, b2-a2))
        errors.append(rep['MaxAPE'])
    pc2 = PatchCollection(patches_list, array=np.array(errors),
                          cmap='coolwarm', edgecolor='black', linewidth=0.5)
    ax2.add_collection(pc2)
    tx = [pt[0] for pt in (r['lower_point'] for r in final_leaves)]
    ty = [pt[1] for pt in (r['lower_point'] for r in final_leaves)]
    ax2.scatter(tx, ty, c='k', s=10, label='tangent pts')
    fig2.colorbar(pc2, ax=ax2, label='MaxAPE (%)')
    ax2.set_xlim(-10,10); ax2.set_ylim(-10,10)
    ax2.set_xlabel('x1'); ax2.set_ylabel('x2')
    ax2.set_title('Final subregions colored by MaxAPE')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    # 3D final enclosure
    plot_final_approximation(final_leaves)

    # Text report
    print(f"\nGot {len(final_leaves)} final subregions:\n")
    for i, rep in enumerate(final_leaves, 1):
        a1, b1, a2, b2 = rep['range']
        print(f"Region {i}: x1∈[{a1:.2f},{b1:.2f}], x2∈[{a2:.2f},{b2:.2f}]  "
              f"MaxAPE={rep['MaxAPE']:.2f}%  "
              f"Tangent@({rep['lower_point'][0]:.3f},{rep['lower_point'][1]:.3f})")
        print("  Lower plane:", sp.pretty(rep['lower_expr']))
        print("  Upper plane:", sp.pretty(rep['upper_expr']))
        print("-"*60)
