import sys
import heapq
import numpy as np
import sympy as sp
from pycparser import parse_file, c_ast
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from functools import lru_cache

# ---------------------- Symbolic & C-Parser Setup ----------------------
x1, x2 = sp.symbols('x1 x2')

class MathFuncExtractor(c_ast.NodeVisitor):
    """Extracts and converts the C 'math' function into Sympy."""
    def __init__(self):
        self.sympy_expr = None
    def visit_FuncDef(self, node):
        if node.decl.name == 'math':
            for stmt in node.body.block_items:
                if isinstance(stmt, c_ast.Return):
                    self.sympy_expr = self._to_sympy(stmt.expr)
                    return
    def _to_sympy(self, node):
        if isinstance(node, c_ast.BinaryOp):
            L, R = self._to_sympy(node.left), self._to_sympy(node.right)
            return {'+': L+R, '-': L-R, '*': L*R, '/': L/R}[node.op]
        if isinstance(node, c_ast.Constant):
            return sp.Float(node.value)
        if isinstance(node, c_ast.ID):
            return {'x1': x1, 'x2': x2}[node.name]
        if isinstance(node, c_ast.FuncCall):
            fn = node.name.name
            args = [self._to_sympy(a) for a in node.args.exprs]
            if fn == 'pow': return args[0]**int(args[1])
            elif fn == 'cbrt': return args[0] ** sp.Rational(1,3)
            raise NotImplementedError(f"Unsupported '{fn}'")
        raise NotImplementedError(type(node))

def load_c_math_function(filename: str) -> sp.Expr:
    ast = parse_file(filename, use_cpp=True)
    ext = MathFuncExtractor()
    ext.visit(ast)
    if ext.sympy_expr is None:
        raise ValueError("No 'math' function found.")
    return ext.sympy_expr

# ---------------------- Dynamic Function Loading ----------------------
if len(sys.argv) < 2:
    print("Usage: python script.py <math_c_file.c>")
    sys.exit(1)

c_file = sys.argv[1]
t_f_expr = load_c_math_function(c_file)
f_true = sp.lambdify((x1, x2), t_f_expr, 'numpy')
grad_syms = [sp.diff(t_f_expr, v) for v in (x1, x2)]
grad_funcs = [sp.lambdify((x1, x2), g, 'numpy') for g in grad_syms]

# ---------------------- Global Parameters ----------------------
X_MIN, X_MAX = 1.00, 100.00
Y_MIN, Y_MAX = 1.00, 10000.00
FULL_AREA = (X_MAX - X_MIN) * (Y_MAX - Y_MIN)

MaxAPE_threshold  = 20.0   # percent
MaxAE_threshold   = 20.0   # absolute error
Percentile_cutoff = 10     # percent for AE vs APE mixing
ERROR_GRID_POINTS = 100    # base grid resolution
MIN_GRID          = 20     # minimum grid resolution
CANDIDATE_GRID    = 10     # tangent sampling per axis
SPLIT_GRID_POINTS = 40     # split-axis sampling resolution
split_history     = []

# ---------------------- Lambdify Cache ----------------------
_plane_fn_cache = {}
def get_plane_fn(expr):
    key = str(expr)
    if key not in _plane_fn_cache:
        _plane_fn_cache[key] = sp.lambdify((x1, x2), expr, 'numpy')
    return _plane_fn_cache[key]

# ---------------------- Core Error Metrics ----------------------
def compute_max_errors_plane(expr_fn, a1, b1, a2, b2, threshold=None):
    """
    Dynamic grid density + early-stop row-wise scan.
    """
    # dynamic grid sizing
    region_area = (b1 - a1) * (b2 - a2)
    ratio = np.clip(region_area / FULL_AREA, 0.0, 1.0)
    gp = max(MIN_GRID, int(ERROR_GRID_POINTS * np.sqrt(ratio)))

    # coarse percentile threshold estimation on small grid
    cg = min(10, gp)
    Xc, Yc = np.meshgrid(
        np.linspace(a1, b1, cg), np.linspace(a2, b2, cg)
    )
    flat_true = np.abs(f_true(Xc, Yc)).ravel()
    thresh = np.percentile(flat_true, Percentile_cutoff)

    maxAPE, maxAE = 0.0, 0.0
    xs = np.linspace(a1, b1, gp)
    xs2 = np.linspace(a2, b2, gp)
    for x in xs:
        tvals = f_true(x, xs2)
        avals = expr_fn(x, xs2)
        abs_err = np.abs(avals - tvals)
        pct_err = abs_err / np.maximum(np.abs(tvals), 1e-6) * 100
        ape_row = np.where(np.abs(tvals) < thresh, abs_err, pct_err)

        row_maxAPE = ape_row.max()
        row_maxAE  = abs_err.max()
        maxAPE = max(maxAPE, row_maxAPE)
        maxAE  = max(maxAE,  row_maxAE)
        if threshold is not None and max(maxAPE, maxAE) > threshold:
            break
    return maxAPE, maxAE

# ---------------------- Plane Construction ----------------------
def find_best_tangent_plane(a1, b1, a2, b2, candidates=CANDIDATE_GRID):
    xs1 = np.linspace(a1, b1, candidates)
    xs2 = np.linspace(a2, b2, candidates)
    best_key, best_expr, best_pt, best_fn = np.inf, None, None, None
    for x10 in xs1:
        for x20 in xs2:
            f0 = f_true(x10, x20)
            g1 = grad_funcs[0](x10, x20)
            g2 = grad_funcs[1](x10, x20)
            expr = f0 + g1*(x1 - x10) + g2*(x2 - x20)
            fn = get_plane_fn(expr)
            mpe, mae = compute_max_errors_plane(fn, a1, b1, a2, b2, threshold=best_key)
            key = max(mpe, mae)
            if key < best_key:
                best_key, best_expr, best_pt, best_fn = key, expr, (x10, x20), fn
    return best_expr, best_fn, best_key, best_pt

def find_upper_plane(a1, b1, a2, b2):
    corners = [(a1,a2), (a1,b2), (b1,a2), (b1,b2)]
    vals = [(pt, f_true(pt[0],pt[1])) for pt in corners]
    top3 = sorted(vals, key=lambda x: x[1], reverse=True)[:3]
    alpha, beta, gamma = sp.symbols('alpha beta gamma')
    eqs = [alpha*x + beta*y + gamma - z for (x,y),z in top3]
    sol = sp.solve(eqs, (alpha, beta, gamma))
    expr = sol[alpha]*x1 + sol[beta]*x2 + sol[gamma]
    fn   = get_plane_fn(expr)
    mpe, mae = compute_max_errors_plane(fn, a1, b1, a2, b2)
    return expr, fn, mpe, mae

# ---------------------- Split Axis Decision ----------------------
def sample_ape(expr_fn, a1, b1, a2, b2):
    xs1 = np.linspace(a1, b1, SPLIT_GRID_POINTS)
    xs2 = np.linspace(a2, b2, SPLIT_GRID_POINTS)
    X1, X2 = np.meshgrid(xs1, xs2)
    true_vals = f_true(X1, X2)
    return np.abs(expr_fn(X1, X2) - true_vals) / np.maximum(np.abs(true_vals), 1e-6) * 100

def compute_error_variation(mape_grid, axis):
    return np.abs(np.diff(mape_grid.mean(axis=axis))).sum()

def choose_split_axis(low_fn, up_fn, a1, b1, a2, b2):
    grid_low = sample_ape(low_fn, a1, b1, a2, b2)
    grid_up  = sample_ape(up_fn,  a1, b1, a2, b2)
    var_x1 = compute_error_variation(grid_low,0) + compute_error_variation(grid_up,0)
    var_x2 = compute_error_variation(grid_low,1) + compute_error_variation(grid_up,1)
    return 'x1' if var_x1 >= var_x2 else 'x2'

# ---------------------- Combined Error Info (cached) ----------------------
@lru_cache(maxsize=None)
def max_error_info(a1, b1, a2, b2):
    low_expr, low_fn, _, low_pt    = find_best_tangent_plane(a1, b1, a2, b2)
    up_expr, up_fn, up_mpe, up_mae = find_upper_plane(a1, b1, a2, b2)
    low_mpe, low_mae = compute_max_errors_plane(low_fn, a1, b1, a2, b2)
    max_ape = max(low_mpe, up_mpe)
    max_ae  = max(low_mae, up_mae)
    return max_ape, max_ae, low_expr, up_expr, low_pt, low_fn, up_fn

# ---------------------- Plotting & Main Loop ----------------------
def plot_final_approximation(final_leaves):
    pts = 80
    xs = np.linspace(X_MIN, X_MAX, pts)
    ys = np.linspace(Y_MIN, Y_MAX, pts)
    X, Y = np.meshgrid(xs, ys)
    Z_true = f_true(X, Y)

    fig = plt.figure(figsize=(10,8))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, Z_true, color='gray', linewidth=0.5,
                      rcount=30, ccount=30, alpha=0.7)
    for rep in final_leaves:
        a1, b1, a2, b2 = rep['range']
        lf, uf        = rep['lower_fn'], rep['upper_fn']
        xs_r = np.linspace(a1, b1, 4)
        ys_r = np.linspace(a2, b2, 4)
        Xr, Yr = np.meshgrid(xs_r, ys_r)
        ax.plot_surface(Xr, Yr, lf(Xr, Yr), alpha=0.3, linewidth=0)
        ax.plot_wireframe(Xr, Yr, uf(Xr, Yr), linewidth=1,
                         rcount=4, ccount=4, alpha=0.8)
    ax.set_title("Final Piecewise Linear Bounds")
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("f(x)")
    plt.tight_layout(); plt.show()

if __name__ == '__main__':
    heap, leaves = [], []
    root = (X_MIN, X_MAX, Y_MIN, Y_MAX)
    r_ape, r_ae, lp, up, pt, lfn, ufn = max_error_info(*root)
    heapq.heappush(heap, (-max(r_ape, r_ae), root, lp, up, pt, lfn, ufn, r_ape, r_ae))
    region_list = [root]

    while heap:
        neg_key, region, low_e, up_e, lp_pt, low_fn, up_fn, c_ape, c_ae = heapq.heappop(heap)
        a1, b1, a2, b2 = region
        if c_ape <= MaxAPE_threshold and c_ae <= MaxAE_threshold:
            leaves.append((region, c_ape, c_ae, low_e, up_e, lp_pt, low_fn, up_fn))
            continue
        axis = choose_split_axis(low_fn, up_fn, a1, b1, a2, b2)
        mid  = (a1+b1)/2 if axis=='x1' else (a2+b2)/2
        split_history.append((axis, a1, b1, a2, b2, mid))
        region_list.remove(region)
        children = ([(a1, mid, a2, b2), (mid, b1, a2, b2)] if axis=='x1'
                    else [(a1, b1, a2, mid), (a1, b1, mid, b2)])
        region_list.extend(children)

        # visualize
        fig, ax = plt.subplots(figsize=(10,8))
        patches, errs = [], []
        for reg in region_list:
            u1, v1, u2, v2 = reg
            patches.append(Rectangle((u1, u2), v1-u1, v2-u2))
            e_ape, e_ae, *_ = max_error_info(*reg)
            errs.append(max(e_ape, e_ae))
        pc = PatchCollection(patches, cmap='coolwarm', alpha=0.6,
                             edgecolor='black', linewidth=0.5)
        pc.set_array(np.array(errs)); ax.add_collection(pc)
        for reg, err in zip(region_list, errs):
            u1, v1, u2, v2 = reg
            ax.text((u1+v1)/2, (u2+v2)/2, f"{err:.1f}%",
                    ha='center', va='center', color='white', fontsize=8)
        for axn, x1r, x2r, y1r, y2r, m in split_history:
            if axn=='x1': ax.plot([m,m], [y1r,y2r], 'r--')
            else:         ax.plot([x1r,x2r], [m,m], 'r--')
        fig.colorbar(pc, ax=ax, label='MaxErr (%)')
        ax.set_xlim(X_MIN, X_MAX); ax.set_ylim(Y_MIN, Y_MAX); ax.set_aspect('auto')
        # optional
        ax.set_xscale('linear')
        ax.set_yscale('log')
        ax.set_title(f"Split {axis}@{mid:.2f}"); plt.tight_layout(); plt.show()

        for child in children:
            c_ape, c_ae, ce_low, ce_up, c_pt, c_lf, c_uf = max_error_info(*child)
            heapq.heappush(heap, (-max(c_ape, c_ae), child,
                                  ce_low, ce_up, c_pt, c_lf, c_uf,
                                  c_ape, c_ae))

    # final leaves report
    final_leaves = []
    for region, ape, ae, le, ue, pt, lf, uf in leaves:
        final_leaves.append({
            'range': region, 'MaxAPE': ape, 'MaxAE': ae,
            'lower_point': pt, 'lower_expr': le, 'upper_expr': ue,
            'lower_fn': lf, 'upper_fn': uf
        })

    print(f"\nGot {len(final_leaves)} final subregions:\n")
    for i, rep in enumerate(final_leaves, 1):
        a1,b1,a2,b2 = rep['range']
        print(f"Region {i}: x1[{a1:.2f},{b1:.2f}] x2[{a2:.2f},{b2:.2f}] "
              f"MaxAPE={rep['MaxAPE']:.2f}% MaxAE={rep['MaxAE']:.3f}")
        print(f"  Tangent@({rep['lower_point'][0]:.3f},{rep['lower_point'][1]:.3f})")
        print("  Lower plane:", sp.pretty(rep['lower_expr']))
        print("  Upper plane:", sp.pretty(rep['upper_expr']))
        print("-"*60)

    # 2D error map
    fig2, ax2 = plt.subplots(figsize=(8,8))
    patches2, errs2, tx, ty = [], [], [], []
    for rep in final_leaves:
        a1,b1,a2,b2 = rep['range']
        patches2.append(Rectangle((a1,a2), b1-a1, b2-a2))
        errs2.append(rep['MaxAPE']); tx.append(rep['lower_point'][0]); ty.append(rep['lower_point'][1])
    pc2 = PatchCollection(patches2, cmap='coolwarm', edgecolor='black', linewidth=0.5)
    pc2.set_array(np.array(errs2)); ax2.add_collection(pc2)
    ax2.scatter(tx, ty, c='k', s=10, label='tangent pts')
    fig2.colorbar(pc2, ax=ax2, label='MaxAPE (%)')
    ax2.set_xlim(X_MIN,X_MAX); ax2.set_ylim(Y_MIN,Y_MAX)
    ax2.set_xlabel('x1'); ax2.set_ylabel('x2')
    ax2.set_title('Final subregions by MaxAPE'); ax2.legend(); plt.tight_layout(); plt.show()

    # 3D final enclosure
    plot_final_approximation(final_leaves)
