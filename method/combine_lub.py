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
from sympy import Symbol
import re

# right after your imports:
x1, x2 = sp.symbols('x1 x2')

# ---------------------- Symbolic & C-Parser Setup ----------------------
class MathFuncExtractor(c_ast.NodeVisitor):
    def __init__(self):
        self.sympy_expr = None
        self.env = {}
        self.param_syms = []

    def visit_FuncDef(self, node):
        proto = node.decl.type
        params = proto.args.params if proto.args else []
        if self.sympy_expr is not None:
            return
        names = [p.name for p in params]
        if len(names) < 1 or len(names) > 2:
            return
        syms = sp.symbols(' '.join(names))
        # ensure we always have a tuple, even for one symbol
        if isinstance(syms, sp.Symbol):
            syms = (syms,)
        self.param_syms = list(syms)
        for n, s in zip(names, syms):
            self.env[n] = s
        for stmt in node.body.block_items or []:
            if isinstance(stmt, c_ast.Decl) and stmt.init is not None:
                self.env[stmt.name] = self._to_sympy(stmt.init)
            elif isinstance(stmt, c_ast.Return):
                self.sympy_expr = self._to_sympy(stmt.expr)
                return

    def _to_sympy(self, node):
        if isinstance(node, c_ast.BinaryOp):
            L = self._to_sympy(node.left)
            R = self._to_sympy(node.right)
            return {'+': L+R, '-': L-R, '*': L*R, '/': L/R}[node.op]
        if isinstance(node, c_ast.Constant):
            return sp.Float(node.value)
        if isinstance(node, c_ast.ID):
            return self.env[node.name]
        if isinstance(node, c_ast.FuncCall):
            fn = node.name.name
            args = [self._to_sympy(a) for a in node.args.exprs]
            if fn == 'pow': return args[0]**args[1]
            if fn == 'cbrt': return args[0]**sp.Rational(1,3)
            raise NotImplementedError(f"Unsupported function '{fn}'")
        raise NotImplementedError(f"Cannot handle AST node {type(node)}")


def load_c_math_function(filename: str):
    ast = parse_file(filename, use_cpp=True)
    extractor = MathFuncExtractor()
    extractor.visit(ast)
    if extractor.sympy_expr is None or len(extractor.param_syms) not in (1,2):
        raise ValueError("C file must define a function with 1 or 2 parameters.")
    return extractor.param_syms, extractor.sympy_expr

# ---------------------- UTILITIES ----------------------
def ensure_array(y, xs):
    arr = np.asarray(y)
    if arr.shape != xs.shape:
        arr = np.full_like(xs, float(arr))
    return arr

# ---------------------- 1D APPROXIMATION ----------------------
def compute_errors_1d(f_true, f_approx, a, b, num_points=200):
    xs = np.linspace(a, b, num_points)
    tv = f_true(xs)
    av = f_approx(xs)
    ape = np.abs(av - tv) / np.maximum(np.abs(tv), 1e-6) * 100
    return ape.max()

def find_best_tangent_1d(expr, var, a, b, candidates=50):
    f_func = sp.lambdify(var, expr, 'numpy')
    f_prime = sp.diff(expr, var)
    f_prime_func = sp.lambdify(var, f_prime, 'numpy')
    xs = np.linspace(a, b, candidates)
    best_mape = np.inf
    best_expr = None
    best_x0 = None
    for x0 in xs:
        y0 = f_func(x0)
        slope = f_prime_func(x0)
        e = y0 + slope*(var - x0)
        fn = sp.lambdify(var, e, 'numpy')
        mape = compute_errors_1d(f_func, fn, a, b)
        if mape < best_mape:
            best_mape, best_expr, best_x0 = mape, e, x0
    return best_expr, best_mape, best_x0

def process_subregion_1d(expr, var, a, b, mape_thresh, min_len, subregions=None, iteration=[0]):
    if subregions is None: subregions = []
    iteration[0] += 1
    c = (a + b) / 2
    f = expr
    # function values
    fa = float(f.subs(var, a))
    fc = float(f.subs(var, c))
    fb = float(f.subs(var, b))
    # upper secants
    UL = fa + (fc-fa)/(c-a)*(var - a)
    UR = fc + (fb-fc)/(b-c)*(var - c)
    # best lower tangents
    LL, mape_ll, x0_ll = find_best_tangent_1d(expr, var, a, c)
    LR, mape_lr, x0_lr = find_best_tangent_1d(expr, var, c, b)
    # errors
    f_func = sp.lambdify(var, expr, 'numpy')
    err_UL = compute_errors_1d(f_func, sp.lambdify(var, UL,'numpy'), a, c)
    err_UR = compute_errors_1d(f_func, sp.lambdify(var, UR,'numpy'), c, b)
    left_err  = max(mape_ll, err_UL)
    right_err = max(mape_lr, err_UR)
    # plot
    xs = np.linspace(a, b, 300)
    plt.figure()
    plt.plot(xs, f_func(xs), label='True')
    plt.plot(xs, ensure_array(sp.lambdify(var, LL,'numpy')(xs),xs), '--', label='LL')
    plt.plot(xs, ensure_array(sp.lambdify(var, UL,'numpy')(xs),xs), ':', label='UL')
    plt.plot(xs, ensure_array(sp.lambdify(var, LR,'numpy')(xs),xs), '--', label='LR')
    plt.plot(xs, ensure_array(sp.lambdify(var, UR,'numpy')(xs),xs), ':', label='UR')
    plt.title(f"1D Iter {iteration[0]} Range [{a:.2f},{b:.2f}]")
    plt.legend(); plt.grid(True); plt.show()
    # recurse
    if left_err > mape_thresh and (c-a) > min_len:
        process_subregion_1d(expr, var, a, c, mape_thresh, min_len, subregions, iteration)
    else:
        subregions.append({
        'range':    (a, c),
        'lower_expr': LL,
        'upper_expr': UL,
        'err':      left_err,
        'lower_x0': x0_ll,
        'var_sym':  var   
    })
    if right_err > mape_thresh and (b-c) > min_len:
        process_subregion_1d(expr, var, c, b, mape_thresh, min_len, subregions, iteration)
    else:
        subregions.append({
        'range':    (c, b),
        'lower_expr': LR,
        'upper_expr': UR,
        'err':      right_err,
        'lower_x0': x0_lr,
        'var_sym':  var   
    })
    return subregions

# ---------------------- C-Code Generation ----------------------
def generate_c_conditions_1d(var_name, subregions):
    cvar = Symbol(var_name)
    lines = ["// 1D piecewise bounds"]
    for i, r in enumerate(subregions):
        a, b     = r['range']
        prefix   = 'if' if i == 0 else 'else if'
        cond     = f"{prefix} ({var_name} >= {a:.2f} && {var_name} < {b:.2f})"
        orig_sym = r['var_sym']                       
        expr_c   = sp.ccode(r['lower_expr'].subs({orig_sym: cvar}))
        lines.append(f"{cond} {{\n    result = {expr_c};\n}}")
    if len(lines)>1:
        lines[-1] = lines[-1].replace('else if','else',1)
    return '\n'.join(lines)

# ---------------------- 2D APPROXIMATION ----------------------
# Parameters
X_MIN, X_MAX = 1.0, 100.0
Y_MIN, Y_MAX = 1.0, 10000.0
MaxAPE_threshold = 20.0
MaxAE_threshold  = 20.0
ERROR_GRID_POINTS = 100
MIN_GRID = 20
CANDIDATE_GRID = 10
SPLIT_GRID_POINTS = 40
FULL_AREA = (X_MAX-X_MIN)*(Y_MAX-Y_MIN)

def get_plane_fn(expr):
    return sp.lambdify((x1, x2), expr, 'numpy')

def compute_max_errors_plane(expr_fn, a1, b1, a2, b2, threshold=None):
    # dynamic grid sizing
    region_area = (b1-a1)*(b2-a2)
    ratio = np.clip(region_area/FULL_AREA, 0.0, 1.0)
    gp = max(MIN_GRID, int(ERROR_GRID_POINTS * np.sqrt(ratio)))
    cg = min(10, gp)
    Xc, Yc = np.meshgrid(np.linspace(a1,b1,cg), np.linspace(a2,b2,cg))
    flat = np.abs(f_true(Xc,Yc)).ravel()
    pct = np.percentile(flat, 10)
    maxAPE, maxAE = 0.0, 0.0
    xs = np.linspace(a1,b1,gp)
    xs2 = np.linspace(a2,b2,gp)
    for x in xs:
        tvals = f_true(x, xs2)
        avals = expr_fn(x, xs2)
        abs_err = np.abs(avals - tvals)
        pct_err = abs_err/np.maximum(np.abs(tvals),1e-6)*100
        ape_row = np.where(np.abs(tvals)<pct, abs_err, pct_err)
        maxAPE = max(maxAPE, ape_row.max())
        maxAE  = max(maxAE, abs_err.max())
        if threshold and max(maxAPE,maxAE)>threshold:
            break
    return maxAPE, maxAE

def find_best_tangent_plane(a1,b1,a2,b2):
    best_key = np.inf
    best = (None,None,None,None)
    xs1 = np.linspace(a1,b1,CANDIDATE_GRID)
    xs2 = np.linspace(a2,b2,CANDIDATE_GRID)
    for x10 in xs1:
        for x20 in xs2:
            f0 = f_true(x10,x20)
            g1 = grad_funcs[0](x10,x20)
            g2 = grad_funcs[1](x10,x20)
            expr = f0 + g1*(x1-x10) + g2*(x2-x20)
            fn = get_plane_fn(expr)
            mpe, mae = compute_max_errors_plane(fn, a1,b1,a2,b2, best_key)
            key = max(mpe, mae)
            if key<best_key:
                best_key=key; best=(expr, fn, key, (x10,x20))
    return best

def find_upper_plane(a1,b1,a2,b2):
    corners = [(a1,a2),(a1,b2),(b1,a2),(b1,b2)]
    vals = [(pt, f_true(pt[0],pt[1])) for pt in corners]
    top3 = sorted(vals, key=lambda x: x[1], reverse=True)[:3]
    alpha,beta,gamma = sp.symbols('alpha beta gamma')
    eqs = [alpha*x1+beta*x2+gamma - z for (x1,x2),z in top3]
    sol = sp.solve(eqs, (alpha,beta,gamma))
    expr = sol[alpha]*x1 + sol[beta]*x2 + sol[gamma]
    fn = get_plane_fn(expr)
    mpe, mae = compute_max_errors_plane(fn, a1,b1,a2,b2)
    return expr, fn, mpe, mae

def sample_ape(expr_fn, a1,b1,a2,b2):
    xs1 = np.linspace(a1,b1,SPLIT_GRID_POINTS)
    xs2 = np.linspace(a2,b2,SPLIT_GRID_POINTS)
    X1,X2 = np.meshgrid(xs1,xs2)
    tv = f_true(X1,X2)
    ape = np.abs(expr_fn(X1,X2)-tv)/np.maximum(np.abs(tv),1e-6)*100
    return ape

def compute_error_variation(grid, axis):
    return np.abs(np.diff(grid.mean(axis=axis))).sum()

def choose_split_axis(low_fn, up_fn, a1,b1,a2,b2):
    gl = sample_ape(low_fn, a1,b1,a2,b2)
    gu = sample_ape(up_fn,  a1,b1,a2,b2)
    var1 = compute_error_variation(gl,0)+compute_error_variation(gu,0)
    var2 = compute_error_variation(gl,1)+compute_error_variation(gu,1)
    return 'x1' if var1>=var2 else 'x2'

@lru_cache(maxsize=None)
def max_error_info(a1,b1,a2,b2):
    low_e = find_best_tangent_plane(a1,b1,a2,b2)
    up_e  = find_upper_plane(a1,b1,a2,b2)
    low_mpe, low_mae = compute_max_errors_plane(low_e[1],a1,b1,a2,b2)
    maxAPE = max(low_mpe, up_e[2])
    maxAE  = max(low_mae, up_e[3])
    return maxAPE, maxAE, low_e[0], up_e[0], low_e[3], low_e[1], up_e[1]

def plot_final_approximation(leaves):
    pts = 80
    xs = np.linspace(X_MIN,X_MAX,pts)
    ys = np.linspace(Y_MIN,Y_MAX,pts)
    X,Y = np.meshgrid(xs,ys)
    Z = f_true(X,Y)
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X,Y,Z,color='gray',alpha=0.7,rcount=30,ccount=30)
    for r in leaves:
        a1,b1,a2,b2 = r['range']
        lf, uf = r['lower_fn'], r['upper_fn']
        xsr = np.linspace(a1,b1,4); ysr = np.linspace(a2,b2,4)
        Xr,Yr = np.meshgrid(xsr,ysr)
        ax.plot_surface(Xr,Yr, lf(Xr,Yr), alpha=0.3)
        ax.plot_wireframe(Xr,Yr, uf(Xr,Yr), alpha=0.8)
    ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('f')
    plt.show()

def generate_c_conditions_2d(var1, var2, leaves):
    lower, upper = [], []
    lower.append("// Lower bound 2D")
    for i, r in enumerate(leaves):
        a1,b1,a2,b2 = r['range']
        prefix = 'if' if i==0 else 'else if'
        cond = f"{prefix} ({var1}>= {a1:.2f} && {var1}< {b1:.2f} && {var2}>= {a2:.2f} && {var2}< {b2:.2f})"
        code = sp.ccode(r['lower_expr'].subs({x1:sp.Symbol(var1), x2:sp.Symbol(var2)}))
        lower.append(f"{cond} {{ bic_target = {code}; }}")
    upper.append("// Upper bound 2D")
    for i, r in enumerate(leaves):
        a1,b1,a2,b2 = r['range']
        prefix = 'if' if i==0 else 'else if'
        cond = f"{prefix} ({var1}>= {a1:.2f} && {var1}< {b1:.2f} && {var2}>= {a2:.2f} && {var2}< {b2:.2f})"
        code = sp.ccode(r['upper_expr'].subs({x1:sp.Symbol(var1), x2:sp.Symbol(var2)}))
        upper.append(f"{cond} {{ bic_target = {code}; }}")
    if len(lower)>1: lower[-1] = lower[-1].replace('else if','else',1)
    if len(upper)>1: upper[-1] = upper[-1].replace('else if','else',1)
    return '\n'.join(lower + [''] + upper)

# ---------------------- Main Dispatcher ----------------------
def main():
    # Expect either:
    #   python script.py math_func.c template.c xmin xmax
    #   python script.py math_func.c template.c xmin xmax ymin ymax
    if len(sys.argv) not in (5, 7):
        print("Usage 1D: python3 script.py <math_c_file.c> <template.c> xmin xmax")
        print("Usage 2D: python3 script.py <math_c_file.c> <template.c> xmin xmax ymin ymax")
        sys.exit(1)

    math_c    = sys.argv[1]
    template  = sys.argv[2]
    syms, expr = load_c_math_function(math_c)

    # Placeholder for the generated if/else block
    cond_block = None

    if len(syms) == 1:
        # 1D mode
        var    = syms[0]
        xmin, xmax = map(float, sys.argv[3:5])

        regs = process_subregion_1d(expr, var, xmin, xmax, 20, 5)
        print(f"1D: {len(regs)} segments")

        cond_block = generate_c_conditions_1d(var.name, regs)

    else:
        # 2D mode
        global x1, x2, f_true, grad_funcs
        x1, x2 = sp.symbols('x1 x2')
        f_true = sp.lambdify((x1, x2), expr, 'numpy')
        grad_syms  = [sp.diff(expr, v) for v in (x1, x2)]
        grad_funcs = [sp.lambdify((x1, x2), g, 'numpy') for g in grad_syms]

        xmin, xmax, ymin, ymax = map(float, sys.argv[3:7])
        root = (xmin, xmax, ymin, ymax)
        heap, leaves = [], []

        # Initialize heap
        r_ape, r_ae, lp, up, pt, lfn, ufn = max_error_info(*root)
        heapq.heappush(heap, (-max(r_ape, r_ae), root, lp, up, pt, lfn, ufn, r_ape, r_ae))

        # Adaptive subdivision
        while heap:
            _, reg, le_lo, le_up, pt, lf, uf, ape, ae = heapq.heappop(heap)
            a1, b1, a2, b2 = reg
            if ape <= MaxAPE_threshold and ae <= MaxAE_threshold:
                leaves.append({
                    'range': reg,
                    'lower_expr': le_lo,
                    'upper_expr': le_up,
                    'lower_fn': lf,
                    'upper_fn': uf
                })
            else:
                axis = choose_split_axis(lf, uf, a1, b1, a2, b2)
                mid = (b1 + a1)/2 if axis=='x1' else (b2 + a2)/2
                children = (
                    [(a1, mid, a2, b2), (mid, b1, a2, b2)]
                    if axis=='x1' else
                    [(a1, b1, a2, mid), (a1, b1, mid, b2)]
                )
                for child in children:
                    c_ape, c_ae, c_le, c_up, _, c_lf, c_uf = max_error_info(*child)
                    heapq.heappush(heap, (
                        -max(c_ape, c_ae), child,
                        c_le, c_up, None,
                        c_lf, c_uf, c_ape, c_ae
                    ))

        print(f"2D: {len(leaves)} regions")
        plot_final_approximation(leaves)
        cond_block = generate_c_conditions_2d(syms[0].name, syms[1].name, leaves)

    # --- Inject into template and write output C file ---
    tpl_src = open(template).read()
    out_src = tpl_src.replace('// INSERT_CONDITIONS_HERE', cond_block)
    out_path = template.replace('.c', '_with_bounds.c')
    with open(out_path, 'w') as outf:
        outf.write(out_src)

    print(f"Generated bounded C file: {out_path}")

if __name__ == '__main__':
    main()
