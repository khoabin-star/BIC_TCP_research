import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# Parameters and Helper Functions
# ============================================================
x_range = 100        # last_max_cwnd in [1, 100]
y_range = 10000      # rtt in [1, 10000]
mape_threshold = 20   # MAPE threshold for stopping the adaptive subdivision
output_file = "../data/data_generated_test.csv"

# ------------------------------------------------------------
# Original functions from your code
# ------------------------------------------------------------
# def input_function(last_max_cwnd, rtt, x=10):
#     return (last_max_cwnd - (410 * ((cubic_root((1 << (10 + 3 * x)) // 410 * (last_max_cwnd - (717 * last_max_cwnd // 1024))) - ((1 << 10) * rtt // 1000)) ** 3) >> (10 + 3 * x)))

def input_function(x, y):
    """
    The function to approximate.
    """
    return x - 0.4 * np.float_power(np.float_power(0.75 * x, 1.0 / 3.0) - y / 1000.0, 3)

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
    x_val = ((v[shift] + 10) << b) >> 6
    x_val = (2 * x_val + (a // (x_val * (x_val - 1))))
    x_val = (x_val * 341) >> 10
    return x_val

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

BITS_PER_LONG = 64

# ------------------------------------------------------------
# Error Variation Function
# ------------------------------------------------------------
def compute_error_variation(data, axis):
    """
    For the given data (which must have a 'mape' column), group by the given axis and
    compute the average MAPE for each group. Then, compute the sum of absolute differences
    (gradient) between consecutive average errors.
    """
    grouped = data.groupby(axis)['mape'].mean().sort_index()
    avg_errors = grouped.values
    gradients = np.abs(np.diff(avg_errors))
    total_gradient = np.sum(gradients)
    return total_gradient

# ------------------------------------------------------------
# Data Generation (if needed)
# ------------------------------------------------------------
def generate_data_file(filename=output_file):
    with open(filename, "w") as file:
        file.write("last_max_cwnd,rtt,result\n")
        for last_max_cwnd in range(1, x_range + 1):
            for rtt in range(1, y_range + 1):
                result = input_function(last_max_cwnd, rtt)
                file.write(f"{last_max_cwnd},{rtt},{result}\n")

generate_data_file()

# Read generated data
data = pd.read_csv(output_file)

# ============================================================
# Adaptive Subdivision and Visualization Functions
# ============================================================
def plot_subregions(subregions, iteration):
    """Plots all current subregions as rectangles with their MAPE values."""
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    for sr in subregions:
        rect = Rectangle((sr['cwnd_lower'], sr['rtt_lower']),
                         sr['cwnd_upper'] - sr['cwnd_lower'],
                         sr['rtt_upper'] - sr['rtt_lower'],
                         fill=False, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        center_cwnd = (sr['cwnd_lower'] + sr['cwnd_upper']) / 2
        center_rtt = (sr['rtt_lower'] + sr['rtt_upper']) / 2
        err_text = f"{sr['mape']:.1f}%" if sr['mape'] is not None else "N/A"
        ax.text(center_cwnd, center_rtt, err_text, ha='center', va='center', fontsize=8, color='red')
    ax.set_xlim(0, x_range + 1)
    ax.set_ylim(0, y_range + 1)
    plt.xlabel("last_max_cwnd")
    plt.ylabel("rtt")
    plt.title(f"Iteration {iteration}: Current Subregions")
    plt.show()


def fit_model_for_subregion(subregion):
    # Fit a linear regression model to the data in the subregion
    X = subregion['data'][["last_max_cwnd", "rtt"]]
    y = subregion['data']["result"]
    if len(X) < 2:
        subregion['mape'] = np.inf
        return
    model = LinearRegression()
    model.fit(X, y)
    coef_last_max_cwnd = model.coef_[0]
    coef_rtt = model.coef_[1]
    intercept = model.intercept_
    # Compute integer-based predictions
    y_pred_int = (coef_last_max_cwnd * X["last_max_cwnd"] + coef_rtt * X["rtt"] + intercept)
    mse_int = mean_squared_error(y, y_pred_int)
    mae_int = mean_absolute_error(y, y_pred_int)
   # Compute per-row MAPE (handle zero y values)
    per_row_mape = np.abs((y - y_pred_int) / np.where(y != 0, y, np.nan)) * 100
    # Assign the computed per-row error back to the data
    subregion["data"] = subregion["data"].assign(mape=per_row_mape)
    # take the average MAPE in the subregion.
    # mape_int = np.nanmean(per_row_mape)

    # Instead of averaging, take the maximum (worst-case) MAPE in the subregion.
    mape_int = np.nanmax(per_row_mape)

    subregion['model'] = {
        'coef_last_max_cwnd': coef_last_max_cwnd,
        'coef_rtt': coef_rtt,
        'intercept': intercept,
    }
    subregion['mse'] = mse_int
    subregion['mae'] = mae_int
    subregion['mape'] = mape_int
    subregion['model_fitted'] = True

# ============================================================
# Main Adaptive Subdivision Loop
# ============================================================
# Initial subregion covering the entire domain
initial_subregion = {
    'cwnd_lower': data['last_max_cwnd'].min(),
    'cwnd_upper': data['last_max_cwnd'].max(),  # using an exclusive upper bound
    'rtt_lower': data['rtt'].min(),
    'rtt_upper': data['rtt'].max(),
    'data': data.copy(),
    'model_fitted': False,
    'mse': None,
    'mae': None,
    'mape': None,
}

subregions = [initial_subregion]
iteration = 0

while True:
    # Fit models for subregions that haven't been processed
    for subregion in subregions:
        if not subregion.get('model_fitted', False):
            fit_model_for_subregion(subregion)
    
    # Identify the subregion with the worst MAPE
    worst_subregion = max(subregions, key=lambda x: x['mape'] if x['mape'] is not None else -np.inf)
    # print("Here is worst subregion", worst_subregion)

    # Stop if worst MAPE is below threshold
    if worst_subregion['mape'] <= mape_threshold:
        print("All subregions have MAPE below threshold. Stopping subdivision.")
        break

    # Compute error variation along both axes within the worst subregion
    x_gradient = compute_error_variation(worst_subregion['data'], 'last_max_cwnd')
    y_gradient = compute_error_variation(worst_subregion['data'], 'rtt')
    # print(f"Error variation: last_max_cwnd = {x_gradient}, rtt = {y_gradient}")
    
    # Split along the axis with the larger error variation using the median
    if x_gradient >= y_gradient:
        split_axis = 'last_max_cwnd'
        split_value = worst_subregion['data']['last_max_cwnd'].median()
        left_data = worst_subregion['data'][worst_subregion['data']['last_max_cwnd'] < split_value]
        right_data = worst_subregion['data'][worst_subregion['data']['last_max_cwnd'] >= split_value]
    else:
        split_axis = 'rtt'
        split_value = worst_subregion['data']['rtt'].median()
        left_data = worst_subregion['data'][worst_subregion['data']['rtt'] < split_value]
        right_data = worst_subregion['data'][worst_subregion['data']['rtt'] >= split_value]
    
    # print(f"Splitting worst subregion along {split_axis} at {split_value}")
    
    # If either side is empty, stop subdividing this region
    if len(left_data) == 0 or len(right_data) == 0:
        print("One side of the split is empty. Stopping further subdivision for this subregion.")
        worst_subregion['mape'] = 0  # Force the threshold condition
        continue
    #plot_subregions(subregions, iteration)
    # Create new subregions based on the chosen split axis
    if split_axis == 'last_max_cwnd':
        left_subregion = {
            'cwnd_lower': worst_subregion['cwnd_lower'],
            'cwnd_upper': split_value,
            'rtt_lower': worst_subregion['rtt_lower'],
            'rtt_upper': worst_subregion['rtt_upper'],
            'data': left_data,
            'model_fitted': False,
            'mse': worst_subregion['mse'],
            'mae': worst_subregion['mae'],
            'mape': worst_subregion['mape'],
        }
        right_subregion = {
            'cwnd_lower': split_value,
            'cwnd_upper': worst_subregion['cwnd_upper'],
            'rtt_lower': worst_subregion['rtt_lower'],
            'rtt_upper': worst_subregion['rtt_upper'],
            'data': right_data,
            'model_fitted': False,
            'mse': worst_subregion['mse'],
            'mae': worst_subregion['mae'],
            'mape': worst_subregion['mape'],
        }
    else:  # splitting along rtt
        left_subregion = {
            'cwnd_lower': worst_subregion['cwnd_lower'],
            'cwnd_upper': worst_subregion['cwnd_upper'],
            'rtt_lower': worst_subregion['rtt_lower'],
            'rtt_upper': split_value,
            'data': left_data,
            'model_fitted': False,
            'mse': worst_subregion['mse'],
            'mae': worst_subregion['mae'],
            'mape': worst_subregion['mape'],
        }
        right_subregion = {
            'cwnd_lower': worst_subregion['cwnd_lower'],
            'cwnd_upper': worst_subregion['cwnd_upper'],
            'rtt_lower': split_value,
            'rtt_upper': worst_subregion['rtt_upper'],
            'data': right_data,
            'model_fitted': False,
            'mse': worst_subregion['mse'],
            'mae': worst_subregion['mae'],
            'mape': worst_subregion['mape'],
        }
    
    # Remove the worst subregion and add the two new ones
    subregions.remove(worst_subregion)
    subregions.extend([left_subregion, right_subregion])
    iteration += 1

#plot_subregions(subregions, iteration)

# ------------------------------------------------------------
# Final Reporting: Print total number of subregions (rectangles) and max mape
# ------------------------------------------------------------
total_subregions = len(subregions)
print(f"Total number of subregions (rectangles): {total_subregions}")
max_mape = max(sr['mape'] for sr in subregions if sr['mape'] is not None)
print(f"Maximum MAPE among all subregions: {max_mape:.6f}%")

# ============================================================
# Final Model Fitting & Reporting
# ============================================================
# Ensure all remaining subregions have a fitted model.
for subregion in subregions:
    if not subregion.get('model_fitted', False):
        fit_model_for_subregion(subregion)

# Report the final piecewise linear models and errors.
for idx, sr in enumerate(subregions):
    print(f"Subregion {idx+1}: last_max_cwnd=({sr['cwnd_lower']}, {sr['cwnd_upper']}], rtt=({sr['rtt_lower']}, {sr['rtt_upper']}]")
    model_info = sr['model']
    print(f"Equation: result = ({model_info['coef_last_max_cwnd']}*last_max_cwnd + {model_info['coef_rtt']}*rtt + {model_info['intercept']})/1000")
    print(f"MAPE: {sr['mape']:.6f}%")
    print("-" * 60)

# ------------------------------------------------------------
# Updated snippet for color map creation (2D) remains the same
# ------------------------------------------------------------
# Extract unique boundaries from final subregions
x_bounds = sorted(set([sr['cwnd_lower'] for sr in subregions] + [sr['cwnd_upper'] for sr in subregions]))
y_bounds = sorted(set([sr['rtt_lower'] for sr in subregions] + [sr['rtt_upper'] for sr in subregions]))

# Create meshgrid for pcolormesh (grid corners)
X, Y = np.meshgrid(x_bounds, y_bounds)

# Initialize a result matrix (Z) that will store the approximation value for each cell center
Z = np.zeros((len(y_bounds) - 1, len(x_bounds) - 1), dtype=float)

# Initialize the error matrix with NaN (for the 2D error colormap)
errors = np.full((len(y_bounds) - 1, len(x_bounds) - 1), np.nan)

# Fill each cell with the subregion's MAPE if the entire cell is inside that subregion
for sr in subregions:
    for i in range(len(y_bounds) - 1):
        for j in range(len(x_bounds) - 1):
            # Coordinates of the cell's corners
            cell_x_low = x_bounds[j]
            cell_x_high = x_bounds[j + 1]
            cell_y_low = y_bounds[i]
            cell_y_high = y_bounds[i + 1]

            # Check if this entire cell is within the subregion
            if (cell_x_low >= sr['cwnd_lower'] and cell_x_high <= sr['cwnd_upper'] and
                cell_y_low >= sr['rtt_lower'] and cell_y_high <= sr['rtt_upper']):
                errors[i, j] = sr['mape']

# Plot 2D error colormap (remains unchanged)
plt.figure(figsize=(15, 10))
plt.pcolormesh(X, Y, errors, shading='auto', cmap='coolwarm')
plt.colorbar(label="Error (%)")
for sr in subregions:
    center_x = (sr['cwnd_lower'] + sr['cwnd_upper']) / 2
    center_y = (sr['rtt_lower'] + sr['rtt_upper']) / 2
    plt.scatter(center_x, center_y, color='black', s=10)
plt.title("Error color of each rectangle")
plt.xlabel("cwnd")
plt.ylabel("rtt")
plt.show()

# ------------------------------------------------------------
# 3D Surface Plot: Populate Z with the piecewise linear approximation
# ------------------------------------------------------------
# For each cell defined by adjacent boundaries, calculate the cell center
# and use the subregion's model that covers that center to compute the prediction.
for i in range(len(y_bounds) - 1):
    for j in range(len(x_bounds) - 1):
        cell_x = (x_bounds[j] + x_bounds[j+1]) / 2
        cell_y = (y_bounds[i] + y_bounds[i+1]) / 2
        
        # For each subregion, check if the cell center is contained within it.
        for sr in subregions:
            if (sr['cwnd_lower'] <= cell_x <= sr['cwnd_upper'] and 
                sr['rtt_lower'] <= cell_y <= sr['rtt_upper']):
                coef = sr['model']
                # Use the model to predict the value at (cell_x, cell_y).
                
                # Here, we assume no additional integer scaling is applied.
                pred = coef['coef_last_max_cwnd'] * cell_x + coef['coef_rtt'] * cell_y + coef['intercept']
                Z[i, j] = pred
                break

# 3D Surface Plot
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')
# Create a new meshgrid for plotting the surface; here X_plot, Y_plot must correspond to the centers of the cells.
# However, using the boundaries (x_bounds, y_bounds) is also common practice.
X_plot, Y_plot = np.meshgrid(x_bounds, y_bounds)
ax.plot_surface(X_plot, Y_plot, np.pad(Z, ((0,1),(0,1)), mode='edge'), cmap='viridis', edgecolor='k', alpha=0.8)
ax.set_title("3D Visualization of Piecewise Linear Approximation")
ax.set_xlabel("last_max_cwnd")
ax.set_ylabel("rtt")
ax.set_zlabel("result")
plt.show()

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

for sr in subregions:
    # Get subregion boundaries
    x1 = sr['cwnd_lower']
    x2 = sr['cwnd_upper']
    y1 = sr['rtt_lower']
    y2 = sr['rtt_upper']
    coef = sr['model']
    # Compute the predicted value at each corner using the subregion's linear model
    z11 = coef['coef_last_max_cwnd'] * x1 + coef['coef_rtt'] * y1 + coef['intercept']
    z12 = coef['coef_last_max_cwnd'] * x1 + coef['coef_rtt'] * y2 + coef['intercept']
    z22 = coef['coef_last_max_cwnd'] * x2 + coef['coef_rtt'] * y2 + coef['intercept']
    z21 = coef['coef_last_max_cwnd'] * x2 + coef['coef_rtt'] * y1 + coef['intercept']
    
    # Create a polygon with the four corners
    verts = [[(x1, y1, z11), (x1, y2, z12), (x2, y2, z22), (x2, y1, z21)]]
    poly = Poly3DCollection(verts, alpha=0.7, edgecolor='k')
    
    # Map the subregion's MAPE (normalized) to a color using the 'viridis' colormap.
    # For example, if sr['mape'] is lower, it will be one color; if higher, another.
    # Normalize the value between 0 and 1.
    norm_val = sr['mape'] / max_mape if max_mape > 0 else 0.0
    face_color = cm.viridis(norm_val)
    poly.set_facecolor(face_color)
    
    ax.add_collection3d(poly)

ax.set_xlabel("last_max_cwnd")
ax.set_ylabel("rtt")
ax.set_zlabel("result")
ax.set_xlim3d(1, 100)
ax.set_ylim3d(1, 10000)
ax.set_zlim3d(1, 300)
ax.set_title("3D Visualization with Exact Subregion Boundaries")
plt.show()

# ============================================================
# Method 1: Cascading if-else conditions in C code
def generate_c_conditions(subregions):
    conditions = []
    for i, subregion in enumerate(subregions):
        coef = subregion['model']
        cwnd_lower = int(subregion['cwnd_lower'])  # Round down
        cwnd_upper = int(subregion['cwnd_upper'])  # Round up
        rtt_lower = int(subregion['rtt_lower'])  # Round down
        rtt_upper = int(subregion['rtt_upper'])  # Round up
        
        condition = f"""
if (last_max_cwnd >= {cwnd_lower} && last_max_cwnd < {cwnd_upper} &&
    rtt >= {rtt_lower} && rtt < {rtt_upper}) {{
    bic_target = ({int(coef['coef_last_max_cwnd'])} * last_max_cwnd + {int(coef['coef_rtt'])} * rtt + {int(coef['intercept'])}) / 1000;
}}
"""
        if i > 0:
            condition = condition.replace("if", "else if")
        
        conditions.append(condition)

    # Automatically convert the last condition to "else" without any condition
    if len(conditions) > 0:
        last_condition = f"""
else {{
    bic_target = ({int(subregions[-1]['model']['coef_last_max_cwnd'])} * last_max_cwnd + {int(subregions[-1]['model']['coef_rtt'])} * rtt + {int(subregions[-1]['model']['intercept'])}) / 1000;
}}
"""
        conditions[-1] = last_condition

    return "\n".join(conditions)

def generate_c_conditions_sorted(subregions):
    # Sort subregions by area: smaller region first.
    sorted_subregions = sorted(subregions, key=lambda sr: (sr['cwnd_upper'] - sr['cwnd_lower']) * (sr['rtt_upper'] - sr['rtt_lower']))
    
    conditions = []
    for i, subregion in enumerate(sorted_subregions):
        coef = subregion['model']
        # Convert boundaries to int; adjust rounding as needed.
        cwnd_lower = int(subregion['cwnd_lower'])
        cwnd_upper = int(subregion['cwnd_upper'])
        rtt_lower = int(subregion['rtt_lower'])
        rtt_upper = int(subregion['rtt_upper'])
        
        condition = f"""
if (last_max_cwnd >= {cwnd_lower} && last_max_cwnd < {cwnd_upper} &&
    rtt >= {rtt_lower} && rtt < {rtt_upper}) {{
    bic_target = ({int(coef['coef_last_max_cwnd'])} * last_max_cwnd + {int(coef['coef_rtt'])} * rtt + {int(coef['intercept'])}) / 1000;
}}
"""
        if i > 0:
            condition = condition.replace("if", "else if", 1)
        conditions.append(condition)
    
    # Optionally, change the last condition to a plain "else" clause.
    if conditions:
        last_condition = f"""
else {{
    bic_target = ({int(sorted_subregions[-1]['model']['coef_last_max_cwnd'])} * last_max_cwnd + {int(sorted_subregions[-1]['model']['coef_rtt'])} * rtt + {int(sorted_subregions[-1]['model']['intercept'])}) / 1000;
}}
"""
        conditions[-1] = last_condition
    
    return "\n".join(conditions)

def generate_c_conditions_sorted_desc(subregions):
    # Sort subregions by area (largest first).
    sorted_subregions = sorted(subregions, key=lambda sr: (sr['cwnd_upper'] - sr['cwnd_lower']) * (sr['rtt_upper'] - sr['rtt_lower']), reverse=True)
    
    conditions = []
    for i, subregion in enumerate(sorted_subregions):
        coef = subregion['model']
        cwnd_lower = int(subregion['cwnd_lower'])
        cwnd_upper = int(subregion['cwnd_upper'])
        rtt_lower = int(subregion['rtt_lower'])
        rtt_upper = int(subregion['rtt_upper'])
        
        condition = f"""
if (last_max_cwnd >= {cwnd_lower} && last_max_cwnd < {cwnd_upper} &&
    rtt >= {rtt_lower} && rtt < {rtt_upper}) {{
    bic_target = ({int(coef['coef_last_max_cwnd'])} * last_max_cwnd + {int(coef['coef_rtt'])} * rtt + {int(coef['intercept'])}) / 1000;
}}
"""
        if i > 0:
            condition = condition.replace("if", "else if", 1)
        conditions.append(condition)
    
    # Optionally, change the last condition to a plain "else" clause.
    if conditions:
        last_condition = f"""
else {{
    bic_target = ({int(sorted_subregions[-1]['model']['coef_last_max_cwnd'])} * last_max_cwnd + {int(sorted_subregions[-1]['model']['coef_rtt'])} * rtt + {int(sorted_subregions[-1]['model']['intercept'])}) / 1000;
}}
"""
        conditions[-1] = last_condition
    
    return "\n".join(conditions)


def insert_conditions_into_template(conditions, template_file_path="./template.c", output_file_path="./output_with_conditions.c"):
    with open(template_file_path, "r") as template_file:
        template_code = template_file.read()

    # Insert the conditions into the template
    code_with_conditions = template_code.replace("// INSERT_CONDITIONS_HERE", conditions)

    # Write the result to a new file
    with open(output_file_path, "w") as output_file:
        output_file.write(code_with_conditions)


# Generate the conditions and insert them into the template
c_conditions = generate_c_conditions_sorted(subregions)
# insert_conditions_into_template(c_conditions)

## ============================================================
## Method 2: Array‐of‐structures with a binary search lookup
# def generate_c_array(subregions):
#     lines = []
#     lines.append("typedef struct {")
#     lines.append("    int cwnd_lower, cwnd_upper;")
#     lines.append("    int rtt_lower, rtt_upper;")
#     lines.append("    int coef_last_max_cwnd;")
#     lines.append("    int coef_rtt;")
#     lines.append("    int intercept;")
#     lines.append("} Subregion;")
#     lines.append("")
#     lines.append("Subregion subregions[] = {")
#     for sr in subregions:
#         coef = sr['model']
#         cwnd_lower = int(sr['cwnd_lower'])
#         cwnd_upper = int(sr['cwnd_upper'])
#         rtt_lower = int(sr['rtt_lower'])
#         rtt_upper = int(sr['rtt_upper'])
#         c0 = int(coef['coef_last_max_cwnd'])
#         c1 = int(coef['coef_rtt'])
#         intercept = int(coef['intercept'])
#         lines.append(f"    {{{cwnd_lower}, {cwnd_upper}, {rtt_lower}, {rtt_upper}, {c0}, {c1}, {intercept}}},")
#     lines.append("};")
#     lines.append("")
#     lines.append("int num_subregions = sizeof(subregions) / sizeof(Subregion);")
#     return "\n".join(lines)

# def generate_c_linear_search_lookup():
#     code = """
# int find_subregion(int last_max_cwnd, int rtt) {
#     for (int i = 0; i < num_subregions; i++) {
#         if (last_max_cwnd >= subregions[i].cwnd_lower &&
#             last_max_cwnd < subregions[i].cwnd_upper &&
#             rtt >= subregions[i].rtt_lower &&
#             rtt < subregions[i].rtt_upper) {
#             return i;
#         }
#     }
#     return -1; // No matching subregion found
# }
# """
#     return code

# def insert_c_array_into_template(c_array_code, search_code, template_file_path="./template1.c", output_file_path="./output_with_c_array.c"):
#     with open(template_file_path, "r") as template_file:
#         template_code = template_file.read()

#     # Replace placeholders in your template with the generated C code segments.
#     code_with_array = template_code.replace("// INSERT_SUBREGION_ARRAY_HERE", c_array_code)
#     code_with_all = code_with_array.replace("// INSERT_BINARY_SEARCH_HERE", search_code)

#     with open(output_file_path, "w") as output_file:
#         output_file.write(code_with_all)

# c_array = generate_c_array(subregions)
# search_code = generate_c_linear_search_lookup()
# insert_c_array_into_template(c_array, search_code)

def merge_subregions(sr1, sr2):
    """
    Merge two subregions by taking the union of their boundaries
    and concatenating their data.
    """
    merged_sr = {}
    # Boundaries: take the min of lower bounds and the max of upper bounds.
    merged_sr['cwnd_lower'] = min(sr1['cwnd_lower'], sr2['cwnd_lower'])
    merged_sr['cwnd_upper'] = max(sr1['cwnd_upper'], sr2['cwnd_upper'])
    merged_sr['rtt_lower'] = min(sr1['rtt_lower'], sr2['rtt_lower'])
    merged_sr['rtt_upper'] = max(sr1['rtt_upper'], sr2['rtt_upper'])
    # Concatenate the dataframes from both subregions:
    merged_sr['data'] = pd.concat([sr1['data'], sr2['data']], ignore_index=True)
    
    # Reset the model and error metrics
    merged_sr['model_fitted'] = False
    merged_sr['mse'] = None
    merged_sr['mae'] = None
    merged_sr['mape'] = None
    return merged_sr


# For the manual merge, we extract them (here we use indices 24 and 34 if zero-indexed).
sr25 = subregions[24]
sr35 = subregions[34]

# Merge the two subregions.
merged_sr = merge_subregions(sr25, sr35)
# The merged subregion should now have:
# last_max_cwnd: [min(25.5, 50.5) = 25.5, max(50.5, 100) = 100]
# rtt: [min(2500.5, 2500.5) = 2500.5, max(5000.5, 5000.5) = 5000.5]

# Fit the linear model for the merged subregion (using your existing function)
fit_model_for_subregion(merged_sr)

# Print the results for the merged subregion:
print("Merged Subregion:")
print(f"last_max_cwnd=({merged_sr['cwnd_lower']}, {merged_sr['cwnd_upper']}], "
      f"rtt=({merged_sr['rtt_lower']}, {merged_sr['rtt_upper']}])")
merged_model = merged_sr['model']
print(f"Equation: result = ({merged_model['coef_last_max_cwnd']}*last_max_cwnd + "
      f"{merged_model['coef_rtt']}*rtt + {merged_model['intercept']})/1000")
print(f"MAPE: {merged_sr['mape']:.6f}%")

# Check if below threshold:
if merged_sr['mape'] <= mape_threshold:
    print("Merged subregion MAPE is below the threshold.")
else:
    print("Merged subregion MAPE exceeds the threshold.")
