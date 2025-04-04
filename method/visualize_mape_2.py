import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# Parameters and Helper Functions
# ============================================================
x_range = 1000        # last_max_cwnd in [1, 100]
y_range = 1000      # rtt in [1, 10000]
mape_threshold = 6   # MAPE threshold for stopping the adaptive subdivision
output_file = "../data/data_generated_test.csv"

# ------------------------------------------------------------
# Original functions from your code
# ------------------------------------------------------------
def input_function(last_max_cwnd, rtt, x=10):
    return (last_max_cwnd - (410 * ((cubic_root((1 << (10 + 3 * x)) // 410 * (last_max_cwnd - (717 * last_max_cwnd // 1024))) - ((1 << 10) * rtt // 1000)) ** 3) >> (10 + 3 * x)))

# def input_function(x, y):
#     """
#     The function to approximate.
#     """
#     return x - 0.4 * np.float_power(np.float_power(0.75 * x, 1.0 / 3.0) - y / 1000.0, 3)

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

# Uncomment to regenerate data if needed.
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
    coef_last_max_cwnd = int(round(model.coef_[0] * 1000))
    coef_rtt = int(round(model.coef_[1] * 1000))
    intercept = int(round(model.intercept_ * 1000))
    # Compute integer-based predictions
    y_pred_int = (coef_last_max_cwnd * X["last_max_cwnd"] + coef_rtt * X["rtt"] + intercept) // 1000
    mse_int = mean_squared_error(y, y_pred_int)
    mae_int = mean_absolute_error(y, y_pred_int)
   # Compute per-row MAPE (handle zero y values)
    per_row_mape = np.abs((y - y_pred_int) / np.where(y != 0, y, np.nan)) * 100
    # Assign the computed per-row error back to the data
    subregion["data"] = subregion["data"].assign(mape=per_row_mape)
    mape_int = np.nanmean(per_row_mape)
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

# Plot initial subregion
# plot_subregions(subregions, iteration)

while True:
    # Fit models for subregions that haven't been processed
    for subregion in subregions:
        if not subregion.get('model_fitted', False):
            fit_model_for_subregion(subregion)
    
    # Identify the subregion with the worst MAPE
    worst_subregion = max(subregions, key=lambda x: x['mape'] if x['mape'] is not None else -np.inf)


    # Stop if worst MAPE is below threshold
    if worst_subregion['mape'] <= mape_threshold:
        print("All subregions have MAPE below threshold. Stopping subdivision.")
        break

    # Compute error variation along both axes within the worst subregion
    x_gradient = compute_error_variation(worst_subregion['data'], 'last_max_cwnd')
    y_gradient = compute_error_variation(worst_subregion['data'], 'rtt')
    print(f"Error variation: last_max_cwnd = {x_gradient}, rtt = {y_gradient}")
    
    if x_gradient >= y_gradient:
        split_axis = 'last_max_cwnd'
        grouped = worst_subregion['data'].groupby('last_max_cwnd')['mape'].mean()
        # Sort by error in descending order
        sorted_candidates = grouped.sort_values(ascending=False)
        print("Here is sorted candidates", sorted_candidates)
        split_value = None
        for candidate in sorted_candidates.index:
            if candidate > worst_subregion['cwnd_lower'] and candidate < worst_subregion['cwnd_upper']:
                split_value = candidate
                break
        # Fallback if no candidate is found
        if split_value is None:
            split_value = worst_subregion['data']['last_max_cwnd'].median()
        
        left_data = worst_subregion['data'][worst_subregion['data']['last_max_cwnd'] < split_value]
        right_data = worst_subregion['data'][worst_subregion['data']['last_max_cwnd'] >= split_value]
    else:
        split_axis = 'rtt'
        grouped = worst_subregion['data'].groupby('rtt')['mape'].mean()
        sorted_candidates = grouped.sort_values(ascending=False)
        print("Here is sorted candidates", sorted_candidates)
        split_value = None
        for candidate in sorted_candidates.index:
            if candidate > worst_subregion['rtt_lower'] and candidate < worst_subregion['rtt_upper']:
                split_value = candidate
                break
        if split_value is None:
            split_value = worst_subregion['data']['rtt'].median()
        
        left_data = worst_subregion['data'][worst_subregion['data']['rtt'] < split_value]
        right_data = worst_subregion['data'][worst_subregion['data']['rtt'] >= split_value]
        print("Here is left data", left_data)
        print("Here is right data", right_data)

    
    print(f"Splitting worst subregion along {split_axis} at {split_value}")
    
    # If either side is empty, stop subdividing this region
    if len(left_data) == 0 or len(right_data) == 0:
        print("One side of the split is empty. Stopping further subdivision for this subregion.")
        worst_subregion['mape'] = 0  # Force the threshold condition
        continue
    plot_subregions(subregions, iteration)
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

plot_subregions(subregions, iteration)
# print("Here is final subregions", subregions)

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
# Updated snippet for color map creation
# ------------------------------------------------------------
# Extract unique boundaries from final subregions
x_bounds = sorted(set([sr['cwnd_lower'] for sr in subregions] + [sr['cwnd_upper'] for sr in subregions]))
y_bounds = sorted(set([sr['rtt_lower'] for sr in subregions] + [sr['rtt_upper'] for sr in subregions]))

# Create meshgrid for pcolormesh (grid corners)
X, Y = np.meshgrid(x_bounds, y_bounds)

# Initialize the error matrix with NaN
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

plt.figure(figsize=(15, 10))
plt.pcolormesh(X, Y, errors, shading='auto', cmap='coolwarm')
plt.colorbar(label="Error (%)")

# Optionally, overlay grid points at the centers of each subregion
for sr in subregions:
    center_x = (sr['cwnd_lower'] + sr['cwnd_upper']) / 2
    center_y = (sr['rtt_lower'] + sr['rtt_upper']) / 2
    plt.scatter(center_x, center_y, color='black', s=10)
    
plt.title("Error color of each rectangle")
plt.xlabel("cwnd")
plt.ylabel("rtt")
plt.show()