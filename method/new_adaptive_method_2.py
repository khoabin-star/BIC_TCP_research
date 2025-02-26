import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Set maximum allowed intervals for each feature.
# For example:
max_last_max_cwnd = 2   # We aim for 2 intervals along last_max_cwnd.
max_rtt = 4             # We aim for 4 intervals along rtt.

# Read the data from the CSV file.
data = pd.read_csv("../data/output1.csv")

# Compute the global ranges.
global_cwnd_range = data['last_max_cwnd'].max() - data['last_max_cwnd'].min()
global_rtt_range = data['rtt'].max() - data['rtt'].min()

# Define the initial subregion (covering the entire dataset) with split-tracking fields.
initial_subregion = {
    'cwnd_lower': data['last_max_cwnd'].min(),
    'cwnd_upper': data['last_max_cwnd'].max(),
    'rtt_lower': data['rtt'].min(),
    'rtt_upper': data['rtt'].max(),
    'data': data,
    'model_fitted': False,
    # Track the number of splits that have been applied along each dimension.
    'num_cwnd_splits': 0,
    'num_rtt_splits': 0,
    'mse': None,
    'mae': None,
    'mape': None,
}

subregions = [initial_subregion]

# ------------------------------
# Adaptive subdivision process:
# We allow a split along a dimension only if:
#   (a) The number of splits along that dimension for the subregion is below the limit, and
#   (b) The subregion’s range in that dimension is larger than the target width.
# ------------------------------
while True:
    # First, ensure that every subregion has a fitted model.
    for subregion in subregions:
        if not subregion['model_fitted']:
            X = subregion['data'][["last_max_cwnd", "rtt"]]
            y = subregion['data']["result"]
            # If there is insufficient data to fit a model, mark error as infinity.
            if len(X) < 2:
                subregion['mse'] = np.inf
                continue
            model = LinearRegression()
            model.fit(X, y)
            # Scale coefficients to integer values (as in your original code).
            coef_last_max_cwnd = int(round(model.coef_[0] * 1000))
            coef_rtt = int(round(model.coef_[1] * 1000))
            intercept = int(round(model.intercept_ * 1000))
            # Compute integer-based predictions.
            y_pred_int = (coef_last_max_cwnd * X["last_max_cwnd"] +
                          coef_rtt * X["rtt"] + intercept) // 1000
            mse_int = mean_squared_error(y, y_pred_int)
            mae_int = mean_absolute_error(y, y_pred_int)
            mape_int = np.mean(np.abs((y - y_pred_int) / y)) * 100  # MAPE in percentage
            subregion['model'] = {
                'coef_last_max_cwnd': coef_last_max_cwnd,
                'coef_rtt': coef_rtt,
                'intercept': intercept,
            }
            subregion['mse'] = mse_int
            subregion['mae'] = mae_int
            subregion['mape'] = mape_int
            subregion['model_fitted'] = True

    # Build a list of subregions that are eligible for further splitting.
    splittable_subregions = []
    for sub in subregions:
        can_split_cwnd = (sub['num_cwnd_splits'] < (max_last_max_cwnd - 1)) and \
            ((sub['cwnd_upper'] - sub['cwnd_lower']) > (global_cwnd_range / max_last_max_cwnd))
        can_split_rtt = (sub['num_rtt_splits'] < (max_rtt - 1)) and \
            ((sub['rtt_upper'] - sub['rtt_lower']) > (global_rtt_range / max_rtt))
        if (can_split_cwnd or can_split_rtt) and (len(sub['data']) >= 2):
            splittable_subregions.append(sub)
    
    # If no subregion can be further split, exit the loop.
    if not splittable_subregions:
        break

    # Select the subregion with the highest MSE (among those splittable).
    worst_subregion = max(splittable_subregions, key=lambda x: x['mse'] if x['mse'] is not None else -np.inf)

    # Determine if we can split along each dimension for this subregion.
    can_split_cwnd = (worst_subregion['num_cwnd_splits'] < (max_last_max_cwnd - 1)) and \
        ((worst_subregion['cwnd_upper'] - worst_subregion['cwnd_lower']) > (global_cwnd_range / max_last_max_cwnd))
    can_split_rtt = (worst_subregion['num_rtt_splits'] < (max_rtt - 1)) and \
        ((worst_subregion['rtt_upper'] - worst_subregion['rtt_lower']) > (global_rtt_range / max_rtt))
    
    # Decide which dimension to split: if both are allowed, choose the one with the larger range.
    cwnd_range = worst_subregion['cwnd_upper'] - worst_subregion['cwnd_lower']
    rtt_range = worst_subregion['rtt_upper'] - worst_subregion['rtt_lower']
    
    if can_split_cwnd and (not can_split_rtt or cwnd_range >= rtt_range):
        # Split along last_max_cwnd.
        split_value = (worst_subregion['cwnd_lower'] + worst_subregion['cwnd_upper']) / 2
        left_data = worst_subregion['data'][worst_subregion['data']['last_max_cwnd'] < split_value]
        right_data = worst_subregion['data'][worst_subregion['data']['last_max_cwnd'] >= split_value]
        if len(left_data) == 0 or len(right_data) == 0:
            worst_subregion['mse'] = -np.inf  # Mark as non-splittable.
            continue
        left_subregion = {
            'cwnd_lower': worst_subregion['cwnd_lower'],
            'cwnd_upper': split_value,
            'rtt_lower': worst_subregion['rtt_lower'],
            'rtt_upper': worst_subregion['rtt_upper'],
            'data': left_data,
            'model_fitted': False,
            'num_cwnd_splits': worst_subregion['num_cwnd_splits'] + 1,
            'num_rtt_splits': worst_subregion['num_rtt_splits'],
            'mse': None,
            'mae': None,
            'mape': None,
        }
        right_subregion = {
            'cwnd_lower': split_value,
            'cwnd_upper': worst_subregion['cwnd_upper'],
            'rtt_lower': worst_subregion['rtt_lower'],
            'rtt_upper': worst_subregion['rtt_upper'],
            'data': right_data,
            'model_fitted': False,
            'num_cwnd_splits': worst_subregion['num_cwnd_splits'] + 1,
            'num_rtt_splits': worst_subregion['num_rtt_splits'],
            'mse': None,
            'mae': None,
            'mape': None,
        }
    elif can_split_rtt:
        # Split along rtt.
        split_value = (worst_subregion['rtt_lower'] + worst_subregion['rtt_upper']) / 2
        lower_data = worst_subregion['data'][worst_subregion['data']['rtt'] < split_value]
        upper_data = worst_subregion['data'][worst_subregion['data']['rtt'] >= split_value]
        if len(lower_data) == 0 or len(upper_data) == 0:
            worst_subregion['mse'] = -np.inf
            continue
        left_subregion = {
            'cwnd_lower': worst_subregion['cwnd_lower'],
            'cwnd_upper': worst_subregion['cwnd_upper'],
            'rtt_lower': worst_subregion['rtt_lower'],
            'rtt_upper': split_value,
            'data': lower_data,
            'model_fitted': False,
            'num_cwnd_splits': worst_subregion['num_cwnd_splits'],
            'num_rtt_splits': worst_subregion['num_rtt_splits'] + 1,
            'mse': None,
            'mae': None,
            'mape': None,
        }
        right_subregion = {
            'cwnd_lower': worst_subregion['cwnd_lower'],
            'cwnd_upper': worst_subregion['cwnd_upper'],
            'rtt_lower': split_value,
            'rtt_upper': worst_subregion['rtt_upper'],
            'data': upper_data,
            'model_fitted': False,
            'num_cwnd_splits': worst_subregion['num_cwnd_splits'],
            'num_rtt_splits': worst_subregion['num_rtt_splits'] + 1,
            'mse': None,
            'mae': None,
            'mape': None,
        }
    else:
        # If neither dimension can be split, break out.
        break

    # Remove the worst subregion and add the new ones.
    subregions.remove(worst_subregion)
    subregions.extend([left_subregion, right_subregion])

# ------------------------------
# Final model fitting for any subregions that haven't been processed.
# ------------------------------
for subregion in subregions:
    if not subregion['model_fitted']:
        X = subregion['data'][["last_max_cwnd", "rtt"]]
        y = subregion['data']["result"]
        if len(X) < 2:
            subregion['mse'] = np.inf
            continue
        model = LinearRegression()
        model.fit(X, y)
        coef_last_max_cwnd = int(round(model.coef_[0] * 1000))
        coef_rtt = int(round(model.coef_[1] * 1000))
        intercept = int(round(model.intercept_ * 1000))
        y_pred_int = (coef_last_max_cwnd * X["last_max_cwnd"] +
                      coef_rtt * X["rtt"] + intercept) // 1000
        mse_int = mean_squared_error(y, y_pred_int)
        mae_int = mean_absolute_error(y, y_pred_int)
        mape_int = np.mean(np.abs((y - y_pred_int) / y)) * 100
        subregion['model'] = {
            'coef_last_max_cwnd': coef_last_max_cwnd,
            'coef_rtt': coef_rtt,
            'intercept': intercept,
        }
        subregion['mse'] = mse_int
        subregion['mae'] = mae_int
        subregion['mape'] = mape_int
        subregion['model_fitted'] = True

# ------------------------------
# Print out information about each subregion.
# ------------------------------
total_mse = 0.0
total_mae = 0.0
total_mape = 0.0

max_mse = -np.inf
max_mae = -np.inf
max_mape = -np.inf
min_mse = np.inf
min_mae = np.inf
min_mape = np.inf

max_mse_index = None
max_mae_index = None
max_mape_index = None
min_mse_index = None
min_mae_index = None
min_mape_index = None

for index, subregion in enumerate(subregions):
    cwnd_lower = subregion['cwnd_lower']
    cwnd_upper = subregion['cwnd_upper']
    rtt_lower = subregion['rtt_lower']
    rtt_upper = subregion['rtt_upper']

    model_info = subregion.get('model', {})
    mse = subregion.get('mse', np.inf)
    mae = subregion.get('mae', np.inf)
    mape = subregion.get('mape', np.inf)

    total_mse += mse
    total_mae += mae
    total_mape += mape

    if mse > max_mse:
        max_mse = mse
        max_mse_index = index
    if mae > max_mae:
        max_mae = mae
        max_mae_index = index
    if mape > max_mape:
        max_mape = mape
        max_mape_index = index

    if mse < min_mse:
        min_mse = mse
        min_mse_index = index
    if mae < min_mae:
        min_mae = mae
        min_mae_index = index
    if mape < min_mape:
        min_mape = mape
        min_mape_index = index

    print(f"Subregion {index + 1}: last_max_cwnd=({cwnd_lower}, {cwnd_upper}], rtt=({rtt_lower}, {rtt_upper}]")
    print(f"Equation: result = ({model_info.get('coef_last_max_cwnd', 0)} * last_max_cwnd + {model_info.get('coef_rtt', 0)} * rtt + {model_info.get('intercept', 0)}) / 1000")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.6f}%")
    print("-" * 60)

if len(subregions) > 0:
    average_mse = total_mse / len(subregions)
    average_mae = total_mae / len(subregions)
    average_mape = total_mape / len(subregions)
    print(f"Average Mean Squared Error (MSE) across all subregions: {average_mse:.6f}")
    print(f"Average Mean Absolute Error (MAE) across all subregions: {average_mae:.6f}")
    print(f"Average Mean Absolute Percentage Error (MAPE) across all subregions: {average_mape:.6f}%")
else:
    print("No subregions to calculate average errors.")

if max_mse_index is not None:
    worst_mse_subregion = subregions[max_mse_index]
    print(f"\nMax MSE: {max_mse:.6f} at subregion last_max_cwnd=({worst_mse_subregion['cwnd_lower']}, {worst_mse_subregion['cwnd_upper']}], "
          f"rtt=({worst_mse_subregion['rtt_lower']}, {worst_mse_subregion['rtt_upper']}])")
if max_mae_index is not None:
    worst_mae_subregion = subregions[max_mae_index]
    print(f"Max MAE: {max_mae:.6f} at subregion last_max_cwnd=({worst_mae_subregion['cwnd_lower']}, {worst_mae_subregion['cwnd_upper']}], "
          f"rtt=({worst_mae_subregion['rtt_lower']}, {worst_mae_subregion['rtt_upper']}])")
if max_mape_index is not None:
    worst_mape_subregion = subregions[max_mape_index]
    print(f"Max MAPE: {max_mape:.6f}% at subregion last_max_cwnd=({worst_mape_subregion['cwnd_lower']}, {worst_mape_subregion['cwnd_upper']}], "
          f"rtt=({worst_mape_subregion['rtt_lower']}, {worst_mape_subregion['rtt_upper']}])")

if min_mse_index is not None:
    best_mse_subregion = subregions[min_mse_index]
    print(f"\nMin MSE: {min_mse:.6f} at subregion last_max_cwnd=({best_mse_subregion['cwnd_lower']}, {best_mse_subregion['cwnd_upper']}], "
          f"rtt=({best_mse_subregion['rtt_lower']}, {best_mse_subregion['rtt_upper']}])")
