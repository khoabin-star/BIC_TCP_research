import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Read the data from the CSV file
data = pd.read_csv("output2.csv")

# Define the maximum number of subregions (intervals)
max_subregions = 25  # You can adjust this number as needed

# Define initial subregion covering the entire data range
initial_subregion = {
    'cwnd_lower': data['last_max_cwnd'].min(),
    'cwnd_upper': data['last_max_cwnd'].max(),
    'rtt_lower': data['rtt'].min(),
    'rtt_upper': data['rtt'].max(),
    'data': data,
    'model_fitted': False,
    'mse': None,
    'mae': None,
    'mape': None,
}

# Initialize the list of subregions
subregions = [initial_subregion]

# Total number of subregions (intervals)
total_subregions = 1

# Start the adaptive subdivision process
while total_subregions < max_subregions:
    # Fit models and compute errors for subregions that haven't been processed yet
    for subregion in subregions:
        if not subregion['model_fitted']:
            # Prepare features and target
            X = subregion['data'][["last_max_cwnd", "rtt"]]
            y = subregion['data']["result"]

            # Check if there is enough data to fit the model
            if len(X) < 2:
                subregion['mse'] = np.inf
                continue

            # Fit a linear regression model for the subregion
            model = LinearRegression()
            model.fit(X, y)

            # Convert coefficients to integer-scaled values
            coef_last_max_cwnd = int(round(model.coef_[0] * 1000))
            coef_rtt = int(round(model.coef_[1] * 1000))
            intercept = int(round(model.intercept_ * 1000))

            # Compute integer-based predictions
            y_pred_int = (coef_last_max_cwnd * X["last_max_cwnd"] + coef_rtt * X["rtt"] + intercept) // 1000

            # Compute error metrics
            mse_int = mean_squared_error(y, y_pred_int)
            mae_int = mean_absolute_error(y, y_pred_int)
            mape_int = np.mean(np.abs((y - y_pred_int) / y)) * 100  # MAPE in percentage

            # Update subregion with model and errors
            subregion['model'] = {
                'coef_last_max_cwnd': coef_last_max_cwnd,
                'coef_rtt': coef_rtt,
                'intercept': intercept,
            }
            subregion['mse'] = mse_int
            subregion['mae'] = mae_int
            subregion['mape'] = mape_int
            subregion['model_fitted'] = True

    # Identify the subregion with the highest MSE
    worst_subregion = max(subregions, key=lambda x: x['mse'] if x['mse'] is not None else -np.inf)

    # Check if the worst subregion can be subdivided
    if len(worst_subregion['data']) < 2 or worst_subregion['mse'] == np.inf:
        # Cannot subdivide further
        break

    # Subdivide the worst subregion
    # Decide which dimension to split based on data range
    cwnd_range = worst_subregion['cwnd_upper'] - worst_subregion['cwnd_lower']
    rtt_range = worst_subregion['rtt_upper'] - worst_subregion['rtt_lower']

    if cwnd_range >= rtt_range:
        # Split along last_max_cwnd
        split_value = (worst_subregion['cwnd_lower'] + worst_subregion['cwnd_upper']) / 2
        left_data = worst_subregion['data'][worst_subregion['data']['last_max_cwnd'] < split_value]
        right_data = worst_subregion['data'][worst_subregion['data']['last_max_cwnd'] >= split_value]

        if len(left_data) == 0 or len(right_data) == 0:
            # Cannot split further along this dimension
            break

        left_subregion = {
            'cwnd_lower': worst_subregion['cwnd_lower'],
            'cwnd_upper': split_value,
            'rtt_lower': worst_subregion['rtt_lower'],
            'rtt_upper': worst_subregion['rtt_upper'],
            'data': left_data,
            'model_fitted': False,
        }
        right_subregion = {
            'cwnd_lower': split_value,
            'cwnd_upper': worst_subregion['cwnd_upper'],
            'rtt_lower': worst_subregion['rtt_lower'],
            'rtt_upper': worst_subregion['rtt_upper'],
            'data': right_data,
            'model_fitted': False,
        }
    else:
        # Split along rtt
        split_value = (worst_subregion['rtt_lower'] + worst_subregion['rtt_upper']) / 2
        lower_data = worst_subregion['data'][worst_subregion['data']['rtt'] < split_value]
        upper_data = worst_subregion['data'][worst_subregion['data']['rtt'] >= split_value]

        if len(lower_data) == 0 or len(upper_data) == 0:
            # Cannot split further along this dimension
            break

        left_subregion = {
            'cwnd_lower': worst_subregion['cwnd_lower'],
            'cwnd_upper': worst_subregion['cwnd_upper'],
            'rtt_lower': worst_subregion['rtt_lower'],
            'rtt_upper': split_value,
            'data': lower_data,
            'model_fitted': False,
        }
        right_subregion = {
            'cwnd_lower': worst_subregion['cwnd_lower'],
            'cwnd_upper': worst_subregion['cwnd_upper'],
            'rtt_lower': split_value,
            'rtt_upper': worst_subregion['rtt_upper'],
            'data': upper_data,
            'model_fitted': False,
        }

    # Remove the worst subregion and add the new ones
    subregions.remove(worst_subregion)
    subregions.extend([left_subregion, right_subregion])
    total_subregions += 1
    if total_subregions >= max_subregions:
        break

# After subdivision, fit models for remaining subregions if not already done
for subregion in subregions:
    if not subregion['model_fitted']:
        # Prepare features and target
        X = subregion['data'][["last_max_cwnd", "rtt"]]
        y = subregion['data']["result"]

        # Check if there is enough data to fit the model
        if len(X) < 2:
            subregion['mse'] = np.inf
            continue

        # Fit a linear regression model for the subregion
        model = LinearRegression()
        model.fit(X, y)

        # Convert coefficients to integer-scaled values
        coef_last_max_cwnd = int(round(model.coef_[0] * 1000))
        coef_rtt = int(round(model.coef_[1] * 1000))
        intercept = int(round(model.intercept_ * 1000))

        # Compute integer-based predictions
        y_pred_int = (coef_last_max_cwnd * X["last_max_cwnd"] + coef_rtt * X["rtt"] + intercept) // 1000

        # Compute error metrics
        mse_int = mean_squared_error(y, y_pred_int)
        mae_int = mean_absolute_error(y, y_pred_int)
        mape_int = np.mean(np.abs((y - y_pred_int) / y)) * 100  # MAPE in percentage

        # Update subregion with model and errors
        subregion['model'] = {
            'coef_last_max_cwnd': coef_last_max_cwnd,
            'coef_rtt': coef_rtt,
            'intercept': intercept,
        }
        subregion['mse'] = mse_int
        subregion['mae'] = mae_int
        subregion['mape'] = mape_int
        subregion['model_fitted'] = True

# Initialize variables to calculate average errors
total_mse = 0
total_mae = 0
total_mape = 0

# Variables to track highest errors
max_mse = -np.inf
max_mse_subregion = None
max_mae = -np.inf
max_mae_subregion = None
max_mape = -np.inf
max_mape_subregion = None

# Variables to track least errors
min_mse = np.inf
min_mse_subregion = None
min_mae = np.inf
min_mae_subregion = None
min_mape = np.inf
min_mape_subregion = None

# Print all piecewise linear models and their errors
for index, subregion in enumerate(subregions):
    cwnd_lower = subregion['cwnd_lower']
    cwnd_upper = subregion['cwnd_upper']
    rtt_lower = subregion['rtt_lower']
    rtt_upper = subregion['rtt_upper']
    model_info = subregion['model']
    mse = subregion['mse']
    mae = subregion['mae']
    mape = subregion['mape']

    # Accumulate errors for averages
    total_mse += mse
    total_mae += mae
    total_mape += mape

    # Update max errors if needed
    if mse > max_mse:
        max_mse = mse
        max_mse_subregion = index

    if mae > max_mae:
        max_mae = mae
        max_mae_subregion = index

    if mape > max_mape:
        max_mape = mape
        max_mape_subregion = index

    # Update min errors if needed
    if mse < min_mse:
        min_mse = mse
        min_mse_subregion = index

    if mae < min_mae:
        min_mae = mae
        min_mae_subregion = index

    if mape < min_mape:
        min_mape = mape
        min_mape_subregion = index

    # Print the integer-based equation
    print(f"Subregion {index + 1}: last_max_cwnd=({cwnd_lower}, {cwnd_upper}], rtt=({rtt_lower}, {rtt_upper}]")
    print(f"Equation: result = ({model_info['coef_last_max_cwnd']} * last_max_cwnd + {model_info['coef_rtt']} * rtt + {model_info['intercept']}) / 1000")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.6f}%")
    print("-" * 60)

# Calculate and print the average errors
if len(subregions) > 0:
    average_mse = total_mse / len(subregions)
    average_mae = total_mae / len(subregions)
    average_mape = total_mape / len(subregions)
    print(f"Average Mean Squared Error (MSE) across all subregions: {average_mse:.6f}")
    print(f"Average Mean Absolute Error (MAE) across all subregions: {average_mae:.6f}")
    print(f"Average Mean Absolute Percentage Error (MAPE) across all subregions: {average_mape:.6f}%")
else:
    print("No subregions to calculate average errors.")

# Print the subregions with the highest errors
worst_mse_subregion = subregions[max_mse_subregion]
print(f"\nMax MSE: {max_mse:.6f} at subregion last_max_cwnd=({worst_mse_subregion['cwnd_lower']}, {worst_mse_subregion['cwnd_upper']}], rtt=({worst_mse_subregion['rtt_lower']}, {worst_mse_subregion['rtt_upper']}])")
worst_mae_subregion = subregions[max_mae_subregion]
print(f"Max MAE: {max_mae:.6f} at subregion last_max_cwnd=({worst_mae_subregion['cwnd_lower']}, {worst_mae_subregion['cwnd_upper']}], rtt=({worst_mae_subregion['rtt_lower']}, {worst_mae_subregion['rtt_upper']}])")
worst_mape_subregion = subregions[max_mape_subregion]
print(f"Max MAPE: {max_mape:.6f}% at subregion last_max_cwnd=({worst_mape_subregion['cwnd_lower']}, {worst_mape_subregion['cwnd_upper']}], rtt=({worst_mape_subregion['rtt_lower']}, {worst_mape_subregion['rtt_upper']}])")

# Print the subregions with the least errors
best_mse_subregion = subregions[min_mse_subregion]
print(f"\nMin MSE: {min_mse:.6f} at subregion last_max_cwnd=({best_mse_subregion['cwnd_lower']}, {best_mse_subregion['cwnd_upper']}], rtt=({best_mse_subregion['rtt_lower']}, {best_mse_subregion['rtt_upper']}])")
best_mae_subregion = subregions[min_mae_subregion]
print(f"Min MAE: {min_mae:.6f} at subregion last_max_cwnd=({best_mae_subregion['cwnd_lower']}, {best_mae_subregion['cwnd_upper']}], rtt=({best_mae_subregion['rtt_lower']}, {best_mae_subregion['rtt_upper']}])")
best_mape_subregion = subregions[min_mape_subregion]
print(f"Min MAPE: {min_mape:.6f}% at subregion last_max_cwnd=({best_mape_subregion['cwnd_lower']}, {best_mape_subregion['cwnd_upper']}], rtt=({best_mape_subregion['rtt_lower']}, {best_mape_subregion['rtt_upper']}])")
