import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Read the data from the CSV file
data = pd.read_csv("output2.csv")

# Define the number of intervals
num_intervals = 4

# Create intervals for last_max_cwnd and rtt (1-1000)
last_max_cwnd_intervals = np.linspace(1, 100000, num_intervals + 1)
rtt_intervals = np.linspace(1, 1000, num_intervals + 1)

# Round the interval bounds to integers
last_max_cwnd_intervals = np.round(last_max_cwnd_intervals).astype(int)
rtt_intervals = np.round(rtt_intervals).astype(int)

# Initialize a dictionary to store the piecewise models and errors
piecewise_models = {}

# Initialize variables to calculate average errors
total_mse = 0
total_mae = 0
total_mape = 0
subregion_count = 0

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

# Fit piecewise linear models for each subregion
for i in range(num_intervals):
    for j in range(num_intervals):
        # Define the bounds for the current subregion
        cwnd_lower, cwnd_upper = last_max_cwnd_intervals[i], last_max_cwnd_intervals[i + 1]
        rtt_lower, rtt_upper = rtt_intervals[j], rtt_intervals[j + 1]

        # Filter data for the current subregion
        subregion_data = data[
            (data["last_max_cwnd"] >= cwnd_lower) & (data["last_max_cwnd"] < cwnd_upper) &
            (data["rtt"] >= rtt_lower) & (data["rtt"] < rtt_upper)
        ]

        # Skip if there's not enough data in the subregion
        if len(subregion_data) < 2:
            print(f"No sufficient data for subregion: last_max_cwnd=[{cwnd_lower}, {cwnd_upper}), rtt=[{rtt_lower}, {rtt_upper})")
            continue

        # Prepare features and target
        X = subregion_data[["last_max_cwnd", "rtt"]]
        y = subregion_data["result"]

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
        
        # Accumulate errors for averages
        total_mse += mse_int
        total_mae += mae_int
        total_mape += mape_int
        subregion_count += 1

        # Store the model and errors in the dictionary
        piecewise_models[(i, j)] = {
            "cwnd_bounds": (cwnd_lower, cwnd_upper),
            "rtt_bounds": (rtt_lower, rtt_upper),
            "coef_last_max_cwnd": coef_last_max_cwnd,
            "coef_rtt": coef_rtt,
            "intercept": intercept,
            "mse": mse_int,
            "mae": mae_int,
            "mape": mape_int,
        }

        # Update max errors if needed
        if mse_int > max_mse:
            max_mse = mse_int
            max_mse_subregion = (i, j)

        if mae_int > max_mae:
            max_mae = mae_int
            max_mae_subregion = (i, j)

        if mape_int > max_mape:
            max_mape = mape_int
            max_mape_subregion = (i, j)

        # Update min errors if needed
        if mse_int < min_mse:
            min_mse = mse_int
            min_mse_subregion = (i, j)

        if mae_int < min_mae:
            min_mae = mae_int
            min_mae_subregion = (i, j)

        if mape_int < min_mape:
            min_mape = mape_int
            min_mape_subregion = (i, j)

# Print all piecewise linear models and their errors
for (i, j), subregion in piecewise_models.items():
    cwnd_lower, cwnd_upper = subregion["cwnd_bounds"]
    rtt_lower, rtt_upper = subregion["rtt_bounds"]
    coef_last_max_cwnd = subregion["coef_last_max_cwnd"]
    coef_rtt = subregion["coef_rtt"]
    intercept = subregion["intercept"]
    mse = subregion["mse"]
    mae = subregion["mae"]
    mape = subregion["mape"]

    # Print the integer-based equation
    print(f"Subregion: last_max_cwnd=[{cwnd_lower}, {cwnd_upper}), rtt=[{rtt_lower}, {rtt_upper})")
    print(f"Equation: result = ({coef_last_max_cwnd} * last_max_cwnd + {coef_rtt} * rtt + {intercept}) / 1000")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.6f}%")
    print("-" * 60)

# Calculate and print the average errors
if subregion_count > 0:
    average_mse = total_mse / subregion_count
    average_mae = total_mae / subregion_count
    average_mape = total_mape / subregion_count
    print(f"Average Mean Squared Error (MSE) across all subregions: {average_mse:.6f}")
    print(f"Average Mean Absolute Error (MAE) across all subregions: {average_mae:.6f}")
    print(f"Average Mean Absolute Percentage Error (MAPE) across all subregions: {average_mape:.6f}%")
else:
    print("No sufficient data in any subregion to calculate average errors.")

# Print the subregions with the highest errors
print(f"\nMax MSE: {max_mse:.6f} at subregion last_max_cwnd=[{last_max_cwnd_intervals[max_mse_subregion[0]]}, {last_max_cwnd_intervals[max_mse_subregion[0] + 1]}), rtt=[{rtt_intervals[max_mse_subregion[1]]}, {rtt_intervals[max_mse_subregion[1] + 1]}))")
print(f"Max MAE: {max_mae:.6f} at subregion last_max_cwnd=[{last_max_cwnd_intervals[max_mae_subregion[0]]}, {last_max_cwnd_intervals[max_mae_subregion[0] + 1]}), rtt=[{rtt_intervals[max_mae_subregion[1]]}, {rtt_intervals[max_mae_subregion[1] + 1]}))")
print(f"Max MAPE: {max_mape:.6f}% at subregion last_max_cwnd=[{last_max_cwnd_intervals[max_mape_subregion[0]]}, {last_max_cwnd_intervals[max_mape_subregion[0] + 1]}), rtt=[{rtt_intervals[max_mape_subregion[1]]}, {rtt_intervals[max_mape_subregion[1] + 1]}))")

# Print the subregions with the least errors
print(f"\nMin MSE: {min_mse:.6f} at subregion last_max_cwnd=[{last_max_cwnd_intervals[min_mse_subregion[0]]}, {last_max_cwnd_intervals[min_mse_subregion[0] + 1]}), rtt=[{rtt_intervals[min_mse_subregion[1]]}, {rtt_intervals[min_mse_subregion[1] + 1]}))")
print(f"Min MAE: {min_mae:.6f} at subregion last_max_cwnd=[{last_max_cwnd_intervals[min_mae_subregion[0]]}, {last_max_cwnd_intervals[min_mae_subregion[0] + 1]}), rtt=[{rtt_intervals[min_mae_subregion[1]]}, {rtt_intervals[min_mae_subregion[1] + 1]}))")
print(f"Min MAPE: {min_mape:.6f}% at subregion last_max_cwnd=[{last_max_cwnd_intervals[min_mape_subregion[0]]}, {last_max_cwnd_intervals[min_mape_subregion[0] + 1]}), rtt=[{rtt_intervals[min_mape_subregion[1]]}, {rtt_intervals[min_mape_subregion[1] + 1]}))")
