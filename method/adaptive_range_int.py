import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Read the data from the CSV file
data = pd.read_csv("../data/output.csv")

# Define the maximum number of subregions
max_subregions = 9

# Initialize the subregions
subregions = [
    {"cwnd_bounds": (1, 500), "rtt_bounds": (1, 500)},
    {"cwnd_bounds": (1, 500), "rtt_bounds": (500, 1000)},
    {"cwnd_bounds": (500, 1000), "rtt_bounds": (1, 500)},
    {"cwnd_bounds": (500, 1000), "rtt_bounds": (500, 1000)},
]

# Function to fit a model and calculate integer-based errors for a subregion
def fit_model_and_calculate_errors(subregion, data):
    cwnd_lower, cwnd_upper = subregion["cwnd_bounds"]
    rtt_lower, rtt_upper = subregion["rtt_bounds"]

    # Filter data for the subregion
    subregion_data = data[
        (data["last_max_cwnd"] >= cwnd_lower) & (data["last_max_cwnd"] < cwnd_upper) &
        (data["rtt"] >= rtt_lower) & (data["rtt"] < rtt_upper)
    ]

    # Skip if there's not enough data
    if len(subregion_data) < 2:
        return None, np.inf, np.inf, np.inf

    # Fit a linear regression model
    X = subregion_data[["last_max_cwnd", "rtt"]]
    y = subregion_data["result"]
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

    return model, mse_int, mae_int, mape_int, coef_last_max_cwnd, coef_rtt, intercept

# Adaptive splitting loop
while len(subregions) < max_subregions:
    # Fit models and calculate errors for all subregions
    errors = []
    for subregion in subregions:
        _, mse, _, _, _, _, _ = fit_model_and_calculate_errors(subregion, data)
        errors.append(mse)

    # Find the subregion with the worst error
    worst_index = np.argmax(errors)
    worst_subregion = subregions[worst_index]

    # Split the worst subregion into 4 new subregions
    cwnd_lower, cwnd_upper = worst_subregion["cwnd_bounds"]
    rtt_lower, rtt_upper = worst_subregion["rtt_bounds"]

    cwnd_mid = (cwnd_lower + cwnd_upper) / 2
    rtt_mid = (rtt_lower + rtt_upper) / 2

    # Create new subregions
    new_subregions = [
        {"cwnd_bounds": (cwnd_lower, cwnd_mid), "rtt_bounds": (rtt_lower, rtt_mid)},
        {"cwnd_bounds": (cwnd_lower, cwnd_mid), "rtt_bounds": (rtt_mid, rtt_upper)},
        {"cwnd_bounds": (cwnd_mid, cwnd_upper), "rtt_bounds": (rtt_lower, rtt_mid)},
        {"cwnd_bounds": (cwnd_mid, cwnd_upper), "rtt_bounds": (rtt_mid, rtt_upper)},
    ]

    # Replace the worst subregion with the new subregions
    subregions.pop(worst_index)
    subregions.extend(new_subregions)

# Calculate total errors and track worst subregions
total_mse, total_mae, total_mape, subregion_count = 0, 0, 0, 0
worst_mse, worst_mae, worst_mape = -np.inf, -np.inf, -np.inf
least_mse, least_mae, least_mape = np.inf, np.inf, np.inf
worst_mse_subregion, worst_mae_subregion, worst_mape_subregion = None, None, None
least_mse_subregion, least_mae_subregion, least_mape_subregion = None, None, None

# Print the final subregions and their errors
for i, subregion in enumerate(subregions):
    model, mse, mae, mape, coef_last_max_cwnd, coef_rtt, intercept = fit_model_and_calculate_errors(subregion, data)
    if model is not None:
        print(f"Subregion {i + 1}: last_max_cwnd={subregion['cwnd_bounds']}, rtt={subregion['rtt_bounds']}")
        print(f"Equation: result = ({coef_last_max_cwnd} * last_max_cwnd + {coef_rtt} * rtt + {intercept}) // 1000")
        print(f"Mean Squared Error (MSE): {mse:.6f}")
        print(f"Mean Absolute Error (MAE): {mae:.6f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.6f}%")
        print("-" * 60)

        total_mse += mse
        total_mae += mae
        total_mape += mape
        subregion_count += 1

        # Track the worst subregions
        if mse > worst_mse:
            worst_mse = mse
            worst_mse_subregion = subregion
        if mae > worst_mae:
            worst_mae = mae
            worst_mae_subregion = subregion
        if mape > worst_mape:
            worst_mape = mape
            worst_mape_subregion = subregion

        # Track the least (minimum) errors
        if mse < least_mse:
            least_mse = mse
            least_mse_subregion = subregion
        if mae < least_mae:
            least_mae = mae
            least_mae_subregion = subregion
        if mape < least_mape:
            least_mape = mape
            least_mape_subregion = subregion

# Calculate and print the average errors
if subregion_count > 0:
    avg_mse = total_mse / subregion_count
    avg_mae = total_mae / subregion_count
    avg_mape = total_mape / subregion_count
    print(f"Average MSE across all subregions: {avg_mse:.6f}")
    print(f"Average MAE across all subregions: {avg_mae:.6f}")
    print(f"Average MAPE across all subregions: {avg_mape:.6f}%")
else:
    print("No sufficient data in any subregion to calculate average errors.")

# Print subregions with highest and least errors
if worst_mse_subregion:
    print("\nSubregion with Highest MSE:")
    print(f"Bounds: last_max_cwnd={worst_mse_subregion['cwnd_bounds']}, rtt={worst_mse_subregion['rtt_bounds']}")
    print(f"Highest MSE: {worst_mse:.6f}")

if worst_mae_subregion:
    print("\nSubregion with Highest MAE:")
    print(f"Bounds: last_max_cwnd={worst_mae_subregion['cwnd_bounds']}, rtt={worst_mae_subregion['rtt_bounds']}")
    print(f"Highest MAE: {worst_mae:.6f}")

if worst_mape_subregion:
    print("\nSubregion with Highest MAPE:")
    print(f"Bounds: last_max_cwnd={worst_mape_subregion['cwnd_bounds']}, rtt={worst_mape_subregion['rtt_bounds']}")
    print(f"Highest MAPE: {worst_mape:.6f}%")

# Print subregions with least errors
if least_mse_subregion:
    print("\nSubregion with Least MSE:")
    print(f"Bounds: last_max_cwnd={least_mse_subregion['cwnd_bounds']}, rtt={least_mse_subregion['rtt_bounds']}")
    print(f"Least MSE: {least_mse:.6f}")

if least_mae_subregion:
    print("\nSubregion with Least MAE:")
    print(f"Bounds: last_max_cwnd={least_mae_subregion['cwnd_bounds']}, rtt={least_mae_subregion['rtt_bounds']}")
    print(f"Least MAE: {least_mae:.6f}")

if least_mape_subregion:
    print("\nSubregion with Least MAPE:")
    print(f"Bounds: last_max_cwnd={least_mape_subregion['cwnd_bounds']}, rtt={least_mape_subregion['rtt_bounds']}")
    print(f"Least MAPE: {least_mape:.6f}%")
