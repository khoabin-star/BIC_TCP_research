import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Read the data from the CSV file
data = pd.read_csv("output.csv")

# Define the number of intervals
num_intervals = 5

# Create intervals for last_max_cwnd and rtt (1-1000)
last_max_cwnd_intervals = np.linspace(1, 1000, num_intervals + 1)
rtt_intervals = np.linspace(1, 1000, num_intervals + 1)

# Round the interval bounds to integers
last_max_cwnd_intervals = np.round(last_max_cwnd_intervals).astype(int)
rtt_intervals = np.round(rtt_intervals).astype(int)

# Initialize a dictionary to store the piecewise models and errors
piecewise_models = {}

# Initialize variables to calculate average MSE
total_mse = 0
subregion_count = 0

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

        # Calculate the error (Mean Squared Error and R² score)
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        # Accumulate MSE and count for average calculation
        total_mse += mse
        subregion_count += 1

        # Store the model and error in the dictionary
        piecewise_models[(i, j)] = {
            "model": model,
            "cwnd_bounds": (cwnd_lower, cwnd_upper),
            "rtt_bounds": (rtt_lower, rtt_upper),
            "mse": mse,
            "r2": r2,
        }

# Print all piecewise linear models and their errors
for (i, j), subregion in piecewise_models.items():
    cwnd_lower, cwnd_upper = subregion["cwnd_bounds"]
    rtt_lower, rtt_upper = subregion["rtt_bounds"]
    model = subregion["model"]
    mse = subregion["mse"]
    r2 = subregion["r2"]

    # Get the coefficients and intercept of the linear model
    coef_last_max_cwnd, coef_rtt = model.coef_
    intercept = model.intercept_

    # Print the equation and error for the subregion
    print(f"Subregion: last_max_cwnd=[{cwnd_lower}, {cwnd_upper}), rtt=[{rtt_lower}, {rtt_upper})")
    print(f"Equation: result = {coef_last_max_cwnd:.6f} * last_max_cwnd + {coef_rtt:.6f} * rtt + {intercept:.6f}")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"R² Score: {r2:.6f}")
    print("-" * 60)

# Calculate and print the average MSE
if subregion_count > 0:
    average_mse = total_mse / subregion_count
    print(f"Average Mean Squared Error (MSE) across all subregions: {average_mse:.6f}")
else:
    print("No sufficient data in any subregion to calculate average MSE.")
