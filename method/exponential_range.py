import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Read the data from the CSV file
data = pd.read_csv("../data/output.csv")

# Define the exponential intervals
last_cwnd_intervals = [1, 10, 100, 1000, 5000, 10000]  # Exponential ranges
rtt_intervals = [1, 10, 100, 400, 700, 1000] 
# Initialize a dictionary to store the piecewise models and errors
piecewise_models = {}

# Initialize variables to calculate average MSE
total_mse = 0
subregion_count = 0

# Fit piecewise linear models for each subregion
for i in range(len(last_cwnd_intervals) - 1):
    for j in range(len(rtt_intervals) - 1):
        # Define the bounds for the current subregion
        cwnd_lower, cwnd_upper = last_cwnd_intervals[i], last_cwnd_intervals[i + 1]
        rtt_lower, rtt_upper = rtt_intervals[j], rtt_intervals[j + 1]
        
        # Filter data for the current subregion (training data)
        subregion_data = data[
            (data["last_max_cwnd"] >= cwnd_lower) & (data["last_max_cwnd"] < cwnd_upper) &
            (data["rtt"] >= rtt_lower) & (data["rtt"] < rtt_upper)
        ]
        
        # Skip if there's not enough data in the subregion
        if len(subregion_data) < 2:
            print(f"No sufficient data for subregion: last_max_cwnd=[{cwnd_lower}, {cwnd_upper}), rtt=[{rtt_lower}, {rtt_upper})")
            continue
        
        # Prepare features and target for training
        X_train = subregion_data[["last_max_cwnd", "rtt"]]
        y_train = subregion_data["result"]
        
        # Fit a linear regression model for the subregion
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Calculate in-sample error (MSE and R² score for the same subregion data)
        y_pred = model.predict(X_train)
        mse = mean_squared_error(y_train, y_pred)
        r2 = r2_score(y_train, y_pred)

        # Accumulate MSE for average calculation
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