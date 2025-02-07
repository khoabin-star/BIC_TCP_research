import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Read the data from the CSV file
data = pd.read_csv("output.csv")

# Define the exponential intervals
intervals = [1, 10, 100, 1000]  # Exponential ranges: [1, 10), [10, 100), [100, 1000]

# Initialize a dictionary to store the piecewise models and errors
piecewise_models = {}

# Fit piecewise linear models for each subregion
for i in range(len(intervals) - 1):
    for j in range(len(intervals) - 1):
        # Define the bounds for the current subregion
        cwnd_lower, cwnd_upper = intervals[i], intervals[i + 1]
        rtt_lower, rtt_upper = intervals[j], intervals[j + 1]
        
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
        
        # Test the model on ALL data points (entire range)
        X_test = data[["last_max_cwnd", "rtt"]]
        y_test = data["result"]
        y_pred = model.predict(X_test)
        
        # Calculate the error (Mean Squared Error and R² score) on the entire dataset
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
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
    print(f"Mean Squared Error (MSE) on entire dataset: {mse:.6f}")
    print(f"R² Score on entire dataset: {r2:.6f}")
    print("-" * 60)