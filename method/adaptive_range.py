import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

data = pd.read_csv("../data/output.csv")

# Define the maximum number of subregions
max_subregions = 9

# Initialize the subregions
subregions = [
    {
        "cwnd_bounds": (1, 500),
        "rtt_bounds": (1, 500),
    },
    {
        "cwnd_bounds": (1, 500),
        "rtt_bounds": (500, 1000),
    },
    {
        "cwnd_bounds": (500, 1000),
        "rtt_bounds": (1, 500),
    },
    {
        "cwnd_bounds": (500, 1000),
        "rtt_bounds": (500, 1000),
    },
]

# Function to fit a model and calculate MSE for a subregion
def fit_model_and_calculate_error(subregion, data):
    cwnd_lower, cwnd_upper = subregion["cwnd_bounds"]
    rtt_lower, rtt_upper = subregion["rtt_bounds"]
    
    # Filter data for the subregion
    subregion_data = data[
        (data["last_max_cwnd"] >= cwnd_lower) & (data["last_max_cwnd"] < cwnd_upper) &
        (data["rtt"] >= rtt_lower) & (data["rtt"] < rtt_upper)
    ]
    
    # Skip if there's not enough data
    if len(subregion_data) < 2:
        return None, np.inf 
    
    # Fit a linear regression model
    X = subregion_data[["last_max_cwnd", "rtt"]]
    y = subregion_data["result"]
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate MSE on the subregion data (in-sample error)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    
    return model, mse

# Adaptive splitting loop
while len(subregions) < max_subregions:
    # Fit models and calculate errors for all subregions
    errors = []
    for subregion in subregions:
        model, mse = fit_model_and_calculate_error(subregion, data)
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

# Calculate total MSE and subregion count for average MSE calculation
total_mse = 0
subregion_count = 0

# Print the final subregions and their errors
for i, subregion in enumerate(subregions):
    model, mse = fit_model_and_calculate_error(subregion, data)
    if model is not None:
        coef_last_max_cwnd, coef_rtt = model.coef_
        intercept = model.intercept_
        print(f"Subregion {i + 1}: last_max_cwnd={subregion['cwnd_bounds']}, rtt={subregion['rtt_bounds']}")
        print(f"Equation: result = {coef_last_max_cwnd:.6f} * last_max_cwnd + {coef_rtt:.6f} * rtt + {intercept:.6f}")
        print(f"Mean Squared Error (MSE): {mse:.6f}")
        print("-" * 60)
        
        total_mse += mse
        subregion_count += 1

# Calculate and print the average MSE
if subregion_count > 0:
    average_mse = total_mse / subregion_count
    print(f"Average Mean Squared Error (MSE) across all subregions: {average_mse:.6f}")
else:
    print("No sufficient data in any subregion to calculate average MSE.")
