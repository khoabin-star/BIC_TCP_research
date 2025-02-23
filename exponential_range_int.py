import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Read the data from the CSV file
data = pd.read_csv("output2.csv")

# Define the number of intervals
num_intervals = 5

# Function to generate exponential intervals for a given range
def generate_exponential_intervals_int(lower, upper, num_intervals):
    log_lower = np.log10(lower)
    log_upper = np.log10(upper)
    
    step = (log_upper - log_lower) / num_intervals  # Calculate the step on log scale
    intervals = [int(round(lower * 10**(i * step))) for i in range(num_intervals + 1)]
    
    return intervals

# Example for last_cwnd range (1 to 10,000) and rtt range (1 to 1000)
last_cwnd_intervals = generate_exponential_intervals_int(1, 100000, num_intervals)
rtt_intervals = generate_exponential_intervals_int(1, 1000, num_intervals)

print("last_cwnd intervals:", last_cwnd_intervals)
print("rtt intervals:", rtt_intervals)

# Initialize a dictionary to store the piecewise models and errors
piecewise_models = {}

# Initialize variables to calculate average errors
total_mse = 0
total_mae = 0
total_mape = 0
subregion_count = 0

# Initialize variables to track the maximum and minimum errors and corresponding subregions
max_mse = -np.inf
max_mae = -np.inf
max_mape = -np.inf
min_mse = np.inf
min_mae = np.inf
min_mape = np.inf
max_mse_subregion = None
max_mae_subregion = None
max_mape_subregion = None
min_mse_subregion = None
min_mae_subregion = None
min_mape_subregion = None

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
        
        # Convert coefficients to integer-friendly format
        coef_last_max_cwnd = round(model.coef_[0] * 1000)
        coef_rtt = round(model.coef_[1] * 1000)
        intercept = round(model.intercept_ * 1000)
        
        # Compute integer-based predictions
        y_pred_int = (coef_last_max_cwnd * X_train["last_max_cwnd"] +
                      coef_rtt * X_train["rtt"] + intercept) / 1000
        
        # Calculate integer-based error metrics
        mse = mean_squared_error(y_train, y_pred_int)
        mae = mean_absolute_error(y_train, y_pred_int)
        mape = np.mean(np.abs((y_train - y_pred_int) / y_train)) * 100

        # Accumulate errors for average calculation
        total_mse += mse
        total_mae += mae
        total_mape += mape
        subregion_count += 1
        
        # Track the maximum errors and corresponding subregions
        if mse > max_mse:
            max_mse = mse
            max_mse_subregion = (cwnd_lower, cwnd_upper, rtt_lower, rtt_upper)
        if mae > max_mae:
            max_mae = mae
            max_mae_subregion = (cwnd_lower, cwnd_upper, rtt_lower, rtt_upper)
        if mape > max_mape:
            max_mape = mape
            max_mape_subregion = (cwnd_lower, cwnd_upper, rtt_lower, rtt_upper)

        # Track the minimum errors and corresponding subregions
        if mse < min_mse:
            min_mse = mse
            min_mse_subregion = (cwnd_lower, cwnd_upper, rtt_lower, rtt_upper)
        if mae < min_mae:
            min_mae = mae
            min_mae_subregion = (cwnd_lower, cwnd_upper, rtt_lower, rtt_upper)
        if mape < min_mape:
            min_mape = mape
            min_mape_subregion = (cwnd_lower, cwnd_upper, rtt_lower, rtt_upper)
        
        # Store the model and error in the dictionary
        piecewise_models[(i, j)] = {
            "cwnd_bounds": (cwnd_lower, cwnd_upper),
            "rtt_bounds": (rtt_lower, rtt_upper),
            "mse": mse,
            "mae": mae,
            "mape": mape,
            "coef_last_max_cwnd": coef_last_max_cwnd,
            "coef_rtt": coef_rtt,
            "intercept": intercept,
        }

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
    
    print(f"Subregion: last_max_cwnd=[{cwnd_lower}, {cwnd_upper}), rtt=[{rtt_lower}, {rtt_upper})")
    print(f"Equation: result = ({coef_last_max_cwnd} * last_max_cwnd + {coef_rtt} * rtt + {intercept}) / 1000")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.6f}%")
    print("-" * 60)

# Calculate and print the average errors
if subregion_count > 0:
    avg_mse = total_mse / subregion_count
    avg_mae = total_mae / subregion_count
    avg_mape = total_mape / subregion_count
    print(f"Average Mean Squared Error (MSE): {avg_mse:.6f}")
    print(f"Average Mean Absolute Error (MAE): {avg_mae:.6f}")
    print(f"Average Mean Absolute Percentage Error (MAPE): {avg_mape:.6f}%")
else:
    print("No sufficient data in any subregion to calculate average errors.")

# Print the subregion with the highest and lowest errors
if max_mse_subregion:
    print(f"Max MSE = {max_mse:.6f} at subregion: last_max_cwnd={max_mse_subregion[0]}-{max_mse_subregion[1]}, rtt={max_mse_subregion[2]}-{max_mse_subregion[3]}")
if max_mae_subregion:
    print(f"Max MAE = {max_mae:.6f} at subregion: last_max_cwnd={max_mae_subregion[0]}-{max_mae_subregion[1]}, rtt={max_mae_subregion[2]}-{max_mae_subregion[3]}")
if max_mape_subregion:
    print(f"Max MAPE = {max_mape:.6f}% at subregion: last_max_cwnd={max_mape_subregion[0]}-{max_mape_subregion[1]}, rtt={max_mape_subregion[2]}-{max_mape_subregion[3]}")

if min_mse_subregion:
    print(f"Min MSE = {min_mse:.6f} at subregion: last_max_cwnd={min_mse_subregion[0]}-{min_mse_subregion[1]}, rtt={min_mse_subregion[2]}-{min_mse_subregion[3]}")
if min_mae_subregion:
    print(f"Min MAE = {min_mae:.6f} at subregion: last_max_cwnd={min_mae_subregion[0]}-{min_mae_subregion[1]}, rtt={min_mae_subregion[2]}-{min_mae_subregion[3]}")
if min_mape_subregion:
    print(f"Min MAPE = {min_mape:.6f}% at subregion: last_max_cwnd={min_mape_subregion[0]}-{min_mape_subregion[1]}, rtt={min_mape_subregion[2]}-{min_mape_subregion[3]}")
