import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

data = pd.read_csv("output.csv")

max_subregions = 9

subregions = [
    {"cwnd_bounds": (1, 500), "rtt_bounds": (1, 500)},
    {"cwnd_bounds": (1, 500), "rtt_bounds": (500, 1000)},
    {"cwnd_bounds": (500, 1000), "rtt_bounds": (1, 500)},
    {"cwnd_bounds": (500, 1000), "rtt_bounds": (500, 1000)},
]

def fit_model_and_calculate_error(subregion, data):
    cwnd_lower, cwnd_upper = subregion["cwnd_bounds"]
    rtt_lower, rtt_upper = subregion["rtt_bounds"]
    
    subregion_data = data[
        (data["last_max_cwnd"] >= cwnd_lower) & (data["last_max_cwnd"] < cwnd_upper) &
        (data["rtt"] >= rtt_lower) & (data["rtt"] < rtt_upper)
    ]
    
    if len(subregion_data) < 2:
        return None, np.inf 
    
    X = subregion_data[["last_max_cwnd", "rtt"]]
    y = subregion_data["result"]
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    
    return model, mse

while len(subregions) < max_subregions:
    errors = []
    for subregion in subregions:
        model, mse = fit_model_and_calculate_error(subregion, data)
        errors.append(mse)
    
    worst_index = np.argmax(errors)
    worst_subregion = subregions[worst_index]
    
    cwnd_lower, cwnd_upper = worst_subregion["cwnd_bounds"]
    rtt_lower, rtt_upper = worst_subregion["rtt_bounds"]
    
    cwnd_mid = (cwnd_lower + cwnd_upper) // 2
    rtt_mid = (rtt_lower + rtt_upper) // 2
    
    new_subregions = [
        {"cwnd_bounds": (cwnd_lower, cwnd_mid), "rtt_bounds": (rtt_lower, rtt_mid)},
        {"cwnd_bounds": (cwnd_lower, cwnd_mid), "rtt_bounds": (rtt_mid, rtt_upper)},
        {"cwnd_bounds": (cwnd_mid, cwnd_upper), "rtt_bounds": (rtt_lower, rtt_mid)},
        {"cwnd_bounds": (cwnd_mid, cwnd_upper), "rtt_bounds": (rtt_mid, rtt_upper)},
    ]
    
    subregions.pop(worst_index)
    subregions.extend(new_subregions)

total_mse = 0
integer_mse = 0
subregion_count = 0

for i, subregion in enumerate(subregions):
    model, mse = fit_model_and_calculate_error(subregion, data)
    if model is not None:
        coef_last_max_cwnd, coef_rtt = model.coef_
        intercept = model.intercept_
        
        # Convert coefficients to integer-based approximations
        scale_factor = 1000
        int_coef_cwnd = int(round(coef_last_max_cwnd * scale_factor))
        int_coef_rtt = int(round(coef_rtt * scale_factor))
        int_intercept = int(round(intercept))
        
        print(f"Subregion {i + 1}: last_max_cwnd={subregion['cwnd_bounds']}, rtt={subregion['rtt_bounds']}")
        print(f"Floating Point Equation: result = {coef_last_max_cwnd:.6f} * last_max_cwnd + {coef_rtt:.6f} * rtt + {intercept:.6f}")
        print(f"Integer Approximation: result = ({int_coef_cwnd} * last_max_cwnd) / {scale_factor} + ({int_coef_rtt} * rtt) / {scale_factor} + {int_intercept}")
        print(f"Mean Squared Error (MSE) for Floating Point: {mse:.6f}")
        
        # Calculate MSE for integer approximation
        X = data[["last_max_cwnd", "rtt"]]
        y = data["result"]
        y_pred_int = (int_coef_cwnd * X["last_max_cwnd"]) // scale_factor + (int_coef_rtt * X["rtt"]) // scale_factor + int_intercept
        int_mse = mean_squared_error(y, y_pred_int)
        
        print(f"Mean Squared Error (MSE) for Integer Approximation: {int_mse:.6f}")
        print("-" * 60)
        
        total_mse += mse
        integer_mse += int_mse
        subregion_count += 1

if subregion_count > 0:
    average_mse = total_mse / subregion_count
    avg_int_mse = integer_mse / subregion_count
    print(f"Average Mean Squared Error (MSE) for Floating Point: {average_mse:.6f}")
    print(f"Average Mean Squared Error (MSE) for Integer Approximation: {avg_int_mse:.6f}")
else:
    print("No sufficient data in any subregion to calculate average MSE.")
