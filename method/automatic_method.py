import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import os
warnings.filterwarnings("ignore", category=FutureWarning)

#Define your parameter here
x_range = 1000
y_range = 1000
max_subregions = 9 

# Define your function here
def input_function(last_max_cwnd, rtt, x=10):
    return (last_max_cwnd - (410 * ((cubic_root((1 << (10 + 3 * x)) // 410 * (last_max_cwnd - (717 * last_max_cwnd // 1024))) - ((1 << 10) * rtt // 1000)) ** 3) >> (10 + 3 * x)))

# Cubic root function (from earlier)
def cubic_root(a):
    v = [
        0, 54, 54, 54, 118, 118, 118, 118, 123, 129, 134, 138, 143, 147, 151, 156,
        157, 161, 164, 168, 170, 173, 176, 179, 181, 185, 187, 190, 192, 194, 197, 199,
        200, 202, 204, 206, 209, 211, 213, 215, 217, 219, 221, 222, 224, 225, 227, 229,
        231, 232, 234, 236, 237, 239, 240, 242, 244, 245, 246, 248, 250, 251, 252, 254
    ]
    b = fls64(a)
    if b < 7:
        return (v[a] + 35) >> 6
    b = ((b * 84) >> 8) - 1
    shift = (a >> (b * 3))
    x = ((v[shift] + 10) << b) >> 6
    x = (2 * x + (a // (x * (x - 1))))
    x = (x * 341) >> 10
    return x

# Function to find the last set bit in a 64-bit integer
def fls64(x):
    if x == 0:
        return 0
    return __fls(x) + 1

def __fls(word):
    num = BITS_PER_LONG - 1
    if BITS_PER_LONG == 64:
        if not (word & (~0 << 32)):
            num -= 32
            word <<= 32
    if not (word & (~0 << (BITS_PER_LONG - 16))):
        num -= 16
        word <<= 16
    if not (word & (~0 << (BITS_PER_LONG - 8))):
        num -= 8
        word <<= 8
    if not (word & (~0 << (BITS_PER_LONG - 4))):
        num -= 4
        word <<= 4
    if not (word & (~0 << (BITS_PER_LONG - 2))):
        num -= 2
        word <<= 2
    if not (word & (~0 << (BITS_PER_LONG - 1))):
        num -= 1
    return num

# Constants
BITS_PER_LONG = 64


output_file = "../data/data_generated.csv"

# Generate data and write to file
def generate_data_file(filename=output_file):
    with open(filename, "w") as file:
        # Write header
        file.write("last_max_cwnd,rtt,result\n")
        
        # Iterate over last_max_cwnd and rtt
        for last_max_cwnd in range(1, x_range + 1):
            for rtt in range(1, y_range + 1):
                result = input_function(last_max_cwnd, rtt)
                # Write data to file
                file.write(f"{last_max_cwnd},{rtt},{result}\n")

def generate_large_data_file(filename="output3.csv", num_samples_cwnd=x_range, num_samples_rtt=y_range):
    with open(filename, "w") as file:
        file.write("last_max_cwnd,rtt,result\n")

       # Generate logarithmic samples
        last_max_cwnd_values = np.unique(np.geomspace(1, 100000, num=num_samples_cwnd).astype(int))
        rtt_values = np.unique(np.geomspace(1, 1000, num=num_samples_rtt).astype(int))

        for last_max_cwnd in last_max_cwnd_values:
            for rtt in rtt_values:
                result = input_function(last_max_cwnd, rtt)
                file.write(f"{last_max_cwnd},{rtt},{result}\n")

# Run with optimized sampling
generate_data_file()

# Read the data from the CSV file
data = pd.read_csv(output_file)

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
            subregion['mape'] = mape_int  #avergae weighted error.
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

def generate_c_conditions(subregions):
    conditions = []
    for i, subregion in enumerate(subregions):
        coef = subregion['model']
        cwnd_lower = int(subregion['cwnd_lower'])  # Round down
        cwnd_upper = int(subregion['cwnd_upper'])  # Round up
        rtt_lower = int(subregion['rtt_lower'])  # Round down
        rtt_upper = int(subregion['rtt_upper'])  # Round up
        
        condition = f"""
if (last_max_cwnd >= {cwnd_lower} && last_max_cwnd < {cwnd_upper} &&
    rtt >= {rtt_lower} && rtt < {rtt_upper}) {{
    result = {int(coef['coef_last_max_cwnd'])} * last_max_cwnd + {int(coef['coef_rtt'])} * rtt + {int(coef['intercept'])};
}}
"""
        if i > 0:
            condition = condition.replace("if", "else if")
        
        conditions.append(condition)

    # Optional: Automatically make the last region "else"
    if len(conditions) > 0:
        conditions[-1] = conditions[-1].replace("else if", "else")

    return "\n".join(conditions)


def insert_conditions_into_template(conditions, template_file_path="./template.c", output_file_path="./output_with_conditions.c"):
    with open(template_file_path, "r") as template_file:
        template_code = template_file.read()

    # Insert the conditions into the template
    code_with_conditions = template_code.replace("// INSERT_CONDITIONS_HERE", conditions)

    # Write the result to a new file
    with open(output_file_path, "w") as output_file:
        output_file.write(code_with_conditions)


# Generate the conditions and insert them into the template
c_conditions = generate_c_conditions(subregions)
insert_conditions_into_template(c_conditions)