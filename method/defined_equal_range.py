import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import os
import subprocess

warnings.filterwarnings("ignore", category=FutureWarning)


# Define the number of intervals for each variable
num_last_max_cwnd_interval = 16
num_rtt_interval = 16
#Define your parameter here
x_range = 1000
y_range = 1000

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


output_file = "../data/data_generated_1.csv"

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

# Dynamically compute the min and max values for each variable
cwnd_min = data["last_max_cwnd"].min()
cwnd_max = data["last_max_cwnd"].max()
rtt_min = data["rtt"].min()
rtt_max = data["rtt"].max()

# Create intervals for last_max_cwnd and rtt using specified ranges
last_max_cwnd_intervals = np.linspace(cwnd_min, cwnd_max, num_last_max_cwnd_interval + 1)
rtt_intervals = np.linspace(rtt_min, rtt_max, num_rtt_interval + 1)

# Convert interval endpoints to integers
last_max_cwnd_intervals = np.round(last_max_cwnd_intervals).astype(int)
rtt_intervals = np.round(rtt_intervals).astype(int)

# Initialize a dictionary to store the piecewise models and errors
piecewise_models = {}

# Initialize accumulators for error metrics
total_mse = 0
total_mae = 0
total_mape = 0
subregion_count = 0

# Variables to track regions with the highest errors
max_mse = -np.inf
max_mse_subregion = None
max_mae = -np.inf
max_mae_subregion = None
max_mape = -np.inf
max_mape_subregion = None

# Variables to track regions with the lowest errors
min_mse = np.inf
min_mse_subregion = None
min_mae = np.inf
min_mae_subregion = None
min_mape = np.inf
min_mape_subregion = None

# Fit piecewise linear models for each subregion defined by the two variables
for i in range(num_last_max_cwnd_interval):
    for j in range(num_rtt_interval):
        # Define the bounds for the current subregion
        cwnd_lower, cwnd_upper = last_max_cwnd_intervals[i], last_max_cwnd_intervals[i + 1]
        rtt_lower, rtt_upper = rtt_intervals[j], rtt_intervals[j + 1]

        # Filter the data corresponding to the current subregion
        subregion_data = data[
            (data["last_max_cwnd"] >= cwnd_lower) & (data["last_max_cwnd"] < cwnd_upper) &
            (data["rtt"] >= rtt_lower) & (data["rtt"] < rtt_upper)
        ]

        # Skip if there's not enough data in the subregion
        if len(subregion_data) < 2:
            print(f"No sufficient data for subregion: last_max_cwnd=[{cwnd_lower}, {cwnd_upper}), rtt=[{rtt_lower}, {rtt_upper})")
            continue

        # Prepare features (X) and target (y)
        X = subregion_data[["last_max_cwnd", "rtt"]]
        y = subregion_data["result"]

        # Fit a linear regression model for the subregion
        model = LinearRegression()
        model.fit(X, y)

        # Convert coefficients and intercept to integer-scaled values (multiplied by 1000)
        coef_last_max_cwnd = int(round(model.coef_[0] * 1000))
        coef_rtt = int(round(model.coef_[1] * 1000))
        intercept = int(round(model.intercept_ * 1000))

        # Compute predictions using integer arithmetic
        y_pred_int = (coef_last_max_cwnd * X["last_max_cwnd"] + coef_rtt * X["rtt"] + intercept) // 1000

        # Compute error metrics
        mse_int = mean_squared_error(y, y_pred_int)
        mae_int = mean_absolute_error(y, y_pred_int)
        mape_int = np.mean(np.abs((y - y_pred_int) / y)) * 100  # expressed as a percentage

        # Accumulate errors across subregions
        total_mse += mse_int
        total_mae += mae_int
        total_mape += mape_int
        subregion_count += 1

        # Store the model details and errors in the dictionary
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

        # Update maximum error trackers
        if mse_int > max_mse:
            max_mse = mse_int
            max_mse_subregion = (i, j)
        if mae_int > max_mae:
            max_mae = mae_int
            max_mae_subregion = (i, j)
        if mape_int > max_mape:
            max_mape = mape_int
            max_mape_subregion = (i, j)

        # Update minimum error trackers
        if mse_int < min_mse:
            min_mse = mse_int
            min_mse_subregion = (i, j)
        if mae_int < min_mae:
            min_mae = mae_int
            min_mae_subregion = (i, j)
        if mape_int < min_mape:
            min_mape = mape_int
            min_mape_subregion = (i, j)

# Print the piecewise linear equation and error metrics for each subregion
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

# Calculate and print average error metrics across subregions
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
print(f"\nMax MSE: {max_mse:.6f} at subregion last_max_cwnd=[{last_max_cwnd_intervals[max_mse_subregion[0]]}, {last_max_cwnd_intervals[max_mse_subregion[0] + 1]}), rtt=[{rtt_intervals[max_mse_subregion[1]]}, {rtt_intervals[max_mse_subregion[1] + 1]})")
print(f"Max MAE: {max_mae:.6f} at subregion last_max_cwnd=[{last_max_cwnd_intervals[max_mae_subregion[0]]}, {last_max_cwnd_intervals[max_mae_subregion[0] + 1]}), rtt=[{rtt_intervals[max_mae_subregion[1]]}, {rtt_intervals[max_mae_subregion[1] + 1]})")
print(f"Max MAPE: {max_mape:.6f}% at subregion last_max_cwnd=[{last_max_cwnd_intervals[max_mape_subregion[0]]}, {last_max_cwnd_intervals[max_mape_subregion[0] + 1]}), rtt=[{rtt_intervals[max_mape_subregion[1]]}, {rtt_intervals[max_mape_subregion[1] + 1]})")

# Print the subregions with the least errors
print(f"\nMin MSE: {min_mse:.6f} at subregion last_max_cwnd=[{last_max_cwnd_intervals[min_mse_subregion[0]]}, {last_max_cwnd_intervals[min_mse_subregion[0] + 1]}), rtt=[{rtt_intervals[min_mse_subregion[1]]}, {rtt_intervals[min_mse_subregion[1] + 1]})")
print(f"Min MAE: {min_mae:.6f} at subregion last_max_cwnd=[{last_max_cwnd_intervals[min_mae_subregion[0]]}, {last_max_cwnd_intervals[min_mae_subregion[0] + 1]}), rtt=[{rtt_intervals[min_mae_subregion[1]]}, {rtt_intervals[min_mae_subregion[1] + 1]})")
print(f"Min MAPE: {min_mape:.6f}% at subregion last_max_cwnd=[{last_max_cwnd_intervals[min_mape_subregion[0]]}, {last_max_cwnd_intervals[min_mape_subregion[0] + 1]}), rtt=[{rtt_intervals[min_mape_subregion[1]]}, {rtt_intervals[min_mape_subregion[1] + 1]})")


def generate_c_conditions(subregions):
    conditions = []
    subregion_keys = list(subregions.keys())
    for i, ((key_i, key_j), subregion) in enumerate(subregions.items()):
        coef_last_max_cwnd = subregion["coef_last_max_cwnd"]
        coef_rtt = subregion["coef_rtt"]
        intercept = subregion["intercept"]
        cwnd_lower, cwnd_upper = subregion["cwnd_bounds"]
        rtt_lower, rtt_upper = subregion["rtt_bounds"]

        condition = f"""
if (last_max_cwnd >= {cwnd_lower} && last_max_cwnd < {cwnd_upper} &&
    rtt >= {rtt_lower} && rtt < {rtt_upper}) {{
    bic_target = ({coef_last_max_cwnd} * last_max_cwnd + {coef_rtt} * rtt + {intercept}) / 1000;
}}
"""
        if i > 0:
            condition = condition.replace("if", "else if")

        conditions.append(condition)

    # Correct the final condition to be 'else' without repeating subregion condition
    if len(conditions) > 0:
        last_condition = f"""
else {{
    bic_target = ({coef_last_max_cwnd} * last_max_cwnd + {coef_rtt} * rtt + {intercept}) / 1000;
}}
"""
        conditions[-1] = last_condition

    return "\n".join(conditions)


def insert_conditions_into_template(conditions, template_file_path="./template.c", output_file_path="./output_with_conditions_1.c"):
    with open(template_file_path, "r") as template_file:
        template_code = template_file.read()

    # Insert the conditions into the template
    code_with_conditions = template_code.replace("// INSERT_CONDITIONS_HERE", conditions)

    # Write the result to a new file
    with open(output_file_path, "w") as output_file:
        output_file.write(code_with_conditions)


# Generate the conditions and insert them into the template
c_conditions = generate_c_conditions(piecewise_models)
insert_conditions_into_template(c_conditions)

def run_klee_on_output():
    c_file_path = "output_with_conditions_1.c"
    bc_file_path = "output_with_conditions_1.bc"
    klee_out_dir = "klee-out-46"

    # Compile the C file into LLVM bitcode
    subprocess.run(["clang", "-I", "../../include", "-emit-llvm", "-c", "-g", "-O0", "-Xclang", "-disable-O0-optnone", c_file_path])

    # Run KLEE on the bitcode file
    subprocess.run(["klee", "--solver-backend=z3", bc_file_path])

    # Collect KLEE stats
    klee_stats_output = subprocess.run(["klee-stats", "--print-all", klee_out_dir], capture_output=True, text=True).stdout
    stats = parse_klee_stats(klee_stats_output)

    print("KLEE Scalability Metrics:")
    print(f"States: {stats.get('States')}")
    print(f"Time(s): {stats.get('Time(s)')}")
    print(f"Instrs: {stats.get('Instrs')}")
    print(f"Mem(MiB): {stats.get('Mem(MiB)')}")

    # Clean up temporary files
    subprocess.run(["rm", bc_file_path])

def parse_klee_stats(stats_output):
    lines = stats_output.split("\n")
    if len(lines) < 4:
        return {}

    header_line = lines[1]
    data_line = lines[3]

    headers = [header.strip() for header in header_line.split("|")[1:-1]]
    data_values = [data.strip() for data in data_line.split("|")[1:-1]]

    return dict(zip(headers, data_values))

def clear_klee_output_directories():
    directory = "."  # Current directory
    directories = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]

    for dir_name in directories:
        if dir_name.startswith("klee-out-") or dir_name == "klee-last":
            dir_path = os.path.join(directory, dir_name)
            subprocess.run(["rm", "-r", dir_path])

if __name__ == "__main__":
    # clear_klee_output_directories()
    run_klee_on_output()

