import matplotlib.pyplot as plt
import numpy as np

# Data for the original code (last-max-cwnd, rtt, time)
original_x = [(10, 10), (50, 50), (100, 100), (200, 200), (500, 500), (1000, 1000)]
original_time = [144.57, 67.74, 32.45, 82.97, 72.36, 118.79]

# Data for the adaptive method with 25 intervals (last-max-cwnd, rtt, time)
adaptive_x = [(10, 10), (50, 50), (100, 100), (200, 200), (500, 500), (1000, 1000)]
adaptive_time = [0.45, 0.45, 0.46, 0.47, 0.86, 1.16]

# Convert x values (last-max-cwnd, rtt) to strings for labeling on x-axis
x_labels = [f'({a[0]}, {a[1]})' for a in original_x]

# Set up the figure and axis for the plot
plt.figure(figsize=(10, 6))

# Plotting the lines for both the original and adaptive method
plt.plot(x_labels, original_time, label='Original Method', marker='o', linestyle='-', color='skyblue')
plt.plot(x_labels, adaptive_time, label='Adaptive Method (9 Intervals)', marker='o', linestyle='-', color='lightcoral')

# Adding labels and title
plt.xlabel('Configuration (last-max-cwnd, rtt)', fontsize=12)
plt.ylabel('Time', fontsize=12)
plt.title('Scalability Comparison: Original vs Adaptive Method', fontsize=14)
plt.legend()

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Display the plot
plt.tight_layout()
plt.show()
