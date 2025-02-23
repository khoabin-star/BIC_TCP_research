import matplotlib.pyplot as plt

# Data preparation remains the same
intervals = [9, 16, 25]

# Data for range up to 1,000
adaptive_mse_1000 = [1.27, 0.70, 0.645030]
equal_mse_1000 = [1.435873, 0.751400, 0.733861]
exponential_mse_1000 = [3.439415, 1.420982, 0.722114]

# Data for range up to 10,000
adaptive_mse_10000 = [24.105028, 12.600162, 9.477788]
equal_mse_10000 = [26.387291, 14.264945, 11.410529]
exponential_mse_10000 = [83.738588, 37.641669, 18.371623]

# Create a figure with two subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

# First subplot: Range up to 1,000
axes[0].plot(intervals, adaptive_mse_1000, marker='o', label='Adaptive Range')
axes[0].plot(intervals, equal_mse_1000, marker='s', label='Equal Range')
axes[0].plot(intervals, exponential_mse_1000, marker='^', label='Exponential Range')
axes[0].set_title('Average MSE vs. Number of Intervals (Range up to 1,000)')
axes[0].set_xlabel('Number of Intervals')
axes[0].set_ylabel('Average MSE')
axes[0].set_xticks(intervals)
axes[0].legend()
axes[0].grid(True)

# Second subplot: Range up to 10,000
axes[1].plot(intervals, adaptive_mse_10000, marker='o', label='Adaptive Range')
axes[1].plot(intervals, equal_mse_10000, marker='s', label='Equal Range')
axes[1].plot(intervals, exponential_mse_10000, marker='^', label='Exponential Range')
axes[1].set_title('Average MSE vs. Number of Intervals (Range up to 10,000)')
axes[1].set_xlabel('Number of Intervals')
axes[1].set_ylabel('Average MSE')
axes[1].set_xticks(intervals)
axes[1].legend()
axes[1].grid(True)

# Adjust layout
plt.tight_layout()
plt.savefig('average_mse_combined.png', dpi=300)
plt.show()
