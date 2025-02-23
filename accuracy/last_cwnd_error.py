import matplotlib.pyplot as plt

# Data for plotting
last_max_cwnd_values = [1000, 10000, 100000]  # last_max_cwnd values

# Range 1: last_max_cwnd: 1000, rtt: 1000
mse_range1 = 0.751400
mae_range1 = 0.621049
mape_range1 = 0.975970

# Range 2: last_max_cwnd: 10000, rtt: 1000
mse_range2 = 14.264945
mae_range2 = 2.996752
mape_range2 = 0.958237

# Range 3: last_max_cwnd: 100000, rtt: 1000
mse_range3 = 911.894497
mae_range3 = 22.680261
mape_range3 = 1.856074

# Plot MSE, MAE, and MAPE for different last_max_cwnd values
plt.figure(figsize=(12, 8))

# Plot MSE
plt.subplot(2, 2, 1)
plt.plot(last_max_cwnd_values, [mse_range1, mse_range2, mse_range3], marker='o', label='MSE', linestyle='-', color='b')
plt.xscale('log')  # Use a logarithmic scale for better visualization
plt.xlabel('last_max_cwnd (log scale)')
plt.ylabel('Average MSE')
plt.title('Average MSE vs. last_max_cwnd')
plt.legend()

# Plot MAE
plt.subplot(2, 2, 2)
plt.plot(last_max_cwnd_values, [mae_range1, mae_range2, mae_range3], marker='o', label='MAE', linestyle='-', color='g')
plt.xscale('log')  # Use a logarithmic scale for better visualization
plt.xlabel('last_max_cwnd (log scale)')
plt.ylabel('Average MAE')
plt.title('Average MAE vs. last_max_cwnd')
plt.legend()

# Plot MAPE
plt.subplot(2, 2, 3)
plt.plot(last_max_cwnd_values, [mape_range1, mape_range2, mape_range3], marker='o', label='MAPE', linestyle='-', color='r')
plt.xscale('log')  # Use a logarithmic scale for better visualization
plt.xlabel('last_max_cwnd (log scale)')
plt.ylabel('Average MAPE')
plt.title('Average MAPE vs. last_max_cwnd')
plt.legend()

# Adjust layout and show
plt.tight_layout()
plt.show()
