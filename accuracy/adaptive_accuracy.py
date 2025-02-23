import matplotlib.pyplot as plt

# Data for plotting
intervals = ['3x3', '4x4', '5x5']  # Number of intervals: 3x3, 4x4, 5x5

# Range 1: last_max_cwnd: 1000, rtt: 1000
mse_range1 = [1.289507, 0.751400]
mae_range1 = [0.834411, 0.621049]
mape_range1 = [1.532996, 0.975970]

# Range 2: last_max_cwnd: 10000, rtt: 1000
mse_range2 = [22.714466, 14.264945, 9.396147]
mae_range2 = [3.424097, 2.996752, 2.376208]
mape_range2 = [1.507825, 0.958237, 0.887165]

# Range 3: last_max_cwnd: 100000, rtt: 1000
mse_range3 = [1253.923302, 911.894497]
mae_range3 = [26.418972, 22.680261]
mape_range3 = [2.871367, 1.856074]

# Plot MSE, MAE, and MAPE for each range
plt.figure(figsize=(12, 8))

# Plot MSE
plt.subplot(2, 2, 1)
plt.plot(intervals, mse_range1 + [None], marker='o', label='Range 1: last_max_cwnd=1000', linestyle='-', color='b')
plt.plot(intervals, mse_range2, marker='o', label='Range 2: last_max_cwnd=10000', linestyle='-', color='g')
plt.plot(intervals, mse_range3 + [None], marker='o', label='Range 3: last_max_cwnd=100000', linestyle='-', color='r')
plt.xlabel('Number of Intervals')
plt.ylabel('Average MSE')
plt.title('Average MSE for Different Ranges')
plt.legend()

# Plot MAE
plt.subplot(2, 2, 2)
plt.plot(intervals, mae_range1 + [None], marker='o', label='Range 1: last_max_cwnd=1000', linestyle='-', color='b')
plt.plot(intervals, mae_range2, marker='o', label='Range 2: last_max_cwnd=10000', linestyle='-', color='g')
plt.plot(intervals, mae_range3 + [None], marker='o', label='Range 3: last_max_cwnd=100000', linestyle='-', color='r')
plt.xlabel('Number of Intervals')
plt.ylabel('Average MAE')
plt.title('Average MAE for Different Ranges')
plt.legend()

# Plot MAPE
plt.subplot(2, 2, 3)
plt.plot(intervals, mape_range1 + [None], marker='o', label='Range 1: last_max_cwnd=1000', linestyle='-', color='b')
plt.plot(intervals, mape_range2, marker='o', label='Range 2: last_max_cwnd=10000', linestyle='-', color='g')
plt.plot(intervals, mape_range3 + [None], marker='o', label='Range 3: last_max_cwnd=100000', linestyle='-', color='r')
plt.xlabel('Number of Intervals')
plt.ylabel('Average MAPE')
plt.title('Average MAPE for Different Ranges')
plt.legend()

# Adjust layout and show
plt.tight_layout()
plt.show()
