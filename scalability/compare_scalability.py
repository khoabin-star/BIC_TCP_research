import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ['Original', 'Adaptive (9 intervals)', 'Adaptive (16 intervals)', 'Adaptive (25 intervals)']
cases = ['(1000, 1000)', '(10,000, 1000)']

# Running times for each method in both cases
running_times = {
    '(1000, 1000)': [221.62, 0.18, 0.45, 0.62],
    '(10,000, 1000)': [369.87, 0.33, 0.42, 0.60]
}

# Positions and width for the bars
x = np.arange(len(methods))
width = 0.35  # Width of the bars

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Bars for each case
rects1 = ax.bar(x - width/2, running_times['(1000, 1000)'], width, label='(1000, 1000)')
rects2 = ax.bar(x + width/2, running_times['(10,000, 1000)'], width, label='(10,000, 1000)')

# Set log scale for better visibility of small values
ax.set_yscale('log')

# Labels and Title
ax.set_xlabel('Methods')
ax.set_ylabel('KLEE Running Time (seconds, log scale)')
ax.set_title('Comparison of KLEE Running Times for Original and Adaptive Methods')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=45)
ax.legend()

# Annotate bars with running time values
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # Offset text above bar
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.show()
