# %%
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV files
high_entropy = pd.read_csv('/root/projects/rl-enzyme-engineering/results/experimental_data/sequence_rl_high_ent/training/reward.csv')
low_entropy = pd.read_csv('/root/projects/rl-enzyme-engineering/results/experimental_data/sequence_rl_low_ent/training/reward.csv')

# Create the plot
plt.figure(figsize=(12, 6))

# Plot high entropy data
plt.plot(high_entropy['Step'], high_entropy['Value'], label='High Entropy', color='blue')

# Plot low entropy data
plt.plot(low_entropy['Step'], low_entropy['Value'], label='Low Entropy', color='red')

# Customize the plot
plt.title('Comparison of High and Low Entropy Coefficient Runs')
plt.xlabel('Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# Optionally, save the plot
# plt.savefig('entropy_comparison.png')

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Read the CSV files
high_entropy = pd.read_csv('/root/projects/rl-enzyme-engineering/results/experimental_data/sequence_rl_high_ent/training/reward.csv')
low_entropy = pd.read_csv('/root/projects/rl-enzyme-engineering/results/experimental_data/sequence_rl_low_ent/training/reward.csv')

# Calculate time in hours
high_entropy['Hours'] = (high_entropy['Wall time'] - high_entropy['Wall time'].iloc[0]) / 3600
low_entropy['Hours'] = (low_entropy['Wall time'] - low_entropy['Wall time'].iloc[0]) / 3600

# Create the plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot high entropy data
ax1.plot(high_entropy['Step'], high_entropy['Value'], label='Entropy Coefficient = 0.1', color='blue')

# Plot low entropy data
ax1.plot(low_entropy['Step'], low_entropy['Value'], label='Entropy Coefficient = 0.01', color='red')

# Customize the primary axis (Steps)
ax1.set_xlabel('Step')
ax1.set_ylabel('Value')
ax1.tick_params(axis='x', labelcolor='black')

# Create a secondary x-axis for time in hours
ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())
ax2.set_xlabel('Time (hours)')

# Set the tick locations and labels for the secondary axis
max_hours = max(high_entropy['Hours'].max(), low_entropy['Hours'].max())
ax2.set_xticks(ax1.get_xticks())
ax2.set_xticklabels([f'{h:.1f}' for h in np.linspace(0, max_hours, len(ax1.get_xticks()))])

# Customize the plot
plt.title('Comparison of High and Low Entropy Coefficient Runs')
ax1.legend()
ax1.grid(True)

# Show the plot
plt.tight_layout()
plt.show()

# Optionally, save the plot
plt.savefig('entropy_comparison_with_time.png')