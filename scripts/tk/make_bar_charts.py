import matplotlib.pyplot as plt
import numpy as np

# Set up data for 8 subplots
num_subplots = 8
agents = ['ORCA', 'ORCA-Backoff', 'MEDIRL', 'LZ-MEDIRL (Ours)' 'Expert']
values = np.random.rand(num_subplots, len(agents))  # Random values between 0 and 1
values[0] = [1.0,1.0,0.7, 1.0]
values[1] = [1.0,1.0,0.7, 1.0]
values[2] = [1.0,1.0,1.0, 1.0]
values[3] = [1.0,1.0,1.0, 1.0]
values[4] = [1.0,1.0,1.0, 1.0]
values[5] = [0.0,0.4,1.0, 1.0]
values[6] = [0.0,1.0,1.0, 1.0]
values[7] = [1.0,1.0,0.1, 0.5]
# Create subplots
fig, axs = plt.subplots(1, num_subplots, figsize=(12, 2))

colors = ['blue', 'orange', 'green', 'violet']

for i in range(num_subplots):
    axs[i].bar(agents, values[i], color=colors)
    axs[i].set_ylim(0, 1)  # Set y-axis limits between 0 and 1
    axs[i].set_title(f'ep. {i + 1}')
    axs[i].set_xticks([])  # Remove x-axis labels
    axs[i].spines['top'].set_visible(False)  # Remove the top spine
    axs[i].spines['right'].set_visible(False)  # Remove the right spine

# Add a common legend
handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
labels = agents
fig.legend(handles, labels, loc='upper right', title='Agents', fontsize='medium')

plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to prevent overlap
plt.savefig('success.png')
plt.show()