import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import os
# Set up data for 8 subplots
agents = ['ORCA', 'ORCA_backoff', 'exp_7.39']
titles_agents = ['ORCA', 'ORCA with backoff', 'MEDIRL (ours)']
agents_frames_max = [len(os.listdir('robo_frames/ORCA/episode_24')), len(os.listdir('robo_frames/ORCA_backoff/episode_24')), len(os.listdir('robo_frames/exp_7.39/episode_24'))]
num_subplots = 8
fig, axs = plt.subplots(num_subplots, 3, figsize=(4,12))

for i in range(num_subplots):
    times = []
    for j in range(3):
        time = np.linspace(0, 520, num_subplots).astype(int)
        time = np.where(time > agents_frames_max[j], agents_frames_max[j]-1, time)
        img = mpimg.imread(f'robo_frames/{agents[j]}/episode_24/{time[i]}.png')
        axs[i, j].imshow(img)
        times.append(time[i])
        axs[-1, j].set_xlabel(titles_agents[j])
        plt.setp(axs[i, j].get_xticklabels(), visible=False)
        plt.setp(axs[i, j].get_yticklabels(), visible=False)
    axs[i, 0].set_ylabel(f'time {np.round(max(times)*0.2)}')
plt.tight_layout()
plt.savefig("traj_time.png")