import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np

# Load data from file
data = pd.read_csv('lorenz_solution.csv')  # replace 'your_file.csv' with the path to your data file
sample_rate = 100
data = data.iloc[::sample_rate]

# Normalize time for color mapping
colors = plt.cm.viridis(np.linspace(0, 1, len(data)))

# Set up a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([data['x'].min(), data['x'].max()])
ax.set_ylim([data['y'].min(), data['y'].max()])
ax.set_zlim([data['z'].min(), data['z'].max()])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Animation of x, y, z over Time')

# Initialize a scatter plot for the animation
scatter = ax.scatter([], [], [], color=colors[0], marker='o')

# Update function for the animation
def update(num):
    ax.clear()
    ax.set_xlim([data['x'].min(), data['x'].max()])
    ax.set_ylim([data['y'].min(), data['y'].max()])
    ax.set_zlim([data['z'].min(), data['z'].max()])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Animation of x, y, z over Time')
    
    # Update scatter plot up to the current frame
    ax.scatter(data['x'][:num], data['y'][:num], data['z'][:num], c=colors[:num], marker='o')
    return scatter,

# Create the animation
ani = FuncAnimation(fig, update, frames=len(data), blit=False)

# Save the animation with a specified FPS
writer = FFMpegWriter(fps=60, metadata={'title': '3D Trajectory Animation'})
ani.save('C:/temp/A17/3d_animation1.mp4', writer=writer)

plt.show()
