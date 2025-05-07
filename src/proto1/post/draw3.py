import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Load data from file
data = pd.read_csv('lorenz_solution.csv')  # replace 'your_file.csv' with the path to your data file
sample_rate = 100
data = data.iloc[::sample_rate]
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

# Initialize a line for the plot
line, = ax.plot([], [], [], lw=2)

# Update function for the animation
def update(num):
    line.set_data(data['x'][:num], data['y'][:num])
    line.set_3d_properties(data['z'][:num])
    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=len(data), blit=True)

# Save the animation
#writer = FFMpegWriter(fps=30, metadata={'title': '3D Trajectory Animation'}, bitrate=1800)
writer = FFMpegWriter(fps=60, metadata={'title': '3D Trajectory Animation'})
ani.save('C:/temp/A17/3d_animation.mp4', writer=writer)

plt.show()
