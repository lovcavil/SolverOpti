import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data from file
data = pd.read_csv('lorenz_solution.csv')  # replace 'your_file.csv' with the path to your data file

# Set up a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the data
ax.plot(data['x'], data['y'], data['z'], label='Trajectory in 3D space')

# Label the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Plot of x, y, and z over Time')

# Optionally, color by time
sc = ax.scatter(data['x'], data['y'], data['z'], c=data['t'], cmap='viridis', marker='o')
plt.colorbar(sc, label='Time (t)')

# Display the plot
plt.show()
