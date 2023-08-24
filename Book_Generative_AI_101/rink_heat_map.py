import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter

# Load the ice hockey rink image
rink_image_path = '/Users/tylerkapp/Desktop/ice_rink_image.png'
rink_image = plt.imread(rink_image_path)

# function for adding hot spots to rink
def add_hot_spot(data, row, col, radius, value=1.0):
    for i in range(max(row - radius, 0), min(row + radius + 1, data.shape[0])):
        for j in range(max(col - radius, 0), min(col + radius + 1, data.shape[1])):
            distance = np.sqrt((i - row)**2 + (j - col)**2)
            if distance <= radius:
                data[i, j] = value

# Define the dimensions of the heat map data
rows, cols = 30, 50

# Create a 2D array with low values (e.g., zeros) for a blue background
heat_map_data = np.zeros((rows, cols))

# Define the dimensions of the heat map data
rows, cols = 30, 50

# Create a 2D array with low values (e.g., zeros) for a blue background
heat_map_data = np.zeros((rows, cols))

# Define the coordinates and radius of the hot spots (red areas) with high values (e.g., ones)
hot_spots = [
    (10, 25, 2), # position 1 lw
    (10, 31, 2), # position 2 lw
    (10, 15, 2), # position ld
    (20, 15, 2), # position rd
    (20, 29, 2), # position c
    (23, 28, 2), # position rw
]

# Set the values at the hot spots
for i, j, radius in hot_spots:
    add_hot_spot(heat_map_data, i, j, radius)

# Smooth the data if desired
heat_map_data = gaussian_filter(heat_map_data, sigma=1)

# Create a figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Show the ice hockey rink image
ax.imshow(rink_image, extent=[0, cols, 0, rows])

# Overlay the heat map on top of the rink image
cax = ax.imshow(heat_map_data, cmap="coolwarm", alpha=0.6, extent=[0, cols, 0, rows])

# Optionally, add a color bar to interpret the heat map values
plt.colorbar(cax)

plt.axis('off')  # Turn off the axis
plt.show()

