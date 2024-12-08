import numpy as np

# Load the camera poses from the .npy file
poses = np.load("poses_with_extras.npy")

# Check the shape of the loaded array
print(poses.shape)

# Example: Accessing the first pose
print(poses[0])  # First row (3x4 pose matrix + additional parameters)