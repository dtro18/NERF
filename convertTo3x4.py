import numpy as np

import os

def parse_images_and_cameras(images_file, cameras_file):
    poses = []
    # Parse cameras.txt
    camera_data = {}
    with open(cameras_file, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            camera_id = int(parts[0])
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))
            
            camera_data[camera_id] = (width, height, params)

    # Parse images.txt
    with open(images_file, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            camera_id = int(parts[8])

            if camera_id not in camera_data:
                print(f"Camera ID {camera_id} from images.txt is not found in cameras.txt.")
                continue
            # Build pose
            pose = build_pose(qw, qx, qy, qz, tx, ty, tz)
            width, height, params = camera_data[camera_id]
            focal_length = params[0]
            z_near, z_far = 0.1, 100.0
            # Append additional parameters
            pose_with_extras = np.concatenate([pose.flatten(), [width, height, focal_length, z_near, z_far]])
            poses.append(pose_with_extras)
    return np.array(poses)

# Convert quaternion into 3x3 rotation matrix
def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
# Add the translation vector as the last column to generate the 3x4 matrix
def build_pose(qw, qx, qy, qz, tx, ty, tz):
    R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    T = np.array([[tx, ty, tz]]).T
    return np.hstack((R, T))

# Example usage
poses = parse_images_and_cameras("D:\\COLMAP\\Colmap Project\\sparse\\0\\images.txt", "D:\\COLMAP\\Colmap Project\\sparse\\0\\cameras.txt")
np.save("poses_with_extras.npy", poses)
