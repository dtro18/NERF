import numpy as np
import math
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
            z_near, z_far = np.array([0]), np.array([105.0])
            # Append additional parameters
            intrinsics = np.array([[height, width, focal_length]]).T
            pose_with_extras = np.concatenate((pose, intrinsics), axis=1).flatten()
            print(pose_with_extras)
            pose_with_extras = np.concatenate((pose_with_extras, z_near, z_far))
            print(pose_with_extras)
    return np.array(poses)

# Assuming they come out normalized...
def normalize_quarternion(qw, qx, qy, qz):
    mag = math.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    return qw / mag, qx / mag, qy / mag, qz / mag

def transpose_rotation_matrix(R):
    return R.transpose()
# Convert quaternion into 3x3 rotation matrix
def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R

def update_translation_vector(Rt, T):
    return -np.dot(Rt, T)

def colmap_coords_to_endonerf_coords(v):
    # Assuming vector is an np array has shape 3 x n
    p = np.array([[0, -1, 0], [1, 0, 0], [0, 0, -1]])
    return np.dot(p, v)
# Uncomment later

# Generate the 3x4 camera-to-world matrix
def build_pose(qw, qx, qy, qz, tx, ty, tz):
    R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    R = transpose_rotation_matrix(R)

    T = np.array([[tx, ty, tz]]).T
    T = update_translation_vector(R, T)

    R = colmap_coords_to_endonerf_coords(R)
    T = colmap_coords_to_endonerf_coords(T)
    return np.hstack((R, T))

# Example usage
poses = parse_images_and_cameras("D:\\COLMAP\\Colmap Project Fixed Camera No Tree\\sparse\\images.txt", "D:\\COLMAP\\Colmap Project Fixed Camera No Tree\\sparse\\cameras.txt")
# np.save("poses_with_extras.npy", poses)


# q0, q1, q2, q3 = 0.851773, 0.0165051, 0.503764, -0.142941
# rotation_matrix = quaternion_to_rotation_matrix(q0, q1, q2, q3)