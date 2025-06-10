from droid.robot_env import RobotEnv
from calibration.calibration_utils import load_calibration_info
import numpy as np
import math
from copy import deepcopy
import os
import sqlite3
import cv2
import shutil
from PIL import Image
import time


def save_images_and_ref(images, camera_poses, output_dir):
    if len(images) != len(camera_poses):
            raise ValueError("Number of images must match number of camera poses")
        
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create image directory
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    # create ref.txt
    ref_txt_path = os.path.join(f"{output_dir}_world", "ref.txt")
    with open(ref_txt_path, "w") as f:
        pass
    
    for idx, (image, pose) in enumerate(zip(images, camera_poses)):
        image_id = idx + 1
        image_name = f"image_{image_id:06d}.jpg"
        image_path = os.path.join(image_dir, image_name)
        
        # Convert to PIL Image and save
        if image.shape[2] == 3:  # RGB
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(image)
        elif image.shape[2] == 4:  # RGBA
            pil_img = Image.fromarray(image[:, :, :3])  # Drop alpha channel
        else:
            raise ValueError(f"Unexpected image format with {image.shape[2]} channels")
        pil_img.save(image_path)
        img_width, img_height = pil_img.size
        
        # Extract pose components
        x, y, z, roll, pitch, yaw = pose
        
        # Convert Euler angles to quaternion
        qw, qx, qy, qz = euler_to_quaternion(roll, pitch, yaw)

        with open(ref_txt_path, "a") as f:
            f.write(f"{image_name} {x} {y} {z}\n")


def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (in radians) to quaternion.
    
    Args:
        roll: Rotation around x-axis
        pitch: Rotation around y-axis
        yaw: Rotation around z-axis
        
    Returns:
        Quaternion as (qw, qx, qy, qz)
    """
    # Roll (x-axis rotation)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    return qw, qx, qy, qz


def sample_top_hemisphere(n, d, center):
    """
    Generate n points uniformly distributed on the top half of a sphere with radius d,
    along with direction vectors pointing toward the origin.
    
    Parameters:
    -----------
    n : int
        Number of points to generate
    d : float
        Radius of the sphere
    
    Returns:
    --------
    poses : ndarray
        Array of shape (n, 6) containing the 3D coordinates of the sampled points and the direction vectors pointing to the origin
    """
    # Use the Fibonacci sphere algorithm for near-uniform distribution
    # Modified to only include the top hemisphere
    
    poses = []
    phi = np.pi * (3 - np.sqrt(5))  # Golden angle in radians
    
    for i in range(n):
        # Generate points in range [0.4,1] for the top hemisphere
        z = 1 - (i / (n - 1) * 0.4) if n > 1 else 0.5
        
        # Convert to elevation angle (0 to π/2 for top hemisphere)
        elevation = np.arccos(z)
        
        # Azimuth angle spread using golden ratio
        azimuth = phi * i
        
        # Convert spherical to Cartesian coordinates
        x = np.sin(elevation) * np.cos(azimuth)
        y = np.sin(elevation) * np.sin(azimuth)
        z = np.cos(elevation)  # This will always be positive (top hemisphere)
        
        # Unit vector (direction)
        unit_vector = np.array([x, y, z])
        
        # Scale by radius for the point position
        point = unit_vector * d + center
        
        # Direction vector pointing to origin (negative of the unit vector)
        look_direction = -unit_vector # z axis

        up_direction = np.array([1, 0, 0])  # Up direction in world coordinates
        look_direction = look_direction / np.linalg.norm(look_direction)  # Normalize the direction vector
        up_direction = up_direction / np.linalg.norm(up_direction)  # Normalize the up direction vector

        right_direction = np.cross(look_direction, up_direction)  # Right vector
        right_direction = right_direction / np.linalg.norm(right_direction)  # Normalize the right vector
        up_direction = np.cross(right_direction, look_direction)
        up_direction = up_direction / np.linalg.norm(up_direction)  # Normalize the up direction vector

        # compute euler angle from rotation matrix
        # rot_mat = np.array([right_vector, up_direction, look_direction]).T
        rot_mat = np.array([up_direction, right_direction, look_direction]).T
        import scipy.spatial.transform as R
        # Convert rotation matrix to euler angles
        euler_angles = R.Rotation.from_matrix(rot_mat).as_euler('xyz', degrees=False)

        poses.append(np.concatenate([point, euler_angles]))
    
    poses = np.array(poses)

    return poses


def scan_scene(env, camera_id, distance, center):
    """
    Scan the scene with the robot wrist camera, return the images with poses.
    """
    
    # Load calibration info
    calibration_pose = load_calibration_info()[camera_id]
    print("Calibration Pose: ", calibration_pose)

    # Generate robot poses
    robot_poses = sample_top_hemisphere(50, distance, center)

    state, _ = env.get_state()
    pose = state["cartesian_position"].copy()
    print("Pose: ", pose)

    # Scan the scene
    images = []
    camera_poses = []
    for robot_pose in robot_poses:
        print("target Pose: ", robot_pose)
        action = np.concatenate([robot_pose, [0]])
        timer = time.time()
        while True:
            position_error = np.linalg.norm(action[:3] - pose[:3])
            orientation_error = action[3:6] - pose[3:6]
            orientation_error[orientation_error > np.pi] = 2 * np.pi - orientation_error[orientation_error > np.pi]
            orientation_error[orientation_error < -np.pi] = 2 * np.pi + orientation_error[orientation_error < -np.pi]
            orientation_error = np.linalg.norm(orientation_error)
            if position_error < 0.05 and orientation_error < 0.5 or time.time() - timer > 4:
                break
            env.update_robot(action, action_space="cartesian_position", gripper_action_space="velocity", blocking=False)
        # env.update_robot(action, action_space="cartesian_velocity", gripper_action_space="velocity", blocking=True)
            state, _ = env.get_state()
            pose = state["cartesian_position"].copy()
            # print("Pose: ", pose)
        time.sleep(2)
        state, _ = env.get_state()
        # cam_obs, _ = env.read_cameras()

        print("Camera ID: ", camera_id)
        # print("cam_obs: ", cam_obs["image"].keys())

        # img = deepcopy(cam_obs["image"][camera_id])
        pose = state["cartesian_position"].copy()
        print("pose: ", pose)
        camera_pose = np.array(pose) + np.array(calibration_pose)
        # save image
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(f"image.jpg", img)
        # images.append(img)
        camera_poses.append(camera_pose)

    return images, camera_poses


def parser():
    import argparse
    parser = argparse.ArgumentParser(description="Scan the scene with the robot wrist camera")
    parser.add_argument("--name", type=str, default="bunny")
    args = parser.parse_args()
    return args

args = parser()

# Make the robot env
env = RobotEnv()
env.gripper_action_space = "velocity"
hand_camera_id = "243222071972"

distance = 0.45
center = np.array([0.6, 0.0, 0.1])

# Scan the scene
images, camera_poses = scan_scene(env, hand_camera_id, distance, center)

# save images and poses into numpy files
np.save("images.npy", images)
np.save("camera_poses.npy", camera_poses)

# save as colmap ref
output_path = os.path.join("../../gaussian/colmap", args.name)
# save_images_and_poses_to_colmap(images, camera_poses, camera_intrinsics, output_path)
save_images_and_ref(images, camera_poses, output_path)