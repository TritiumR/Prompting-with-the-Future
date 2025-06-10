import numpy as np
import cv2
import glob
import sys
import os
import yaml
from colmap.read_write_colmap import read_cameras_binary, read_images_binary, read_points3D_binary, write_cameras_binary, write_images_binary, write_points3D_binary
from scipy.spatial.transform import Rotation as Rot

CHESSBOARD_ROWS = 7
CHESSBOARD_COLS = 10 
# CHESSBOARD_ROWS = 4
# CHESSBOARD_COLS = 7 
# SQUARE_SIZE = 0.016625  # meters
# careful!!
# SQUARE_SIZE = 0.0096
# SQUARE_SIZE = 0.014
SQUARE_SIZE = 0.0111547
# CALIBRATION_IMAGE_NAME = '00000.png'


def calibrate_camera(folder):
    print("Calibrating camera...")
    # Termination criteria for corner subpixel refinement
    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)

    # Prepare object points based on the real chessboard dimensions
    objp = np.zeros((CHESSBOARD_ROWS * CHESSBOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_COLS, 0:CHESSBOARD_ROWS].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane
    image_size = None

    images_folder = os.path.join(folder, 'images')
    # Get list of calibration images
    images = glob.glob(f'{images_folder}/*.png')

    if len(images) == 0:
        print("No images found in the 'images' directory.")
        sys.exit(0)

    # Loop over the images and detect chessboard corners
    for img_file in images:
        img = cv2.imread(img_file)
        if img is None:
            print(f"Could not read image {img_file}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]  # width, height

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_COLS, CHESSBOARD_ROWS), None)

        # If found, add object points and image points (after refining them)
        if ret:
            objpoints.append(objp)
            # Refine the corner positions
            corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_subpix)
            img = cv2.drawChessboardCorners(img, (CHESSBOARD_COLS, CHESSBOARD_ROWS), corners_subpix, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(200)
        else:
            continue

    # Check if we have enough data
    if len(objpoints) == 0 or len(imgpoints) == 0:
        print("Not enough data to perform calibration.")
        sys.exit(0)

    # Calibration flags to fix higher order distortion coefficients (if desired)
    flags = cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6

    # Perform camera calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None, flags=flags)

    # Check if calibration was successful
    if not ret:
        print("Calibration failed.")
        sys.exit(0)

    data = {'camera_matrix': np.asarray(camera_matrix).tolist(), 'dist_coeff': np.asarray(dist_coeffs).tolist()}

    # Extract camera intrinsic parameters
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    # Extract distortion coefficients
    # dist_coeffs = [k1, k2, p1, p2, k3]
    # Since we fixed k3, k4, k5, k6, we only need k1, k2, p1, p2
    k1 = dist_coeffs[0, 0]
    k2 = dist_coeffs[0, 1]
    p1 = dist_coeffs[0, 2]
    p2 = dist_coeffs[0, 3]

    # Prepare COLMAP camera parameters
    camera_id = 1
    model = 'OPENCV'
    width, height = image_size

    params = [fx, fy, cx, cy, k1, k2, p1, p2]
    params_str = ' '.join(map(str, params))
    line = f"{camera_id} {model} {width} {height} {params_str}\n"

    # Write to 'cameras.txt' in COLMAP format
    with open(f'{folder}/chessboard_cameras.txt', 'w') as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        f.write(f'# Number of cameras: 1\n')
        f.write(line)

    print("Calibration successful. Camera parameters saved to 'chessboard_cameras.txt'.")
      
       
def load_camera_parameters(filename='cameras.txt'):
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('#') or line.strip() == '':
                continue
            tokens = line.strip().split()
            camera_id, model, width, height = tokens[:4]
            params = list(map(float, tokens[4:]))
            fx, fy, cx, cy, k1, k2, p1, p2 = params
            camera_matrix = np.array([[fx, 0, cx],
                                      [0, fy, cy],
                                      [0,  0,  1]], dtype=np.float64)
            dist_coeffs = np.array([k1, k2, p1, p2], dtype=np.float64)
            image_size = (int(width), int(height))
            return camera_matrix, dist_coeffs, image_size
    print("Camera parameters not found.")
    sys.exit(0)


def calibrate_pose(folder):
    print("Estimating camera pose...")
    # Prepare object points based on the real chessboard dimensions
    objp = np.zeros((CHESSBOARD_ROWS * CHESSBOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_COLS, 0:CHESSBOARD_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE  # Scale object points to real-world dimensions

    # Load camera parameters
    camera_matrix, dist_coeffs, image_size = load_camera_parameters(f'{folder}/chessboard_cameras.txt')

    images_folder = os.path.join(folder, 'images')
    # Get list of images to estimate pose
    images = glob.glob(f'{images_folder}/*.png')

    if len(images) == 0:
        print("No images found in the 'images' directory.")
        sys.exit(0)

    # Prepare output file for camera poses
    pose_output = open(f'{folder}/chessboard_poses.txt', 'w')
    pose_output.write('# Image filename, rotation vector (Rodrigues), translation vector\n')

    # Loop over the images and estimate pose
    for img_file in images:
        img = cv2.imread(img_file)
        if img is None:
            print(f"Could not read image {img_file}")
            return None, None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_COLS, CHESSBOARD_ROWS), None)

        if ret:
            # Refine the corner positions
            criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
            corners_subpix = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

            # Solve the PnP problem to obtain rotation and translation vectors
            retval, rvec, tvec = cv2.solvePnP(objp, corners_subpix, camera_matrix, dist_coeffs)

            # Optionally, draw the axes on the image
            axis_length = SQUARE_SIZE * 3  # Length of the axes to be drawn
            axis = np.float32([
                [0, 0, 0],
                [axis_length, 0, 0],
                [0, axis_length, 0],
                [0, 0, -axis_length]
            ]).reshape(-1, 3)

            # Project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)

            # Draw the axes on the image
            corner = tuple(corners_subpix[0].ravel().astype(int))
            img = cv2.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (255,0,0), 3)
            img = cv2.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (0,255,0), 3)
            img = cv2.line(img, corner, tuple(imgpts[3].ravel().astype(int)), (0,0,255), 3)

            # save the image
            cv2.imwrite(f'{folder}/pose_estimation.png', img)

            # Write the pose to the output file
            rvec_str = ' '.join(map(str, rvec.flatten()))
            tvec_str = ' '.join(map(str, tvec.flatten()))
            pose_output.write(f"{img_file} {rvec_str} {tvec_str}\n")
        else:
            # print(f"Chessboard not detected in image {img_file}")
            continue
    
    pose_output.close()
    print("Camera pose estimation complete. Poses saved to 'chessboard_poses.txt'.")


if __name__ == "__main__":
    import sys
    name = sys.argv[1]
    suffix = sys.argv[2]
    folder = f'../../gaussians/data/colmap/{name}'
    out_folder = f'../../gaussians/data/colmap/{name}_{suffix}'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    calibrate_camera(folder)

    calibrate_pose(folder)

    # Read chessboard poses to get the real-world camera centers
    chessboard_poses = {}
    with open(f'{folder}/chessboard_poses.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('#') or line.strip() == '':
                continue
            tokens = line.strip().split()
            image_name = os.path.basename(tokens[0])  # Get image filename
            rvec = np.array(list(map(float, tokens[1:4]))).reshape(3, 1)
            tvec = np.array(list(map(float, tokens[4:7]))).reshape(3, 1)
            # Convert rotation vector to rotation matrix
            R_world, _ = cv2.Rodrigues(rvec)
            t_world = tvec
            # Compute camera center in chessboard coordinate system
            C_world = -R_world.T @ t_world
            # Store camera center
            chessboard_poses[image_name] = C_world.flatten()
    
    # Save the camera centers to a file
    with open(f'{out_folder}/ref.txt', 'w') as f:
        for image_name, C_world in chessboard_poses.items():
            f.write(f"{image_name} {C_world[0]} {C_world[1]} {C_world[2]}\n")

    print("Camera centers saved to 'ref.txt'.")

    