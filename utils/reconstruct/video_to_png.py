import cv2
import os
import argparse


def extract_frames(video_path, output_folder, save_interval):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    id = 0

    while success:
        # Define the output filename
        frame_filename = os.path.join(output_folder, f"{id:05d}.png")
        # Save the current frame as a PNG image
        if count % save_interval == 0:
            cv2.imwrite(frame_filename, image)
            id += 1
        # Read the next frame from the video
        success, image = vidcap.read()
        count += 1

    vidcap.release()
    print(f"Extracted {id} frames to '{output_folder}'")


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='', help='name of the scene')
    parser.add_argument('--save_interval', type=int, default=60, help='save interval')

    return parser.parse_args()


# take two arguments: video_path and output_folder from the command line
if __name__ == '__main__':
    args = parser_args()
    name = args.name
    save_interval = args.save_interval

    # any file ends with .MOV will be considered as video file
    video_name = [f for f in os.listdir(f'./colmap/{name}/') if f.endswith('.MOV')][0]
    video_path = f'../../gaussians/colmap/{name}/{video_name}'
    if save_interval != 60:
        output_folder = f'../../gaussians/colmap/{name}_{save_interval}/extracted_frames'
    else:
        output_folder = f'../../gaussians/colmap/{name}/extracted_frames' 
    print('Extracting frames from video...')
    extract_frames(video_path, output_folder, save_interval)