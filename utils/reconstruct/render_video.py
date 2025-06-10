from pytorch3d.renderer import (
    look_at_view_transform,
    RasterizationSettings
)

import torch
import numpy as np
import open3d as o3d
import cv2
import os
import argparse

from self_render import render_meshes_with_projection
from utils.camera import get_up_direction


def camera(num_views, device):
    distance = 1.2
    elev_range = torch.linspace(75, -75, num_views)

    elevs = []
    azims = []
    smooth_elev_range = torch.linspace(0, 75, num_views + 1)[:-1]
    smooth_azim_range = torch.linspace(50, -75, num_views + 1)[:-1]
    for elev,  azim in zip(smooth_elev_range, smooth_azim_range):
        elevs.append(180 - elev)
        azims.append(azim)

    odd_even = 0
    for elev in elev_range:
        if odd_even == 0:
            azim_range = torch.linspace(-75, 75, num_views)
            odd_even = 1
        else:
            azim_range = torch.linspace(75, -75, num_views)
            odd_even = 0
        
        for azim in azim_range:
            elevs.append(180 - elev)
            azims.append(azim)

    elev = torch.tensor(np.array(elevs), device=device).float()
    azim = torch.tensor(np.array(azims), device=device).float()

    # up = get_up_direction(elev, azim)
    # up[:, 0] = -up[:, 0]
    up = torch.tensor(np.array([[0, 0, -1]]), device=device).float()
    at = torch.tensor(np.array([[0, 0, 0]]), device=device).float()
    R, T = look_at_view_transform(dist=distance, elev=elev, azim=azim, up=up, at=at, device=device)

    return R, T


def render_parser():
    parser = argparse.ArgumentParser(description='Render a video from a mesh')
    parser.add_argument('--name', type=str, default='bunny', help='Name of the mesh')
    parser.add_argument('--iteration', type=int, default=None, help='iterations of the gaussian splatting')
    return parser


if __name__ == "__main__":
    import sys
    import sys

    args = render_parser().parse_args()
    name = args.name

    iteration = args.iteration

    if iteration is not None:
        mesh_path = f"../../gaussians/output/{name}/train/ours_{iteration}/fuse_post.ply"
    else:
        mesh_floder = f'../../gaussians/output/{name}/train/'
        name_list = os.listdir(mesh_floder)
        max_iteration = np.max([int(name.split('_')[-1]) for name in name_list])

        mesh_path = f"../../gaussians/output/{name}/train/ours_{max_iteration}/fuse_post.ply"

    video_path = mesh_path.replace('.ply', '.mp4')

    device = torch.device("cuda:0")

    # Load an scene mesh (e.g., an OBJ file)
    scene_mesh = o3d.io.read_triangle_mesh(mesh_path)

    R, T = camera(20, device)

    images = []
    per_pixel_vertices = []
    # careful! batch_size can only be 1, todo: fix batch_size > 1
    batch_size = 20
    print('rendering...')
    for i in range(len(R) // batch_size):
        batch_images, batch_per_pixel_vertices = render_meshes_with_projection([scene_mesh], R[i * batch_size:(i + 1) * batch_size], T[i * batch_size:(i + 1) * batch_size], device, rotate_num=batch_size)
        # save images
        # vis_img = batch_images[0] * 255
        # vis_img = vis_img.astype(np.uint8)[..., :3]
        # cv2.imwrite(f'tmp_images/{i}.png', vis_img)
        # print('batch_per_pixel_vertices:', batch_per_pixel_vertices.shape)
        images.append(batch_images)
        per_pixel_vertices.append(batch_per_pixel_vertices)
    
    images = np.concatenate(images, axis=0)
    per_pixel_vertices = np.concatenate(per_pixel_vertices, axis=0)

    # save per_pixel_vertices
    file_path = mesh_path.replace('.ply', '.npy')
    np.save(file_path, per_pixel_vertices)

    print('Images:', images.shape)
    print('Projected vertices positions:', per_pixel_vertices.shape)

    width, height = images[0].shape[1], images[0].shape[0]

    # make .mp4 video
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for image in images:
        # print('Image:', image.shape, image.min(), image.max())
        image = (image * 255).astype(np.uint8)[:, :, :3]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video.write(image)
    
    video.release()