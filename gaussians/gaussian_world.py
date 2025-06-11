import sys
sys.path.append('gaussians')

import torch
from scene import Scene
from scene.cameras import Camera
import os
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, get_combined_args_by_path
from gaussian_renderer import GaussianModel
from utils.mesh_utils import GaussianExtractor
import numpy as np
import cv2
from scipy.spatial import KDTree
import open3d as o3d

class GaussianWorld:
    def __init__(self, name, parser, post_process=False):
        model = ModelParams(parser, sentinel=True)
        pipeline = PipelineParams(parser)
        parser.add_argument("--iteration", default=30000, type=int)

        args = get_combined_args_by_path(parser, f'gaussians/output/{name}')
        args.model_path = f'gaussians/output/{name}'
        args.source_path = f'gaussians/data/colmap/{name}'

        self.args = args

        point_cloud_path = os.path.join(args.model_path, 'point_cloud')
        name_list = os.listdir(point_cloud_path)
        max_iteration = np.max([int(name.split('_')[-1]) for name in name_list])
        args.iteration = max_iteration

        dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
        # print('dataset:', vars(dataset))
        gaussians = GaussianModel(dataset.sh_degree)

        # careful with the resolution scales, no training images are loaded
        # scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, resolution_scales = [])
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, skip=True)
        # print('scene finished')
        bg_color = [1, 1, 1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        gaussian_points = gaussians._xyz.detach().cpu().numpy()
        # print('gaussian_points:', gaussian_points.shape)

        self.object_gaussians = []
        # careful with the threshold
        threshold = 0.06
        box_threshold = 0.0015
        # careful! for angry birds
        # threshold = 0.06
        # box_threshold = 0.00
        mesh_list = os.listdir(f'gaussians/output/{name}/train/ours_{max_iteration}/')
        for mesh_file in mesh_list:
            if not mesh_file.endswith(".ply"):
                continue
            if mesh_file.endswith("_convex.ply"):
                continue
            if mesh_file.endswith("_convex_normalize.ply"):
                continue
            if mesh_file.endswith("_normalize.ply"):
                continue
            if mesh_file.startswith("fuse"):
                continue
            if mesh_file == "background.ply":
                continue
            if mesh_file == "background_convex.ply":
                continue

            object_path = os.path.join(f'gaussians/output/{name}/train/ours_{max_iteration}', mesh_file)
            object_mesh = o3d.io.read_triangle_mesh(object_path)
            object_vertices = np.asarray(object_mesh.vertices)

            max_x, max_y, max_z = np.max(object_vertices, axis=0) + box_threshold
            min_x, min_y, min_z = np.min(object_vertices, axis=0) - box_threshold

            in_box_gaussian_points_id = np.where((gaussian_points[:, 0] >= min_x) & (gaussian_points[:, 0] <= max_x) &
                                                (gaussian_points[:, 1] >= min_y) & (gaussian_points[:, 1] <= max_y) &
                                                (gaussian_points[:, 2] >= min_z) & (gaussian_points[:, 2] <= max_z))[0]
            # print('in_box_gaussian_points:', in_box_gaussian_points_id.shape)

            in_box_gaussian_points = gaussian_points[in_box_gaussian_points_id]

            kdtree = KDTree(object_vertices)
            # distances, indices = kdtree.query(gaussian_points, k=1)
            distances, indices = kdtree.query(in_box_gaussian_points, k=1)
            # print('distances:', distances.shape)

            object_indices = np.where(distances < threshold)[0]
            object_indices = in_box_gaussian_points_id[object_indices]
            # print('object_indices:', object_indices.shape)
            self.object_gaussians.append(object_indices)

        if post_process:
            # remove gaussians outside the workspace
            box = [-0.48, 0.15, -0.55, 0.55, -0.5, 0.03]
            out_of_box = (gaussians._xyz[:, 0] < box[0]) | (gaussians._xyz[:, 0] > box[1]) | (gaussians._xyz[:, 1] < box[2]) | (gaussians._xyz[:, 1] > box[3]) | (gaussians._xyz[:, 2] < box[4]) | (gaussians._xyz[:, 2] > box[5])
            # remove the gaussians outside the box
            gaussians._xyz = gaussians._xyz[~out_of_box]
            gaussians._features_dc = gaussians._features_dc[~out_of_box]
            gaussians._features_rest = gaussians._features_rest[~out_of_box]
            gaussians._scaling = gaussians._scaling[~out_of_box]
            gaussians._rotation = gaussians._rotation[~out_of_box]
            gaussians._opacity = gaussians._opacity[~out_of_box]

            new_index = np.cumsum(~out_of_box.cpu().numpy()) - 1
            for i in range(len(self.object_gaussians)):
                self.object_gaussians[i] = new_index[self.object_gaussians[i]]

            # cover the QR code
            patch_box = np.array([[-0.05, -0.03, -0.1], [0.16, 0.13, 0.1]])
            in_box = (gaussians._xyz[:, 0] > patch_box[0][0]) & (gaussians._xyz[:, 0] < patch_box[1][0]) & (gaussians._xyz[:, 1] > patch_box[0][1]) & (gaussians._xyz[:, 1] < patch_box[1][1]) & (gaussians._xyz[:, 2] > patch_box[0][2]) & (gaussians._xyz[:, 2] < patch_box[1][2])
            # remove the patch gaussians
            gaussians._xyz = gaussians._xyz[~in_box]
            gaussians._features_dc = gaussians._features_dc[~in_box]
            gaussians._features_rest = gaussians._features_rest[~in_box]
            gaussians._scaling = gaussians._scaling[~in_box]
            gaussians._rotation = gaussians._rotation[~in_box]
            gaussians._opacity = gaussians._opacity[~in_box]

            new_index = np.cumsum(~in_box.cpu().numpy()) - 1

            for i in range(len(self.object_gaussians)):
                self.object_gaussians[i] = new_index[self.object_gaussians[i]]

            off_set = np.array([0.0, 0.18, 0.0])

            source_box = patch_box - off_set
            source_index = (gaussians._xyz[:, 0] > source_box[0][0]) & (gaussians._xyz[:, 0] < source_box[1][0]) & (gaussians._xyz[:, 1] > source_box[0][1]) & (gaussians._xyz[:, 1] < source_box[1][1]) & (gaussians._xyz[:, 2] > source_box[0][2]) & (gaussians._xyz[:, 2] < source_box[1][2])
            source_gaussians = gaussians._xyz[source_index] + torch.tensor([0.0, 0.18, -0.001], device=gaussians._xyz.device)
            source_features_dc = gaussians._features_dc[source_index]
            source_features_rest = gaussians._features_rest[source_index]
            source_scaling = gaussians._scaling[source_index]
            source_rotation = gaussians._rotation[source_index]
            source_opacity = gaussians._opacity[source_index]

            # add the source gaussians
            gaussians._xyz = torch.cat((gaussians._xyz, source_gaussians), dim=0)
            gaussians._features_dc = torch.cat((gaussians._features_dc, source_features_dc), dim=0)
            gaussians._features_rest = torch.cat((gaussians._features_rest, source_features_rest), dim=0)
            gaussians._scaling = torch.cat((gaussians._scaling, source_scaling), dim=0)
            gaussians._rotation = torch.cat((gaussians._rotation, source_rotation), dim=0)
            gaussians._opacity = torch.cat((gaussians._opacity, source_opacity), dim=0)

            # add patches under the objects
            for object_id in range(len(self.object_gaussians)):
                object_indices = self.object_gaussians[object_id]
                object_xyz = gaussians._xyz[object_indices]
                object_min_x, object_min_y, _ = np.min(object_xyz.cpu().numpy(), axis=0)
                object_max_x, object_max_y, _ = np.max(object_xyz.cpu().numpy(), axis=0)

                border = 0.01

                patch_box = np.array([[object_min_x - border, object_min_y - border, -0.1], [object_max_x + border, object_max_y + border, 0.1]])
                in_box = (gaussians._xyz[:, 0] > patch_box[0][0]) & (gaussians._xyz[:, 0] < patch_box[1][0]) & (gaussians._xyz[:, 1] > patch_box[0][1]) & (gaussians._xyz[:, 1] < patch_box[1][1]) & (gaussians._xyz[:, 2] > patch_box[0][2]) & (gaussians._xyz[:, 2] < patch_box[1][2])

                for i in range(len(self.object_gaussians)):
                    in_box[self.object_gaussians[i]] = False
                
                # remove the patch gaussians
                gaussians._xyz = gaussians._xyz[~in_box]
                gaussians._features_dc = gaussians._features_dc[~in_box]
                gaussians._features_rest = gaussians._features_rest[~in_box]
                gaussians._scaling = gaussians._scaling[~in_box]
                gaussians._rotation = gaussians._rotation[~in_box]
                gaussians._opacity = gaussians._opacity[~in_box]
                
                new_index = np.cumsum(~in_box.cpu().numpy()) - 1

                for i in range(len(self.object_gaussians)):
                    self.object_gaussians[i] = new_index[self.object_gaussians[i]]

                offset = object_max_y - object_min_y + border + 0.01

                source_box = patch_box - np.array([0.0, offset, 0.0])
                source_index = (gaussians._xyz[:, 0] > source_box[0][0]) & (gaussians._xyz[:, 0] < source_box[1][0]) & (gaussians._xyz[:, 1] > source_box[0][1]) & (gaussians._xyz[:, 1] < source_box[1][1]) & (gaussians._xyz[:, 2] > source_box[0][2]) & (gaussians._xyz[:, 2] < source_box[1][2])
                source_gaussians = gaussians._xyz[source_index] + torch.tensor([0.0, offset, 0.0], device=gaussians._xyz.device)
                source_features_dc = gaussians._features_dc[source_index]
                source_features_rest = gaussians._features_rest[source_index]
                source_scaling = gaussians._scaling[source_index]
                source_rotation = gaussians._rotation[source_index]
                source_opacity = gaussians._opacity[source_index]

                # add the source gaussians
                gaussians._xyz = torch.cat((gaussians._xyz, source_gaussians), dim=0).float()
                gaussians._features_dc = torch.cat((gaussians._features_dc, source_features_dc), dim=0).float()
                gaussians._features_rest = torch.cat((gaussians._features_rest, source_features_rest), dim=0).float()
                gaussians._scaling = torch.cat((gaussians._scaling, source_scaling), dim=0).float()
                gaussians._rotation = torch.cat((gaussians._rotation, source_rotation), dim=0).float()
                gaussians._opacity = torch.cat((gaussians._opacity, source_opacity), dim=0).float()

        self.gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)    

        self.gaussExtractor.viewpoint_stack = scene.getTrainCameras()
        self.gaussExtractor.estimate_bounding_sphere()
        self.radius = self.gaussExtractor.radius
        self.center = self.gaussExtractor.center

        print("--- Gaussian World Built ---")
    

    def render(self, Rs, Ts, image_size, FoV, device, object_states=[], rotate_num=0):
        viewpoint_cams = []
        image = torch.ones((1, image_size, image_size), device=device)
        for idx, cam in enumerate(Rs):
            uid = idx
            R = Rs[idx]
            T = Ts[idx]
            # print('R:', R)
            viewpoint_cam = Camera(colmap_id=uid, R=R, T=T, 
                                   FoVx=FoV, FoVy=FoV, 
                                   image=image, gt_alpha_mask=None,
                                   image_name="", uid=uid, data_device=device)
            viewpoint_cams.append(viewpoint_cam)

        self.gaussExtractor.reconstruction(viewpoint_cams, radius=self.radius, center=self.center, object_gaussians=self.object_gaussians, object_states=object_states)
        rgbmaps = np.array(self.gaussExtractor.rgbmaps)
        depthmaps = np.array(self.gaussExtractor.depthmaps)
        alphamaps = np.array(self.gaussExtractor.alphamaps)

        rgbmaps = np.clip(rgbmaps.transpose(0, 2, 3, 1), 0.0, 1.0)
        depthmaps = depthmaps.transpose(0, 2, 3, 1)[..., 0]
        # print('depthmaps:', depthmaps.shape)
        alphamaps = alphamaps.transpose(0, 2, 3, 1)[..., 0]

        # careful, rotating the images
        for i in range(rotate_num):
            rgbmaps[i] = cv2.rotate(rgbmaps[i], cv2.ROTATE_90_CLOCKWISE)
            depthmaps[i] = cv2.rotate(depthmaps[i], cv2.ROTATE_90_CLOCKWISE)
            alphamaps[i] = cv2.rotate(alphamaps[i], cv2.ROTATE_90_CLOCKWISE)
        
        return rgbmaps, depthmaps, alphamaps
    