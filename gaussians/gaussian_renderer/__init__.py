#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussians.scene.gaussian_model import GaussianModel
from gaussians.utils.sh_utils import eval_sh
from gaussians.utils.point_utils import depth_to_normal
from e3nn import o3


def transform_shs(shs_feat, rotation_matrix):
    shs_feat = shs_feat.cpu()
    rotation_matrix = rotation_matrix.cpu()
    ## rotate shs
    P = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=shs_feat.dtype) # switch axes: yzx -> xyz
    permuted_rotation_matrix = torch.linalg.inv(P) @ rotation_matrix @ P
    rot_angles = o3._rotation.matrix_to_angles(permuted_rotation_matrix)
    
    # Construction coefficient
    D_1 = o3.wigner_D(1, rot_angles[0], -rot_angles[1], rot_angles[2])
    D_2 = o3.wigner_D(2, rot_angles[0], -rot_angles[1], rot_angles[2])
    D_3 = o3.wigner_D(3, rot_angles[0], -rot_angles[1], rot_angles[2])

    # rotation of the shs features
    shs_feat[:, :3] = D_1 @ shs_feat[:, :3]
    shs_feat[:, 3:8] = D_2 @ shs_feat[:, 3:8]
    shs_feat[:, 8:15] = D_3 @ shs_feat[:, 8:15]
    return shs_feat


def transform_quaternion(q):
  """
  Transforms a quaternion to a coordinate system rotated 180 degrees around the y-axis.
  Parameters:
    q (torch.Tensor): Original quaternion [w, i, j, k].
  Returns:
    torch.Tensor: Transformed quaternion [w', i', j', k'].
  """
  # Ensure the quaternion is a PyTorch tensor
  if not isinstance(q, torch.Tensor):
    q = torch.tensor(q, dtype=torch.float32)
  # Quaternion representing 180° rotation about the y-axis
  q_y = torch.tensor([0.0, 0.0, -1.0, 0.0], dtype=q.dtype, device=q.device) # [w, i, j, k]
  q_y_conjugate = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=q.dtype, device=q.device) # Conjugate of q_y
  # Perform the conjugation: q' = q_y * q * q_y_conjugate
  def quaternion_multiply(q1, q2):
    """
    Multiplies two quaternions.
    Parameters:
      q1 (torch.Tensor): First quaternion [w, i, j, k].
      q2 (torch.Tensor): Second quaternion [w, i, j, k].
    Returns:
      torch.Tensor: Result of quaternion multiplication.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return torch.tensor([
      w1*w2 - x1*x2 - y1*y2 - z1*z2,
      w1*x2 + x1*w2 + y1*z2 - z1*y2,
      w1*y2 - x1*z2 + y1*w2 + z1*x2,
      w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=q1.dtype, device=q1.device)
  q_prime = quaternion_multiply(quaternion_multiply(q_y, q), q_y_conjugate)
  q_prime = q_prime / torch.norm(q_prime) # Normalize the quaternion
  return q_prime


def quaternion_multiply_batch(a, b):
    if a.ndim == 1 or a.size(0) == 1:
        a = a.view(1, 4).expand(b.size(0), 4)
    elif b.ndim == 1 or b.size(0) == 1:
        b = b.view(1, 4).expand(a.size(0), 4)
    return torch.stack([
        a[:,0]*b[:,0] - a[:,1]*b[:,1] - a[:,2]*b[:,2] - a[:,3]*b[:,3],
        a[:,0]*b[:,1] + a[:,1]*b[:,0] + a[:,2]*b[:,3] - a[:,3]*b[:,2],
        a[:,0]*b[:,2] - a[:,1]*b[:,3] + a[:,2]*b[:,0] + a[:,3]*b[:,1],
        a[:,0]*b[:,3] + a[:,1]*b[:,2] - a[:,2]*b[:,1] + a[:,3]*b[:,0]
    ], dim=1)


def rotate_points_with_quaternion(points, quaternion):
    """
    Rotate 3D points using a quaternion with PyTorch tensors.
    
    Args:
    points (torch.Tensor): Tensor of 3D points with shape (n, 3)
    quaternion (torch.Tensor): Rotation quaternion [w, x, y, z]
    
    Returns:
    torch.Tensor: Rotated 3D points with shape (n, 3)
    """
    # Ensure inputs are torch tensors
    points = points.float() if not isinstance(points, torch.Tensor) else points
    quaternion = quaternion.float() if not isinstance(quaternion, torch.Tensor) else quaternion
    
    # Normalize the quaternion
    q = quaternion / torch.norm(quaternion)
    
    # Extract quaternion components
    w, x, y, z = q
    
    # Rotation matrix from quaternion
    rotation_matrix = torch.tensor([
        [1 - 2*y**2 - 2*z**2,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ], dtype=points.dtype, device=points.device)
    
    # Rotate points
    rotated_points = torch.matmul(points, rotation_matrix.T)
    
    return rotated_points


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, radius=None, center=None, object_gaussians=[], object_states=[]):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz.clone()
    means2D = screenspace_points
    opacity = pc.get_opacity.clone()

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    shs = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation.clone()
        shs = pc.get_features.clone()

    # follow the simulator to update the object states
    old_opacity = opacity.clone()
    for obj_gaussian, obj_state in zip(object_gaussians, object_states):
        gaussain_translation = obj_state[:3].clone()
        rotation = obj_state[3:7].to(means3D.device).clone()
        rotation = transform_quaternion(rotation)

        middle = (means3D[obj_gaussian].min(dim=0)[0] + means3D[obj_gaussian].max(dim=0)[0]) / 2

        # careful! could be problematic if the middle of gaussian is different from the middle of the object mesh
        means3D[obj_gaussian] -= middle
        means3D[obj_gaussian] = rotate_points_with_quaternion(means3D[obj_gaussian], rotation)
        means3D[obj_gaussian] += middle

        new_rotation = quaternion_multiply_batch(
            rotation.to(rotations[obj_gaussian].device),
            rotations[obj_gaussian]
        )
        new_rotation = new_rotation / torch.norm(new_rotation, dim=-1, keepdim=True)
        rotations[obj_gaussian] = new_rotation

        # rotating shs
        obj_shs = pc.get_features[obj_gaussian].clone()
        rotation_matrix = o3.quaternion_to_matrix(rotation)
        non_base_shs = obj_shs[:, 1:]
        new_shs = transform_shs(non_base_shs, rotation_matrix).to(obj_shs.device)
        new_shs = torch.cat([obj_shs[:, :1], new_shs], dim=1)
        shs[obj_gaussian] = new_shs

        # translation
        gaussain_translation = obj_state[:3].clone()
        gaussain_translation[0] = -gaussain_translation[0]
        gaussain_translation[2] = -gaussain_translation[2]
        means3D[obj_gaussian] += gaussain_translation.to(means3D.device)
        
    if center is not None and radius is not None:
        out_of_radius = torch.norm(means3D - center, dim=1) > radius
        opacity[out_of_radius] = 0.0
        # keep the opacity of the object gaussians
        for obj_gaussian in object_gaussians:
            opacity[obj_gaussian] = old_opacity[obj_gaussian]
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
    # shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            # print('here!!!')
            if shs is not None:
                shs = shs
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color
    
    rendered_image, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets =  {"render": rendered_image,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
    }


    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1; 
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()


    rets.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
    })

    return rets