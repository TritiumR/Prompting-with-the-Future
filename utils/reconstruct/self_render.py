# Data structures and functions for rendering
import torch
import cv2
import numpy as np
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesVertex
)

raster_settings = RasterizationSettings(
    image_size=512, 
    blur_radius=0.0, 
    faces_per_pixel=20, 
)


def get_cameras(R, T, device):
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=1.0, zfar=100.0)
    return cameras


def render_meshes_with_projection(mesh_list, R, T, device, znear=0.01, zfar=100.0, rotate_num=0):
    raster_settings = RasterizationSettings(
        image_size=1024, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )

    batch_size = len(R)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=znear, zfar=zfar)
    lights = PointLights(device=device, location=[[0.0, 0.0, -1.0]])

    # Create a Phong renderer by composing a rasterizer and a shader.
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )
    shader = SoftPhongShader(
        device=device,
        cameras=cameras,
        lights=lights
    )
    renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=shader
    )

    # Create a PyTorch3D Meshes object with textures
    pytorch_mesh_list = []
    for mesh in mesh_list:
        verts = torch.tensor(np.array(mesh.vertices), device=device).float()
        faces = torch.tensor(np.array(mesh.triangles), device=device).long()
        verts_rgb = torch.tensor(np.array(mesh.vertex_colors), device=device).float()
        texture = TexturesVertex(verts_features=[verts_rgb])
        mesh = Meshes(verts=[verts], faces=[faces], textures=texture)
        pytorch_mesh_list.append(mesh)

    meshes = join_meshes_as_scene(pytorch_mesh_list)
    meshes = meshes.extend(batch_size)

    fragments = renderer.rasterizer(meshes)
    images = shader(fragments, meshes)

    images = images.cpu().numpy()

    # Get face indices per pixel: [B, H, W, K]
    pixel_faces = fragments.pix_to_face
    
    # Create mask for valid pixels (face index >= 0)
    pixel_mask = pixel_faces >= 0
    
    # Get faces tensor: [B, F, 3]
    faces = meshes.faces_padded()
    # print('Faces: ', faces.shape)
    
    # Initialize output tensor for vertex indices
    B, H, W, K = pixel_faces.shape
    pixel_vertex_indices = torch.full(
        (B, H, W, K, 3),
        -1,
        dtype=torch.long,
        device=device
    )

    # For each batch
    for b in range(batch_size):
        # Get valid face indices for this batch
        valid_faces = pixel_faces[b][pixel_mask[b]]
        
        # Get vertex indices for these faces
        if len(valid_faces) > 0:
            # Get the vertices for each face: [num_valid_pixels, 3]
            face_vertices = faces[b, valid_faces - b * faces.shape[1]]
            
            # Place the vertex indices back in the full image
            pixel_vertex_indices[b][pixel_mask[b]] = face_vertices

    return images, pixel_vertex_indices.cpu().numpy()