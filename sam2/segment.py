import torch
import sys
sys.path.append("../../")
from sam2.sam2_video_predictor import SAM2VideoPredictor
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from utils.prompt_gpt import generate_segment_names, get_names
import numpy as np
import torch
import requests
import cv2
from PIL import Image
import requests
import re
import open3d as o3d
from scipy.spatial import KDTree
import os
import argparse
import base64


def get_momol_prompt(image, instruction):
    """
    Get the all objects names relevant to the instruction from the image by prompting GPT
    """
    prompt_file = "../../prompts/segment_all_name_system_prompt.txt"
    with open(prompt_file, "r") as f:
        system_prompt = f.read()

    name_list = None

    # content = generate_segment_names(system_prompt, image, instruction)
    # name_list = get_names(content)

    try_times = 0
    while name_list is None and try_times < 5:
        try:
            content = generate_segment_names(system_prompt, image, instruction)
            name_list = get_names(content)
        except:
            print("Error in generating prompts, retrying...")
            try_times += 1
            continue

    return name_list


def post_process_mesh(mesh, cluster_to_keep=1):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = mesh
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    # print("num vertices raw {}".format(len(mesh.vertices)))
    # print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0, triangles_to_remove


def shrink_contour(mask, kernel_size=3, iterations=1):
    """
    Shrinks the contour of a binary segmentation mask using erosion.

    Parameters:
    - mask (numpy.ndarray): Binary segmentation mask (values: 0 or 255).
    - kernel_size (int): Size of the erosion kernel. Larger size means more shrinking.
    - iterations (int): Number of erosion iterations. More iterations mean more shrinking.

    Returns:
    - numpy.ndarray: Segmentation mask with shrunken contour.
    """
    if len(mask.shape) != 2:
        raise ValueError("Input mask must be a binary 2D array.")

    # Create the erosion kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Apply erosion
    shrunken_mask = cv2.erode(mask, kernel, iterations=iterations)

    return shrunken_mask


def point(image, text_list):
    # load the processor
    processor = AutoProcessor.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        use_fast=True,
        torch_dtype='auto',
        device_map='cpu'
    )

    # load the model
    model = AutoModelForCausalLM.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='cpu'
    )

    # process the image and text
    # text = "Please point to all movable objects on the table"
    # text = "Please point to the bread"
    # text = "Please point to the cup on the table"
    # text = "Please point to the shoes"
    # text = "Please point to the drum stick"
    # text = "Please point to the cucumber on the table"
    # text = "Please point to charger"
    # text = "Please point to egg box"
    generated_text_list = []
    for text in text_list:
        text = f"Please point to the {text}"
        # text = "Please point to the tennis ball on the table"
        # text = "Please point to toy blocks"
        inputs = processor.process(
            images=[image],
            text=text,
        )

        # move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

        # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )

        # only get generated tokens; decode them to text
        generated_tokens = output[0,inputs['input_ids'].size(1):]
        generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # print the generated text
        print(generated_text)

        generated_text_list.append(generated_text)

    # free the model to save memory
    del model
    del processor
    torch.cuda.empty_cache()
    # torch.cuda.ipc_collect()

    return generated_text_list
    

def segment(video_path, points, mesh, per_pixel_vertices):
    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-large")

    points = np.array(points, dtype=np.float32)
    labels = np.array([[1]] * len(points), np.int32)
    # list from 0 to len(points)
    all_object_ids = list(range(len(points)))

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        mesh_vertices = np.asarray(mesh.vertices)
        mesh_faces = np.asarray(mesh.triangles)
        vertices_labels = np.zeros((len(mesh_vertices)), dtype=np.int32) - 1
        state = predictor.init_state(video_path)

        # boxes = np.array(boxes, dtype=np.float32)

        # add new prompts and instantly get the output on the same frame
        for obj_id in all_object_ids:
            frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, frame_idx=0, obj_id=obj_id, points=points[obj_id], labels=labels[obj_id])

        # propagate the prompts to get masklets throughout the video
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            per_pixel_vertice = per_pixel_vertices[frame_idx]
            for obj_id in object_ids:
                mask = masks[obj_id]
                object_mask = (mask[0] > 0.0).cpu().numpy()
                shrinked_mask = shrink_contour((object_mask * 255).astype(np.uint8), kernel_size=7, iterations=4)
                shrinked_mask = shrinked_mask > 0
                vertices_mask = per_pixel_vertice[shrinked_mask].reshape(-1)
                # print('vertices_mask:', vertices_mask.shape)
                vertices_labels[vertices_mask] = obj_id

            if frame_idx % 10 != 0:
                continue
            
            vis_mask = np.zeros((masks[0].shape[1], masks[0].shape[2]), dtype=np.uint8)
            for object_id in object_ids:
                mask = masks[object_id]
                # print('mask:', mask.shape)
                # mask_min = mask.min()
                # mask_max = mask.max()
                object_mask = (mask > 0.0).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)

                # print('object_mask:', object_mask.shape)
                vis_mask[object_mask[:, :, 0] > 0] = 255 / len(object_ids) * (object_id + 1)
                # print(object_mask.shape)
                # cv2.imshow("Object mask", object_mask * 255)
                # cv2.imwrite(f"images/mask_{frame_idx}_{object_id}.png", object_mask * 255)
            # cv2.imwrite(f"images/mask_{frame_idx}.png", vis_mask)

    threshold = 0.005
    object_meshes = []
    for obj_id in all_object_ids:
        object_vertices = mesh_vertices[vertices_labels == obj_id]
        object_min = object_vertices.min(axis=0)
        object_max = object_vertices.max(axis=0)
        in_box_vertices_ids = np.where((mesh_vertices[:, 0] >= object_min[0]) & (mesh_vertices[:, 0] <= object_max[0]) & 
                                        (mesh_vertices[:, 1] >= object_min[1]) & (mesh_vertices[:, 1] <= object_max[1]) & 
                                        (mesh_vertices[:, 2] >= object_min[2]) & (mesh_vertices[:, 2] <= object_max[2]))[0]

        in_box_vertices = mesh_vertices[in_box_vertices_ids]

        kdtree = KDTree(object_vertices)
        # distances, indices = kdtree.query(gaussian_points, k=1)
        distances, indices = kdtree.query(in_box_vertices, k=1)
        # print('distances:', distances.shape)

        object_indices = np.where(distances < threshold)[0]
        object_indices = in_box_vertices_ids[object_indices]
        # print('object_indices:', object_indices.shape)

        # mask the inbox vertices
        vertices_labels[object_indices] = obj_id

        # save object mesh
        # print('object_mesh:', object_mesh.vertices.shape)
        # Convert sub_vertex_ids to a set for fast lookup
        object_vertex_ids_set = set(object_indices)

        # Find the faces that are fully contained in the sub_vertex_ids
        mask = np.all(np.isin(mesh_faces, list(object_vertex_ids_set)), axis=1)
        object_faces = mesh_faces[mask]

        # Extract the subset of vertices that are actually used in the sub_faces
        used_vertex_ids = np.unique(object_faces)
        # print('used_vertex_ids:', used_vertex_ids.shape)
        # object_vertex_map = {old_id: new_id for new_id, old_id in enumerate(used_vertex_ids)}

        object_vertex_map = np.zeros(mesh_vertices.shape[0], dtype=int) - 1  # Default to -1 for unmapped vertices
        object_vertex_map[used_vertex_ids] = np.arange(len(used_vertex_ids))

        # Remap the face indices to the new vertex numbering
        remapped_faces = object_vertex_map[object_faces].astype(np.int32)
        # print('remapped_faces:', remapped_faces.shape, remapped_faces.min(), remapped_faces.max())

        object_vertices = mesh_vertices[used_vertex_ids].astype(np.float64)
        # print('object_vertices: ', object_vertices.shape)

        object_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(object_vertices),
            triangles=o3d.utility.Vector3iVector(remapped_faces)
        )

        post_object_mesh, triangles_to_remove = post_process_mesh(object_mesh)
        # print('triangles_to_remove:', triangles_to_remove.shape)

        removed_face_id = np.where(triangles_to_remove)[0]

        # print('removed_face_id:', removed_face_id.shape)

        removed_vertices = np.unique(remapped_faces[removed_face_id])

        # print('removed_vertices:', removed_vertices.shape)

        give_back_to_background = used_vertex_ids[removed_vertices]

        vertices_labels[give_back_to_background] = -1

        object_meshes.append(post_object_mesh)

    background_indices = np.where(vertices_labels == -1)[0]
    print('background_indices:', background_indices.shape)

    background_vertex_ids_set = set(background_indices)

    mask = np.all(np.isin(mesh_faces, list(background_vertex_ids_set)), axis=1)
    background_faces = mesh_faces[mask]

    used_vertex_ids = np.unique(background_faces)

    background_vertex_map = np.zeros(mesh_vertices.shape[0], dtype=int) - 1  # Default to -1 for unmapped vertices
    background_vertex_map[used_vertex_ids] = np.arange(len(used_vertex_ids))

    remapped_faces = background_vertex_map[background_faces].astype(np.int32)
    # print('remapped_faces:', remapped_faces.shape, remapped_faces.min(), remapped_faces.max())

    background_vertices = mesh_vertices[used_vertex_ids].astype(np.float64)
    # print('object_vertices: ', object_vertices.shape)

    background_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(background_vertices),
        triangles=o3d.utility.Vector3iVector(remapped_faces)
    )

    post_background_mesh, triangles_to_remove = post_process_mesh(background_mesh)

    object_meshes.append(post_background_mesh)

    return object_meshes


def render_parser():
    parser = argparse.ArgumentParser(description='Segment objects from a mesh')
    parser.add_argument('--name', type=str, default='bunny', help='Name of the mesh')
    parser.add_argument('--instruction', type=str, default='bunny', help='Name of the object that you want to segment')
    parser.add_argument('--iteration', type=int, default=None, help='iterations of the gaussian splatting')
    return parser


if __name__ == "__main__":
    import sys
    parser = render_parser()
    args = parser.parse_args()
    name = args.name
    instruction = args.instruction

    iteration = args.iteration
    if iteration is not None:
        mesh_path = f"../output/{name}/train/ours_{iteration}/fuse_post.ply"
        folder = f"../output/{name}/train/ours_{iteration}"
    else:
        mesh_floder = f'../output/{name}/train/'
        name_list = os.listdir(mesh_floder)
        max_iteration = np.max([int(name.split('_')[-1]) for name in name_list])

        mesh_path = f"../output/{name}/train/ours_{max_iteration}/fuse_post.ply"
        folder = f"../output/{name}/train/ours_{max_iteration}"

    video_path = mesh_path.replace(".ply", ".mp4")

    device = torch.device("cuda:0")

    pattern = r'x\d*="([\d.]+)" y\d*="([\d.]+)"'

    video = cv2.VideoCapture(video_path)
    first_frame = video.read()[1]
    first_frame = Image.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))

    # save image
    image_path = mesh_path.replace(".ply", ".png")
    first_frame.save(image_path)

    # molmo prompt
    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    text_list = get_momol_prompt(encoded_image, instruction)

    print('text_list:', text_list)
    # exit(0)

    # get object points
    points_list = point(first_frame, text_list)
    name_list = []
    pairs_list = []
    for text_id, points in enumerate(points_list):
        pairs = re.findall(pattern, points)
        pairs_list += pairs
        if len(pairs) == 1:
            name_list += [text_list[text_id]]
        else:
            name_list += [text_list[text_id] + str(i) for i in range(len(pairs))]

    print('pairs_list:', pairs_list)
    # # Convert matches to a structured list of float pairs
    coordinate_pairs = [[[float(x), float(y)]] for x, y in pairs_list]

    image_size = first_frame.size
    for coordinate_pair in coordinate_pairs:
        coordinate_pair[0][0] = int(coordinate_pair[0][0] / 100 * image_size[0])
        coordinate_pair[0][1] = int(coordinate_pair[0][1] / 100 * image_size[1])

    # coordinate_pairs = [[[248, 245]], [[250, 336]], [[291, 420]]]

    scene_mesh = o3d.io.read_triangle_mesh(mesh_path)
    per_pixel_vertices = np.load(mesh_path.replace('.ply', '.npy'))
    print('per_pixel_vertices:', per_pixel_vertices.shape)


    if len(coordinate_pairs) == 0:
        o3d.io.write_triangle_mesh(f"{folder}/background.ply", scene_mesh)
        exit()

    object_meshes = segment(video_path, coordinate_pairs, scene_mesh, per_pixel_vertices)

    for i, object_mesh in enumerate(object_meshes):
        if i == len(object_meshes) - 1:
            o3d.io.write_triangle_mesh(f"{folder}/background.ply", object_mesh)
        else:
            o3d.io.write_triangle_mesh(f"{folder}/{name_list[i]}.ply", object_mesh)

    print(f"Segment {len(object_meshes) - 1} objects.")
    

