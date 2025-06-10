#!/usr/bin/env python3

try:
    import trimesh
except ModuleNotFoundError:
    print("trimesh is required. Please install with `pip install trimesh`")
    exit(1)


import numpy as np
import sys
import os
import argparse
import coacd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        # required=True,
        help="input model loaded by trimesh. Supported formats: glb, gltf, obj, off, ply, stl, etc.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        # required=True,
        help="output model exported by trimesh. Supported formats: glb, gltf, obj, off, ply, stl, etc.",
    )
    parser.add_argument("--quiet", action="store_true", help="do not print logs")
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.01,
        help="termination criteria in [0.01, 1] (0.01: most fine-grained; 1: most coarse)",
    )
    parser.add_argument(
        "-pm",
        "--preprocess-mode",
        type=str,
        default="auto",
        help="No remeshing before running CoACD. Only suitable for manifold input.",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=50,
        help="surface samping resolution for Hausdorff distance computation",
    )
    parser.add_argument(
        "-nm",
        "--no-merge",
        action="store_true",
        help="If merge is enabled, try to reduce total number of parts by merging.",
    )
    parser.add_argument(
        "-d",
        "--decimate",
        action="store_true",
        help="If decimate is enabled, reduce total number of vertices per convex hull to max_ch_vertex.",
    )
    parser.add_argument(
        "-dt",
        "--max-ch-vertex",
        type=int,
        default=256,
        help="max # vertices per convex hull, works only when decimate is enabled",
    )
    parser.add_argument(
        "-ex",
        "--extrude",
        action="store_true",
        help="If extrude is enabled, extrude the neighboring convex hulls along the overlap face (other faces are unchanged).",
    )
    parser.add_argument(
        "-em",
        "--extrude-margin",
        type=float,
        default=0.01,
        help="extrude margin, works only when extrude is enabled",
    )
    parser.add_argument(
        "-c",
        "--max-convex-hull",
        type=int,
        default=-1,
        help="max # convex hulls in the result, -1 for no limit, works only when merge is enabled",
    )
    parser.add_argument(
        "-mi",
        "--mcts-iteration",
        type=int,
        default=150,
        help="Number of MCTS iterations.",
    )
    parser.add_argument(
        "-md",
        "--mcts-max-depth",
        type=int,
        default=3,
        help="Maximum depth for MCTS search.",
    )
    parser.add_argument(
        "-mn",
        "--mcts-node",
        type=int,
        default=20,
        help="Number of cut candidates for MCTS.",
    )
    parser.add_argument(
        "-pr",
        "--prep-resolution",
        type=int,
        default=50,
        help="Preprocessing resolution.",
    )
    parser.add_argument(
        "--pca",
        action="store_true",
        help="Use PCA to align input mesh. Suitable for non-axis-aligned mesh.",
    )
    parser.add_argument(
        "-am",
        "--apx-mode",
        type=str,
        default="ch",
        help="Approximation shape mode (ch/box).",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="folder name to decompose all models in the folder",
    )
    parser.add_argument("--iteration", type=int, default=None, help="iteration number of gaussian splatting")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    args = parser.parse_args()

    if args.quiet:
        coacd.set_log_level("error")

    iteration = args.iteration
    if iteration is not None:
        folder = f'../../gaussians/output/{args.name}/train/ours_{iteration}'
    else:
        mesh_floder = f'../../gaussians/output/{args.name}/train/'
        name_list = os.listdir(mesh_floder)
        max_iteration = np.max([int(name.split('_')[-1]) for name in name_list])
        folder = f'../../gaussians/output/{args.name}/train/ours_{max_iteration}'

    mesh_list = os.listdir(folder)

    for mesh_file in mesh_list:
        if not mesh_file.endswith(".ply"):
            continue
        if mesh_file.endswith("_convex.ply"):
            continue
        if mesh_file.endswith("_convex_normalize.ply"):
            continue
        if mesh_file.endswith("_normalize.ply"):
            continue
        if mesh_file == "background.ply":
            continue
        if mesh_file.startswith("fuse"):
            continue
        
        mesh_name = os.path.join(folder, mesh_file)

        input_file = mesh_name
        output_file = mesh_name.replace(".ply", "_convex.ply")
        print(f"Processing {input_file} -> {output_file}")

        if not os.path.isfile(input_file):
            break

        mesh = trimesh.load(input_file, force="mesh")
        mesh = coacd.Mesh(mesh.vertices, mesh.faces)
        result = coacd.run_coacd(
            mesh,
            threshold=0.1,
            max_convex_hull=-1,
            preprocess_mode=args.preprocess_mode,
            preprocess_resolution=50,
            resolution=50,
            mcts_nodes=args.mcts_node,
            mcts_iterations=args.mcts_iteration,
            mcts_max_depth=args.mcts_max_depth,
            pca=args.pca,
            merge=not args.no_merge,
            decimate=args.decimate,
            max_ch_vertex=args.max_ch_vertex,
            extrude=args.extrude,
            extrude_margin=args.extrude_margin,
            apx_mode=args.apx_mode,
            seed=args.seed,
        )
        mesh_parts = []
        for vs, fs in result:
            mesh_parts.append(trimesh.Trimesh(vs, fs))
        scene = trimesh.Scene()
        np.random.seed(0)
        for p in mesh_parts:
            p.visual.vertex_colors[:, :3] = (np.random.rand(3) * 255).astype(np.uint8)
            scene.add_geometry(p)
        scene.export(output_file)

    # # careful!! ignore background
    # exit(0)

    mesh_name = os.path.join(folder, f"background.ply")
    input_file = mesh_name
    output_file = mesh_name.replace(".ply", "_convex.ply")

    mesh = trimesh.load(input_file, force="mesh")
    mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    result = coacd.run_coacd(
        mesh,
        threshold=args.threshold,
        max_convex_hull=-1,
        preprocess_mode=args.preprocess_mode,
        preprocess_resolution=50,
        resolution=50,
        mcts_nodes=args.mcts_node,
        mcts_iterations=args.mcts_iteration,
        mcts_max_depth=args.mcts_max_depth,
        pca=args.pca,
        merge=not args.no_merge,
        decimate=args.decimate,
        max_ch_vertex=args.max_ch_vertex,
        extrude=args.extrude,
        extrude_margin=args.extrude_margin,
        apx_mode=args.apx_mode,
        seed=args.seed,
    )
    mesh_parts = []
    for vs, fs in result:
        mesh_parts.append(trimesh.Trimesh(vs, fs))
    scene = trimesh.Scene()
    np.random.seed(0)
    for p in mesh_parts:
        p.visual.vertex_colors[:, :3] = (np.random.rand(3) * 255).astype(np.uint8)
        scene.add_geometry(p)
    scene.export(output_file)

