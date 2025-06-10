import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import copy
import base64
import cv2
import time
import argparse
import multiprocessing

from mesh.mesh_world import MeshWorld
from utils.robot import Robot, Robotiq
from utils.prompt import *
from utils.camera import get_up_direction, create_wrist_camera, fixed_to_gripper_gaussian

from utils.prompt_gpt import *
from self_render import render_meshes, get_cameras, render_meshes_with_depth
from gaussian.gaussian_world import GaussianWorld

from pytorch3d.renderer import (
    look_at_view_transform,
    RasterizationSettings
)

def robo4d_parse():
    parser = argparse.ArgumentParser(description="Robo4D")
    parser.add_argument("--instruction", type=str, default="push the laptop screen")
    parser.add_argument("--name", type=str, default='test')
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--total_steps", type=int, default=10)
    parser.add_argument("--overlay", action="store_true")
    parser.add_argument("--pivot", action="store_true")
    parser.add_argument("--repeat_time", type=int, default=1)
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--camera_view_id", type=int, default=0)
    parser.add_argument("--view_num", type=int, default=4)
    parser.add_argument("--random_view", action="store_true")
    parser.add_argument("--prompt_version", type=int, default=0)
    parser.add_argument("--temperature", type=int, default=0)
    parser.add_argument("--current", action="store_true")
    parser.add_argument("--example_number", type=int, default=0)
    parser.add_argument("--view_example_number", type=int, default=0)
    parser.add_argument("--use_history", action="store_true")
    parser.add_argument("--max_history", type=int, default=4)
    parser.add_argument("--hack_view", type=int, default=-1)
    parser.add_argument("--qwen", action="store_true")
    parser.add_argument("--change_view", action="store_true")
    parser.add_argument("--change_everytime", action="store_true")
    parser.add_argument("--change_to_others", action="store_true")
    parser.add_argument("--only_hand", action="store_true")
    parser.add_argument("--plane_action", action="store_true")
    parser.add_argument("--sample_open_close", action="store_true")
    parser.add_argument("--open_close", action="store_true")
    parser.add_argument("--subgoal", action="store_true")
    parser.add_argument("--subgoal_image", action="store_true")
    parser.add_argument("--wrist_camera", action="store_true")
    parser.add_argument("--cem_iteration", type=int, default=3)
    parser.add_argument("--num_sample_each_group", type=int, default=6)
    parser.add_argument("--num_sample_actions", type=int, default=36)
    parser.add_argument("--num_sample_vlm", type=int, default=36)
    parser.add_argument("--rotation", action="store_true")
    parser.add_argument("--rotation_or_translation", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--gripper_from_sim", action="store_true")
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--try_grasp", action="store_true")
    parser.add_argument("--try_release", action="store_true")
    parser.add_argument("--gripper_visual_prompt", action="store_true")
    parser.add_argument("--zoom_in", action="store_true")
    parser.add_argument("--keep_ratio", action="store_true")
    parser.add_argument("--stage", action="store_true")
    parser.add_argument("--ABCD", action="store_true")
    parser.add_argument("--abcd", action="store_true")
    parser.add_argument("--uniform", action="store_true")
    parser.add_argument("--no_gripper_when_grasping", action="store_true")
    parser.add_argument("--close_gripper", action="store_true")
    parser.add_argument("--only_z", action="store_true")
    parser.add_argument("--replan", action="store_true")
    parser.add_argument("--success", action="store_true")
    parser.add_argument("--just_grasp_it", action="store_true")
    parser.add_argument("--scene_name", type=str, default=None)
    parser.add_argument("--background_name", type=str, default=None)
    parser.add_argument("--gaussian_iteration", type=int, default=None)
    parser.add_argument("--scene_id", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=10)
    return parser

parser = robo4d_parse()
args = parser.parse_args()

assert not (args.random_view and args.plane_action), 'random view plane action not implemented'

image_size = args.image_size  # Size of the output image
raster_settings = RasterizationSettings(
    image_size=image_size, 
    blur_radius=0.0, 
    faces_per_pixel=20, 
)
znear = 0.01
zfar = 100
FoV = 60

after_image_weight = 0.5

if args.scene_name is None:
    output_path = os.path.join('results', f'{args.instruction}/{args.scene_id}/{args.name}')
else:
    output_path = os.path.join('results', f'{args.instruction}/{args.scene_name}/{args.name}')

if not os.path.exists(output_path):
    os.makedirs(output_path)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

system_prompt = get_prompt(args)
close_gripper_prompt = get_close_gripper_prompt(args)

if args.subgoal:
    subgoal_prompt = get_subgoal_prompt(args)
    args.change_view = True

if args.change_view:    
    select_prompt = get_view_prompt(args)
    change_prompt = get_change_prompt(args)

if args.gripper_from_sim or args.try_grasp or args.try_release:
    grasp_prompt = get_grasp_prompt(args)
    release_prompt = get_release_prompt(args)

if args.stage:
    stage_prompt = get_stage_prompt(args)

examples = None
if args.example_number > 0:
    examples = get_example(args)

if args.view_example_number > 0:
    view_examples = get_view_example(args)

if args.success:
    success_prompt = get_success_prompt(args)

previous_instruction = args.instruction

distance = 1.5
if args.instruction == 'close the laptop':
    scene_name = 'home2_world'
    robot_translation = [-0.4, -0.1, 0]
    action_ids = [5, 5, 5]
    ori_gt_action_id = 0
elif args.instruction == 'touch the laptop':
    scene_name = 'home1_world'
    robot_translation = [-0.3, -0.1, 0]
    action_ids = []
    ori_gt_action_id = 0
    depth_truc = 4
elif args.instruction == 'grasp the laptop screen':
    scene_name = 'home1_world'
    robot_translation = [-0.4, 0.1, 0]
    action_ids = [5, 5, 5]
    ori_gt_action_id = 5
    depth_truc = 4
elif args.instruction == 'push the chair':
    scene_name = 'office1_world'
    robot_translation = [-0.4, -0.1, 0]
    action_ids = [5, 5, 5, 1, 1]
    ori_gt_action_id = 0
    depth_truc = 4
elif args.instruction == "type 'hello world' with the keyboard":
    scene_name = 'keyboard_world'
    robot_translation = [-0.5, -0.1, 0.2]
    action_ids = [5, 5, 5, 1]
    ori_gt_action_id = 0
    depth_truc = 4
elif args.instruction == "type 'robo4D' with the keyboard":
    scene_name = 'keyboard1_world'
    robot_translation = [-0.5, -0.1, 0.2]
    action_ids = [5, 5, 5, 1]
    ori_gt_action_id = 0
    depth_truc = 4
elif args.instruction == "press the keyboard":
    scene_name = 'office3_world'
    robot_translation = [-1.0, -0.0, 0.0]
    action_ids = [5, 5, 5, 5]
    ori_gt_action_id = 0
elif args.instruction == "put the black pen on the paper into the pencil box":
    scene_name = 'pen_to_box_world'
    robot_translation = [-0.6, -0.0, 0.0]
    action_ids = [5, 5]
    ori_gt_action_id = 0
elif args.instruction == "put the orange airpod into the pencil box":
    scene_name = 'pen_to_box_world'
    robot_translation = [-0.8, -0.0, 0.0]
    action_ids = [5, 5, 9]
    ori_gt_action_id = 0
elif args.instruction == "pick up the orange airpod":
    scene_name = 'pen_to_box_world'
    robot_translation = [-0.88, -0.1, -0.]
    action_ids = [2]
elif args.instruction == "pick up the orange airpods":
    scene_name = 'pen_to_box_world'
    robot_translation = [-0.9, 0.0, -0.05]
    action_ids = [5]
    ori_gt_action_id = 0
# elif args.instruction == "toast the bread":
#     scene_name = 'bread1_world'
#     robot_translation = [-0.4, -0.0, -0.0]
#     action_ids = [2, 2, 2, 3]
elif args.instruction == "toast the bread":
    scene_name = 'bread_1_world'
    robot_translation = [-0.38, -0.0, -0.0]
    action_ids = [2, 1, 3]
elif args.instruction == "toasting the bread":
    scene_name = 'bread2_world'
    robot_translation = [-0.4, -0.0, -0.0]
    action_ids = [2, 2, 2, 3]
elif args.instruction == "toasting a bread":
    scene_name = 'bread_clean_1_world'
    distance = 1.8
    robot_translation = [-0.4, -0.0, -0.0]
    action_ids = [2, 2]
elif args.instruction == "pick up the bread":
    scene_name = 'bread1_world'
    robot_translation = [-0.4, -0.0, -0.0]
    action_ids = [2, 2]
elif args.instruction == "toast bread":
    scene_name = 'new_bread_1_world'
    robot_translation = [-0.38, -0.0, -0.0]
    action_ids = [2]
elif args.instruction == "toasting bread":
    scene_name = 'new_bread_2_world'
    robot_translation = [-0.38, 0.1, -0.0]
    action_ids = [2]
# elif args.instruction == "water the plant with the cup":
#     scene_name = 'water_world'
#     robot_translation = [-0.4, -0.0, -0.2]
#     action_ids = [5, 9, 12]
# elif args.instruction == "water the plant with the green cup":
#     scene_name = 'plant_1_world'
#     robot_translation = [-0.3, 0.1, 0.05]
#     action_ids = [2, 3]
elif args.instruction == "hang the cup to the mug holder":
    scene_name = 'cup_world'
    robot_translation = [-0.4, -0.0, -0.0]
    action_ids = [2, 2]
elif args.instruction == "hang a cup to the mug holder":
    scene_name = 'cup_clean_1_world'
    robot_translation = [-0.4, -0.0, -0.0]
    action_ids = [2, 2]
elif args.instruction == "water the plant with the white cup":
    scene_name = 'water_world_test'
    robot_translation = [-0.4, -0.0, -0.2]
    action_ids = [5, 9, 12]
elif args.instruction == "put the orange airpods into the pencil box":
    scene_name = 'pen_to_box_world'
    robot_translation = [-0.88, -0.1, -0.]
    action_ids = [2]
elif args.instruction == "pair up the shoes":
    scene_name = 'shoes_1_world'
    robot_translation = [-0.38, 0.13, 0.04]
    action_ids = [2, 2]
elif args.instruction == "pair up the shoe":
    scene_name = 'shoes_3_world'
    robot_translation = [-0.35, 0., 0.]
    action_ids = [2, 8, 8, 8, 8, 8]
elif args.instruction == "pair up the brown shoes":
    scene_name = 'shoes_4_world'
    robot_translation = [-0.40, 0., 0.]
    action_ids = [2, 1, 1, 1, 1]
elif args.instruction == "pair up the shoes":
    scene_name = 'shoes_4_world'
    robot_translation = [-0.40, 0., 0.]
    action_ids = [2, 1, 1, 1, 1]
elif args.instruction == "pair up white shoes":
    scene_name = 'shoes_4_world'
    robot_translation = [-0.40, 0., 0.]
    action_ids = [2]
elif args.instruction == "move the separated white shoe to its pair":
    scene_name = 'shoes_5_world'
    robot_translation = [-0.40, 0., 0.]
    action_ids = [2]
elif args.instruction == "move the brown shoe to its pair on the white board":
    scene_name = 'shoes_6_world'
    robot_translation = [-0.40, 0., 0.]
    action_ids = [2]
elif args.instruction == "hit the purple drum with the drumstick":
    scene_name = 'drum_1_world'
    robot_translation = [-0.35, 0., 0.0]
    action_ids = [2]
elif args.instruction == "water the plant with the blue cup":
    scene_name = 'plant_4_world'
    robot_translation = [-0.35, 0.0, 0.0]
    action_ids = [2, 3]
elif args.instruction == "water the plant with the cup":
    scene_name = 'plant_5_world'
    robot_translation = [-0.35, 0.1, 0.0]
    action_ids = [2]
elif args.instruction == "water the plant with the green cup":
    scene_name = 'water_3_world'
    robot_translation = [-0.35, 0.0, 0.0]
    action_ids = [2]
elif args.instruction == "pick up the sprite":
    scene_name = 'water_world_test'
    robot_translation = [-0.38, -0.0, -0.005]
    action_ids = [3, 3, 5]
elif args.instruction == "press the space on the keyboard":
    scene_name = 'space_world'
    robot_translation = [-0.38, 0.0, 0.0]
    action_ids = [2]
elif args.instruction == "put the green cucumber into the basket":
    scene_name = 'basket_world'
    robot_translation = [-0.38, 0.0, 0.0]
    action_ids = [8, 8]
elif args.instruction == "put all vegetables into the basket":
    scene_name = 'basket_world'
    robot_translation = [-0.38, 0.0, 0.0]
    action_ids = [2, 11, 11, 11]
    # action_ids = [2, 2, 2]
elif args.instruction == "unplug the charger":
    scene_name = 'charger_1_world'
    robot_translation = [-0.35, 0.0, 0.0]
    action_ids = [2, 4, 4]
    # action_ids = [2, 2, 2]
elif args.instruction == "push egg carton into dustpan":
    scene_name = 'egg_1_world'
    robot_translation = [-0.35, 0.0, 0.0]
    action_ids = [2, 3, 3, 1, 1, 1, 1, 5]
elif args.instruction == "push the egg carton into dustpan":
    scene_name = 'egg_3_world'
    robot_translation = [-0.35, 0.0, 0.0]
    action_ids = [2]
    # action_ids = [2, 2, 2]
elif args.instruction == "play the lowest pitch with the drum stick":
    scene_name = 'tune_2_world'
    robot_translation = [-0.38, 0.0, 0.0]
    action_ids = [2, 5, 0, 4]
    # action_ids = [2, 2, 2]
elif args.instruction == "sweep the trash off the table":
    scene_name = 'sweep_1_world'
    robot_translation = [-0.38, 0.0, 0.0]
    action_ids = [2, 5, 0, 4]
    # action_ids = [2, 2, 2]
elif args.instruction == "wipe the tea with the sponge":
    scene_name = 'sponge_1_world'
    robot_translation = [-0.38, 0.0, 0.0]
    action_ids = [2]
    # action_ids = [2, 2, 2]
elif args.instruction == "put the tennis ball into the tennis ball tube":
    scene_name = 'tennis_1_world'
    robot_translation = [-0.38, 0.0, 0.0]
    action_ids = [2]
    # action_ids = [2, 2, 2]
else:
    if args.scene_name is not None:
        scene_name = args.scene_name
        robot_translation = [-0.38, 0.0, 0.0]
        action_ids = [2]
    else:
        raise('instruction not simple')

if args.scene_id != 0:
    scene_name = f'{scene_name.split("_")[0]}_{args.scene_id}_world'

if args.scene_name is not None:
    scene_name = args.scene_name

max_replan = 5

# build gaussian world
gaussian_world = GaussianWorld(scene_name, parser, gaussian_iteration=args.gaussian_iteration, background_name=args.background_name)
# center = gaussian_world.center.cpu().numpy()
# careful! camera distance
center = np.array([0, 0, 0])
radius = gaussian_world.radius * distance

print('center: ', center)
print('radius: ', radius)

change = None
close_gripper = False
times = 0
while change is None and times < 5:
    try:
        times += 1
        content = generate_close_gripper(close_gripper_prompt)
        close_gripper = get_close_gripper(content)
        change = True
    except Exception as e:
        print('catched', e)
        pass

with open(f'{output_path}/close_gripper_content.txt', 'w') as f:
    f.write(content)

if close_gripper:
    print('!!! Keep Gripper Closed !!!')

if args.only_hand:
    robot_uids = 'PandaRobotiqHand'
else:
    robot_uids = 'PandaRobotiq'
# build mesh world
mesh_world = MeshWorld(scene_name, num_envs=args.num_sample_actions, scene_traslation=-np.array(robot_translation), radius=radius, \
                       image_size=image_size, record_video=args.record_video, robot_uids=robot_uids, need_render=True, dir=output_path, \
                        close_gripper=close_gripper, gaussian_iteration=args.gaussian_iteration, background_name=args.background_name)

if args.qwen:
    from prompt_qwen import Qwen, qwen_prompt_helper
    qwen = Qwen(model_size="2B")

elev = torch.tensor([-70, 0, 70, 0], device=device)
azim = torch.tensor([0, 70, 0, 0], device=device)
up = get_up_direction(elev, azim)
at = torch.tensor(center[None], device=device).float()
R_fixed, T_fixed = look_at_view_transform(dist=radius, elev=elev, azim=azim, up=up, at=at, device=device)
up[:, 0] = -up[:, 0]
at[:, 0] = -at[:, 0]
elev = 180 - elev
R_gaussian_fixed, T_gaussian_fixed = look_at_view_transform(dist=radius, elev=elev, azim=azim, up=up, at=at, device="cpu")
R_gaussian_fixed = R_gaussian_fixed.numpy()
T_gaussian_fixed = T_gaussian_fixed.numpy()

plane_action_none_dimensions = [1, 0, 1, 2]

success = False

for time_id in range(args.repeat_time):
    if success:
        break
    trajectory = []
    history = []
    excute_frames = []
    history_object_states = []
    output_actions = []
    mesh_world.grasping_now = False
    mesh_world.grasping_pos = -1.0
    replan_time = 0
    if close_gripper:
        mesh_world.grasping_now = True
        mesh_world.grasping_pos = 1.0
        output_actions.append('grasp')
    else:
        mesh_world.grasping_pos = -1.0

    joint_angles = np.array([
        # FR3 Joints
        0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, np.pi / 4,
        # Robotiq Joints (4 Passive, 2 Active)
        0, 0, 0, 0, 0, 0
    ])

    trajectory.append(torch.tensor(joint_angles))

    for action_id in action_ids:
        joint_angles, robot_images, robot_depth_images = mesh_world.one_step_action_batch(trajectory[-1], action_id, disable_self_collisions=False)
        trajectory[0] = joint_angles

    output_actions.append(joint_angles.cpu().numpy().tolist())

    encoded_image = None

    robot_images, robot_depth_images = mesh_world.get_image_depth()
    current_robot_images, current_robot_depths = robot_images, robot_depth_images
    rgbmaps, depthmaps, alphamaps = gaussian_world.render(R_gaussian_fixed, T_gaussian_fixed, image_size, -FoV / 180.0 * np.pi, device, rotate_num=4)
    depthmaps[np.where(depthmaps == 0)] = zfar
    current_robot_depths[np.where(current_robot_depths == 0)] = zfar

    current_robot_images = current_robot_images[:, 0, ...]
    current_robot_depths = current_robot_depths[:, 0, ..., 0]

    robot_mask = np.where((np.any(current_robot_images != 0, axis=-1)) * (current_robot_depths < depthmaps), 1, 0)
    current_images = np.where(robot_mask[:, :, :, None], current_robot_images, rgbmaps)

    excute_frames.append(current_images[args.camera_view_id])

    if not args.white_bg:
        current_images[np.where((current_images == [1, 1, 1]).all(axis=3))] = [0, 0, 0]

    encoded_images = []
    if args.subgoal:
        if args.subgoal_image:
            for i in range(len(current_images)):
                plt.imsave(f'{output_path}/{time_id}_subgoal_view_{i + 1}.png', current_images[i])
                with open(f'{output_path}/{time_id}_subgoal_view_{i + 1}.png', 'rb') as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                encoded_images.append([encoded_image])

        subgoals = None
        try_time = 0
        while subgoals is None and try_time < 5:
            try_time += 1
            try:
                if args.qwen:
                    content = qwen.generate_subgoals(encoded_image, subgoal_prompt)
                else:
                    content = generate_subgoals(encoded_images, subgoal_prompt)
                subgoals = get_subgoals(content)
            except Exception as e:
                print('catched', e)
                pass
        
        with open(f'{output_path}/{time_id}_subgoal_content.txt', 'w') as f:
            f.write(content)
        
        print('subgoals: ', subgoals)
        
        stages_text = ""
        for goal_id, goal in enumerate(subgoals):
            stages_text += f'{goal_id + 1}. {goal}\n'

        if args.stage:
            stage_prompt = stage_prompt.replace('<subgoal>', stages_text)
        
        subgoal_id = 0

    view_id = args.camera_view_id
    object_states = []
    object_transformations = []
    while len(trajectory) <= args.total_steps:
        grasp = release = False

        # render the current state
        robot_images, robot_depth_images = mesh_world.get_image_depth()
        current_robot_images, current_robot_depths = robot_images, robot_depth_images
        rgbmaps, depthmaps, alphamaps = gaussian_world.render(R_gaussian_fixed, T_gaussian_fixed, image_size, -FoV / 180.0 * np.pi, device, object_states=object_transformations, rotate_num=4)
        depthmaps[np.where(depthmaps == 0)] = zfar
        current_robot_depths[np.where(current_robot_depths == 0)] = zfar

        current_robot_images = current_robot_images[:, 0, ...]
        current_robot_depths = current_robot_depths[:, 0, ..., 0]

        robot_mask = np.where((np.any(current_robot_images != 0, axis=-1)) * (current_robot_depths < depthmaps), 1, 0)
        current_images = np.where(robot_mask[:, :, :, None], current_robot_images, rgbmaps)

        if not args.white_bg:
            current_images[np.where((current_images == [1, 1, 1]).all(axis=3))] = [0, 0, 0]

        for i in range(len(current_images)):
            plt.imsave(f'{output_path}/{time_id}_{len(trajectory)}_view_{i + 1}.png', current_images[i])

        if args.success:
            encoded_images = []
            for i in range(len(current_images)):
                with open(f'{output_path}/{time_id}_{len(trajectory)}_view_{i + 1}.png', 'rb') as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                encoded_images.append([encoded_image])
            
            try_time = 0
            change = None
            while change is None and try_time < 5:
                try_time += 1
                try:
                    if args.qwen:
                        content = qwen.generate_success(encoded_images, success_prompt)
                    else:
                        content = generate_success(encoded_images, success_prompt)
                    success = get_success(content)
                    change = True
                except Exception as e:
                    print('catched', e)
                    pass

            with open(f'{output_path}/{time_id}_{len(trajectory)}_success_content.txt', 'w') as f:
                f.write(content)
            
            if success:
                # test again to make sure
                try_time = 0
                change = None
                while change is None and try_time < 5:
                    try_time += 1
                    try:
                        if args.qwen:
                            content = qwen.generate_success(encoded_images, success_prompt)
                        else:
                            content = generate_success(encoded_images, success_prompt)
                        success = get_success(content)
                        change = True
                    except Exception as e:
                        print('catched', e)
                        pass

                if success:
                    print('!!! Success !!!')
                    break

        if args.subgoal and args.stage:
            encoded_images = []
            for i in range(len(current_images)):
                with open(f'{output_path}/{time_id}_{len(trajectory)}_view_{i + 1}.png', 'rb') as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                encoded_images.append([encoded_image])

            try_time = 0
            change = None
            while change is None and try_time < 5:
                try_time += 1
                try:
                    if args.qwen:
                        content = qwen.select_stage(encoded_images, stage_prompt, grasping=mesh_world.grasping_now)
                    else:
                        content = select_stage(encoded_images, stage_prompt, grasping=mesh_world.grasping_now)
                    stage = get_stage(content)
                    change = True
                except Exception as e:
                    print('catched', e)
                    pass
                
            with open(f'{output_path}/{time_id}_{len(trajectory)}_stage_content.txt', 'w') as f:
                f.write(content)

            subgoal_id = stage - 1

            print('current stage: ', stage)
            
        # give subgoal to system prompt
        if args.subgoal:
            system_prompt = system_prompt.replace(previous_instruction, subgoals[subgoal_id])
            # print('system_prompt: ', system_prompt)
            if args.change_view:
                select_prompt = select_prompt.replace(previous_instruction, subgoals[subgoal_id])
                change_prompt = change_prompt.replace(previous_instruction, subgoals[subgoal_id])
            if args.release or args.try_release:
                release_prompt = release_prompt.replace(previous_instruction, subgoals[subgoal_id])
            if args.try_grasp:
                grasp_prompt = grasp_prompt.replace(previous_instruction, subgoals[subgoal_id])
                # print('select_prompt: ', select_prompt)
                # print('change_prompt: ', change_prompt)
            # print('system_prompt: ', system_prompt)
            previous_instruction = subgoals[subgoal_id]

        if args.gripper_from_sim:
            will_grasp = will_release = False
            if not mesh_world.grasping_now:
                joint_angles_list, action_object_transformations, will_grasp, robot_images, robot_depth_images = mesh_world.grasp()
            else:
                joint_angles_list, action_object_transformations, robot_images, robot_depth_images = mesh_world.release()
                will_release = True
            
            if will_grasp or will_release:
                # careful! seeing from the front, todo: fix camera view to the gripper
                current_robot_images, current_robot_depths = robot_images[1:2], robot_depth_images[1:2]
                # R_gripper, T_gripper, R_gripper_gaussian, T_gripper_gaussian = fixed_to_gripper_gaussian(hand_position, radius, [1], device)
                # current_robot_images, current_robot_depths = render_meshes_with_depth(mesh_list, R_gripper, T_gripper, device, raster_settings=raster_settings, rotate_num=0)

                rgbmaps, depthmaps, alphamaps = gaussian_world.render(R_gaussian_fixed[1:2], T_gaussian_fixed[1:2], image_size, -FoV / 180.0 * np.pi, device, object_states=action_object_transformations[0], rotate_num=1)
                depthmaps[np.where(depthmaps == 0)] = zfar
                current_robot_depths[np.where(current_robot_depths == 0)] = zfar

                current_robot_images = current_robot_images[:, 0, ...]
                current_robot_depths = current_robot_depths[:, 0, ..., 0]

                robot_mask = np.where((np.any(current_robot_images != 0, axis=-1)) * (current_robot_depths < depthmaps), 1, 0)
                current_images = np.where(robot_mask[:, :, :, None], current_robot_images, rgbmaps)

                # current_images = current_images[0]
                if not args.white_bg:
                    current_images[np.where((current_images == [1, 1, 1]).all(axis=3))] = [0, 0, 0]
                encoded_images = []
                for i in range(len(current_images)):
                    plt.imsave(f'{output_path}/{time_id}_{len(trajectory)}_gripper_{i + 1}.png', current_images[i])
                    with open(f'{output_path}/{time_id}_{len(trajectory)}_gripper_{i + 1}.png', 'rb') as image_file:
                        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                    encoded_images.append([encoded_image])

                change = None
                grasp = release = False
                try_time = 0
                while change is None and try_time < 5:
                    try_time += 1
                    try:
                        if will_grasp:
                            if args.qwen:
                                content = qwen.generate_grasp(encoded_images, grasp_prompt)
                            else:
                                content = generate_grasp(encoded_images, grasp_prompt)
                            grasp = get_grasp(content)
                        else:
                            if args.qwen:
                                content = qwen.generate_release(encoded_images, release_prompt)
                            else:
                                content = generate_release(encoded_images, release_prompt)
                            release = get_release(content)
                        change = True
                    except Exception as e:
                        print('catched', e)
                        pass
                
                with open(f'{output_path}/{time_id}_{len(trajectory)}_grasp_content.txt', 'w') as f:
                    f.write(content)

                if grasp:
                    print('grasp!')
                    joint_angles_list, action_object_transformations, will_grasp, root_images, robot_depth_images = mesh_world.grasp(non_stop=True)
                if release:
                    print('release!')
                    joint_angles_list, action_object_transformations, root_images, robot_depth_images = mesh_world.release(non_stop=True)

        if args.release and mesh_world.grasping_now:
            # if args.subgoal and subgoal_id != len(subgoals) - 1:
            #     print('skip release')
            joint_angles_list, action_object_transformations, robot_images, robot_depth_images = mesh_world.release()

            # careful! seeing from the front, todo: fix camera view to the gripper
            current_robot_images, current_robot_depths = robot_images, robot_depth_images
            # R_gripper, T_gripper, R_gripper_gaussian, T_gripper_gaussian = fixed_to_gripper_gaussian(hand_position, radius, [1], device)
            # current_robot_images, current_robot_depths = render_meshes_with_depth(mesh_list, R_gripper, T_gripper, device, raster_settings=raster_settings, rotate_num=0)

            rgbmaps, depthmaps, alphamaps = gaussian_world.render(R_gaussian_fixed, T_gaussian_fixed, image_size, -FoV / 180.0 * np.pi, device, object_states=action_object_transformations[0], rotate_num=4)
            depthmaps[np.where(depthmaps == 0)] = zfar
            current_robot_depths[np.where(current_robot_depths == 0)] = zfar

            current_robot_images = current_robot_images[:, 0, ...]
            current_robot_depths = current_robot_depths[:, 0, ..., 0]

            robot_mask = np.where((np.any(current_robot_images != 0, axis=-1)) * (current_robot_depths < depthmaps), 1, 0)
            current_images = np.where(robot_mask[:, :, :, None], current_robot_images, rgbmaps)

            # current_images = current_images[0]
            if not args.white_bg:
                current_images[np.where((current_images == [1, 1, 1]).all(axis=3))] = [0, 0, 0]
                
            encoded_images = []
            for i in range(len(current_images)):
                plt.imsave(f'{output_path}/{time_id}_{len(trajectory)}_release_{i + 1}.png', current_images[i])
                with open(f'{output_path}/{time_id}_{len(trajectory)}_release_{i + 1}.png', 'rb') as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                encoded_images.append([encoded_image])

            change = None
            release = False
            try_time = 0
            while change is None and try_time < 5:
                try_time += 1
                try:
                    if args.qwen:
                        content = qwen.generate_release(encoded_images, release_prompt)
                    else:
                        content = generate_release(encoded_images, release_prompt)
                    release = get_release(content)
                    change = True
                except Exception as e:
                    print('catched', e)
                    pass
            
            with open(f'{output_path}/{time_id}_{len(trajectory)}_release_content.txt', 'w') as f:
                f.write(content)

            if release:
                print('release!')
                output_actions.append('release')
                joint_angles_list, action_object_transformations, root_images, robot_depth_images = mesh_world.release(non_stop=True)
        
        if grasp or release:
            continue

        # chage view or not
        change = False
        if args.change_view and not args.change_everytime:
            encoded_images = []
            with open(f'{output_path}/{time_id}_{len(trajectory)}_view_{view_id + 1}.png', 'rb') as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            encoded_images.append([encoded_image])

            change = None
            try_time = 0
            while change is None and try_time < 5:
                try_time += 1
                try:
                    if args.qwen:
                        content = qwen.generate_change(encoded_images, change_prompt)
                    else:
                        content = generate_change(encoded_images, change_prompt)
                    change = get_change(content)
                except Exception as e:
                    print('catched', e)
                    pass
            
            with open(f'{output_path}/{time_id}_{len(trajectory)}_change_content.txt', 'w') as f:
                f.write(content)

        # select view
        if change or args.change_everytime:
            print('change view')

            encoded_images = []
            for i in range(len(R_fixed)):
                if args.change_to_others and i == view_id:
                    continue
                # plt.imsave(f'{output_path}/{time_id}_{len(trajectory)}_view_{i + 1}.png', view_images[i])
                with open(f'{output_path}/{time_id}_{len(trajectory)}_view_{i + 1}.png', 'rb') as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                encoded_images.append([encoded_image])
            
            chosen_view_id = None
            try_time = 0
            while chosen_view_id is None and try_time < 5:
                try_time += 1
                try:
                    if args.view_example_number > 0:
                        if args.qwen:
                            content = qwen.select_view(encoded_images, select_prompt, examples=view_examples)
                        else:
                            content = simple_select_view(encoded_images, select_prompt, examples=view_examples)
                    else:
                        if args.qwen:
                            content = qwen.select_view(encoded_images, select_prompt)
                        else:
                            content = simple_select_view(encoded_images, select_prompt)
                    chosen_view_id = get_view(content)
                except Exception as e:
                    print('catched', e)
                    pass
            
            if not args.stage and chosen_view_id == -1:
                if args.subgoal:
                    print('subgoal achieved')
                    subgoal_id += 1
                    if subgoal_id >= len(subgoals):
                        print('goal achieved')
                        break
                else:
                    print('goal achieved')
                    break
                continue

            # save content
            with open(f'{output_path}/{time_id}_{len(trajectory)}_view_content.txt', 'w') as f:
                f.write(content)
            
            chosen_view_id = chosen_view_id - 1
            if args.change_to_others and chosen_view_id >= view_id:
                chosen_view_id = chosen_view_id + 1

            view_id = chosen_view_id

            print('best view_id: ', view_id)

            args.camera_view_id = view_id

        R_gaussian = R_gaussian_fixed[view_id: view_id + 1]
        T_gaussian = T_gaussian_fixed[view_id: view_id + 1]

        action_dimenstions = 6
        means = np.zeros(args.horizon * action_dimenstions)
        variances = np.ones(args.horizon * action_dimenstions) * 2
        # careful! rotation variance
        # rotation_variance = 1.5
        # rotation_variance = 1.0
        # only move parallel to the image plane
        if args.plane_action:
            variances[np.arange(args.horizon * action_dimenstions) % action_dimenstions == plane_action_none_dimensions[view_id]] = 0

        print('variances: ', variances)

        covariance = np.zeros((len(means), len(means)))
        np.fill_diagonal(covariance, variances)

        for iteration in range(args.cem_iteration):
            prev_time = time.time()
            # Render future results
            prompt = []

            samples = np.random.multivariate_normal(means, covariance, size=args.num_sample_actions)
            grasp = False
            release = False

            if args.uniform:
                if iteration == 0:
                    action_scale = 1.5
                    for sample_id in range(args.num_sample_actions):
                        samples[sample_id] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                        samples[sample_id, sample_id % 6] = action_scale
                        action_scale = -action_scale
                # print('samples: ', samples)

            if args.try_grasp and not mesh_world.grasping_now:
                joint_angles_list, action_object_transformations, post_samples, robot_images, robot_depth_images, grasp_object_transformations, grasp_robot_images, grasp_robot_depth_images, is_grasping = mesh_world.sample_trajectory_distribution_batch(samples.reshape(args.num_sample_actions, args.horizon, action_dimenstions), try_grasp=True)
                if is_grasping.sum():
                    print(f'{is_grasping.sum()} could grasp!')
                    grasp_object_transformations = grasp_object_transformations[is_grasping]
                    grasp_robot_images = grasp_robot_images[:, is_grasping]
                    grasp_robot_depth_images = grasp_robot_depth_images[:, is_grasping]
                    rgbmaps, depthmaps, alphamaps = [], [], []
                    for grasp_id in range(len(grasp_object_transformations)):
                        rgbmap, depthmap, alphamap = gaussian_world.render(R_gaussian_fixed, T_gaussian_fixed, image_size, -FoV / 180.0 * np.pi, device, object_states=grasp_object_transformations[grasp_id], rotate_num=4)
                        depthmap[np.where(depthmap == 0)] = zfar

                        rgbmaps.append(rgbmap[:])
                        depthmaps.append(depthmap[:])
                        alphamaps.append(alphamap[:])

                    rgbmaps = np.stack(rgbmaps)
                    depthmaps = np.stack(depthmaps)
                    alphamaps = np.stack(alphamaps)

                    rgbmaps = rgbmaps.transpose(1, 0, 2, 3, 4)
                    depthmaps = depthmaps.transpose(1, 0, 2, 3)
                    alphamaps = alphamaps.transpose(1, 0, 2, 3)

                    robot_mask = np.where((np.any(grasp_robot_images != 0, axis=-1)) * (grasp_robot_depth_images[..., 0] < depthmaps), 1, 0)
                    images = np.where(robot_mask[:, :, :, :, None], grasp_robot_images, rgbmaps)

                    if not args.white_bg:
                        images[np.where((images == [1, 1, 1]).all(axis=4))] = [0, 0, 0]

                    for grasp_id in range(len(grasp_object_transformations)):
                        img_views = images[:, grasp_id]
                        # print('img_views: ', img_views.shape)
                        encoded_images = []
                        for idx, img in enumerate(img_views):
                            plt.imsave(f'{output_path}/{time_id}_{len(trajectory)}_{iteration}_grasp_{grasp_id + 1}_view_{idx + 1}.png', img)
                            with open(f'{output_path}/{time_id}_{len(trajectory)}_{iteration}_grasp_{grasp_id + 1}_view_{idx + 1}.png', 'rb') as image_file:
                                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                            encoded_images.append(encoded_image)

                        change = None
                        try_time = 0
                        while change is None and try_time < 5:
                            try_time += 1
                            try:
                                if args.qwen:
                                    content = qwen.generate_grasp(encoded_images, grasp_prompt)
                                else:
                                    content = generate_grasp(encoded_images, grasp_prompt)
                                grasp = get_grasp(content)
                                change = True
                            except Exception as e:
                                print('catched', e)
                                pass
                        
                        with open(f'{output_path}/{time_id}_{len(trajectory)}_{iteration}_grasp_{grasp_id + 1}_content.txt', 'w') as f:
                            f.write(content)

                        if grasp:
                            means = post_samples[is_grasping][grasp_id]
                            break
                    if grasp:
                        break

            elif args.try_release and mesh_world.grasping_now and subgoal_id == len(subgoals) - 1:
                joint_angles_list, action_object_transformations, post_samples, robot_images, robot_depth_images, release_object_transformations, release_robot_images, release_robot_depth_images = mesh_world.sample_trajectory_distribution_batch(means.reshape(args.horizon, action_dimenstions)[None], try_release=True)

                rgbmaps, depthmaps, alphamaps = [], [], []
                for release_id in range(len(release_object_transformations)):
                    rgbmap, depthmap, alphamap = gaussian_world.render(R_gaussian_fixed, T_gaussian_fixed, image_size, -FoV / 180.0 * np.pi, device, object_states=release_object_transformations[release_id], rotate_num=4)
                    depthmap[np.where(depthmap == 0)] = zfar

                    rgbmaps.append(rgbmap[:])
                    depthmaps.append(depthmap[:])
                    alphamaps.append(alphamap[:])

                rgbmaps = np.stack(rgbmaps)
                depthmaps = np.stack(depthmaps)
                alphamaps = np.stack(alphamaps)

                rgbmaps = rgbmaps.transpose(1, 0, 2, 3, 4)
                depthmaps = depthmaps.transpose(1, 0, 2, 3)
                alphamaps = alphamaps.transpose(1, 0, 2, 3)

                robot_mask = np.where((np.any(release_robot_images != 0, axis=-1)) * (release_robot_depth_images[..., 0] < depthmaps), 1, 0)
                images = np.where(robot_mask[:, :, :, :, None], release_robot_images, rgbmaps)

                if not args.white_bg:
                    images[np.where((images == [1, 1, 1]).all(axis=4))] = [0, 0, 0]
                
                processes = []
                queue = multiprocessing.Queue()
                best_of_each_group = []
                for release_id in range(len(release_object_transformations)):
                    img_views = images[:, release_id]
                    # print('img_views: ', img_views.shape)
                    encoded_images = []
                    for idx, img in enumerate(img_views):
                        plt.imsave(f'{output_path}/{time_id}_{len(trajectory)}_{iteration}_release_{release_id + 1}_view_{idx + 1}.png', img)
                        with open(f'{output_path}/{time_id}_{len(trajectory)}_{iteration}_release_{release_id + 1}_view_{idx + 1}.png', 'rb') as image_file:
                            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                        encoded_images.append([encoded_image])
                    # if args.current:

                    p = multiprocessing.Process(target=prompt_release_helper, args=(release_id, queue, encoded_images, release_prompt, args))
                    processes.append(p)
                    p.start()

                for p in processes:
                    p.join()

                for p in processes:
                    release_id, release, content = queue.get()

                    with open(f'{output_path}/{time_id}_{len(trajectory)}_{iteration}_{release_id + 1}_release.txt', 'w') as f:
                        f.write(content)
                    if release:
                        means = post_samples[release_id]
                        print('release!')
                        break
                if release:
                    break

            else:
                joint_angles_list, action_object_transformations, post_samples, robot_images, robot_depth_images = mesh_world.sample_trajectory_distribution_batch(samples.reshape(args.num_sample_actions, args.horizon, action_dimenstions))
            
            robot_images = robot_images[view_id]
            robot_depth_images = robot_depth_images[view_id]

            if args.num_sample_actions > args.num_sample_vlm:
                joint_angles_list = joint_angles_list[: args.num_sample_vlm]
                action_object_transformations = action_object_transformations[: args.num_sample_vlm]
                post_samples = post_samples[: args.num_sample_vlm]
                robot_images = robot_images[: args.num_sample_vlm]
                robot_depth_images = robot_depth_images[: args.num_sample_vlm]
            # print('post_samples: ', len(post_samples))

            print('step: ', len(trajectory), 'iteration: ', iteration, 'simulate time: ', time.time() - prev_time)
            prev_time = time.time()

            rgbmaps, depthmaps, alphamaps = [], [], []
            for act_id, joint_angles in enumerate(joint_angles_list):
                rgbmap, depthmap, alphamap = gaussian_world.render(R_gaussian, T_gaussian, image_size, -FoV / 180.0 * np.pi, device, object_states=action_object_transformations[act_id], rotate_num=1)
                depthmap[np.where(depthmap == 0)] = zfar

                rgbmaps.append(rgbmap[0])
                depthmaps.append(depthmap[0])
                alphamaps.append(alphamap[0])

            rgbmaps = np.stack(rgbmaps)
            depthmaps = np.stack(depthmaps)
            alphamaps = np.stack(alphamaps)

            robot_mask = np.where((np.any(robot_images != 0, axis=-1)) * (robot_depth_images[..., 0] < depthmaps), 1, 0)
            if args.no_gripper_when_grasping and mesh_world.grasping_now:
                images = np.where(robot_mask[:, :, :, None], rgbmaps, rgbmaps)
            else:
                images = np.where(robot_mask[:, :, :, None], robot_images, rgbmaps)

            # turn white background to black
            if not args.white_bg:
                images[np.where((images == [1, 1, 1]).all(axis=3))] = [0, 0, 0]

            vis_images = []
            for act_id, img in enumerate(images):
                vis_images.append(img.copy())
                plt.imsave(f'{output_path}/{time_id}_{len(trajectory)}_{act_id + 1}.png', img)
                
                with open(f'{output_path}/{time_id}_{len(trajectory)}_{act_id + 1}.png', 'rb') as image_file:
                    prompt.append([base64.b64encode(image_file.read()).decode('utf-8')])
            
            print('iteration: ', iteration, 'render time: ', time.time() - prev_time)

            group_num = len(joint_angles_list) // args.num_sample_each_group

            vis_all_images = []
            for group_id in range(group_num):
                vis_group_images = vis_images[group_id * args.num_sample_each_group: (group_id + 1) * args.num_sample_each_group]
                vis_group_images = np.concatenate(vis_group_images, axis=1)
                vis_all_images.append(vis_group_images)
            vis_all_images = np.concatenate(vis_all_images, axis=0)

            plt.imsave(f'{output_path}/{time_id}_{len(trajectory)}_{iteration}_all.png', vis_all_images)

            prev_time = time.time()

            processes = []
            queue = multiprocessing.Queue()
            best_of_each_group = []
            if args.qwen:
                for group_id in range(group_num):
                    group_prompt = prompt[group_id * args.num_sample_each_group: (group_id + 1) * args.num_sample_each_group]
                    group_id, answer, content = qwen_prompt_helper(group_id, qwen, group_prompt, system_prompt, args, history, examples)
                    best_id = answer - 1
                    best_of_each_group.append(group_id * args.num_sample_each_group + best_id)

                    with open(f'{output_path}/{time_id}_{len(trajectory)}_{iteration}_{group_id}_response.txt', 'w') as f:
                        f.write(content)
            else:
                for group_id in range(group_num):
                    group_prompt = prompt[group_id * args.num_sample_each_group: (group_id + 1) * args.num_sample_each_group]
                    # if args.current:

                    if args.qwen:
                        p = multiprocessing.Process(target=qwen_prompt_helper, args=(group_id, qwen, queue, group_prompt, system_prompt, args, history, examples))
                    else:
                        p = multiprocessing.Process(target=prompt_helper, args=(group_id, queue, group_prompt, system_prompt, args, history, examples, mesh_world.grasping_now))
                    processes.append(p)
                    p.start()

                for p in processes:
                    p.join()

                for p in processes:
                    group_id, answer, content = queue.get()
                    best_id = answer - 1
                    best_of_each_group.append(group_id * args.num_sample_each_group + best_id)

                    with open(f'{output_path}/{time_id}_{len(trajectory)}_{iteration}_{group_id}_response.txt', 'w') as f:
                        f.write(content)
            
            elite_samples = post_samples[best_of_each_group].reshape(-1, args.horizon * action_dimenstions)
            means = np.mean(elite_samples, axis=0)
            covariance = np.cov(elite_samples, rowvar=False)
            covariance = np.diag(np.diag(covariance))

            print('iteration: ', iteration, 'prompt time: ', time.time() - prev_time)
            # print('iteration: ', iteration, 'means: ', means)
            start_time = time.time()

            # print(f'chosen_action: {possible_direction_name_list[chosen_id]}')
            # object_states = action_object_states[chosen_id]

        print('mean: ', means)
        joint_angles_list, action_object_transformations, robot_images, robot_depth_images = mesh_world.sample_trajectory_distribution_batch(means.reshape(args.horizon, -1)[None], non_stop=True)

        # save actions
        output_actions.append(joint_angles_list[0].cpu().numpy().tolist())

        if args.try_grasp and grasp:
            print('grasp!')
            if args.just_grasp_it:
                grasp_joint_angles_list, grasp_action_object_transformations, grasp_robot_images, grasp_robot_depth_images = mesh_world.set_grasp_state(grasp_id)
            else:
                grasp_joint_angles_list, grasp_action_object_transformations, will_grasp, grasp_robot_images, grasp_robot_depth_images = mesh_world.grasp(non_stop=True)

            if args.just_grasp_it or will_grasp:
                joint_angles_list, action_object_transformations, robot_images, robot_depth_images = grasp_joint_angles_list, grasp_action_object_transformations, grasp_robot_images, grasp_robot_depth_images
                output_actions.append('grasp')
                output_actions.append(joint_angles_list[0].cpu().numpy().tolist())
        
        if args.try_release and release:
            print('release!')
            release_joint_angles_list, release_action_object_transformations, release_robot_images, release_robot_depth_images = mesh_world.release(non_stop=True)
            joint_angles_list, action_object_transformations, robot_images, robot_depth_images = release_joint_angles_list, release_action_object_transformations, release_robot_images, release_robot_depth_images
            output_actions.append('release')
            output_actions.append(joint_angles_list[0].cpu().numpy().tolist())

        if args.record_video:
            mesh_world.env.flush_video()

        # object_states = action_object_states[0]
        object_transformations = action_object_transformations[0]

        # mesh_list, wrist_eyes, wrist_up, wrist_ats, hand_position, robot_images, robot_depth_images = robot.joint_angle_to_meshes(joint_angles_list[0], offset=robot_translation, only_hand=args.only_hand)

        # robot_images, robot_depths = render_meshes_with_depth(mesh_list, R, T, device, raster_settings=raster_settings, rotate_num=1)
        # robot_images = robot_images[view_id]
        # robot_depths = robot_depth_images[view_id]
        rgbmaps, depthmaps, alphamaps = gaussian_world.render(R_gaussian, T_gaussian, image_size, -FoV / 180.0 * np.pi, device, object_states=object_transformations, rotate_num=1)
        depthmaps[np.where(depthmaps == 0)] = zfar
        robot_depth_images[np.where(robot_depth_images == 0)] = zfar

        robot_images = robot_images[view_id, 0, ...]
        robot_depth_images = robot_depth_images[view_id, 0, ..., 0]

        robot_mask = np.where((np.any(robot_images != 0, axis=-1)) * (robot_depth_images < depthmaps), 1, 0)
        images = np.where(robot_mask[:, :, :, None], robot_images, rgbmaps)
        
        # if args.overlay:

        images = images[:, :, :, :3]

        # turn white background to black
        if not args.white_bg:
            images[np.where((images == [1, 1, 1]).all(axis=3))] = [0, 0, 0]

        # save trajectory image
        plt.imsave(f'{output_path}/{time_id}_{len(trajectory)}.png', images[0])
        with open(f'{output_path}/{time_id}_{len(trajectory)}.png', 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        excute_frames.append(images[0])
        history.append(encoded_image)
        if len(history) > args.max_history:
            history.pop(0)
        trajectory.append(joint_angles_list[0])

        if args.replan and replan_time < max_replan:
            if mesh_world.object_drop:
                print('object drop replan!')
                trajectory = trajectory[:-1]
                history = history[:-1]
                excute_frames = excute_frames[:-1]
                output_actions = output_actions[:-1]
                mesh_world.history_states = mesh_world.history_states[:-1]
                mesh_world.object_drop = False
                mesh_world.grasping_now = True
                mesh_world.grasping_pos = mesh_world.prev_grasping_pos
                replan_time += 1

            for obj_id in range(len(mesh_world.env.unwrapped.objects)):
                object_translation_distance = torch.norm(object_transformations[obj_id, :3]).cpu()
                print('object_translation_distance: ', object_translation_distance)
                if object_translation_distance > 0.03 and not 'grasp' in output_actions:
                    print('object move replan!')
                    trajectory = trajectory[:-1]
                    history = history[:-1]
                    excute_frames = excute_frames[:-1]
                    output_actions = output_actions[:-1]
                    mesh_world.history_states = mesh_world.history_states[:-1]
                    replan_time += 1
        
        # print('best trajectory: ', trajectory[-1])
        # history_object_states.append(action_object_states[0])

    # concatenate excute_frames to one image and save it
    blank_line = np.ones((image_size, image_size // 20, 3))
    for i in range(len(excute_frames)):
        if i == 0:
            image = excute_frames[i]
        else:
            image = np.concatenate([image, blank_line, excute_frames[i]], axis=1)
    # image = np.concatenate(excute_frames, axis=1)
    plt.imsave(f'{output_path}/{time_id}_trajectory.png', image)

    # save actions
    with open(f'{output_path}/{time_id}_actions.txt', 'w') as f:
        for action in output_actions:
            f.write(f'{action}\n')

    mesh_world.reset()

