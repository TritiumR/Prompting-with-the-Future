import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import base64
import time
import argparse
import multiprocessing

from meshes.mesh_world import MeshWorld
from utils.prompt import *
from utils.camera import get_up_direction

from utils.prompt_gpt import *
from gaussians.gaussian_world import GaussianWorld

from pytorch3d.renderer import look_at_view_transform

def robo4d_parse():
    parser = argparse.ArgumentParser(description="Robo4D")
    parser.add_argument("--scene_name", type=str, default="basket_world")
    parser.add_argument("--instruction", type=str, default="put the green cucumber into the basket")
    parser.add_argument("--name", type=str, default='demo')
    parser.add_argument("--image_size", type=int, default=500)
    parser.add_argument("--total_steps", type=int, default=10)
    parser.add_argument("--camera_view_id", type=int, default=1)
    parser.add_argument("--plane_action", action="store_true")
    parser.add_argument("--cem_iteration", type=int, default=3)
    parser.add_argument("--num_sample_each_group", type=int, default=6)
    parser.add_argument("--num_sample_actions", type=int, default=81)
    parser.add_argument("--num_sample_vlm", type=int, default=36)
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--try_release", action="store_true")
    parser.add_argument("--replan", action="store_true")
    return parser

parser = robo4d_parse()
args = parser.parse_args()

# render settings
image_size = args.image_size
znear = 0.01
zfar = 100
FoV = 60

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
subgoal_prompt = get_subgoal_prompt(args)
select_prompt = get_view_prompt(args)
grasp_prompt = get_grasp_prompt(args)
release_prompt = get_release_prompt(args)
stage_prompt = get_stage_prompt(args)
success_prompt = get_success_prompt(args)

previous_instruction = args.instruction

distance = 1.5
scene_name = args.scene_name
robot_translation = [-0.45, 0.0, 0.0]
action_ids = [2]

max_replan = 5

gaussian_world = GaussianWorld(scene_name, parser, post_process=True)
center = np.array([0, 0, 0])
radius = gaussian_world.radius * distance

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

robot_uids = 'PandaRobotiqHand'

# camera config
elev = torch.tensor([-70, 0, 70, 0], device=device)
azim = torch.tensor([0, 70, 0, 0], device=device)
up = get_up_direction(elev, azim)
at = torch.tensor(center[None], device=device).float()

cameras_config = []
for i in range(4):
    cameras_config.append({
        'elev': elev[i].item(),
        'azim': azim[i].item(),
    })

# gaussian camera config
R_fixed, T_fixed = look_at_view_transform(dist=radius, elev=elev, azim=azim, up=up, at=at, device=device)
up[:, 0] = -up[:, 0]
at[:, 0] = -at[:, 0]
elev = 180 - elev
R_gaussian_fixed, T_gaussian_fixed = look_at_view_transform(dist=radius, elev=elev, azim=azim, up=up, at=at, device="cpu")
R_gaussian_fixed = R_gaussian_fixed.numpy()
T_gaussian_fixed = T_gaussian_fixed.numpy()

plane_action_dimensions = [[0, 2], [1, 2], [0, 2], [0, 1]]

# build mesh world
mesh_world = MeshWorld(scene_name, num_envs=args.num_sample_actions, scene_traslation=-np.array(robot_translation), radius=radius, \
                       image_size=image_size, record_video=args.record_video, robot_uids=robot_uids, need_render=True, dir=output_path, \
                        close_gripper=close_gripper, cameras_config=cameras_config)

success = False
trajectory = []
history = []
excute_frames = []
history_object_states = []
output_actions = []
replan_time = 0

if close_gripper:
    output_actions.append('grasp')

initial_joint_angles = mesh_world.agent.robot.get_qpos()[0].cpu().numpy().tolist()
print('initial_joint_angles: ', initial_joint_angles)
trajectory.append(torch.tensor(initial_joint_angles))

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

encoded_images = []
for i in range(len(current_images)):
    plt.imsave(f'{output_path}/subgoal_view_{i + 1}.png', current_images[i])
    with open(f'{output_path}/subgoal_view_{i + 1}.png', 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    encoded_images.append([encoded_image])

subgoals = None
try_time = 0
while subgoals is None and try_time < 5:
    try_time += 1
    try:
        content = generate_subgoals(encoded_images, subgoal_prompt)
        subgoals = get_subgoals(content)
    except Exception as e:
        print('catched', e)
        pass

with open(f'{output_path}/subgoal_content.txt', 'w') as f:
    f.write(content)

print('subgoals: ', subgoals)

stages_text = ""
for goal_id, goal in enumerate(subgoals):
    stages_text += f'{goal_id + 1}. {goal}\n'

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

    for i in range(len(current_images)):
        plt.imsave(f'{output_path}/{len(trajectory)}_view_{i + 1}.png', current_images[i])

    encoded_images = []
    for i in range(len(current_images)):
        with open(f'{output_path}/{len(trajectory)}_view_{i + 1}.png', 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        encoded_images.append([encoded_image])
    
    # check success
    try_time = 0
    change = None
    while change is None and try_time < 5:
        try_time += 1
        try:
            content = generate_success(encoded_images, success_prompt)
            success = get_success(content)
            change = True
        except Exception as e:
            print('catched', e)
            pass

    with open(f'{output_path}/{len(trajectory)}_success_content.txt', 'w') as f:
        f.write(content)
    
    if success:
        # test again to make sure
        try_time = 0
        change = None
        while change is None and try_time < 5:
            try_time += 1
            try:
                content = generate_success(encoded_images, success_prompt)
                success = get_success(content)
                change = True
            except Exception as e:
                print('catched', e)
                pass

        if success:
            print('!!! Success !!!')
            break

    encoded_images = []
    for i in range(len(current_images)):
        with open(f'{output_path}/{len(trajectory)}_view_{i + 1}.png', 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        encoded_images.append([encoded_image])

    try_time = 0
    change = None
    while change is None and try_time < 5:
        try_time += 1
        try:
            content = select_stage(encoded_images, stage_prompt, grasping=mesh_world.grasping_now)
            stage = get_stage(content)
            change = True
        except Exception as e:
            print('catched', e)
            pass
        
    with open(f'{output_path}/{len(trajectory)}_stage_content.txt', 'w') as f:
        f.write(content)

    subgoal_id = stage - 1

    print('current stage: ', stage)
        
    # give subgoal to system prompt
    system_prompt = system_prompt.replace(previous_instruction, subgoals[subgoal_id])
    select_prompt = select_prompt.replace(previous_instruction, subgoals[subgoal_id])
    release_prompt = release_prompt.replace(previous_instruction, subgoals[subgoal_id])
    grasp_prompt = grasp_prompt.replace(previous_instruction, subgoals[subgoal_id])
    previous_instruction = subgoals[subgoal_id]

    # sample release action
    if mesh_world.grasping_now:
        joint_angles_list, action_object_transformations, robot_images, robot_depth_images = mesh_world.release()

        current_robot_images, current_robot_depths = robot_images, robot_depth_images

        rgbmaps, depthmaps, alphamaps = gaussian_world.render(R_gaussian_fixed, T_gaussian_fixed, image_size, -FoV / 180.0 * np.pi, device, object_states=action_object_transformations[0], rotate_num=4)
        depthmaps[np.where(depthmaps == 0)] = zfar
        current_robot_depths[np.where(current_robot_depths == 0)] = zfar

        current_robot_images = current_robot_images[:, 0, ...]
        current_robot_depths = current_robot_depths[:, 0, ..., 0]

        robot_mask = np.where((np.any(current_robot_images != 0, axis=-1)) * (current_robot_depths < depthmaps), 1, 0)
        current_images = np.where(robot_mask[:, :, :, None], current_robot_images, rgbmaps)

        encoded_images = []
        for i in range(len(current_images)):
            plt.imsave(f'{output_path}/{len(trajectory)}_release_{i + 1}.png', current_images[i])
            with open(f'{output_path}/{len(trajectory)}_release_{i + 1}.png', 'rb') as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            encoded_images.append([encoded_image])

        change = None
        release = False
        try_time = 0
        while change is None and try_time < 5:
            try_time += 1
            try:
                content = generate_release(encoded_images, release_prompt)
                release = get_release(content)
                change = True
            except Exception as e:
                print('catched', e)
                pass
        
        with open(f'{output_path}/{len(trajectory)}_release_content.txt', 'w') as f:
            f.write(content)

        if release:
            print('release!')
            output_actions.append('release')
            joint_angles_list, action_object_transformations, root_images, robot_depth_images = mesh_world.release(non_stop=True)
    
    if release:
        continue

    # select view
    encoded_images = []
    for i in range(len(R_fixed)):
        if i == view_id:
            continue
        with open(f'{output_path}/{len(trajectory)}_view_{i + 1}.png', 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        encoded_images.append([encoded_image])
    
    chosen_view_id = None
    try_time = 0
    while chosen_view_id is None and try_time < 5:
        try_time += 1
        try:
            content = simple_select_view(encoded_images, select_prompt)
            chosen_view_id = get_view(content)
        except Exception as e:
            print('catched', e)
            pass

    # save content
    with open(f'{output_path}/{len(trajectory)}_view_content.txt', 'w') as f:
        f.write(content)
    
    chosen_view_id = chosen_view_id - 1
    if chosen_view_id >= view_id:
        chosen_view_id = chosen_view_id + 1

    view_id = chosen_view_id

    print('selected view_id: ', view_id)

    args.camera_view_id = view_id

    R_gaussian = R_gaussian_fixed[view_id: view_id + 1]
    T_gaussian = T_gaussian_fixed[view_id: view_id + 1]

    action_dimenstions = 6
    means = np.zeros(action_dimenstions)
    variances = np.ones(action_dimenstions) * 0.5
    translation_variance = 1.5
    # careful! rotation variance
    # rotation_variance = 1.5
    rotation_variance = 1.0
    # only move parallel to the image plane
    if args.plane_action:
        variances[plane_action_dimensions[view_id]] = translation_variance
    else:
        variances[: 3] = translation_variance

    variances[3:] = rotation_variance

    covariance = np.zeros((action_dimenstions, action_dimenstions))
    np.fill_diagonal(covariance, variances)

    for iteration in range(args.cem_iteration):
        prev_time = time.time()
        prompt = []

        samples = np.random.multivariate_normal(means, covariance, size=args.num_sample_actions)
        grasp = False
        release = False

        if not mesh_world.grasping_now:
            joint_angles_list, action_object_transformations, post_samples, robot_images, robot_depth_images, grasp_object_transformations, grasp_robot_images, grasp_robot_depth_images, is_grasping = mesh_world.sample_action_distribution_batch(samples, try_grasp=True)
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

                for grasp_id in range(len(grasp_object_transformations)):
                    img_views = images[:, grasp_id]
                    # print('img_views: ', img_views.shape)
                    encoded_images = []
                    for idx, img in enumerate(img_views):
                        plt.imsave(f'{output_path}/{len(trajectory)}_{iteration}_grasp_{grasp_id + 1}_view_{idx + 1}.png', img)
                        with open(f'{output_path}/{len(trajectory)}_{iteration}_grasp_{grasp_id + 1}_view_{idx + 1}.png', 'rb') as image_file:
                            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                        encoded_images.append(encoded_image)

                    change = None
                    try_time = 0
                    while change is None and try_time < 5:
                        try_time += 1
                        try:
                            content = generate_grasp(encoded_images, grasp_prompt)
                            grasp = get_grasp(content)
                            change = True
                        except Exception as e:
                            print('catched', e)
                            pass
                    
                    with open(f'{output_path}/{len(trajectory)}_{iteration}_grasp_{grasp_id + 1}_content.txt', 'w') as f:
                        f.write(content)

                    if grasp:
                        means = post_samples[is_grasping][grasp_id]
                        break
                if grasp:
                    break

        elif args.try_release and mesh_world.grasping_now and subgoal_id == len(subgoals) - 1:
            joint_angles_list, action_object_transformations, post_samples, robot_images, robot_depth_images, release_object_transformations, release_robot_images, release_robot_depth_images = mesh_world.sample_action_distribution_batch(samples, try_release=True)

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

            processes = []
            queue = multiprocessing.Queue()
            best_of_each_group = []
            for release_id in range(len(release_object_transformations)):
                img_views = images[:, release_id]
                encoded_images = []
                for idx, img in enumerate(img_views):
                    plt.imsave(f'{output_path}/{len(trajectory)}_{iteration}_release_{release_id + 1}_view_{idx + 1}.png', img)
                    with open(f'{output_path}/{len(trajectory)}_{iteration}_release_{release_id + 1}_view_{idx + 1}.png', 'rb') as image_file:
                        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                    encoded_images.append([encoded_image])

                p = multiprocessing.Process(target=prompt_release_helper, args=(release_id, queue, encoded_images, release_prompt, args))
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            for p in processes:
                release_id, release, content = queue.get()

                with open(f'{output_path}/{len(trajectory)}_{iteration}_{release_id + 1}_release.txt', 'w') as f:
                    f.write(content)
                if release:
                    means = post_samples[release_id]
                    print('release!')
                    break
            if release:
                break

        else:
            joint_angles_list, action_object_transformations, post_samples, robot_images, robot_depth_images = mesh_world.sample_action_distribution_batch(samples)
        
        robot_images = robot_images[view_id]
        robot_depth_images = robot_depth_images[view_id]

        if args.num_sample_actions > args.num_sample_vlm:
            joint_angles_list = joint_angles_list[: args.num_sample_vlm]
            action_object_transformations = action_object_transformations[: args.num_sample_vlm]
            post_samples = post_samples[: args.num_sample_vlm]
            robot_images = robot_images[: args.num_sample_vlm]
            robot_depth_images = robot_depth_images[: args.num_sample_vlm]

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
        images = np.where(robot_mask[:, :, :, None], robot_images, rgbmaps)

        vis_images = []
        for act_id, img in enumerate(images):
            vis_images.append(img.copy())
            plt.imsave(f'{output_path}/{len(trajectory)}_{act_id + 1}.png', img)
            
            with open(f'{output_path}/{len(trajectory)}_{act_id + 1}.png', 'rb') as image_file:
                prompt.append([base64.b64encode(image_file.read()).decode('utf-8')])
        
        print('iteration: ', iteration, 'render time: ', time.time() - prev_time)

        group_num = len(joint_angles_list) // args.num_sample_each_group

        vis_all_images = []
        for group_id in range(group_num):
            vis_group_images = vis_images[group_id * args.num_sample_each_group: (group_id + 1) * args.num_sample_each_group]
            vis_group_images = np.concatenate(vis_group_images, axis=1)
            vis_all_images.append(vis_group_images)
        vis_all_images = np.concatenate(vis_all_images, axis=0)

        plt.imsave(f'{output_path}/{len(trajectory)}_{iteration}_all.png', vis_all_images)

        prev_time = time.time()

        processes = []
        queue = multiprocessing.Queue()
        best_of_each_group = []
        
        for group_id in range(group_num):
            group_prompt = prompt[group_id * args.num_sample_each_group: (group_id + 1) * args.num_sample_each_group]

            p = multiprocessing.Process(target=prompt_helper, args=(group_id, queue, group_prompt, system_prompt, mesh_world.grasping_now))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        for p in processes:
            group_id, answer, content = queue.get()
            best_id = answer - 1
            best_of_each_group.append(group_id * args.num_sample_each_group + best_id)

            with open(f'{output_path}/{len(trajectory)}_{iteration}_{group_id}_response.txt', 'w') as f:
                f.write(content)
        
        elite_samples = post_samples[best_of_each_group]
        means = np.mean(elite_samples, axis=0)
        covariance = np.cov(elite_samples, rowvar=False)

        print('iteration: ', iteration, 'prompt time: ', time.time() - prev_time)
        start_time = time.time()

    joint_angles_list, action_object_transformations, robot_images, robot_depth_images = mesh_world.sample_action_distribution_batch(means[None], non_stop=True)

    # save actions
    output_actions.append(joint_angles_list[0].cpu().numpy().tolist())

    if grasp:
        print('grasp!')
        grasp_joint_angles_list, grasp_action_object_transformations, grasp_robot_images, grasp_robot_depth_images = mesh_world.set_grasp_state(grasp_id)
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

    object_transformations = action_object_transformations[0]

    rgbmaps, depthmaps, alphamaps = gaussian_world.render(R_gaussian, T_gaussian, image_size, -FoV / 180.0 * np.pi, device, object_states=object_transformations, rotate_num=1)
    depthmaps[np.where(depthmaps == 0)] = zfar
    robot_depth_images[np.where(robot_depth_images == 0)] = zfar

    robot_images = robot_images[view_id, 0, ...]
    robot_depth_images = robot_depth_images[view_id, 0, ..., 0]

    robot_mask = np.where((np.any(robot_images != 0, axis=-1)) * (robot_depth_images < depthmaps), 1, 0)
    images = np.where(robot_mask[:, :, :, None], robot_images, rgbmaps)
    
    images = images[:, :, :, :3]

    # save trajectory image
    plt.imsave(f'{output_path}/{len(trajectory)}.png', images[0])
    with open(f'{output_path}/{len(trajectory)}.png', 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    excute_frames.append(images[0])
    trajectory.append(joint_angles_list[0])

    if args.replan and replan_time < max_replan:
        if mesh_world.object_drop:
            print('object drop replan!')
            trajectory = trajectory[:-1]
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
                excute_frames = excute_frames[:-1]
                output_actions = output_actions[:-1]
                mesh_world.history_states = mesh_world.history_states[:-1]
                replan_time += 1
    
# concatenate excute_frames to one image and save it
blank_line = np.ones((image_size, image_size // 20, 3))
for i in range(len(excute_frames)):
    if i == 0:
        image = excute_frames[i]
    else:
        image = np.concatenate([image, blank_line, excute_frames[i]], axis=1)
plt.imsave(f'{output_path}/trajectory.png', image)

# save actions
with open(f'{output_path}/actions.txt', 'w') as f:
    for action in output_actions:
        f.write(f'{action}\n')

mesh_world.reset()

