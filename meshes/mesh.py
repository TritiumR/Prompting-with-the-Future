"""
Code for a minimal environment/task with just a robot being loaded. We recommend copying this template and modifying as you need.

At a high-level, ManiSkill tasks can minimally be defined by how the environment resets, what agents/objects are
loaded, goal parameterization, and success conditions

Environment reset is comprised of running two functions, `self._reconfigure` and `self.initialize_episode`, which is auto
run by ManiSkill. As a user, you can override a number of functions that affect reconfiguration and episode initialization.

Reconfiguration will reset the entire environment scene and allow you to load/swap assets and agents.

Episode initialization will reset the positions of all objects (called actors), articulations, and agents,
in addition to initializing any task relevant data like a goal

See comments for how to make your own environment and what each required function should do
"""

from typing import Any, Dict, Union

import os
import numpy as np
import torch
import torch.random
import open3d as o3d

from mani_skill.agents.robots import Fetch, Panda
from utils.robot.panda_robotiq.PandaRobotiqHand import PandaRobotiqHand
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
# from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig


# careful, what's the max_episode_steps?
@register_env("Mesh", max_episode_steps=1000)
class MeshEnv(BaseEnv):
    """
    Task Description
    ----------------
    A simple task where the there is a laptop and a panda robot.

    Randomizations
    --------------
    No randomizations are applied in this task.

    Success Conditions
    ------------------
    -No success conditions are defined in this task.

    Visualization: https://maniskill.readthedocs.io/en/latest/tasks/index.html#pushcube-v1
    """

    SUPPORTED_ROBOTS = ["panda", "fetch", "PandaRobotiqHand"]

    # Specify some supported robot types
    agent: Union[Panda, Fetch, PandaRobotiqHand]

    # set some commonly used values
    cube_half_size = 0.02

    def __init__(self, *args, robot_uids="PandaRobotiq", robot_init_qpos_noise=0.00, scene_name='home1', scene_translation=[0,0,0], radius=1.0, image_size=512, need_render=False, record_video=False, gaussian_iteration=None, background_name=None, cameras_config=None, use_joint=False, demo=False, **kwargs):
        # specifying robot_uids="panda" as the default means gym.make("PushCube-v1") will default to using the panda arm.
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.scene_name = scene_name
        self.scene_translation = scene_translation
        self.radius = radius
        self.image_size = image_size
        self.need_render = need_render
        self.record_video = record_video
        self.lift_height = 0.01
        self.no_object = False
        self.gaussian_iteration = gaussian_iteration
        self.background_name = background_name
        self.cameras_config = cameras_config
        self.use_joint = use_joint
        self.demo = demo
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # Specify default simulation/gpu memory configurations to override any default values
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        # follow the camera config in the main.py
        
        sensor_configs = []
        for i, camera_config in enumerate(self.cameras_config):
            elev = camera_config['elev'] / 180.0 * np.pi
            azim = camera_config['azim'] / 180.0 * np.pi
            eye = np.array([self.radius * np.sin(azim), self.radius * np.sin(elev), self.radius * np.cos(elev) * np.sin(np.pi / 2 - azim)])
            eye += self.scene_translation
            pose = sapien_utils.look_at(eye=eye, target=self.scene_translation)
            sensor_configs.append(CameraConfig(f"view_{i}", pose=pose, width=self.image_size, height=self.image_size, fov=70 / 180 * np.pi, near=0.01, far=100))
        return sensor_configs
        
    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.2, 1.2, 0.6], [0.3, 0.3, 0.2])
        return CameraConfig(
            "render_camera", pose=pose, width=500, height=500, fov=1, near=0.01, far=100
        )

    def _load_scene(self, options: dict):
        # we then add the cube that we want to push and give it a color and size using a convenience build_cube function
        # we specify the body_type to be "dynamic" as it should be able to move when touched by other objects / the robot

        xyz = self.scene_translation
        xyz = torch.tensor(xyz)[None]

        # rotation axis
        axis = np.array([0, 1, 0])
        axis = axis / np.linalg.norm(axis)
        # rotation angle
        angle = np.pi
        q = [np.cos(angle / 2), np.sin(angle / 2) * axis[0], np.sin(angle / 2) * axis[1], np.sin(angle / 2) * axis[2]]

        obj_pose = Pose.create_from_pq(p=xyz, q=q)

        background_builder = self.scene.create_actor_builder()
        self.objects = []

        if self.gaussian_iteration is not None:
            max_iteration = self.gaussian_iteration
        else:
            mesh_floder = f'./gaussians/output/{self.scene_name}/train/'
            name_list = os.listdir(mesh_floder)
            max_iteration = np.max([int(name.split('_')[-1]) for name in name_list])

        if self.background_name is not None:
            collision_path = f'./gaussians/output/{self.background_name}/train/ours_{max_iteration}/background_convex.ply'
            visual_path = f'./gaussians/output/{self.background_name}/train/ours_{max_iteration}/background.ply'
        else:
            collision_path = f'./gaussians/output/{self.scene_name}/train/ours_{max_iteration}/background_convex.ply'
            visual_path = f'./gaussians/output/{self.scene_name}/train/ours_{max_iteration}/background.ply'

        background_builder.add_multiple_convex_collisions_from_file(filename=collision_path)

        if self.record_video or self.use_joint or self.demo:
            background_builder.add_visual_from_file(filename=visual_path)
        
        static_friction = 0.99
        dynamic_friction = 0.9

        background_builder.initial_pose = obj_pose
        self.background = background_builder.build_static(name='scene')

        self.object_offset = []
        mesh_list = os.listdir(f'./gaussians/output/{self.scene_name}/train/ours_{max_iteration}/')

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
            
            mesh_name = mesh_file[:-4]

            object_builder = self.scene.create_actor_builder()
            collision_path = f'./gaussians/output/{self.scene_name}/train/ours_{max_iteration}/{mesh_name}_convex.ply'
            visual_path = f'./gaussians/output/{self.scene_name}/train/ours_{max_iteration}/{mesh_name}.ply'

            mesh = o3d.io.read_triangle_mesh(collision_path)
            min_bound = mesh.get_min_bound()
            max_bound = mesh.get_max_bound()
            mesh_middle = (min_bound + max_bound) / 2

            # normalize the mesh
            mesh.translate(-mesh_middle)

            # save
            new_collision_path = f'./gaussians/output/{self.scene_name}/train/ours_{max_iteration}/{mesh_name}_convex_normalize.ply'
            o3d.io.write_triangle_mesh(new_collision_path, mesh)
            
            visual_mesh = o3d.io.read_triangle_mesh(visual_path)
            visual_mesh.translate(-mesh_middle)
            new_visual_path = f'./gaussians/output/{self.scene_name}/train/ours_{max_iteration}/{mesh_name}_normalize.ply'
            o3d.io.write_triangle_mesh(new_visual_path, visual_mesh)

            center = np.array(mesh_middle)
            center[0] = -center[0]
            center[2] = -center[2]
            self.object_offset.append(center)

            object_builder.add_multiple_convex_collisions_from_file(filename=new_collision_path)

            if self.record_video or self.use_joint or self.demo:
                object_builder.add_visual_from_file(filename=new_visual_path)

            for r in object_builder.collision_records:
                r.material.static_friction = static_friction
                r.material.dynamic_friction = dynamic_friction

            object_xyz = xyz + torch.tensor([0.0, 0.0, self.lift_height]) + torch.tensor(center)
            obj_pose = Pose.create_from_pq(p=object_xyz, q=q)
            object_builder.initial_pose = obj_pose
            self.objects.append(object_builder.build_dynamic(name=mesh_name))

            current_mass = self.objects[-1].get_mass()

        if len(self.objects) == 0:
            self.no_object = True
            print('no object')
            self.cube = actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=np.array([12, 42, 160, 255]) / 255,
                name="cube",
                body_type="dynamic",
            )
            self.objects.append(self.cube)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # use the torch.device context manager to automatically create tensors on CPU or CUDA depending on self.device, the device the environment runs on
        with torch.device(self.device):
            # the initialization functions where you as a user place all the objects and initialize their properties
            # are designed to support partial resets, where you generate initial state for a subset of the environments.
            # this is done by using the env_idx variable, which also tells you the batch size
            b = len(env_idx)

            # when using scene builders, you must always call .initialize on them so they can set the correct poses of objects in the prebuilt scene
            xyz = torch.tensor(self.scene_translation)[None].repeat(b, 1)

            xyz += torch.tensor([0.0, 0.0, self.lift_height])
            # rotation axis
            axis = np.array([0, 1, 0])
            axis = axis / np.linalg.norm(axis)
            # rotation angle
            angle = np.pi
            q = [np.cos(angle / 2), np.sin(angle / 2) * axis[0], np.sin(angle / 2) * axis[1], np.sin(angle / 2) * axis[2]]

            if self.no_object:
                xyz = torch.tensor(self.scene_translation)[None].repeat(b, 1)
                xyz += torch.tensor([0.0, 0.0, -1.0])
                q = [1, 0, 0, 0]
                cube_pose = Pose.create_from_pq(p=xyz, q=q)
                self.objects[0].set_pose(cube_pose)
                return

            for obj_id, obj in enumerate(self.objects):
                obj_pose = Pose.create_from_pq(p=xyz + torch.tensor(self.object_offset[obj_id]), q=q)

                obj.set_pose(obj_pose)

    def _get_obs_extra(self, info: Dict):
        # some useful observation info for solving the task includes the pose of the tcp (tool center point) which is the point between the
        # grippers of the robot
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )

        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        # assign rewards to parallel environments that achieved success to the maximum of 3.
        return 0.0

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 3.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
