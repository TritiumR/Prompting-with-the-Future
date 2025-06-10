import gymnasium as gym
import mani_skill.envs
import meshes.mesh
import numpy as np
import torch
from mani_skill.utils.wrappers.record import RecordEpisode
import copy
from scipy.spatial.transform import Rotation as R


def quaternion_multiply(a, b):
    return torch.tensor([
        a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
        a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
        a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
        a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]
    ])


def compute_quaternion_rotation(q1, q2):
    """
    Compute the rotation between two quaternions.
    
    Args:
    q1 (list or np.array): First quaternion [w, i, j, k]
    q2 (list or np.array): Second quaternion [w, i, j, k]
    
    Returns:
    np.array: Rotation quaternion representing the rotation from q1 to q2
    """
    # Convert inputs to numpy arrays
    q1 = np.array(q1)
    q2 = np.array(q2)
    
    # Normalize the quaternions
    q1_norm = q1 / np.linalg.norm(q1)
    q2_norm = q2 / np.linalg.norm(q2)
    
    # Compute the relative rotation quaternion
    # The relative rotation is q2 * q1_conjugate
    # For a quaternion a + bi + cj + dk, conjugate is a - bi - cj - dk
    q1_conjugate = np.array([q1_norm[0], -q1_norm[1], -q1_norm[2], -q1_norm[3]])
    
    # Quaternion multiplication
    def quaternion_multiply(a, b):
        return np.array([
            a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
            a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
            a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
            a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]
        ])
    
    # Compute relative rotation quaternion
    rotation_quaternion = quaternion_multiply(q2_norm, q1_conjugate)
    
    # Normalize the rotation quaternion
    rotation_quaternion /= np.linalg.norm(rotation_quaternion)
        
    return rotation_quaternion


def compute_quaternion_rotation_batch(q1, q2):
    """
    Compute the rotation between a quaternion and batch of quaternions using PyTorch.
    
    Args:
    q1 (torch.Tensor): First quaternion [w, i, j, k]
    q2 (torch.Tensor): Batch of quaternions or single quaternion [w, i, j, k]
    
    Returns:
    torch.Tensor: Batch of rotation quaternions representing the rotation from q1 to q2
    """
    # Ensure inputs are torch tensors
    q1 = q1.float() if not isinstance(q1, torch.Tensor) else q1.float()
    q2 = q2.float() if not isinstance(q2, torch.Tensor) else q2.float()
    
    # Normalize the first quaternion
    q1_norm = q1 / torch.norm(q1)
    
    # Normalize the second quaternion(s)
    if q2.dim() == 1:
        # Single quaternion case
        q2_norm = q2 / torch.norm(q2)
        q2_norm = q2_norm.unsqueeze(0)  # Add batch dimension
    else:
        # Batch of quaternions
        q2_norm = q2 / torch.norm(q2, dim=1, keepdim=True)
    
    # Compute the conjugate of q1
    q1_conjugate = torch.tensor([q1_norm[0], -q1_norm[1], -q1_norm[2], -q1_norm[3]], 
                                 dtype=q1.dtype, device=q1.device)
    
    # Quaternion multiplication function
    def quaternion_multiply(a, b):
        """
        Batched quaternion multiplication.
        a: [w, x, y, z] or [batch, w, x, y, z]
        b: [w, x, y, z] or [batch, w, x, y, z]
        """
        # Ensure a is 2D (either single or batched)
        if b.dim() == 1:
            b = b.unsqueeze(0)
        
        # Expand conjugate to match batch size if needed
        if a.dim() == 2:
            b = b.expand(b.size(0), -1)
        
        return torch.stack([
            a[:,0]*b[:,0] - a[:,1]*b[:,1] - a[:,2]*b[:,2] - a[:,3]*b[:,3],
            a[:,0]*b[:,1] + a[:,1]*b[:,0] + a[:,2]*b[:,3] - a[:,3]*b[:,2],
            a[:,0]*b[:,2] - a[:,1]*b[:,3] + a[:,2]*b[:,0] + a[:,3]*b[:,1],
            a[:,0]*b[:,3] + a[:,1]*b[:,2] - a[:,2]*b[:,1] + a[:,3]*b[:,0]
        ], dim=1)
    
    # Compute relative rotation quaternion(s)
    rotation_quaternion = quaternion_multiply(q2_norm, q1_conjugate.unsqueeze(0))
    
    # Normalize the rotation quaternion(s)
    rotation_quaternion /= torch.norm(rotation_quaternion, dim=1, keepdim=True)
    
    # Optional: Compute rotation angle(s)
    rotation_angle = 2 * torch.acos(rotation_quaternion[:,0])
    
    return rotation_quaternion


def euler_to_quaternion(euler_angle) -> tuple:
    """
    Convert Euler angles to quaternion.
    
    Args:
        roll (float): Rotation angle around the x-axis in radians.
        pitch (float): Rotation angle around the y-axis in radians.
        yaw (float): Rotation angle around the z-axis in radians.
        
    Returns:
        tuple: Quaternion as (w, x, y, z).
    """
    roll, pitch, yaw = euler_angle
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr *sp * cy

    return (w, x, y, z)


def quaternion_diff_to_euler(q1_batch, q2_batch, order='xyz', degrees=False):
    """
    Compute Euler angles representing relative rotation from q1 to q2.

    Parameters:
        q1_batch (np.ndarray): shape (N, 4), each row is (w, x, y, z).
        q2_batch (np.ndarray): shape (N, 4), each row is (w, x, y, z).
        order (str): order of Euler angle axes, e.g., 'xyz'.
        degrees (bool): whether to return angles in degrees (True) or radians (False).

    Returns:
        np.ndarray: shape (N, 3), each row is the Euler angles representing q_rel = q2 * q1^-1.
    """
    assert q1_batch.shape == q2_batch.shape
    assert q1_batch.shape[1] == 4

    # Convert to (x, y, z, w) for scipy
    q1_xyzw = np.hstack([q1_batch[:, 1:], q1_batch[:, :1]])
    q2_xyzw = np.hstack([q2_batch[:, 1:], q2_batch[:, :1]])

    r1 = R.from_quat(q1_xyzw)
    r2 = R.from_quat(q2_xyzw)

    # Relative rotation r_rel = r2 * r1^-1
    r_rel = r2 * r1.inv()
    return r_rel.as_euler(order, degrees=degrees)


class MeshWorld:
    def __init__(self, name='home1', num_envs=1, scene_traslation=np.array([0,0,0]), radius=1.0, image_size=500, rotate_mode='body', record_video=False, robot_uids='panda', need_render=False, dir=None, close_gripper=False, use_joint=False, gaussian_iteration=None, background_name=None, cameras_config=None, demo=False):
        assert rotate_mode in ['body', 'root']

        if use_joint:
            control = 'pd_joint_pos'
        else:
            control = 'pd_ee_delta_pose'

        self.env = gym.make(
            "Mesh", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
            num_envs=num_envs,
            obs_mode="rgb+depth+segmentation", # there is also "state_dict", "rgbd", ...
            control_mode=control, # there is also "pd_joint_delta_pos", ...
            render_mode="rgb_array", # there is also "rgb_array", "human", ...
            robot_uids=robot_uids,
            scene_name=name,
            scene_translation=scene_traslation,
            radius=radius,
            image_size=image_size,
            need_render=need_render,
            record_video=record_video,
            gaussian_iteration=gaussian_iteration,
            background_name=background_name,
            cameras_config=cameras_config,
            use_joint=use_joint,
            demo=demo
        )

        self.num_envs = num_envs
        self.need_render = need_render
        self.num_cameras = 4 if cameras_config is None else len(cameras_config)
        self.robot_uids = robot_uids
        self.record_video = record_video
        self.close_gripper = close_gripper
        self.use_joint = use_joint
        self.demo = demo

        self.stage = 0

        if dir is not None:
            output_dir = f"{dir}/videos"
        else:
            output_dir = "videos"
        # careful! save video, which is slow
        if record_video:
            self.env = RecordEpisode(self.env, output_dir=output_dir, save_trajectory=False, trajectory_name="trajectory", save_video=True, video_fps=30, max_steps_per_video=5000)

        _, _ = self.env.reset(seed=0) # reset with a seed for determinism

        # print('env: ', self.env)
        self.agent = self.env.unwrapped.agent
        self.joint_limits = np.array([[-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671],
                                      [2.9671, 1.8326, 2.9671, 0.0, 2.9671, 3.8223, 2.9671]])
        initial_joint_angles = np.array([
            # FR3 Joints
            0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, 0,
            # Robotiq Joints (4 Passive, 2 Active)
            0, 0, 0, 0, 0, 0
        ])

        self.scene_translation = scene_traslation

        self.grasping = []

        if self.close_gripper:
            self.grasping_pos = 1.0
            self.grasping_now = True
        else:
            self.grasping_pos = -1.0
            self.grasping_now = False

        self.object_drop = False
        self.all_crash = False

        for obj_id, obj in enumerate(self.env.unwrapped.objects):
            object_state = obj.get_state()[0]

        # stable the environment for 1s
        for _ in range(20):
            self.env.step(None)
        
        # self.object_offset = []
        self.object_init_state = []
        for obj_id, obj in enumerate(self.env.unwrapped.objects):
            object_state = obj.get_state()[0]
            self.object_init_state.append(object_state)
            self.grasping.append(False)

        self.done = False
        self.min_force = 0.1
        self.max_angle = 80
        self.crash_moving_threshold = 1
        self.crash_gripper_threshold = 1
        self.joint_threshold = 0.05

        self.agent.robot.set_qpos(initial_joint_angles)
        if self.num_envs > 1:
            self.env.unwrapped.scene._gpu_apply_all()
            self.env.unwrapped.scene._gpu_fetch_all()
        
        self.history_states = []
        self.history_states.append(self.env.unwrapped.get_state()[0][None])

        print('--- mesh world built ---')

    def sample_action_distribution_batch(self, samples, non_stop=False, try_grasp=False, try_release=False, need_info=False):
        ''' sample delte actions, return joint angles '''
        gripper_delta = self.grasping_pos
        # print('gripper_delta before sample: ', gripper_delta)
        samples = np.clip(samples, -10, 10)

        # delta_scales = np.array([0.1, 0.1, 0.1, np.pi / 4, np.pi / 4, np.pi / 4])
        delta_scales = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
        keep_still = np.array([0, 0, 0, 0, 0, 0, gripper_delta])

        if non_stop:
            # print('samples: ', samples.shape)
            samples = samples.repeat(self.num_envs, 0)
            # print('samples: ', samples.shape)

        sampled_delta_actions = samples * delta_scales

        # append gripper_delta to all actions
        sampled_delta_actions = np.concatenate([sampled_delta_actions, np.ones((samples.shape[0], 1)) * gripper_delta], axis=-1)

        times = 30

        current_states = []

        # self.env.unwrapped.set_state_dict(self.current_state_dict)
        self.env.unwrapped.set_state(self.history_states[-1])

        for obj in self.env.unwrapped.objects:
            obj_state = obj.get_state()
            current_states.append(obj_state)
        current_states = torch.stack(current_states, dim=0)

        touched = []
        for obj in self.env.unwrapped.objects:
            touched.append([False] * self.num_envs)
        touched = np.array(touched)

        # collision = np.array([False] * self.num_envs)
        collision = np.ones(self.num_envs, dtype=np.float32)
        all_crash_mask = np.array([False] * self.num_envs, dtype=bool)
        all_collision_mask = np.array([False] * self.num_envs, dtype=bool)

        prev_qpos = self.agent.robot.get_qpos()
        joint_angles_list = prev_qpos

        for i in range(times):

            obs, _, _, _, _ = self.env.step(sampled_delta_actions)

            # if not non_stop:
            # check if IK failed
            current_qpos = self.agent.robot.get_qpos()
            crash_mask = (np.min(current_qpos[:, :7].cpu().numpy() - self.joint_limits[0], axis=-1) < self.joint_threshold) | (np.min(self.joint_limits[1] - current_qpos[:, :7].cpu().numpy(), axis=-1) < self.joint_threshold)
            # print('crash_mask: ', crash_mask.shape)
            # print('max_diff: ', torch.max(current_qpos - prev_qpos, dim=-1)[0])
            # crash_mask = torch.max(current_qpos - prev_qpos, dim=-1)[0] > self.crash_moving_threshold
            crash_mask = crash_mask
            # print('crash_mask: ', crash_mask.sum())
            # collision[crash_mask] = True
            # save uncrashed joint angles
            new_crash = crash_mask & (~all_crash_mask)
            collision[new_crash] = i / times
            joint_angles_list[new_crash] = prev_qpos[new_crash]
            sampled_delta_actions[new_crash] = keep_still
            all_crash_mask[new_crash] = True

            # check collision
            collision_force = self.check_collision()

            # collision[left_collision > self.min_force] = True
            # collision[right_collision > self.min_force] = True
            collision_mask = collision_force > self.min_force
            collision_mask = collision_mask.astype(bool)
            
            new_collision = collision_mask & (~all_collision_mask)
            collision[new_collision] = i / times
            joint_angles_list[new_collision] = prev_qpos[new_collision]
            sampled_delta_actions[new_collision] = keep_still
            all_collision_mask[new_collision] = True

            # check touch
            for obj_id, obj in enumerate(self.env.unwrapped.objects):
                left_touch, right_touch = self.get_gripper_force(obj)

                touched[obj_id][left_touch > self.min_force] = True
                touched[obj_id][right_touch > self.min_force] = True

            prev_qpos = current_qpos
                
        # if not non_stop:
        # print(f'{all_collision_mask.sum()} collision')
        # print(f'{all_crash_mask.sum()} crash')
        # print(f'at {np.where(all_crash_mask)} crash')
        # print(f'{touched.sum()} touched')

        if all_crash_mask.sum() == self.num_envs:
            # print('all crash')
            # reset to the initial state
            self.all_crash = True

        # careful! only wait for 1 second
        not_still_times = 0
        while not_still_times < 20:
            obs, _, _, _, _ = self.env.step(None)
            not_still_times += 1

        obs = copy.deepcopy(obs)
        # joint_angles_list = self.agent.robot.get_qpos()
        joint_angles_list[collision == 1] = self.agent.robot.get_qpos()[collision == 1]

        object_states = []
        object_transformations = []
        for obj_id, obj in enumerate(self.env.unwrapped.objects):
            states = obj.get_state()
            # print('object states: ', states.shape)

            transformations = states.clone()

            transformations[:, :3] = transformations[:, :3] - self.object_init_state[obj_id][:3]
            # print('translations: ', transformations[:, :3])
            # print('translations: ', torch.norm(transformations[:, :3], dim=-1))
            new_quaternions = transformations[:, 3:7]
            old_quaternion = self.object_init_state[obj_id][3:7]

            # print('new_quaternions: ', new_quaternions.shape)
            # print('old_quaternion: ', old_quaternion.shape)

            rotation_quaternions = compute_quaternion_rotation_batch(old_quaternion, new_quaternions)
            transformations[:, 3:7] = rotation_quaternions

            object_states.append(states)
            object_transformations.append(transformations)

        action_object_transformations = torch.stack(object_transformations, dim=0).transpose(0, 1)

        if non_stop:
            # self.current_state_dict = self.env.unwrapped.get_state_dict()
            self.history_states.append(self.env.unwrapped.get_state()[0][None])

            # careful!! if object droped, open gripper
            is_grasping = False
            for obj in self.env.unwrapped.objects:
                grasping = self.agent.is_grasping(obj, min_force=self.min_force // 2, max_angle=self.max_angle).cpu().numpy()[0]
                if grasping:
                    is_grasping = True
                    break

            if not is_grasping and not self.close_gripper and self.grasping_now:
                print('object dropped')
                self.object_drop = True
                self.prev_grasping_pos = self.grasping_pos
                self.grasping_pos = -1.0
                self.grasping_now = False

            if self.need_render:
                images, depth_images = self.render_image_depth(obs)
                if need_info:
                    infos = self.get_info_by_name()
                    return joint_angles_list, action_object_transformations, images, depth_images, infos
                return joint_angles_list, action_object_transformations, images, depth_images
            
            if need_info:
                infos = self.get_info_by_name()
                return joint_angles_list, action_object_transformations, infos

            return joint_angles_list, action_object_transformations
        
        post_samples = samples * collision[:, None]

        if try_grasp:
            grasp_object_transformations, grasp_obs, is_grasping = self.try_grasp()

        if try_release:
            release_object_transformations, release_obs = self.try_release()

        if self.need_render:
            images, depth_images = self.render_image_depth(obs)
            if try_grasp:
                if is_grasping.sum() > 0:
                    print('which_is_grasping: ', np.where(is_grasping))
                    grasp_images, grasp_depth_images = self.render_image_depth(grasp_obs)
                else:
                    grasp_images = None
                    grasp_depth_images = None
                if need_info:
                    infos = self.get_info_by_name()
                    return joint_angles_list, action_object_transformations, post_samples, images, depth_images, grasp_object_transformations, grasp_images, grasp_depth_images, is_grasping, infos
                return joint_angles_list, action_object_transformations, post_samples, images, depth_images, grasp_object_transformations, grasp_images, grasp_depth_images, is_grasping
            
            if try_release:
                release_images, release_depth_images = self.render_image_depth(release_obs)
                if need_info:
                    infos = self.get_info_by_name()
                    return joint_angles_list, action_object_transformations, post_samples, images, depth_images, release_object_transformations, release_images, release_depth_images, infos
                return joint_angles_list, action_object_transformations, post_samples, images, depth_images, release_object_transformations, release_images, release_depth_images

            if need_info:
                infos = self.get_info_by_name()
                return joint_angles_list, action_object_transformations, post_samples, images, depth_images, infos

            return joint_angles_list, action_object_transformations, post_samples, images, depth_images
        
        if need_info:
            infos = self.get_info_by_name()
            return joint_angles_list, action_object_transformations, post_samples, infos

        return joint_angles_list, action_object_transformations, post_samples

    def reset(self):
        self.env.reset()

    def try_grasp(self):
        ''' try to grasp at different positions '''
        grasp_action = np.array([0., 0., 0., 0., 0., 0., 0.])

        grasp_action = grasp_action[None].repeat(self.num_envs, 0)
        test_firm_action = np.array([0., 0., 0.2, 0., 0., 0., 0.])
        test_firm_action = test_firm_action[None].repeat(self.num_envs, 0)

        times = 30

        grasp_intervals = np.linspace(-1, 1, times)

        all_crash_mask = np.array([False] * self.num_envs, dtype=bool)
        min_force = self.min_force
        max_angle = self.max_angle
        is_grasping = np.zeros(self.num_envs, dtype=bool)
        prev_qpos = self.agent.robot.get_qpos()
        for i in range(times):
            # careful!!! always use 0.1 to grasp
            grasp_action[~(is_grasping | all_crash_mask), -1] = grasp_intervals[i]
            obs, _, _, _, _ = self.env.step(grasp_action)

            current_qpos = self.agent.robot.get_qpos()
            crash_mask = torch.max(current_qpos - prev_qpos, dim=-1)[0] > self.crash_gripper_threshold
            crash_mask = crash_mask.cpu().numpy()
            all_crash_mask[crash_mask] = True
            prev_qpos = current_qpos

            for obj in self.env.unwrapped.objects:
                grasping = self.agent.is_grasping(obj, min_force=min_force, max_angle=max_angle).cpu().numpy()
                new_grasping = grasping & (~is_grasping)
                is_grasping[grasping] = True
                # careful!!! always use 0.1 to grasp
                test_firm_action[new_grasping, -1] = grasp_intervals[i]

        is_grasping[all_crash_mask] = False

        print(f'{all_crash_mask.sum()} crash')
        
        print(f'{is_grasping.sum()} grasping after trying grasping')

        if is_grasping.sum() == 0:
            return None, None, is_grasping

        all_crash_mask = np.array([False] * self.num_envs, dtype=bool)
        up_times = 0
        while up_times < 20:
            obs, _, _, _, _ = self.env.step(test_firm_action)
            current_qpos = self.agent.robot.get_qpos()
            crash_mask = (np.min(current_qpos[:, :7].cpu().numpy() - self.joint_limits[0], axis=-1) < self.joint_threshold) | (np.min(self.joint_limits[1] - current_qpos[:, :7].cpu().numpy(), axis=-1) < self.joint_threshold)
            test_firm_action[crash_mask, :6] = np.array([0., 0., 0., 0., 0., 0.])
            all_crash_mask[crash_mask] = True
            up_times += 1

        is_grasping = np.zeros(self.num_envs, dtype=bool)
        for obj in self.env.unwrapped.objects:
            grasping = self.agent.is_grasping(obj, min_force=min_force, max_angle=max_angle).cpu().numpy()
            is_grasping[grasping] = True

        is_grasping[all_crash_mask] = False

        print(f'{all_crash_mask.sum()} crash')

        print(f'{is_grasping.sum()} grasping after lifting')

        if is_grasping.sum() == 0:
            return None, None, is_grasping

        object_transformations = []
        for obj_id, obj in enumerate(self.env.unwrapped.objects):
            states = obj.get_state()

            transformations = states.clone()

            transformations[:, :3] = transformations[:, :3] - self.object_init_state[obj_id][:3]
            new_quaternions = transformations[:, 3:7]
            old_quaternion = self.object_init_state[obj_id][3:7]
            rotation_quaternions = compute_quaternion_rotation_batch(old_quaternion, new_quaternions)
            transformations[:, 3:7] = rotation_quaternions

            object_transformations.append(transformations)

        action_object_transformations = torch.stack(object_transformations, dim=0).transpose(0, 1)


        self.try_grasp_states = self.env.unwrapped.get_state()[is_grasping]
        self.try_grasp_pos = test_firm_action[is_grasping, -1]

        return action_object_transformations, obs, is_grasping
    
    def set_grasp_state(self, grasp_id, need_info=False):
        ''' set the grasp state '''
        self.env.unwrapped.set_state(self.try_grasp_states[grasp_id][None])
        self.history_states.append(self.try_grasp_states[grasp_id][None])
        self.grasping_pos = self.try_grasp_pos[grasp_id]
        self.grasping_now = True

        joint_angles_list = self.agent.robot.get_qpos()

        object_states = []
        object_transformations = []
        for obj_id, obj in enumerate(self.env.unwrapped.objects):
            states = obj.get_state()

            transformations = states.clone()

            transformations[:, :3] = transformations[:, :3] - self.object_init_state[obj_id][:3]
            new_quaternions = transformations[:, 3:7]
            old_quaternion = self.object_init_state[obj_id][3:7]
            rotation_quaternions = compute_quaternion_rotation_batch(old_quaternion, new_quaternions)
            transformations[:, 3:7] = rotation_quaternions

            object_states.append(states)
            object_transformations.append(transformations)

        action_object_transformations = torch.stack(object_transformations, dim=0).transpose(0, 1)

        if self.need_render:
            images, depth_images = self.get_image_depth()
            if need_info:
                infos = self.get_info_by_name()
                return joint_angles_list, action_object_transformations, images, depth_images, infos
            return joint_angles_list, action_object_transformations, images, depth_images
    
    def release(self, non_stop=False):
        ''' open the gripper to release the object '''
        grasp_action = np.array([0., 0., 0., 0., 0., 0., 0.])

        grasp_action = grasp_action[None].repeat(self.num_envs, 0)

        times = 20
        grasp_intervals = np.linspace(self.grasping_pos, -1, times)

        self.env.unwrapped.set_state(self.history_states[-1])

        current_states = []
        for obj in self.env.unwrapped.objects:
            obj_state = obj.get_state()
            current_states.append(obj_state)
        current_states = torch.stack(current_states, dim=0)

        for i in range(times):
            grasp_action[:, -1] = grasp_intervals[i]
            obs, _, _, _, _ = self.env.step(grasp_action)
        
        # wait for 2 seconds
        not_still_times = 0
        while not_still_times < 40:
            obs, reward, terminated, truncated, info = self.env.step(None)
            not_still_times += 1

        object_states = []
        object_transformations = []
        for obj_id, obj in enumerate(self.env.unwrapped.objects):
            states = obj.get_state()

            transformations = states.clone()

            transformations[:, :3] = transformations[:, :3] - self.object_init_state[obj_id][:3]
            new_quaternions = transformations[:, 3:7]
            old_quaternion = self.object_init_state[obj_id][3:7]
            rotation_quaternions = compute_quaternion_rotation_batch(old_quaternion, new_quaternions)
            transformations[:, 3:7] = rotation_quaternions

            object_states.append(states)
            object_transformations.append(transformations)

        action_object_transformations = torch.stack(object_transformations, dim=0).transpose(0, 1)

        joint_angles_list = self.agent.robot.get_qpos()

        if non_stop:
            self.history_states.append(self.env.unwrapped.get_state()[0][None])
            self.grasping_pos = -1.
            self.grasping_now = False
            print('grasping_pos: ', self.grasping_pos)

        if self.need_render:
            images, depth_images = self.render_image_depth(obs)
            return joint_angles_list, action_object_transformations, images, depth_images

        return joint_angles_list, action_object_transformations

    def try_release(self):
        ''' open the gripper to release the object '''
        grasp_action = np.array([0., 0., 0., 0., 0., 0., 0.])

        grasp_action = grasp_action[None].repeat(self.num_envs, 0)

        times = 20
        grasp_intervals = np.linspace(self.grasping_pos, -1, times)

        for i in range(times):
            grasp_action[:, -1] = grasp_intervals[i]
            obs, _, _, _, _ = self.env.step(grasp_action)

        # wait for 2 seconds
        not_still_times = 0
        while not_still_times < 40:
            obs, _, _, _, _ = self.env.step(None)
            not_still_times += 1

        object_states = []
        object_transformations = []
        for obj_id, obj in enumerate(self.env.unwrapped.objects):
            states = obj.get_state()

            transformations = states.clone()

            transformations[:, :3] = transformations[:, :3] - self.object_init_state[obj_id][:3]
            new_quaternions = transformations[:, 3:7]
            old_quaternion = self.object_init_state[obj_id][3:7]
            rotation_quaternions = compute_quaternion_rotation_batch(old_quaternion, new_quaternions)
            transformations[:, 3:7] = rotation_quaternions

            object_states.append(states)
            object_transformations.append(transformations)

        action_object_transformations = torch.stack(object_transformations, dim=0).transpose(0, 1)

        return action_object_transformations, obs

    def render_image_depth(self, obs):
        ''' render image and depth '''
        images = []
        depth_images = []
        for i in range(self.num_cameras):
            name = f'view_{i}'
            rgb = obs['sensor_data'][name]['rgb'].cpu().numpy() / 255.
            depth = obs['sensor_data'][name]['depth'].cpu().numpy() / 1000.
            # if object visual is loaded, we need to mask out the object
            if self.record_video:
                segment = obs['sensor_data'][name]['segmentation'].cpu().numpy()
                gripper_mask = (segment >= 11) & (segment <= 22)
                depth[~gripper_mask] = 1e10
            images.append(rgb)
            depth_images.append(depth)

        images = np.stack(images, axis=0)
        depth_images = np.stack(depth_images, axis=0)

        return images, depth_images

    def get_image_depth(self, no_set_state=False):
        ''' get image and depth '''
        if not no_set_state:
            self.env.unwrapped.set_state(self.history_states[-1])

        obs, _, _, _, _ = self.env.step(None)
        images = []
        depth_images = []
        for i in range(self.num_cameras):
            name = f'view_{i}'
            rgb = obs['sensor_data'][name]['rgb'].cpu().numpy() / 255.
            depth = obs['sensor_data'][name]['depth'].cpu().numpy() / 1000.
            # if object visual is loaded, we need to mask out the object
            if self.record_video:
                segment = obs['sensor_data'][name]['segmentation'].cpu().numpy()
                if self.demo:
                    gripper_mask = (segment >= 0) & (segment <= 22)
                else:
                    gripper_mask = (segment >= 11) & (segment <= 22)
                depth[~gripper_mask] = 1e10
            images.append(rgb)
            depth_images.append(depth)

        images = np.stack(images, axis=0)
        depth_images = np.stack(depth_images, axis=0)

        return images, depth_images
    
    def get_gripper_force(self, obj):
        ''' get gripper force '''
        left_link, right_link = self.agent.get_finger_links()
        left_force = torch.norm(self.env.unwrapped.scene.get_pairwise_contact_forces(left_link, obj), dim=1).cpu()
        right_force = torch.norm(self.env.unwrapped.scene.get_pairwise_contact_forces(right_link, obj), dim=1).cpu()

        return left_force, right_force

    def get_info(self):
        ''' get reward '''
        infos = dict()

        gripper_position = self.agent.get_gripper_position().cpu().numpy()
        infos['gripper_position'] = gripper_position
        # print('gripper_pose: ', gripper_pose)
        # careful! only for one object
        infos['object_poses'] = []
        for obj_id, obj in enumerate(self.env.unwrapped.objects):
            # object_state = obj.get_state().cpu().numpy()
            # object_rotation = object_state[:, 3:7]
            # apply to offset
            # object_offset = self.object_offset[obj_id] 
            object_pose = obj.get_state().cpu().numpy() #+ self.object_offset[obj_id]
            infos['object_poses'].append(object_pose)

        return infos

    def check_collision(self):
        ''' check collision '''
        forces = []
        for link in self.agent.robot.get_links():
            force = self.env.unwrapped.scene.get_pairwise_contact_forces(link, self.env.unwrapped.background)
            force = torch.norm(force, dim=1).cpu().numpy()
            forces.append(force)
        
        forces = np.stack(forces, axis=0)
        max_force = np.max(forces, axis=0)
        return max_force

    def sample_trajectory_distribution_batch(self, samples, non_stop=False, try_grasp=False, try_release=False, need_info=False):
        ''' sample delte trajectory 
            samples: (num_envs, horizon, 6)
        '''
        gripper_delta = self.grasping_pos

        speed = 0.1

        delta_scales = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]) * speed * 5
        keep_still = np.array([0., 0., 0., 0., 0., 0., gripper_delta])

        if non_stop:
            samples = samples.repeat(self.num_envs, 0)

        sampled_delta_actions = samples * delta_scales
        
        # append gripper_delta to all actions
        sampled_delta_actions = np.concatenate([sampled_delta_actions, np.ones((samples.shape[0], samples.shape[1], 1)) * gripper_delta], axis=-1)

        times = int(samples.shape[1] / speed)

        current_states = []

        # self.env.unwrapped.set_state_dict(self.current_state_dict)
        self.env.unwrapped.set_state(self.history_states[-1])

        for obj in self.env.unwrapped.objects:
            obj_state = obj.get_state()
            current_states.append(obj_state)
        current_states = torch.stack(current_states, dim=0)

        touched = []
        for obj in self.env.unwrapped.objects:
            touched.append([False] * self.num_envs)
        touched = np.array(touched)

        # collision = np.array([False] * self.num_envs)
        collision = np.ones(self.num_envs, dtype=int) * times
        all_crash_mask = np.array([False] * self.num_envs, dtype=bool)
        all_collision_mask = np.array([False] * self.num_envs, dtype=bool)

        prev_qpos = self.agent.robot.get_qpos()
        joint_angles_list = prev_qpos

        for i in range(times):
            step_sampled_delta_actions = sampled_delta_actions[:, int(i * speed)]

            obs, _, _, _, _ = self.env.step(step_sampled_delta_actions)

            # if not non_stop:
            # check if IK failed
            current_qpos = self.agent.robot.get_qpos()
            crash_mask = (np.min(current_qpos[:, :7].cpu().numpy() - self.joint_limits[0], axis=-1) < self.joint_threshold) | (np.min(self.joint_limits[1] - current_qpos[:, :7].cpu().numpy(), axis=-1) < self.joint_threshold)
            # print('crash_mask: ', crash_mask.shape)
            # print('max_diff: ', torch.max(current_qpos - prev_qpos, dim=-1)[0])
            # crash_mask = torch.max(current_qpos - prev_qpos, dim=-1)[0] > self.crash_moving_threshold
            crash_mask = crash_mask
            # print('crash_mask: ', crash_mask.sum())
            # collision[crash_mask] = True
            # save uncrashed joint angles
            new_crash = crash_mask & (~all_crash_mask)
            collision[new_crash] = i
            joint_angles_list[new_crash] = prev_qpos[new_crash]
            sampled_delta_actions[new_crash, int(i * speed):] = keep_still
            all_crash_mask[new_crash] = True

            # check collision
            collision_force = self.check_collision()

            # collision[left_collision > self.min_force] = True
            # collision[right_collision > self.min_force] = True
            collision_mask = collision_force > self.min_force
            collision_mask = collision_mask.astype(bool)
            
            new_collision = collision_mask & (~all_collision_mask)
            collision[new_collision] = i
            joint_angles_list[new_collision] = prev_qpos[new_collision]
            sampled_delta_actions[new_collision, int(i * speed):] = keep_still
            all_collision_mask[new_collision] = True

            # check touch
            for obj_id, obj in enumerate(self.env.unwrapped.objects):
                left_touch, right_touch = self.get_gripper_force(obj)

                touched[obj_id][left_touch > self.min_force] = True
                touched[obj_id][right_touch > self.min_force] = True

            prev_qpos = current_qpos
                
        print(f'{all_crash_mask.sum()} crash')
        print(f'{touched.sum()} touched')

        if all_crash_mask.sum() == self.num_envs:
            print('all crash')
            self.all_crash = True

        # wait for 1 second
        not_still_times = 0
        while not_still_times < 20:
            obs, _, _, _, _ = self.env.step(None)
            not_still_times += 1

        obs = copy.deepcopy(obs)
        joint_angles_list[collision == times] = self.agent.robot.get_qpos()[collision == times]

        object_states = []
        object_transformations = []
        for obj_id, obj in enumerate(self.env.unwrapped.objects):
            states = obj.get_state()

            transformations = states.clone()

            transformations[:, :3] = transformations[:, :3] - self.object_init_state[obj_id][:3]
            new_quaternions = transformations[:, 3:7]
            old_quaternion = self.object_init_state[obj_id][3:7]

            rotation_quaternions = compute_quaternion_rotation_batch(old_quaternion, new_quaternions)
            transformations[:, 3:7] = rotation_quaternions

            object_states.append(states)
            object_transformations.append(transformations)

        action_object_transformations = torch.stack(object_transformations, dim=0).transpose(0, 1)

        if non_stop:
            self.history_states.append(self.env.unwrapped.get_state()[0][None])
            is_grasping = False
            for obj in self.env.unwrapped.objects:
                grasping = self.agent.is_grasping(obj, min_force=self.min_force // 2, max_angle=self.max_angle).cpu().numpy()[0]
                if grasping:
                    is_grasping = True
                    break

            if not is_grasping and not self.close_gripper and self.grasping_now:
                print('object dropped')
                self.object_drop = True
                self.prev_grasping_pos = self.grasping_pos
                self.grasping_pos = -1.0
                self.grasping_now = False

            if self.need_render:
                images, depth_images = self.render_image_depth(obs)
                if need_info:
                    infos = self.get_info_by_name()
                    return joint_angles_list, action_object_transformations, images, depth_images, infos
                return joint_angles_list, action_object_transformations, images, depth_images
            
            if need_info:
                infos = self.get_info_by_name()
                return joint_angles_list, action_object_transformations, infos

            return joint_angles_list, action_object_transformations
        
        post_samples = samples
        for early_id, early_stop in enumerate(collision):
            post_samples[early_id, int(early_stop * speed):] = keep_still[:6]

        if try_grasp:
            grasp_object_transformations, grasp_obs, is_grasping = self.try_grasp()

        if try_release:
            release_object_transformations, release_obs = self.try_release()

        if self.need_render:
            images, depth_images = self.render_image_depth(obs)
            if try_grasp:
                if is_grasping.sum() > 0:
                    print('which_is_grasping: ', np.where(is_grasping))
                    grasp_images, grasp_depth_images = self.render_image_depth(grasp_obs)
                else:
                    grasp_images = None
                    grasp_depth_images = None
                if need_info:
                    infos = self.get_info_by_name()
                    return joint_angles_list, action_object_transformations, post_samples, images, depth_images, grasp_object_transformations, grasp_images, grasp_depth_images, is_grasping, infos
                return joint_angles_list, action_object_transformations, post_samples, images, depth_images, grasp_object_transformations, grasp_images, grasp_depth_images, is_grasping
            
            if try_release:
                release_images, release_depth_images = self.render_image_depth(release_obs)
                if need_info:
                    infos = self.get_info_by_name()
                    return joint_angles_list, action_object_transformations, post_samples, images, depth_images, release_object_transformations, release_images, release_depth_images, infos
                return joint_angles_list, action_object_transformations, post_samples, images, depth_images, release_object_transformations, release_images, release_depth_images

            if need_info:
                infos = self.get_info_by_name()
                return joint_angles_list, action_object_transformations, post_samples, images, depth_images, infos

            return joint_angles_list, action_object_transformations, post_samples, images, depth_images
        
        if need_info:
            infos = self.get_info_by_name()
            return joint_angles_list, action_object_transformations, post_samples, infos

        return joint_angles_list, action_object_transformations, post_samples
