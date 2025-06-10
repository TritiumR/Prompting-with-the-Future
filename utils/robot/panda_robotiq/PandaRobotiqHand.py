import os
import numpy as np
import sapien
import torch
from copy import deepcopy

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.controllers.base_controller import BaseController
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs import Pose


_HERE = os.path.dirname(os.path.abspath(__file__))
_ASSET_DIR = _HERE

@register_agent()
class PandaRobotiqHand(BaseAgent):
    """
    To test this agent, use the `scripts/test_robot.py` script with the following command:
    python scripts/test_robot.py -r MyPanda -c pd_ee_pose
    """

    uid = "PandaRobotiqHand"
    urdf_path = os.path.join(_ASSET_DIR, "panda_robotiq_maniskill_hand.urdf")
    disable_self_collisions = False
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            left_inner_finger_pad=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            right_inner_finger_pad=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([
                # FR3 Joints
                0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, np.pi / 4,
                # Robotiq Joints (4 Passive, 2 Active)
                0, 0, 0, 0, 0, 0
            ]),
            pose=sapien.Pose(),
            # EE Pose for this QPos is 
            # p: 0, 0.615, 0.2732, 
            # p(Relative to the base): 0.615, 0.0, 0.2732,
            # q: -0.2706, 0.6533, 0.2706, 0.6533
            # Euler: pi/2, pi/4, pi/2
            # If euler is pi/2, pi/2, pi/2 then the gripper is aligned with its base
        ),
        droid_reset=Keyframe(
            qpos=np.array([
                # FR3 Joints https://github.com/droid-dataset/droid/blob/c5737e40a6b18859b5b78dbcdbf1e3b3f5e461be/config/fr3/franka_panda.yaml#L7
                -0.13935425877571106, -0.020481698215007782, -0.05201413854956627, -2.0691256523132324, 0.05058913677930832, 2.0028650760650635, -0.9167874455451965,
                # Robotiq Joints (4 Passive, 2 Active)
                0, 0, 0, 0, 0, 0
            ]),
            pose=sapien.Pose()
            
            # EE Pose for this qpos is
            # p: tensor([[ 0.5287, -0.0994,  0.4364]])
            # q: tensor([[ 0.2659, -0.6686, -0.2209, -0.6584]])
            # euler: tensor([[-1.5579,  0.8678, -1.5159]]) => ~= -pi/2, pi/4, -pi/2
        ),
    )

    randomize_qpos_reset : bool = False # Parameter to randomize the qpos reset
    randomize_qpos_reset_sigma : float = 0.0 # Parameter to randomize the qpos reset

    arm_joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7"
    ]
    # root_joint_names = [
    #     "root_x_axis_joint",
    #     "root_y_axis_joint",
    #     "root_z_axis_joint",
    #     "root_x_rot_joint",
    #     "root_y_rot_joint",
    #     "root_z_rot_joint",
    # ]
    # arm_kp_joint = np.array([
    #     40, 30, 50, 25, 35, 25, 10
    # ], dtype=np.float32) # https://github.com/droid-dataset/droid/blob/c5737e40a6b18859b5b78dbcdbf1e3b3f5e461be/config/fr3/franka_hardware.yaml#L10
    # arm_kd_joint = np.array([
    #     4, 6, 5, 5, 3, 2, 1
    # ], dtype=np.float32) # https://github.com/droid-dataset/droid/blob/c5737e40a6b18859b5b78dbcdbf1e3b3f5e461be/config/fr3/franka_hardware.yaml#L11
    arm_kp_joint = np.array([
        4500, 4500, 3500, 3500, 2000, 2000, 2000
    ],dtype=np.float32) * 0.9 # https://github.com/google-deepmind/mujoco_menagerie/blob/bd9709b540d58e1dcf417e4ffeffc7d54318280d/franka_fr3/fr3.xml#L148
    # We use Kp, Kd values from mujoco menagerie values, but we dampen them by 0.9x to account for the fact that Mujoco has joint damping value 0.1
    
    arm_kd_joint = np.array([
        450, 450, 350, 350, 200, 200, 200
    ])

    arm_force_limit = np.array([
        86, 86, 86, 86, 11.5, 11.5, 11.5
    ]) # https://github.com/droid-dataset/droid/blob/c5737e40a6b18859b5b78dbcdbf1e3b3f5e461be/config/fr3/franka_hardware.yaml#L68

    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 1.0

    ee_link_name = "robotiq_85_tcp"
    
    gripper_width = 0.085 # 8.5 cm
    gripper_depth = 0.036 # 3.6 cm

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm Controllers
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            joint_names=self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_kp_joint,
            damping=self.arm_kd_joint,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            joint_names=self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_kp_joint,
            damping=self.arm_kd_joint,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.075, # https://github.dev/droid-dataset/droid/blob/main/droid/robot_ik/robot_ik_solver.py#L13
            pos_upper=0.075,
            stiffness=self.arm_kp_joint,
            damping=self.arm_kd_joint,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            # lower=[-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671],
            # upper=[2.9671, 1.8326, 2.9671, 0.0, 2.9671, 3.8223, 2.9671],
            pos_lower=-0.075,
            pos_upper=0.075,
            rot_lower=0.15, # NOTE (yunhao): Please keep this positive, as maniskill uses this as max angular velocity along any direction
            rot_upper=0.15, # https://github.dev/droid-dataset/droid/blob/main/droid/robot_ik/robot_ik_solver.py#L14
            stiffness=self.arm_kp_joint,
            damping=self.arm_kd_joint,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=[0, -0.5, 0], # NOTE(yunhao) Use any workspace limit here
            pos_upper=[0.7, 0.5, 0.5], 
            rot_lower=-np.pi,
            rot_upper=np.pi,
            stiffness=self.arm_kp_joint,
            damping=self.arm_kd_joint,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=False,
            normalize_action=False, # NOTE (yunhao): Droid didn't set any normalization
            # https://github.com/droid-dataset/droid/blob/main/config/fr3/franka_hardware.yaml#L33
        )

        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_kp_joint,
            self.arm_kd_joint,
            self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_kp_joint,
            self.arm_kd_joint,
            self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            lower = -1.0,
            upper = 1.0,
            damping = self.arm_kd_joint,
            force_limit = self.arm_force_limit,
            friction = 1.0,
            normalize_action= True, 
            # nullspace_gain=0.025,  # Gain for nullspace correction 
            # regularization_weight=1e-2,  # Regularization weight for nullspace
            # nullspace_reference=[0.0] * len(self.arm_joint_names),  # Default nullspace target positions # Cartesian Velocity Limits
            # max_lin_delta=0.075,  # Maximum linear velocity in Cartesian space 
            # max_rot_delta=0.15,  # Maximum rotational velocity in Cartesian space
        )


        # -------------------------------------------------------------------------- #
        # Gripper Controllers (Robotiq)
        # -------------------------------------------------------------------------- #
        # base_pd_joint_pos = PDJointPosControllerConfig(
        #     joint_names=self.root_joint_names,
        #     lower=None,
        #     upper=None,
        #     stiffness=1e3,
        #     damping=1e2,
        #     force_limit=100,
        #     normalize_action=False,
        # )
        # base_pd_joint_delta_pos = PDJointPosControllerConfig(
        #     joint_names=self.root_joint_names,
        #     lower=-0.1,
        #     upper=0.1,
        #     stiffness=1e3,
        #     damping=1e2,
        #     force_limit=100,
        #     use_delta=True,
        # )

        # Active and Passive Finger Joints for Mimicking Robotiq Gripper Control
        passive_finger_joint_names = [
            "left_inner_knuckle_joint",
            "right_inner_knuckle_joint",
            "left_inner_finger_joint",
            "right_inner_finger_joint",
        ]
        passive_finger_joints = PassiveControllerConfig(
            joint_names=passive_finger_joint_names,
            damping=0,
            friction=0,
        )

        finger_joint_names = ["left_outer_knuckle_joint", "right_outer_knuckle_joint"]
        finger_mimic_pd_joint_pos = PDJointPosMimicControllerConfig(
            joint_names=finger_joint_names,
            lower=None,
            upper=None,
            stiffness=1e5,
            damping=1e3,
            force_limit=self.gripper_force_limit,
            friction=0.05,
            normalize_action=False,
        ) # NOTE (Yunhao): Gripper Position for droid is [0, 1], we may need someway to normalize our actions to this range
        finger_mimic_pd_joint_delta_pos = PDJointPosMimicControllerConfig(
            joint_names=finger_joint_names,
            lower=-0.25 * 0.81, # https://github.com/droid-dataset/droid/blob/c5737e40a6b18859b5b78dbcdbf1e3b3f5e461be/droid/franka/robot.py#L150 and https://github.com/droid-dataset/droid/blob/c5737e40a6b18859b5b78dbcdbf1e3b3f5e461be/droid/robot_ik/robot_ik_solver.py#L12
            upper=0.25 * 0.81,
            stiffness=1e5,
            damping=1e3,
            force_limit=self.gripper_force_limit,
            normalize_action=True,
            friction=0.05,
            use_delta=True,
        )

        # -------------------------------------------------------------------------- #
        # Controller Configuration Dictionary
        # -------------------------------------------------------------------------- #
        return dict(
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos,
                # base=base_pd_joint_pos,
                finger=finger_mimic_pd_joint_pos,
                passive_finger_joints=passive_finger_joints,
            ),
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos,
                # base=base_pd_joint_delta_pos,
                finger=finger_mimic_pd_joint_delta_pos,
                passive_finger_joints=passive_finger_joints,
            ),
            pd_ee_delta_pos=dict(
                arm=arm_pd_ee_delta_pos,
                # base=base_pd_joint_delta_pos,
                finger=finger_mimic_pd_joint_delta_pos,
                passive_finger_joints=passive_finger_joints,
            ),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose,
                # base=base_pd_joint_delta_pos,
                finger=finger_mimic_pd_joint_delta_pos,
                passive_finger_joints=passive_finger_joints,
            ),
            pd_ee_pose=dict(
                arm=arm_pd_ee_pose, 
                # base=base_pd_joint_delta_pos,
                finger=finger_mimic_pd_joint_pos,
                passive_finger_joints=passive_finger_joints,
            ),
            pd_joint_vel=dict(
                arm=arm_pd_joint_vel, 
                finger=finger_mimic_pd_joint_delta_pos,
                passive_finger_joints=passive_finger_joints,
            ),
            pd_joint_pos_vel=dict(
                arm=arm_pd_joint_pos_vel, 
                finger=finger_mimic_pd_joint_delta_pos,
                passive_finger_joints=passive_finger_joints,
            ),
            pd_joint_delta_pos_vel=dict(
                arm=arm_pd_joint_delta_pos_vel, 
                finger=finger_mimic_pd_joint_delta_pos,
                passive_finger_joints=passive_finger_joints,
            ),
        )
    def _after_loading_articulation(self):
        outer_finger = self.robot.active_joints_map["right_inner_finger_joint"]
        inner_knuckle = self.robot.active_joints_map["right_inner_knuckle_joint"]
        pad = outer_finger.get_child_link()
        lif = inner_knuckle.get_child_link()

        # the next 4 magic arrays come from https://github.com/haosulab/cvpr-tutorial-2022/blob/master/debug/robotiq.py which was
        # used to precompute these poses for drive creation
        p_f_right = [-1.6048949e-08, 3.7600022e-02, 4.3000020e-02]
        p_p_right = [1.3578170e-09, -1.7901104e-02, 6.5159947e-03]
        p_f_left = [-1.8080145e-08, 3.7600014e-02, 4.2999994e-02]
        p_p_left = [-1.4041154e-08, -1.7901093e-02, 6.5159872e-03]

        right_drive = self.scene.create_drive(
            lif, sapien.Pose(p_f_right), pad, sapien.Pose(p_p_right)
        )
        right_drive.set_limit_x(0, 0)
        right_drive.set_limit_y(0, 0)
        right_drive.set_limit_z(0, 0)

        outer_finger = self.robot.active_joints_map["left_inner_finger_joint"]
        inner_knuckle = self.robot.active_joints_map["left_inner_knuckle_joint"]
        pad = outer_finger.get_child_link()
        lif = inner_knuckle.get_child_link()

        left_drive = self.scene.create_drive(
            lif, sapien.Pose(p_f_left), pad, sapien.Pose(p_p_left)
        )
        left_drive.set_limit_x(0, 0)
        left_drive.set_limit_y(0, 0)
        left_drive.set_limit_z(0, 0)
        
        self.gripper_left_outer_knuckle = self.robot.active_joints_map["left_outer_knuckle_joint"]
        self.gripper_right_outer_knuckle = self.robot.active_joints_map["right_outer_knuckle_joint"]
        self.tcp = sapien_utils.get_obj_by_name(self.robot.get_links(), self.ee_link_name)

    def reset(self, init_qpos: torch.Tensor = None):
        qpos_limit = self.robot.get_qlimits()
        init_qpos = init_qpos or self.keyframes["rest"].qpos.copy()

        if not isinstance(init_qpos, torch.Tensor):
            init_qpos = torch.as_tensor(init_qpos, device=self.device)
        
        if init_qpos.ndim == 1:
            init_qpos = init_qpos.unsqueeze(0)

        if self.randomize_qpos_reset:
            if not isinstance(self.randomize_qpos_reset_sigma, torch.Tensor):
                qpos_delta = torch.full((7,), self.randomize_qpos_reset_sigma, device=self.device)
            else:
                qpos_delta = self.randomize_qpos_reset_sigma.to(self.device)
            
            qpos_delta = torch.normal(0, qpos_delta)
            if qpos_delta.ndim == 1:
                qpos_delta = qpos_delta.unsqueeze(0)

            qpos = init_qpos
            init_qpos = torch.concat([
                torch.clamp(qpos[..., :7] + qpos_delta, qpos_limit[..., :7, 0], qpos_limit[..., :7, 1]),
                qpos[..., 7:]
            ], dim=-1)
        
        # Set the robot to the rest pose by default
        super().reset(init_qpos)

    def get_proprioception(self):
        proprio = super().get_proprioception()
        arm_controller : BaseController = self.controller.controllers['arm']
        if hasattr(arm_controller, "ee_pose_at_base"):
            ee_pose : Pose = arm_controller.ee_pose_at_base
            proprio['end_effector_pos'] = ee_pose.p
            proprio['end_effector_quaternion'] = ee_pose.q
            # NOTE (yunhao): For some reason when we convert the quaternion to matrix and then back to euler angles, the euler angles can turn into NaNs. 
            # I wonder if it is because floating has rounding errors?
            # proprio['end_effector_euler'] = matrix_to_euler_angles(quaternion_to_matrix(ee_pose.q), 'XYZ')
            # assert not torch.any(torch.isnan(proprio['end_effector_euler'])), f"End effector euler angles are NaN: {proprio['end_effector_euler']}"
        
        # NOTE (yunhao): Hardcoded gripper limit as [0, 0.81], we can use self.robot().get_qlimits() to get the limits but why bother
        # Actually for the qpos sum, we first need to divide by 2 to get the average qpos, then divide by 0.81 to get the normalized value, then time 2 to map to [0, 2] and finally subtract 1 to get to [-1, 1]
        # But I simplified this to just divide by 0.81
        # NOTE (yunhao): get_limits() outputs (n_joint, 2) tensor
        # gripper_limit = self.gripper_left_outer_knuckle.get_limits()[0]
        # Now we can use gripper_limit[0] and gripper_limit[1] to get the upper-lower limits
        # gripper_pos is between [-1, 1] where -1 is open and 1 is closed
        proprio['gripper_pos'] = (self.gripper_left_outer_knuckle.qpos + self.gripper_right_outer_knuckle.qpos) / 0.81 - 1
        
        return proprio

    def is_grasping(self, object: Actor, min_force=0.2, max_angle=85):
        """Check if the Robotiq gripper is grasping an object.

        Args:
            object (Actor): The object to check if the gripper is grasping.
            min_force (float, optional): Minimum force before the gripper is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
        """
        # Obtain the left and right finger links specifically for Robotiq gripper
        left_finger_link = sapien_utils.get_obj_by_name(self.robot.get_links(), "left_inner_finger_pad")
        right_finger_link = sapien_utils.get_obj_by_name(self.robot.get_links(), "right_inner_finger_pad")
        
        # Get the contact forces between each finger and the object
        l_contact_forces = left_finger_link.get_net_contact_forces()
        r_contact_forces = right_finger_link.get_net_contact_forces()
        
        # Calculate the force magnitudes for each contact force
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)
        
        # Get the direction vectors for each finger link's pose
        ldirection = left_finger_link.pose.to_transformation_matrix()[..., :3, 1]  # Y-axis of the left finger link
        rdirection = right_finger_link.pose.to_transformation_matrix()[..., :3, 1]  # Y-axis for right finger
        # Note (Yunhao): Tested that this is actually correct (Positive Y Axis for Right Finger), probably the URDF already rotates the finger links
        
        # Calculate angles between contact forces and finger opening directions
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        
        # Check if the force magnitude and angle satisfy grasping conditions
        lflag = torch.logical_and(lforce >= min_force, torch.rad2deg(langle) <= max_angle)
        rflag = torch.logical_and(rforce >= min_force, torch.rad2deg(rangle) <= max_angle)

        # Return True if both fingers satisfy the grasping conditions
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold: float = 0.2):
        """Check if the robot's joints are nearly static (i.e., below the threshold velocity).

        Args:
            threshold (float): The maximum joint velocity considered to be static. Defaults to 0.2.

        Returns:
            bool: True if the robot is static, False otherwise.
        """
        # Retrieve the joint velocities
        qvel = self.robot.get_qvel()
        
        # Ignore the last four joints as these are associated with the Robotiq gripper
        # (assuming the last four entries correspond to gripper mechanics)
        qvel_relevant = qvel[..., :-4]

        # Check if all joint velocities are below the threshold
        return torch.max(torch.abs(qvel_relevant), dim=1)[0] <= threshold

    def get_robot_mask(self, segmentation : torch.Tensor):
        """
        Get the segmentation mask for the robot's links.
        Args:
            segmentation (torch.Tensor): The segmentation tensor with shape (N, H, W, [1]).
        Returns:
            A dictionary containing the segmentation masks for each link. Each mask has shape (N, H, W, [1]).
            Also a mask for the entire robot with shape (N, H, W, [1]).
        """
        link_masks = {}
        all_mask = torch.zeros_like(segmentation, device=segmentation.device, dtype=torch.bool)
        for link in self.robot.get_links():
            link_seg_id = link._objs[0].entity.per_scene_id
            link_mask = torch.eq(segmentation, link_seg_id)
            
            link_masks[link.name] = link_mask
            all_mask = torch.logical_or(all_mask, link_mask)
        return link_masks, all_mask
    
    def get_link_id_map(self):
        link_id_map = {}
        for link in self.robot.get_links():
            link_id_map[link.name] = link._objs[0].entity.per_scene_id
        return link_id_map

    # @property
    # def _sensor_configs(self):
    #     return [
    #         CameraConfig(
    #             uid="hand_camera",  # Unique identifier for the camera
    #             pose=Pose.create_from_pq(
    #                 p=torch.tensor([0.1, 0, 0]),
    #                 q=matrix_to_quaternion(euler_angles_to_matrix(torch.tensor([0, np.pi / 2, np.pi]), 'XYZ')),
    #             ),
    #             width=512,  # Image width
    #             height=512,  # Image height
    #             fov=np.pi / 3,  # Field of view (90 degrees)
    #             near=0.01,  # Near clipping plane
    #             far=100,  # Far clipping plane
    #             #TODO: maybe can add a camera link to urdf? can be addd to any link defined in urdf
    #             mount=self.robot.links_map["panda_link8"], 
    #         )
    #     ]

    def build_grasp_pose(
        self,
        approaching : np.ndarray,
        closing : np.ndarray,
        center : np.ndarray,
    ) -> sapien.Pose:
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3

        # Force the gripper to always be at least upright
        if approaching[2] < 0:
            approaching = -approaching
        
        ortho = np.cross(approaching, closing)
        T = np.eye(4)
        T[:3, :3] = np.stack([approaching, closing, ortho], axis=1)
        T[:3, 3] = center
        return Pose.create(sapien.Pose(T))

    @property
    def current_closing_direction(self) -> torch.Tensor:
        closing = self.tcp.pose.to_transformation_matrix()[..., :3, 1]
        return closing

    def get_finger_links(self):
        left_inner_finger = sapien_utils.get_obj_by_name(self.robot.get_links(), "left_inner_finger_pad")
        right_inner_finger = sapien_utils.get_obj_by_name(self.robot.get_links(), "right_inner_finger_pad")
        return left_inner_finger, right_inner_finger

    def get_gripper_position(self):
        return self.tcp.pose.get_p()
    
    def get_gripper_rotation(self):
        return self.tcp.pose.get_q()