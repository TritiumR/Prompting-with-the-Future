import torch
import numpy as np
from pytorch3d.renderer import look_at_view_transform

def get_up_direction(elev, azim):
    device = elev.device
    up = []
    z_axis = torch.tensor([0., 0., -1.], device=device).float()

    for i in range(len(elev)):
        ele = elev[i]
        azi = azim[i]
        look_at = torch.tensor([torch.cos(ele / 180 * np.pi) * torch.sin(azi / 180 * np.pi), 
                                torch.sin(ele / 180 * np.pi), 
                                torch.cos(ele / 180 * np.pi) * torch.cos(azi / 180 * np.pi)], device=device).float()
        product = torch.cross(z_axis, -look_at)
        if torch.norm(product) < 1e-6:
            product = torch.tensor([0., -1., 0.], device=device)
        up.append(product)

    up = torch.stack(up, dim=0)
    return up

def get_up_direction_sim(elev, azim):
    device = elev.device
    up = []
    z_axis = torch.tensor([1., 0., 0.], device=device).float()

    for i in range(len(elev)):
        ele = elev[i]
        azi = azim[i]
        look_at = torch.tensor([torch.cos(ele / 180 * np.pi) * torch.sin(azi / 180 * np.pi), 
                                torch.sin(ele / 180 * np.pi), 
                                torch.cos(ele / 180 * np.pi) * torch.cos(azi / 180 * np.pi)], device=device).float()
        product = torch.cross(z_axis, -look_at)
        if torch.norm(product) < 1e-6:
            product = torch.tensor([0., -1., 0.], device=device)
        up.append(product)

    up = torch.stack(up, dim=0)
    return up

def get_back_direction(elev, azim):
    device = elev.device
    up = []
    z_axis = torch.tensor([-1., 0., 0.], device=device).float()

    for i in range(len(elev)):
        ele = elev[i]
        azi = azim[i]
        look_at = torch.tensor([torch.cos(ele / 180 * np.pi) * torch.sin(azi / 180 * np.pi), 
                                torch.sin(ele / 180 * np.pi), 
                                torch.cos(ele / 180 * np.pi) * torch.cos(azi / 180 * np.pi)], device=device).float()
        product = torch.cross(z_axis, -look_at)
        if torch.norm(product) < 1e-6:
            product = torch.tensor([0., -1., 0.], device=device)
        up.append(product)

    up = torch.stack(up, dim=0)
    return up


def create_wrist_camera(wrist_eyes, wrist_up, wrist_ats, device):
    eye = torch.tensor(wrist_eyes, device=device).float()
    up = torch.tensor(wrist_up[None], device=device).float()
    at = torch.tensor(wrist_ats, device=device).float()
    R_wrist, T_wrist = look_at_view_transform(eye=eye, up=up, at=at, device=device)

    return R_wrist, T_wrist

def fixed_to_gripper(hand_position, distance, view_ids, device):
    eyes = []
    ups = []
    for view_id in view_ids:
        if view_id == 0:
            eye = torch.tensor(hand_position + np.array([0.0, -distance, 0.0]), device=device).float()
            up = torch.tensor([0., 0., 1.], device=device).float()
        elif view_id == 1:
            eye = torch.tensor(hand_position + np.array([distance, 0.0, 0.0]), device=device).float()
            up = torch.tensor([0., 0., 1.], device=device).float()
        elif view_id == 2:
            eye = torch.tensor(hand_position + np.array([0.0, distance, 0.0]), device=device).float()
            up = torch.tensor([0., 0., 1.], device=device).float()
        elif view_id == 3:
            eye = torch.tensor(hand_position + np.array([0.0, 0.0, distance]), device=device).float()
            up = torch.tensor([1., 0., 0.], device=device).float()
        else:
            raise ValueError("Invalid view_id")

        eyes.append(eye)
        ups.append(up)

    # print('len(eyes)', len(eyes))
    # print('len(ups)', len(ups))
    eyes = torch.stack(eyes, dim=0)
    ups = torch.stack(ups, dim=0)
    # print('len(eyes)', len(eyes))
    # print('len(ups)', len(ups))
    at = torch.tensor(np.array([hand_position]), device=device).float()
    # print('len(at)', len(at))

    R_gripper, T_gripper = look_at_view_transform(eye=eyes, up=ups, at=at, device=device)

    return R_gripper, T_gripper


def fixed_to_gripper_gaussian(hand_position, distance, view_ids, device):
    eyes = []
    ups = []
    for view_id in view_ids:
        if view_id == 0:
            eye = torch.tensor(hand_position + np.array([0.0, -distance, 0.0]), device=device).float()
            up = torch.tensor([0., 0., 1.], device=device).float()
        elif view_id == 1:
            eye = torch.tensor(hand_position + np.array([distance, 0.0, 0.0]), device=device).float()
            up = torch.tensor([0., 0., 1.], device=device).float()
        elif view_id == 2:
            eye = torch.tensor(hand_position + np.array([0.0, distance, 0.0]), device=device).float()
            up = torch.tensor([0., 0., 1.], device=device).float()
        elif view_id == 3:
            eye = torch.tensor(hand_position + np.array([0.0, 0.0, distance]), device=device).float()
            up = torch.tensor([1., 0., 0.], device=device).float()
        else:
            raise ValueError("Invalid view_id")

        eyes.append(eye)
        ups.append(up)

    # print('len(eyes)', len(eyes))
    # print('len(ups)', len(ups))
    eyes = torch.stack(eyes, dim=0)
    ups = torch.stack(ups, dim=0)
    # print('len(eyes)', len(eyes))
    # print('len(ups)', len(ups))
    at = torch.tensor(np.array([hand_position]), device=device).float()
    # print('len(at)', len(at))

    R_gripper, T_gripper = look_at_view_transform(eye=eyes, up=ups, at=at, device=device)

    eyes[:, 2] = 2 * at[:, 2] - eyes[:, 2]
    eyes[:, 0] = 2 * at[:, 0] - eyes[:, 0]
    ups[:, 2] = -ups[:, 2]
    ups[:, 0] = -ups[:, 0]
    at[:, 2] = -at[:, 2]
    at[:, 0] = -at[:, 0]

    R_gaussian, T_gaussian = look_at_view_transform(eye=eyes, up=ups, at=at, device="cpu")

    return R_gripper, T_gripper, R_gaussian.numpy(), T_gaussian.numpy()