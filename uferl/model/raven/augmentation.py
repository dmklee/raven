import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import torchvision.transforms.functional as tvf

from uferl.model.common.rotation_transformer import RotationTransformer

'''
We can apply SE(2)/SE(3) transformations to the world (gripper pose, camera pose, output gripper pose)

We can apply SE(2) augmentations to images/rays jointly

We can apply SE(2) augmentations to camera poses and images??
'''

def center_crop(images, intrinsics, extrinsics, crop_shape):
    '''
    images: B, C, H, W
    intrinsics: B, 3, 3
    '''
    B, C, H, W = images.shape
    H_out, W_out = crop_shape
    theta = torch.eye(3).repeat(B, 1, 1).to(images.device)
    theta[:, 0, 0] = W_out / W
    theta[:, 1, 1] = H_out / H

    new_images = tvf.center_crop(img=images, output_size=crop_shape)
    new_intrinsics = torch.bmm(torch.linalg.inv(theta), intrinsics)
    return new_images, new_intrinsics, extrinsics


def random_crop_and_rotate(images, intrinsics, extrinsics, crop_shape, rotation_range=(-10, 10), c4_rots=True):
    '''
    images: B, C, H, W
    intrinsics: B, 3, 3
    '''
    assert len(images.shape) == 4
    assert len(intrinsics.shape) == 3

    B, C, H, W = images.shape
    H_out, W_out = crop_shape

    tops = torch.randint(0, H - H_out + 1, (B,))
    lefts = torch.randint(0, W - W_out + 1, (B,))
    if c4_rots:
        base_angles = 90. * torch.randint(0, 4, (B,)).float().to(images.device)
    else:
        base_angles = 0.
    angle_offsets = torch.FloatTensor(B).to(images.device).uniform_(rotation_range[0], rotation_range[1])
    angles = base_angles + angle_offsets

    cos_angles = torch.cos(torch.deg2rad(angles))
    sin_angles = torch.sin(torch.deg2rad(angles))

    scale_w = W_out / W
    scale_h = H_out / H

    theta = torch.eye(3).repeat(B, 1, 1).to(images.device)
    theta[:, 0, 0] = cos_angles * scale_w
    theta[:, 0, 1] = -sin_angles * scale_h
    theta[:, 1, 0] = sin_angles * scale_w
    theta[:, 1, 1] = cos_angles * scale_h
    theta[:, 0, 2] = -(lefts + W_out / 2 - W/2) / (W/2)
    theta[:, 1, 2] = -(tops + H_out / 2 - H/2) / (H/2)

    grid = F.affine_grid(theta[:, :2], (B, 1, H_out, W_out), align_corners=False)
    new_images = F.grid_sample(images, grid, align_corners=False)

    r = torch.stack([
        cos_angles, sin_angles, -sin_angles, cos_angles
    ], dim=1).reshape(-1, 2, 2)
    theta[:, :2, 2] = torch.einsum('nij,nj->ni', r, theta[:, :2, 2])
    theta[:, 0, 0] = scale_w
    theta[:, 1, 1] = scale_h
    theta[:, 0, 1] = 0
    theta[:, 1, 0] = 0

    new_intrinsics = torch.bmm(torch.linalg.inv(theta), intrinsics)

    new_extrinsics = extrinsics
    theta *= 0
    theta[:, 0, 0] = cos_angles
    theta[:, 0, 1] = -sin_angles
    theta[:, 1, 0] = sin_angles
    theta[:, 1, 1] = cos_angles
    theta[:, 2, 2] = 1.0

    new_extrinsics[:, :3, :3] = extrinsics[:, :3, :3] @ theta

    return new_images, new_intrinsics, new_extrinsics


class Augmenter(nn.Module):
    def __init__(
        self,
        rot_aug_magn=[0, 0, 2*np.pi], # in euler xyz order
        trans_aug_magn=[0, 0, 0],
        crop_shape=(76, 76),
        img_rot_range=(-0, 0),
        img_c4_rots=False,
        n_arms: int=1,
    ):
        super().__init__()
        self.n_arms = n_arms
        self.rot_aug_magn = rot_aug_magn
        self.trans_aug_magn = trans_aug_magn
        self.crop_shape = crop_shape
        self.img_rot_range = img_rot_range
        self.img_c4_rots = img_c4_rots

        self.quat2mtx = RotationTransformer('quaternion', 'matrix')
        self.input2mtx = RotationTransformer('rotation_6d', 'matrix')

    def sample_se3_transform(self, B):
        tfm = torch.zeros((B, 4, 4), dtype=torch.float32)
        tfm[:, 3, 3] = 1

        euler_angles = np.random.uniform([0, 0, 0], self.rot_aug_magn, size=(B, 3))
        rots = Rotation.from_euler('xyz', euler_angles).as_matrix().astype(np.float32)
        tfm[:, :3, :3] = torch.from_numpy(rots)

        tfm[:, 0, 3].uniform_(-self.trans_aug_magn[0], self.trans_aug_magn[0])
        tfm[:, 1, 3].uniform_(-self.trans_aug_magn[1], self.trans_aug_magn[1])
        tfm[:, 2, 3].uniform_(-self.trans_aug_magn[2], self.trans_aug_magn[2])

        return tfm

    def apply_global_transform(self, global_tfm, obs_dict, action=None):
        new_obs_dict = {}
        for name, obs in obs_dict.items():
            if name == 'gravity':
                new_obs_dict[name] = torch.einsum('bij,bj->bi', global_tfm[:, :3, :3], obs)

            elif name.endswith('_pos'):
                pos_homog = torch.cat([
                    obs, torch.ones((*obs.shape[:-1], 1), dtype=obs.dtype, device=obs.device)
                ], dim=-1)
                new_obs_dict[name] = torch.einsum('bij,btj->bti', global_tfm[:, :3], pos_homog)

            elif name.endswith('_quat'):
                mtx = self.quat2mtx.forward(obs[:, :, [3, 0, 1, 2]])
                mtx = torch.einsum('bij,btjk->btik', global_tfm[:, :3, :3], mtx)
                new_obs_dict[name] = self.quat2mtx.inverse(mtx)[:, :, [1, 2, 3, 0]]

            elif name.endswith('_extrinsic'):
                new_obs_dict[name] = torch.einsum('bij,btjk->btik', global_tfm, obs)

            else: # qpos, intrinsic, image
                new_obs_dict[name] = obs

        if action is not None:
            action_pos, action_rot, action_grip = torch.split(action, [3, 6, action.shape[-1] - 9], dim=-1)

            action_pos_homog = torch.cat([
                action_pos, torch.ones((*action_pos.shape[:-1], 1), dtype=action_pos.dtype, device=action_pos.device)
            ], dim=-1)
            new_action_pos = torch.einsum('bij,btnj->btni', global_tfm[:, :3], action_pos_homog)

            action_rotmtx = self.input2mtx.forward(action_rot)
            new_action_rotmtx = torch.einsum('bij,btnjk->btnik', global_tfm[:, :3, :3], action_rotmtx)
            new_action_rot = self.input2mtx.inverse(new_action_rotmtx)

            new_action = torch.cat([new_action_pos, new_action_rot, action_grip], dim=-1)
        else:
            new_action = None

        return new_obs_dict, new_action

    def forward(self, obs_dict, action=None):
        value = next(iter(obs_dict.values()))
        B = value.shape[0]
        device = value.device

        if 'gravity' in obs_dict:
            obs_dict['gravity'] = obs_dict['gravity'].repeat(B, 1)

        if self.training:
            # sample transform
            global_tfm = self.sample_se3_transform(B).to(device)
            obs_dict, action = self.apply_global_transform(global_tfm, obs_dict, action)

        cameras = [a.removesuffix('_image') for a in obs_dict.keys() if a.endswith('_image')]
        for cam in cameras:
            img = obs_dict[f'{cam}_image']
            intrinsic = obs_dict[f'{cam}_intrinsic']
            extrinsic = obs_dict[f'{cam}_extrinsic']

            if self.training:
                new_img, new_intrinsic, new_extrinsic = random_crop_and_rotate(
                    img.flatten(0, 1), intrinsic.flatten(0, 1), extrinsic.flatten(0, 1),
                    self.crop_shape, self.img_rot_range, self.img_c4_rots
                )
            else:
                new_img, new_intrinsic, new_extrinsic = center_crop(
                    img.flatten(0, 1), intrinsic.flatten(0, 1), extrinsic.flatten(0, 1), self.crop_shape
                )

            obs_dict[f'{cam}_image'] = new_img.view(*img.shape[:2], *new_img.shape[1:])
            obs_dict[f'{cam}_intrinsic'] = new_intrinsic.view(intrinsic.shape)
            obs_dict[f'{cam}_extrinsic'] = new_extrinsic.view(extrinsic.shape)

        return obs_dict, action

## OLD
# def old_random_crop_and_rotate(images, intrinsics, extrinsics, crop_shape, rotation_range=(-10, 10), c4_rots=True):
    # '''
    # images: B, C, H, W
    # intrinsics: B, 3, 3
    # '''
    # assert len(images.shape) == 4
    # assert len(intrinsics.shape) == 3

    # B, C, H, W = images.shape
    # H_out, W_out = crop_shape

    # tops = torch.randint(0, H - H_out + 1, (B,))
    # lefts = torch.randint(0, W - W_out + 1, (B,))
    # if c4_rots:
        # base_angles = 90. * torch.randint(0, 4, (B,)).float()
    # else:
        # base_angles = 0.
    # angle_offsets = torch.FloatTensor(B).uniform_(rotation_range[0], rotation_range[1])
    # angles = base_angles + angle_offsets

    # cos_angles = torch.cos(torch.deg2rad(angles))
    # sin_angles = torch.sin(torch.deg2rad(angles))

    # scale_w = W_out / W
    # scale_h = H_out / H

    # theta = torch.eye(3).repeat(B, 1, 1).to(images.device)
    # theta[:, 0, 0] = cos_angles * scale_w
    # theta[:, 0, 1] = -sin_angles * scale_h
    # theta[:, 1, 0] = sin_angles * scale_w
    # theta[:, 1, 1] = cos_angles * scale_h
    # theta[:, 0, 2] = -(lefts + W_out / 2 - W/2) / (W/2)
    # theta[:, 1, 2] = -(tops + H_out / 2 - H/2) / (H/2)

    # grid = F.affine_grid(theta[:, :2], (B, 1, H_out, W_out), align_corners=False)
    # # import matplotlib.pyplot as plt
    # # f, ax = plt.subplots(1, 2)
    # # ax[0].imshow(grid[0,..., 0], vmin=-1, vmax=1)
    # # ax[1].imshow(grid[0,..., 1], vmin=-1, vmax=1)
    # # plt.show()
    # new_images = F.grid_sample(images, grid, align_corners=False)

    # theta[:, :2, 2] = torch.einsum('nji,nj->ni', theta[:, :2, :2], theta[:, :2, 2])
    # theta[:, 0, 0] = scale_w
    # theta[:, 1, 1] = scale_h
    # theta[:, 0, 1] = 0
    # theta[:, 1, 0] = 0

    # new_intrinsics = torch.bmm(torch.linalg.inv(theta), intrinsics)
    # # print('new_intrinsics')
    # # print(new_intrinsics[0].numpy())

    # new_extrinsics = extrinsics
    # theta *= 0
    # theta[:, 0, 0] = cos_angles
    # theta[:, 0, 1] = -sin_angles
    # theta[:, 1, 0] = sin_angles
    # theta[:, 1, 1] = cos_angles
    # theta[:, 2, 2] = 1.0

    # extrinsics[:, :3, :3] = extrinsics[:, :3, :3] @ theta

    # return new_images, new_intrinsics, extrinsics


if __name__ == "__main__":
    # Example usage:
    B, C, H, W = 4, 3, 64, 64
    output_size = (32, 32)

    images = torch.randn(B, C, H, W)
    transformed_images, transform_info = random_crop_and_rotate(images, output_size)

    print(f"Original images shape: {images.shape}")
    print(f"Transformed images shape: {transformed_images.shape}")
    print(f"Transform info: {transform_info}")
