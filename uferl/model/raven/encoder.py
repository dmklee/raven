from typing import Union, Tuple, Optional, Dict, Callable

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange, repeat

from uferl.model.common.rotation_transformer import RotationTransformer
from uferl.model.pos_emb import SinusoidalPosEmb
from uferl.sensors import get_opencv_rays

from uferl.model.raven.utils import parse_obs
from uferl.model.raven.layers import TemperatureAdjustableSoftmax, GTEncoderBlock, ResNet18, GTAttention, NonGTAttention


class RGBEncoder(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        img_encoder: Callable,
        s_out: int,
        r_out: int,
        p_out: int,
        patch_dropout: float=0.0,
        use_depth: bool=False
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.patch_dropout = patch_dropout

        if isinstance(img_encoder, nn.Module):
            self.img_encoder = img_encoder
        else:
            self.img_encoder = img_encoder()


        fdim, h, w = self.get_fmap_shape()
        self.img_fdim = fdim

        self.use_depth = use_depth
        if use_depth:
            self.depth_enc = nn.Sequential(
                nn.Conv2d(1, fdim, kernel_size=16, stride=16, padding=2),
                nn.ReLU(True),
                nn.Conv2d(fdim, fdim, kernel_size=1, stride=1),
            )

        self.ray_encoder = nn.Sequential(
            nn.Linear(3*fdim if use_depth else 2*fdim, fdim),
            nn.ReLU(True),
            nn.Linear(fdim, s_out + 3*r_out + 3*p_out),
        )
        emb = torch.exp(torch.arange(fdim//2) * -np.log(10000) / (fdim //2 - 1 ) )
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        self.register_buffer(
            'ray_emb', emb.view(1, 1, -1),
        )

        ray_dirs = get_opencv_rays(h, w)
        self.ray_dirs = nn.Parameter(ray_dirs, requires_grad=False)

        self.euler2mtx = RotationTransformer('euler_angles', 'matrix', from_convention='YXZ')

    def get_fmap_shape(self) -> Tuple[int, int, int]:
        dummy_img = torch.zeros((1, *self.input_shape), dtype=torch.float32)
        fmap = self.img_encoder(dummy_img)
        return fmap.shape[1:]

    def forward(self, rgb: Tensor, extrinsic: Tensor, intrinsic: Tensor, depth=None) -> Tuple[Tensor, Tensor]:
        '''Encodes image

        img: float tensor of shape B, T, C, H, W
        extrinsic: float tensor of shape B, T, 4, 4 (camera pose; aka camera2world)
        intrinsic: float tensor of shape B, T, 3, 3 (aka camera2pixel)

        return:
            feats: float tensor of shape B, T, H'*W', s_out+3*r_out+3*p_out
            g_se3: float tensor of shape B, T, 4, 4
        '''
        b, t = rgb.shape[:2]
        rgb = rearrange(rgb, "b t c h w -> (b t) c h w")
        fmap = self.img_encoder(rgb)
        feats = rearrange(fmap, "(b t) c h w -> b t (h w) c", b=b)

        if self.use_depth:
            depth_feats = rearrange(
                self.depth_enc(
                    rearrange(depth, "b t c h w -> (b t) c h w")
                ), "(b t) c h w -> b t (h w) c", b=b
            )
            feats = torch.cat([depth_feats, feats, self.ray_emb.expand_as(feats)], dim=-1)
        else:
            feats = torch.cat([feats, self.ray_emb.expand_as(feats)], dim=-1)

        feats = self.ray_encoder(feats)

        pixel2camera = torch.linalg.inv(intrinsic)
        ray_dirs = torch.einsum(
            'btij, rj->btri', pixel2camera, self.ray_dirs
        )
        eulers = torch.stack([
            torch.arctan(ray_dirs[..., 0]/ray_dirs[..., 2]),
            -torch.arctan(ray_dirs[..., 1]/ray_dirs[..., 2]),
            torch.zeros_like(ray_dirs[..., 0])
        ], dim=-1)
        ray_rotmtx = self.euler2mtx.forward(eulers)

        if self.training and self.patch_dropout > 0.:
            ind = np.arange(feats.shape[2])
            np.random.shuffle(ind)
            ind = ind[:int(feats.shape[2]*(1-self.patch_dropout))]
            feats = feats[:, :, ind]
            ray_rotmtx = ray_rotmtx[:, :, ind]

        extrinsic = repeat(extrinsic, 'b t u v -> b t hw u v', hw=feats.shape[2]).clone()
        extrinsic[..., :3, :3] = extrinsic[..., :3, :3] @ ray_rotmtx
        return feats, extrinsic


class NonSE3RGBEncoder(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        img_encoder: Callable,
        s_out: int,
        r_out: int,
        p_out: int,
        patch_dropout: float=0.0,
        use_depth=False,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.patch_dropout = patch_dropout

        assert use_depth == False

        if isinstance(img_encoder, nn.Module):
            self.img_encoder = img_encoder
        else:
            self.img_encoder = img_encoder()

        assert use_depth == False

        fdim, h, w = self.get_fmap_shape()
        self.img_fdim = fdim

        self.ray_encoder = nn.Sequential(
            nn.Linear(3*fdim, fdim),
            nn.ReLU(True),
            nn.Linear(fdim, s_out + 3*r_out + 3*p_out),
        )
        emb = torch.exp(torch.arange(fdim//2) * -np.log(10000) / (fdim //2 - 1 ) )
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        self.register_buffer(
            'ray_emb', emb.view(1, 1, -1),
        )

        ray_dirs = get_opencv_rays(h, w)
        self.ray_dirs = nn.Parameter(ray_dirs, requires_grad=False)

        self.ray_dir_enc = nn.Sequential(
            nn.Linear(2, 4*fdim),
            nn.GELU(),
            nn.Linear(4*fdim, fdim),
        )

        self.euler2mtx = RotationTransformer('euler_angles', 'matrix', from_convention='YXZ')

    def get_fmap_shape(self) -> Tuple[int, int, int]:
        dummy_img = torch.zeros((1, *self.input_shape), dtype=torch.float32)
        fmap = self.img_encoder(dummy_img)
        return fmap.shape[1:]

    def forward(self, rgb: Tensor, extrinsic: Tensor, intrinsic: Tensor, depth=None) -> Tuple[Tensor, Tensor]:
        '''Encodes image

        img: float tensor of shape B, T, C, H, W
        extrinsic: float tensor of shape B, T, 4, 4 (camera pose; aka camera2world)
        intrinsic: float tensor of shape B, T, 3, 3 (aka camera2pixel)

        return:
            feats: float tensor of shape B, T, H'*W', s_out+3*r_out+3*p_out
            g_se3: float tensor of shape B, T, 4, 4
        '''
        b, t = rgb.shape[:2]
        rgb = rearrange(rgb, "b t c h w -> (b t) c h w")
        fmap = self.img_encoder(rgb)
        feats = rearrange(fmap, "(b t) c h w -> b t (h w) c", b=b)



        pixel2camera = torch.linalg.inv(intrinsic)
        ray_dirs = torch.einsum(
            'btij, rj->btri', pixel2camera, self.ray_dirs
        )

        feats = torch.cat([feats, self.ray_emb.expand_as(feats), self.ray_dir_enc(ray_dirs[..., :2])], dim=-1)
        feats = self.ray_encoder(feats)

        if self.training and self.patch_dropout > 0.:
            ind = np.arange(feats.shape[2])
            np.random.shuffle(ind)
            ind = ind[:int(feats.shape[2]*(1-self.patch_dropout))]
            feats = feats[:, :, ind]

        extrinsic = repeat(extrinsic, 'b t u v -> b t hw u v', hw=feats.shape[2]).clone()
        return feats, extrinsic


class GravityVecEncoder(nn.Module):
    def __init__(self, s_out: int, r_out: int, p_out: int):
        super().__init__()
        self.scalars = nn.Parameter(
            torch.randn(1, s_out), requires_grad=True
        )
        self.magnitudes = nn.Parameter(
            torch.randn(r_out), requires_grad=True
        )
        self.positions = nn.Parameter(
            torch.zeros(1, 3*p_out), requires_grad=False
        )

    def forward(self, vec):
        B, _ = vec.shape
        vec = vec.repeat(1, self.magnitudes.shape[0])
        magnitudes = self.magnitudes.repeat_interleave(3).unsqueeze(0)
        feats = torch.cat([
            self.scalars.repeat(B, 1),
            vec * magnitudes,
            self.positions.repeat(B, 1),
        ], dim=-1)

        g_se3 = torch.eye(4).repeat(B, 1, 1).to(vec.device)
        return feats, g_se3


class GripperPoseEncoder(nn.Module):
    def __init__(self, s_out: int, r_out: int, p_out: int, grip_dim: int, hidden_dim: int=256):
        super().__init__()
        self.quat2mtx = RotationTransformer('quaternion', 'matrix')

        self.linear = nn.Sequential(
            nn.Linear(grip_dim + hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, s_out + 3*r_out + 3 * p_out)
        )

        emb = torch.exp(torch.arange(hidden_dim//2) * -np.log(10000) / (hidden_dim //2 - 1 ) )
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        self.register_buffer(
            'emb', emb.view(1, 1, -1),
        )

    def forward(self, pos, quat, opening):
        '''
        pos: B, T, 3
        quat: B, T, 4
        opening: B, T, 2

        returns:
            feats: float tensor of shape B, T, C
            g_se3: float tensor of shape B, T, 4, 4
        '''
        B, T = opening.shape[:2]
        g_se3 = torch.eye(4).repeat(B, T, 1, 1).to(opening.device)
        g_se3[..., :3, :3] = self.quat2mtx.forward(quat[..., [3, 0, 1, 2]])
        g_se3[..., :3, 3] = pos

        feats = self.linear(torch.cat([opening, self.emb.expand(B, T, -1)], dim=-1))

        return feats, g_se3


class RavenObsEnc(nn.Module):
    def __init__(
        self,
        obs_shape_meta,
        n_obs_steps,
        crop_shape,
        img_encoder,
        n_geoms,
        n_scalars,
        num_heads=8,
        adjust_attn_temp=False,
        patch_dropout=0.0,
        attn_dropout=0.0,
        proj_bias=False,
        ff_dropout=0.0,
        n_blocks: int=2,
        attention_fn: nn.Module=GTAttention,
        add_abs_pos_enc: bool = False, # only for non equiv ablation
        ray_rep='se3',
    ):
        super().__init__()
        self.cameras, self.grippers, gravity_obs = parse_obs(obs_shape_meta)

        self.n_obs_steps = n_obs_steps
        self.n_scalars = n_scalars
        self.n_geoms = n_geoms

        self.time_emb = nn.Parameter(
            SinusoidalPosEmb(n_scalars)(torch.arange(1, n_obs_steps+1)).unsqueeze(0).unsqueeze(2),
            requires_grad=False
        )
        self.time_mlp = nn.Linear(2*n_scalars, n_scalars)

        if add_abs_pos_enc:
            self.abs_pos_enc_mlp = nn.Sequential(
                nn.Linear(16, n_scalars + 3*n_geoms),
                nn.GELU(),
                nn.Linear(n_scalars + 3 * n_geoms, n_scalars + 3*n_geoms),
            )
        else:
            self.abs_pos_enc_mlp = None

        self.encoders = nn.ModuleDict()

        rgb_encoder_fn = RGBEncoder if ray_rep == 'se3' else NonSE3RGBEncoder
        for cam in self.cameras:
            self.encoders[cam] = rgb_encoder_fn(
                input_shape=(obs_shape_meta[f'{cam}_image']['shape'][0], *crop_shape),
                img_encoder=img_encoder,
                s_out=n_scalars,
                r_out=n_geoms//2,
                p_out=n_geoms//2,
                patch_dropout=patch_dropout,
                use_depth=any('depth' in k for k in obs_shape_meta.keys())
            )

        for gripper in self.grippers:
            self.encoders[gripper] = GripperPoseEncoder(
                s_out=n_scalars,
                r_out=n_geoms//2,
                p_out=n_geoms//2,
                grip_dim=obs_shape_meta[f'{gripper}_gripper_qpos']['shape'][0]
            )

        if gravity_obs:
            self.encoders['gravity'] = GravityVecEncoder(n_scalars, n_geoms//2, n_geoms//2)

        self.backbone = nn.ModuleList([
            GTEncoderBlock(
                s_dim=n_scalars,
                r_dim=n_geoms//2,
                p_dim=n_geoms//2,
                num_heads=num_heads,
                adjust_attn_temp=adjust_attn_temp,
                attn_dropout=attn_dropout,
                proj_bias=proj_bias,
                ff_dropout=ff_dropout,
                attention_fn=attention_fn,
            ) for _ in range(n_blocks)
        ])

        self.final_layer = nn.Linear(n_scalars+3*n_geoms, n_scalars+3*n_geoms)

    def forward(self, nobs: Dict[str, Tensor]):
        '''
        returns: mv representation of scene, size: B, F, 3
                 s representation of scene, size: B, F
        '''
        token_feats, token_g_se3 = [], []

        # encode images
        for cam in self.cameras:
            img = nobs[f'{cam}_image']
            extrinsic = nobs[f'{cam}_extrinsic']
            intrinsic = nobs[f'{cam}_intrinsic']
            depth = nobs.get(f'{cam}_depth', None)
            new_feats, new_g_se3 = self.encoders[cam](img, extrinsic, intrinsic, depth)
            token_feats.append(new_feats)
            token_g_se3.append(new_g_se3)

        for grip in self.grippers:
            pos = nobs[f'{grip}_eef_pos']
            quat = nobs[f'{grip}_eef_quat']
            opening = nobs[f'{grip}_gripper_qpos']
            new_feats, new_g_se3 = self.encoders[grip](
                pos, quat, opening
            )
            token_feats.append(new_feats.unsqueeze(2))
            token_g_se3.append(new_g_se3.unsqueeze(2))

        token_feats = torch.cat(token_feats, dim=2)
        token_g_se3 = torch.cat(token_g_se3, dim=2)

        scal_feats = token_feats[..., :self.n_scalars]
        t_emb = self.time_emb.expand_as(scal_feats)
        token_feats[..., :self.n_scalars] = self.time_mlp(
            torch.cat([scal_feats, t_emb], dim=3)
        )
        if self.abs_pos_enc_mlp is not None:
            token_feats += self.abs_pos_enc_mlp(rearrange(token_g_se3, 'b t n u v -> b t n (u v)'))

        # ToDo: encode time then fold it in
        token_feats = rearrange(token_feats, 'b t n c -> b (t n) c')
        token_g_se3 = rearrange(token_g_se3, 'b t n u v -> b (t n) u v')

        if 'gravity' in self.encoders:
            gravity_feats, gravity_g_se3 = self.encoders['gravity'](nobs['gravity'])
            token_feats = torch.cat([token_feats, gravity_feats.unsqueeze(1)], dim=1)
            token_g_se3 = torch.cat([token_g_se3, gravity_g_se3.unsqueeze(1)], dim=1)

        for backbone_layer in self.backbone:
            token_feats = backbone_layer(token_feats, token_g_se3)

        token_feats = self.final_layer(token_feats)

        return token_feats, token_g_se3



class TranslationCanonicalizer(nn.Module):
    def __init__(self, s_dim, r_dim, p_dim):
        super().__init__()
        self.s_dim = s_dim
        self.r_dim = r_dim
        self.p_dim = p_dim

    def forward(self, feats, g_se3):
        '''
        feats: token feats of shape B, N, s_dim+3*(r_dim+p_dim)
        g_se3: token group elems of shape B, N, 4, 4

        returns tuple of:
            geom_cond: (B, self.r_dim+self.p_dim, 3)
            scal_cond: (B, self.s_dim)
            offset: (B, 1, 3)
        '''
        s_feats, rp_feats = torch.split(
            feats,
            [self.s_dim, 3*(self.r_dim+self.p_dim)],
            dim=-1
        )
        rp_feats = rearrange(rp_feats, 'b n (c v) -> b n c v', v=3)

        # transform rp_feats
        rp_feats = torch.einsum('bnuv,bncv->bncu', g_se3[..., :3, :3], rp_feats)
        rp_feats[:, :, self.r_dim:] += g_se3[..., :3, 3].unsqueeze(2)

        offset = rp_feats[:, :, self.r_dim:].mean((1, 2), keepdim=True)

        # remove offset from positional feats
        rp_feats[:, :, self.r_dim:] -= offset

        geom_cond = rp_feats.mean(1)
        scal_cond = s_feats.mean(1)

        return geom_cond, scal_cond, offset.squeeze(1)


class SE3Canonicalizer(nn.Module):
    def __init__(self, s_dim, r_dim, p_dim):
        super().__init__()
        self.s_dim = s_dim
        self.r_dim = r_dim
        self.p_dim = p_dim

    def forward(self, feats, g_se3):
        '''
        feats: token feats of shape B, N, s_dim+3*(r_dim+p_dim)
        g_se3: token group elems of shape B, N, narms, 4, 4

        returns tuple of:
            geom_cond: (B, self.r_dim+self.p_dim, 3)
            scal_cond: (B, self.s_dim)
            offset: (B, 1, 3)
        '''
        s_feats, rp_feats = torch.split(
            feats,
            [self.s_dim, 3*(self.r_dim+self.p_dim)],
            dim=-1
        )
        rp_feats = rearrange(rp_feats, 'b n (a c v) -> b n a c v', a=g_se3.shape[2], v=3)

        # transform rp_feats
        rp_feats = torch.einsum('bnauv,bnacv->bnacu', g_se3[..., :3, :3], rp_feats)
        rp_feats[..., self.r_dim//2:, :] += g_se3[..., :3, 3].unsqueeze(-2)

        feats = torch.cat([s_feats, rp_feats.flatten(2)], dim=-1)

        cond = feats.mean(1)

        return cond
