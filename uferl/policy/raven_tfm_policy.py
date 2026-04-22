from typing import Dict, Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from uferl.model.common.normalizer import LinearNormalizer
from uferl.policy.base_image_policy import BaseImagePolicy
from uferl.model.common.rotation_transformer import RotationTransformer

from uferl.model.diffpo.positional_embedding import SinusoidalPosEmb
from uferl.model.raven.tfm_decoders import RavenTfmDecoder
from uferl.model.raven.layers import GTAttention, NonGTAttention
from uferl.model.raven.encoder import RavenObsEnc
from uferl.model.raven.augmentation import Augmenter


def get_timesteps(schedule: str, k_steps: int, exp_scale: float = 1.0):
    t = torch.linspace(0, 1, k_steps + 1)[:-1]
    if schedule == "linear":
        dt = torch.ones(k_steps) / k_steps
    elif schedule == "cosine":
        dt = torch.cos(t * torch.pi) + 1
        dt /= torch.sum(dt)
    elif schedule == "exp":
        dt = torch.exp(-t * exp_scale)
        dt /= torch.sum(dt)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    t0 = torch.cat((torch.zeros(1), torch.cumsum(dt, dim=0)[:-1]))
    return t0, dt


class BaseRavenPolicy(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        # task params
        horizon,
        n_action_steps,
        n_obs_steps,
        # augmentation
        augmenter: Augmenter,
        # arch
        img_encoder: torch.nn.Module,
        enc_n_geoms=64,
        enc_n_scalars=64,
        enc_n_blocks=2,
        enc_n_heads=8,
        enc_adjust_attn_temp=False,
        enc_patch_dropout=0.0,
        enc_attn_dropout=0.0,
        enc_proj_bias=False,
        enc_ff_dropout=0.0,
        enc_attn_fn=GTAttention,
        enc_add_abs_pos_enc=False,
        enc_ray_rep='se3',
        add_gravity_vec=False,
        dec_n_geoms=64,
        dec_n_scalars=64,
        dec_n_blocks=8,
        dec_n_heads=8,
        dec_attn_dropout=0.0,
        dec_attn_fn=GTAttention,
        dec_add_abs_pos_enc=False,
        loss_weights=dict(xyz=10.0, rot6d=10.0, grip=1.0),
        diffusion_step_embed_dim=128,
        num_inference_steps=10,
        **kwargs
    ):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        self.n_arms = 2 if action_shape[0] > 10 else 1
        self.action_dim = action_shape[0]
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.kwargs = kwargs
        self.n_obs_steps = n_obs_steps
        self.num_inference_steps = num_inference_steps
        obs_shape_meta = shape_meta['obs']

        self.enc_n_scalars = enc_n_scalars
        self.enc_n_geoms = enc_n_geoms

        if add_gravity_vec:
            obs_shape_meta['gravity'] = {'shape': [3]}
            self.gravity_vec = torch.nn.Parameter(
                torch.FloatTensor([0, 0, 1.]).unsqueeze(0),
                requires_grad=False
            )
        else:
            self.gravity_vec = None

        self.augmenter = augmenter

        self.enc = RavenObsEnc(
            obs_shape_meta=obs_shape_meta,
            n_obs_steps=n_obs_steps,
            crop_shape=augmenter.crop_shape,
            img_encoder=img_encoder,
            n_geoms=enc_n_geoms,
            n_scalars=enc_n_scalars,
            n_blocks=enc_n_blocks,
            num_heads=enc_n_heads,
            adjust_attn_temp=enc_adjust_attn_temp,
            attn_dropout=enc_attn_dropout,
            patch_dropout=enc_patch_dropout,
            proj_bias=enc_proj_bias,
            ff_dropout=enc_ff_dropout,
            attention_fn=enc_attn_fn,
            add_abs_pos_enc=enc_add_abs_pos_enc,
            ray_rep=enc_ray_rep,
        )

        self.decoder = RavenTfmDecoder(
            n_scalars=dec_n_scalars,
            n_geoms=dec_n_geoms,
            horizon=horizon,
            n_blocks=dec_n_blocks,
            num_heads=dec_n_heads,
            n_arms=self.n_arms,
            attention_fn=dec_attn_fn,
            add_abs_pos_enc=dec_add_abs_pos_enc,
        )

        if dec_n_geoms != enc_n_geoms or dec_n_scalars != enc_n_scalars:
            self.cond_proj = nn.Sequential(
                nn.Linear(enc_n_scalars + 3*enc_n_geoms, dec_n_scalars + 3*dec_n_geoms),
                # nn.RMSNorm(dec_n_scalars + 3*dec_n_geoms),
            )
        else:
            self.cond_proj = None

        self.proj_in = nn.Sequential(
            nn.Linear(self.action_dim//self.n_arms + 128 + diffusion_step_embed_dim + 8, dec_n_scalars + 3*dec_n_geoms),
            # nn.RMSNorm(dec_n_scalars + 3*dec_n_geoms),
        )

        self.action_time_emb = nn.Sequential(
            SinusoidalPosEmb(128),
            nn.Linear(128, 128 * 4),
            nn.Mish(),
            nn.Linear(4*128, 128),
        )
        self.arm_emb = nn.Sequential(
            SinusoidalPosEmb(8),
            nn.Linear(8, 8 * 4),
            nn.Mish(),
            nn.Linear(4*8, 8),
        )

        self.proj_out = nn.Sequential(
            nn.RMSNorm(dec_n_scalars + 3*dec_n_geoms),
            nn.Linear(dec_n_scalars + 3*dec_n_geoms, self.action_dim//self.n_arms),
        )

        self.timestep_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(4*diffusion_step_embed_dim, diffusion_step_embed_dim),
        )

        self.normalizer = LinearNormalizer()
        self.sixd2mtx = RotationTransformer('rotation_6d', 'matrix')
        self.quat2mtx = RotationTransformer('quaternion', 'matrix')
        self.l_w = loss_weights

        print(f"Enc params: {sum(p.numel() for p in self.enc.parameters())*1e-6:.2f}M")
        print(f"Dec params: {sum(p.numel() for p in self.decoder.parameters())*1e-6:.2f}M")


    def get_optimizer(
            self,
            weight_decay: float,
            learning_rate: float,
            betas: Tuple[float, float],
            eps: float
        ) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            weight_decay=weight_decay,
            lr=learning_rate,
            betas=betas,
            eps=eps
        )
        return optimizer

    def set_normalizer(self, normalizer: LinearNormalizer):
        action_stats = normalizer['action'].get_input_stats()
        mean_action_stats = action_stats['mean'].reshape(self.n_arms, -1).mean(0)
        self.ws_center = nn.Parameter(
            mean_action_stats[:3], requires_grad=False
        )

        max_action_stats = action_stats['max'].reshape(self.n_arms, -1).max(0)[0]
        min_action_stats = action_stats['min'].reshape(self.n_arms, -1).min(0)[0]
        self.ws_scale = 0.5 * (max_action_stats[:3] - min_action_stats[:3]).max().item()

        rel_action_stats = normalizer['rel_action'].get_input_stats()
        max_rel_action_stats = rel_action_stats['max'].reshape(self.n_arms, -1)[:, :3]
        min_rel_action_stats = rel_action_stats['min'].reshape(self.n_arms, -1)[:, :3]

        self.rel_action_scale = nn.Parameter(
            0.5 * (max_rel_action_stats - min_rel_action_stats).unsqueeze(0).unsqueeze(0),
            requires_grad=False
        )

        self.normalizer.load_state_dict(normalizer.state_dict())

    def _norm_obs(self, obs_dict):
        for k, v in obs_dict.items():
            if k.endswith('_image'):
                obs_dict[k] = 2.0 * v - 1.0
            elif k.endswith('_depth'):
                obs_dict[k] = v / self.ws_scale
            elif k.endswith('_intrinsic'):
                obs_dict[k] *= self.ws_scale
            elif k.endswith('_extrinsic'):
                obs_dict[k][..., :3, 3] = (v[..., :3, 3] - self.ws_center) / self.ws_scale
            elif k.endswith('_pos'):
                obs_dict[k][..., :3] = (v[..., :3] - self.ws_center) / self.ws_scale

        return obs_dict

    def _norm_action(self, action):
        '''action: (B, horizon, narms, action_dim)'''
        naction = action
        naction[..., :3] = (naction[..., :3] - self.ws_center) / self.ws_scale
        return naction

    def _denorm_action(self, naction):
        '''naction: (B, horizon, narms, action_dim)'''
        action = naction
        action[..., :3] = action[..., :3] * self.ws_scale + self.ws_center
        return action

    def split_action(self, action):
        '''
        action: B, T, narms*action_dim
        returns: B, T, narms, action_dim
        '''
        B, T = action.shape[:2]
        return action.reshape(B, T, self.n_arms, -1)

    def tfm_action(self, action, tfm):
        '''
        action: B, T, narms, action_dim
        tfm B, T, narms, 4, 4
        ret: tfm @ action
        '''
        action_pos, action_rot, action_grip = torch.split(action, [3, 6, action.shape[-1] - 9], dim=-1)

        action_pos_homog = torch.cat([
            action_pos,
            torch.ones(
                (*action_pos.shape[:-1], 1), dtype=action_pos.dtype, device=action_pos.device
            )
        ], dim=-1)
        new_action_pos = torch.einsum('btaij,btaj->btai', tfm[:, :, :, :3, :], action_pos_homog)

        action_rotmtx = self.sixd2mtx.forward(action_rot)
        new_action_rotmtx = torch.einsum('btaij,btajk->btaik', tfm[:, :, :, :3, :3], action_rotmtx)
        new_action_rot = self.sixd2mtx.inverse(new_action_rotmtx)

        new_action = torch.cat([new_action_pos, new_action_rot, action_grip], dim=-1)
        return new_action

    def get_ee_pose(self, nobs, B, T):
        g_se3 = torch.eye(4).repeat(B, self.n_arms, 1, 1).to(self.device)
        for r in range(self.n_arms):
            # take most recent timestep
            pos = nobs[f'robot{r}_eef_pos'][:, -1]
            quat = nobs[f'robot{r}_eef_quat'][:, -1]

            g_se3[:, r, :3, :3] = self.quat2mtx.forward(quat[:, [3, 0, 1, 2]])
            g_se3[:, r, :3, 3] = pos

        # B, T, narms, 4, 4
        g_se3 = g_se3.unsqueeze(1).expand(-1, T, -1, -1, -1)
        return g_se3


class RavenFMPolicy_rel(BaseRavenPolicy):
    def __init__(
        self,
        shape_meta: dict,
        # task params
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps,
        # augmentation
        augmenter: Augmenter,
        # arch
        img_encoder: torch.nn.Module,
        enc_n_geoms=64,
        enc_n_scalars=64,
        enc_n_blocks=2,
        enc_n_heads=8,
        enc_adjust_attn_temp=False,
        enc_patch_dropout=0.0,
        enc_attn_dropout=0.0,
        enc_proj_bias=False,
        enc_ff_dropout=0.0,
        enc_attn_fn=GTAttention,
        enc_add_abs_pos_enc=False,
        enc_ray_rep='se3',
        add_gravity_vec=False,
        dec_n_geoms=64,
        dec_n_scalars=64,
        dec_n_blocks=8,
        dec_n_heads=8,
        dec_attn_dropout=0.0,
        dec_attn_fn=GTAttention,
        dec_add_abs_pos_enc=False,
        loss_weights=dict(xyz=10.0, rot6d=10.0, grip=1.0),
        diffusion_step_embed_dim=128,

        pos_emb_scale=20.0,
        noise_scale=1.0,
        flow_schedule="linear",
        exp_scale=4.0,
        # these args
        **kwargs
    ):
        super().__init__(
            shape_meta=shape_meta,
            # task params
            horizon=horizon,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            # augmentation
            augmenter=augmenter,
            # arch
            img_encoder=img_encoder,
            enc_n_geoms=enc_n_geoms,
            enc_n_scalars=enc_n_scalars,
            enc_n_blocks=enc_n_blocks,
            enc_n_heads=enc_n_heads,
            enc_adjust_attn_temp=enc_adjust_attn_temp,
            enc_patch_dropout=enc_patch_dropout,
            enc_attn_dropout=enc_attn_dropout,
            enc_proj_bias=enc_proj_bias,
            enc_ff_dropout=enc_ff_dropout,
            enc_attn_fn=enc_attn_fn,
            enc_add_abs_pos_enc=enc_add_abs_pos_enc,
            enc_ray_rep=enc_ray_rep,
            add_gravity_vec=add_gravity_vec,
            dec_n_geoms=dec_n_geoms,
            dec_n_scalars=dec_n_scalars,
            dec_n_blocks=dec_n_blocks,
            dec_n_heads=dec_n_heads,
            dec_attn_dropout=dec_attn_dropout,
            dec_attn_fn=dec_attn_fn,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            num_inference_steps=num_inference_steps,
            loss_weights=loss_weights,
        )
        self.pos_emb_scale = pos_emb_scale
        self.noise_scale = noise_scale
        self.flow_schedule = flow_schedule
        self.exp_scale = exp_scale

    @torch.no_grad()
    def predict_action(self, obs_dict):
        assert 'past_action' not in obs_dict # not implemented yet

        if self.gravity_vec is not None:
            obs_dict['gravity'] = self.gravity_vec

        nobs = self._norm_obs(obs_dict)
        nobs = self.augmenter(nobs)[0]

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        # build input
        ee_g_se3 = self.get_ee_pose(nobs, B, T)
        token_feats, token_g_se3 = self.enc(nobs)
        if self.cond_proj is not None:
            token_feats = self.cond_proj(token_feats)

        # run sampling
        t0, dt = get_timesteps(
            self.flow_schedule,
            self.num_inference_steps,
            exp_scale=self.exp_scale
        )
        trajectory = torch.randn(size=(B, T, self.n_arms, Da//self.n_arms), device=self.device)

        for i in range(self.num_inference_steps):
            timesteps = self.pos_emb_scale * t0[i] * torch.ones((B), device=self.device)

            pred_vel = self.proj_out(
                self.decoder(
                    self.proj_in(torch.cat([
                        trajectory,
                        self.action_time_emb(torch.arange(T).to(self.device).unsqueeze(1)).expand(B, -1, self.n_arms, -1),
                        self.timestep_encoder(timesteps.unsqueeze(1).unsqueeze(1)).expand(-1, T, self.n_arms, -1),
                        self.arm_emb(torch.arange(self.n_arms).to(self.device)).expand(B, T, -1, -1),
                    ], dim=-1)).flatten(1, 2),
                    ee_g_se3.flatten(1, 2),
                    token_feats,
                    token_g_se3,
                )
            )
            pred_vel = pred_vel.view(B, T, self.n_arms, -1)

            trajectory = trajectory.detach().clone() + pred_vel * dt[i]

        # convert relative to absolute
        trajectory[:, :, :, :3] *= self.rel_action_scale / self.ws_scale
        trajectory = self.tfm_action(trajectory, ee_g_se3)

        # unnormalize prediction
        action_pred = self._denorm_action(trajectory).flatten(-2)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    def compute_loss(self, batch, z0=None, t=None):
        assert 'valid_mask' not in batch

        obs = batch['obs']
        action = self.split_action(batch['action'])
        B, T = action.shape[:2]

        if self.gravity_vec is not None:
            obs['gravity'] = self.gravity_vec

        nobs = self._norm_obs(obs)
        nactions = self._norm_action(action)

        #augment on the normalized so its centered in workspace!
        nobs, nactions = self.augmenter(nobs, nactions)

        ee_g_se3 = self.get_ee_pose(nobs, B, T).clone()
        rel_trajectory = self.tfm_action(nactions, torch.linalg.inv(ee_g_se3))
        rel_trajectory[:, :, :, :3] *= self.ws_scale / self.rel_action_scale

        token_feats, token_g_se3 = self.enc(nobs)
        if self.cond_proj is not None:
            token_feats = self.cond_proj(token_feats)

        z0 = torch.randn(rel_trajectory.shape, device=self.device)
        t = torch.rand((B, 1, 1, 1), device=self.device)
        z1 = rel_trajectory

        z_t = t * z1 + (1.0 - t) * z0
        target_vel = z1 - z0
        timesteps = t.squeeze() * self.pos_emb_scale

        pred_vel = self.proj_out(
            self.decoder(
                self.proj_in(torch.cat([
                    z_t,
                    self.action_time_emb(torch.arange(T).to(self.device).unsqueeze(1)).expand(B, -1, self.n_arms, -1),
                    self.timestep_encoder(timesteps.unsqueeze(1).unsqueeze(1)).expand(-1, T, self.n_arms, -1),
                    self.arm_emb(torch.arange(self.n_arms).to(self.device)).expand(B, T, -1, -1),
                ], dim=-1)).flatten(1, 2),
                ee_g_se3.flatten(1, 2),
                token_feats,
                token_g_se3,
            )
        ).view(B, T, self.n_arms, -1)

        loss_xyz = F.mse_loss(pred_vel[..., :3], target_vel[..., :3])
        loss_rot6d = F.mse_loss(pred_vel[..., 3:9], target_vel[..., 3:9])
        loss_grip = F.mse_loss(pred_vel[..., 9:], target_vel[..., 9:])

        loss = (
            self.l_w["xyz"] * loss_xyz
            + self.l_w["rot6d"] * loss_rot6d
            + self.l_w["grip"] * loss_grip
        )
        return loss
