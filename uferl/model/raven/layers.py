from typing import Optional

import numpy as np
import torch
from torch import nn, Tensor
from einops import rearrange
from torchvision import models as vision_models

from uferl.common.pytorch_util import replace_submodules


class TemperatureAdjustableSoftmax(nn.Module):
    def __init__(self, init_tau=1.0, dim=-1):
        super().__init__()
        self.tau = nn.Parameter(torch.Tensor([init_tau]))
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, x):
        return self.softmax(x/self.tau)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float=0.0):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0. else nn.Identity(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout) if dropout > 0. else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)


class ResNet18(nn.Module):
    def __init__(self, pretrained=False, use_groupnorm=True):
        super().__init__()
        net = vision_models.resnet18(pretrained=pretrained)
        net.maxpool = nn.Identity()

        self.resnet = torch.nn.Sequential(*list(net.children())[:-2])

        if use_groupnorm:
            replace_submodules(
                root_module=self.resnet,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16,
                    num_channels=x.num_features)
            )

    def forward(self, x):
        y = self.resnet(x)
        return y


class ResNet34(nn.Module):
    def __init__(self, pretrained=False, use_groupnorm=True):
        super().__init__()
        net = vision_models.resnet34(pretrained=pretrained)
        net.maxpool = nn.Identity()

        self.resnet = torch.nn.Sequential(*list(net.children())[:-2])

        if use_groupnorm:
            replace_submodules(
                root_module=self.resnet,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16, num_channels=x.num_features
                ),
            )

    def forward(self, x):
        y = self.resnet(x)
        return y


class GTAttention(nn.Module):
    def __init__(
        self,
        s_dim,
        r_dim,
        p_dim,
        num_heads=4,
        proj_bias=False,
        attn_dropout=0.0,
        adjust_attn_temp=False,
    ):
        super().__init__()
        self.s_dim = s_dim
        self.r_dim = r_dim
        self.p_dim = p_dim

        self.num_heads = num_heads
        full_dim = s_dim + 3*r_dim + 3*p_dim

        # self.scal_weight = nn.Parameter(torch.FloatTensor((1,)))
        # self.rot_weight = nn.Parameter(torch.FloatTensor((1,)))
        # self.pos_weight = nn.Parameter(torch.FloatTensor((1,)))

        self.attn_dropout = nn.Dropout(attn_dropout) if attn_dropout > 0. else nn.Identity()

        self.attend = TemperatureAdjustableSoftmax(dim=-1) if adjust_attn_temp else nn.Softmax(dim=-1)

        self.to_q = nn.Linear(full_dim, full_dim, bias=proj_bias)
        self.to_kv = nn.Linear(full_dim, 2*full_dim, bias=proj_bias)

        self.to_out = nn.Linear(full_dim, full_dim, bias=proj_bias)

    def apply_tfm(self, x, g):
        '''
        x: B, H, N, C
        g: B, N, 4, 4
        '''
        assert x.shape[0] == g.shape[0]
        x_s, x_rp = torch.split(x, [self.s_dim//self.num_heads, (3*self.r_dim + 3*self.p_dim)//self.num_heads], dim=-1)

        x_rp = rearrange(x_rp, 'b h n (c v) -> b h n c v', v=3)

        x_rp = torch.einsum('bnuv,bhncv->bhncu', g[..., :3, :3], x_rp)
        x_rp[:, :, :, -self.p_dim//self.num_heads:] += g[..., :3, 3].unsqueeze(1).unsqueeze(3)
        x = torch.cat([x_s, x_rp.flatten(-2)], dim=-1)
        return x

    def forward(self, x, g_x, y=None, g_y=None, attn_mask=None):
        '''
        x: float tensor of shape B, N, s_out+3*r_out+3*p_out
        g_x: float tensor of shape B, N, 4, 4
        y: float tensor of shape B, M, s_out+3*r_out+3*p_out
        g_y: float tensor of shape B, M, 4, 4

        ToDo: need to transform V and O if doing cross attention
        '''
        q = self.to_q(x)

        if y is None:
            y = x
            g_y = g_x

        k, v = torch.split(self.to_kv(y), y.shape[-1], dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))

        q = self.apply_tfm(q, g_x)
        k = self.apply_tfm(k, g_y)
        v = self.apply_tfm(v, g_y)

        if self.p_dim > 0:
            split = (self.s_dim + 3 * self.r_dim)//self.num_heads
            (q_sr, q_p), (k_sr, k_p) = map(
                lambda t: torch.split(t, split, dim=-1), (q, k)
            )
            dotprod_sim = q_sr @ k_sr.transpose(-1, -2)
            euclid_sim = q_p @ k_p.transpose(-1, -2) \
                            - 0.5 * q_p.pow(2).sum(-1).unsqueeze(-1) \
                            - 0.5 * k_p.pow(2).sum(-1).unsqueeze(-2)
            full_sim = dotprod_sim / np.sqrt(q_sr.shape[-1]) \
                        + euclid_sim / np.sqrt(q_p.shape[-1])
        else:
            full_sim = q @ k.transpose(-1, -2) / np.sqrt(q.shape[-1])

        full_sim = self.attn_dropout(full_sim)

        if attn_mask is not None:
            full_sim = full_sim.masked_fill(attn_mask == True, float('-inf'))

        attn = torch.softmax(full_sim, dim=-1)

        out = attn @ v

        out = self.apply_tfm(out, torch.linalg.inv(g_x))
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class GTDecoderBlock(nn.Module):
    def __init__(
        self,
        s_dim,
        r_dim,
        p_dim,
        num_heads:int=4,
        adjust_attn_temp: bool=False,
        attn_dropout: float=0.0,
        proj_bias: bool=True,
        ff_dropout: float=0.0,
        ff_factor: int=4,
        norm_fn: nn.Module=nn.RMSNorm,
        norm_first: bool=True,
        attention_fn: nn.Module=GTAttention,
    ):
        super().__init__()
        full_dim = s_dim + 3*(r_dim + p_dim)
        # ToDo: should there be dropout too?
        self.norm1 = norm_fn(full_dim)
        self.norm2 = norm_fn(full_dim)
        self.norm3 = norm_fn(full_dim)
        self.norm_first = norm_first

        self.attn1 = attention_fn(
            s_dim, r_dim, p_dim,
            num_heads=num_heads,
            proj_bias=proj_bias,
            attn_dropout=attn_dropout,
            adjust_attn_temp=adjust_attn_temp,
        )
        self.attn2 = attention_fn(
            s_dim, r_dim, p_dim,
            num_heads=num_heads,
            proj_bias=proj_bias,
            attn_dropout=attn_dropout,
            adjust_attn_temp=adjust_attn_temp,
        )
        self.ff = FeedForward(
            full_dim, ff_factor*full_dim, dropout=ff_dropout
        )

    def forward(self, x, g_x, y, g_y, attn_mask=None):
        if self.norm_first:
            x = x + self.attn1(self.norm1(x), g_x, attn_mask=attn_mask)
            x = x + self.attn2(self.norm2(x), g_x, y, g_y)
            x = x + self.ff(self.norm3(x))
        else:
            x = self.norm1(x + self.attn1(x, g), attn_mask=attn_mask)
            x = self.norm2(x + self.attn2(x, g_x, y, g_y))
            x = self.norm3(x + self.ff(x))

        return x


class GTEncoderBlock(nn.Module):
    def __init__(
        self,
        s_dim,
        r_dim,
        p_dim,
        num_heads:int=4,
        adjust_attn_temp: bool=False,
        attn_dropout: float=0.0,
        proj_bias: bool=True,
        ff_dropout: float=0.0,
        ff_factor: int=2,
        norm_fn: nn.Module=nn.RMSNorm,
        norm_first: bool=True,
        attention_fn: nn.Module=GTAttention,
    ):
        super().__init__()
        full_dim = s_dim + 3*(r_dim + p_dim)
        self.norm1 = norm_fn(full_dim)
        self.norm2 = norm_fn(full_dim)
        self.norm_first = norm_first

        self.attn = attention_fn(
            s_dim, r_dim, p_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_bias=proj_bias,
            adjust_attn_temp=adjust_attn_temp,
        )
        self.ff = FeedForward(
            full_dim, ff_factor*full_dim, dropout=ff_dropout
        )

    def forward(self, x, g):
        if self.norm_first:
            x = x + self.attn(self.norm1(x), g)
            x = x + self.ff(self.norm2(x))
        else:
            x = self.norm1(x + self.attn(x, g))
            x = self.norm2(x + self.ff(x))
        return x


class NonGTAttention(nn.Module):
    '''Implementation of normal dot-product attention that follows the same inputs/outputs as GTAttention so
    it can be swapped in easily for ablation experiments
    '''
    def __init__(
        self,
        s_dim,
        r_dim,
        p_dim,
        num_heads=4,
        proj_bias=False,
        attn_dropout=0.0,
        adjust_attn_temp=False,
    ):
        super().__init__()
        self.s_dim = s_dim
        self.r_dim = r_dim
        self.p_dim = p_dim

        self.num_heads = num_heads
        full_dim = s_dim + 3*r_dim + 3*p_dim

        self.attn_dropout = nn.Dropout(attn_dropout) if attn_dropout > 0. else nn.Identity()

        self.attend = TemperatureAdjustableSoftmax(dim=-1) if adjust_attn_temp else nn.Softmax(dim=-1)

        self.to_q = nn.Linear(full_dim, full_dim, bias=proj_bias)
        self.to_kv = nn.Linear(full_dim, 2*full_dim, bias=proj_bias)

        self.to_out = nn.Linear(full_dim, full_dim, bias=proj_bias)

    def forward(self, x, g_x, y=None, g_y=None, attn_mask=None):
        '''
        x: float tensor of shape B, N, s_out+3*r_out+3*p_out
        g_x: float tensor of shape B, N, 4, 4
        y: float tensor of shape B, M, s_out+3*r_out+3*p_out
        g_y: float tensor of shape B, M, 4, 4

        ToDo: need to transform V and O if doing cross attention
        '''
        q = self.to_q(x)

        if y is None:
            y = x
            g_y = g_x

        k, v = torch.split(self.to_kv(y), y.shape[-1], dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))

        full_sim = q @ k.transpose(-1, -2) / np.sqrt(q.shape[-1])

        full_sim = self.attn_dropout(full_sim)

        if attn_mask is not None:
            full_sim = full_sim.masked_fill(attn_mask == True, float('-inf'))

        attn = torch.softmax(full_sim, dim=-1)

        out = attn @ v

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)



if __name__ == "__main__":
    from scipy.spatial.transform import Rotation
    def sample_se3(*shape):
        T = torch.eye(4).repeat(*shape, 1, 1)
        T[..., :3, :3] = torch.from_numpy(
            Rotation.random(np.prod(shape)).as_matrix()
        ).view(*shape, 3, 3)
        T[..., :3, 3].uniform_(-1, 1)
        return T

    #test equivariance here
    B, N, C_s, C_r, C_p = 12, 18, 11, 5, 7
    M = 4
    H = 3

    x = torch.rand((B, N, H*(C_s + 3*C_r + 3*C_p)))
    g_x = sample_se3(B, N)

    y = torch.rand((B, M, H*(C_s + 3*C_r + 3*C_p)))
    g_y = sample_se3(B, M)

    module = GTAttention(
        H*C_s, H*C_r, H*C_p, num_heads=H
    )

    out = module(x, g_x, y, g_y)

    G = sample_se3(B, 1)

    tfm_out = module(x, G @ g_x, y, G @ g_y)
    print((tfm_out - out).abs().max())
    print(out[0, 0, :3])
    print(tfm_out[0, 0, :3])
