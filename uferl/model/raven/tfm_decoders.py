
import torch
from torch import nn
from einops import rearrange
from uferl.model.raven.layers import GTDecoderBlock, GTAttention
from uferl.model.diffpo.positional_embedding import SinusoidalPosEmb


class RavenTfmDecoder(nn.Module):
    def __init__(
        self,
        n_scalars,
        n_geoms,
        horizon: int,
        n_blocks: int,
        num_heads: int=8,
        n_arms: int=1,
        adjust_attn_temp: bool=False,
        attn_dropout: float=0.0,
        proj_bias: bool=True,
        ff_dropout: float=0.0,
        norm_fn: nn.Module=nn.RMSNorm,
        norm_first: bool=True,
        attention_fn: nn.Module=GTAttention,
        add_abs_pos_enc: bool=False,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            GTDecoderBlock(
                s_dim=n_scalars,
                r_dim=n_geoms//2,
                p_dim=n_geoms//2,
                num_heads=num_heads,
                adjust_attn_temp=adjust_attn_temp,
                attn_dropout=attn_dropout,
                proj_bias=proj_bias,
                ff_dropout=ff_dropout,
                norm_fn=norm_fn,
                norm_first=norm_first,
                attention_fn=attention_fn,
            ) for _ in range(n_blocks)
        ])

        t = torch.arange(horizon).expand(n_arms, -1).flatten()
        a, b = torch.meshgrid(t, t, indexing='xy')
        causal_mask = a > b
        # causal_mask[:, -1] = False # no masking on time token
        self.register_buffer(
            'causal_mask', causal_mask.unsqueeze(0).unsqueeze(0),
        )

        if add_abs_pos_enc:
            self.abs_pos_enc_mlp = nn.Sequential(
                nn.Linear(16, n_scalars + 3*n_geoms),
                nn.GELU(),
                nn.Linear(n_scalars + 3 * n_geoms, n_scalars + 3*n_geoms),
            )
        else:
            self.abs_pos_enc_mlp = None

        # self.apply(self._init_weights)

    # def _init_weights(self, module):
        # if isinstance(module, (nn.Linear, nn.Embedding)):
            # torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # if isinstance(module, nn.Linear) and module.bias is not None:
                # torch.nn.init.zeros_(module.bias)
        # elif isinstance(module, nn.LayerNorm):
            # torch.nn.init.zeros_(module.bias)
            # torch.nn.init.ones_(module.weight)
        # elif isinstance(module, nn.RMSNorm):
            # torch.nn.init.ones_(module.weight)

    def forward(self, sample, sample_g_se3, cond, cond_g_se3, time_cond=None, time_g_se3=None):
        # if time_cond is not None:
            # sample = torch.cat([sample, time_cond], dim=1)
            # sample_g_se3 = torch.cat([sample_g_se3, time_g_se3], dim=1)

        if self.abs_pos_enc_mlp is not None:
            cond_abs_pos = self.abs_pos_enc_mlp(
                rearrange(cond_g_se3, '... a b -> ... (a b)')
            )
            cond = cond + cond_abs_pos

        for block in self.blocks:
            sample = block(
                sample, sample_g_se3, cond, cond_g_se3,
                attn_mask=self.causal_mask
            )

        # if time_cond is not None:
            # sample = sample[:, :-1]
            # sample_g_se3 = sample_g_se3[:, :-1]


        return sample

