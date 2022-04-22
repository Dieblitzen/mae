# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.vision_transformer import PatchEmbed
from util.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid


class ChannelsVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, channel_embed=256, **kwargs):
        super().__init__(**kwargs)
        img_size = kwargs['img_size']
        patch_size = kwargs['patch_size']
        in_c = kwargs['in_chans']
        embed_dim = kwargs['embed_dim']

        self.patch_embed = PatchEmbed(img_size, patch_size, 1, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Positional and channel embed
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim - channel_embed))
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.channel_embed = nn.Parameter(torch.zeros(1, in_c, channel_embed))
        channel_embed = get_1d_sincos_pos_embed_from_grid(self.channel_embed.shape[-1], torch.arange(in_c).numpy())
        self.channel_embed.data.copy_(torch.from_numpy(channel_embed).float().unsqueeze(0))

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        # x is (N, C, H, W)
        b, c, h, w = x.shape

        # Call to contiguous to ensure correct reshaping
        x_inp = x.view(b * c, 1, h, w)
        x_embed = self.patch_embed(x_inp).contiguous()  # (N*C, L, D)
        _, L, D = x_embed.shape
        x = x_embed.view(b, c, L, D)  # (N, C, L, D)

        # add channel embed
        channel_embed = self.channel_embed.unsqueeze(2)  # (1, c, 1, cD)
        pos_embed = self.pos_embed[:, 1:, :].unsqueeze(1)  # (1, 1, L, pD)

        # Channel embed same across (x,y) position, and pos embed same across channel (c)
        channel_embed = channel_embed.expand(-1, -1, pos_embed.shape[2], -1)  # (1, c, L, cD)
        pos_embed = pos_embed.expand(-1, channel_embed.shape[1], -1, -1)  # (1, c, L, pD)
        pos_channel = torch.cat((pos_embed, channel_embed), dim=-1)  # (1, c, L, D)

        # add pos embed w/o cls token
        x = x + pos_channel  # (N, c, L, D)
        x = x.view(b, -1, D)  # (N, c*L, D)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.pos_embed[:, :1, :] + self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (N, 1 + c*L, D)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def vit_base_patch16(**kwargs):
    model = ChannelsVisionTransformer(
        patch_size=28, channel_embed=256, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = ChannelsVisionTransformer(
        patch_size=28, channel_embed=256, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = ChannelsVisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model