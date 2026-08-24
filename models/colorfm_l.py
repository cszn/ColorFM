# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from functools import partial
import math
import multiprocessing
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from lightning.pytorch.utilities.rank_zero import rank_zero_warn
from torch.nn.init import trunc_normal_
from models.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block
import numpy as np
import torch.nn.functional as F

try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    memory_efficient_attention = None

if (
    memory_efficient_attention is None
    and multiprocessing.current_process().name == "MainProcess"
):
    rank_zero_warn(
        "xFormers is not available; using the PyTorch attention fallback."
    )

# logger = logging.getLogger("dinov2")


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = np.linspace(0, drop_path_rate, depth).tolist()  # stochastic depth decay rule

        if ffn_layer == "mlp":
            # logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            # logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            # logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        # self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        # if masks is not None:
            # x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class ParamGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(384 * 2, 384 * 2, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(384 * 2, 384, 3, 2, 1),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.mat_dim = 16

        self.layer1 = nn.Sequential(
            nn.Linear(384, 256),
            nn.SiLU(),
            nn.Linear(256, self.mat_dim * 4) 
        )
        self.layer2 = nn.Sequential(
            nn.Linear(384, 256),
            nn.SiLU(),
            nn.Linear(256, self.mat_dim * self.mat_dim)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(384, 256),
            nn.SiLU(),
            nn.Linear(256, self.mat_dim * self.mat_dim) 
        )

        self.layer4 = nn.Sequential(
            nn.Linear(384, 256),
            nn.SiLU(),
            nn.Linear(256, self.mat_dim * 3) 
        )

    def forward(self, f, shift):
        f = f.permute(0, 2, 1).reshape(-1, 384, 16, 16).contiguous()
        shift = shift.permute(0, 2, 1).reshape(-1, 384, 16, 16).contiguous()
        f = torch.cat([f, shift], dim=1)
        feat = self.gap(self.fuse(f))  
        feat = feat.flatten(1)

        kernel1 = self.layer1(feat)
        kernel1 = kernel1.view(-1, 4, self.mat_dim) 

        kernel2 = self.layer2(feat)
        kernel2 = kernel2.view(-1, self.mat_dim, self.mat_dim)

        kernel3 = self.layer3(feat) 
        kernel3 = kernel3.view(-1, self.mat_dim, self.mat_dim)

        kernel4 = self.layer4(feat)
        kernel4 = kernel4.view(-1, self.mat_dim, 3) 
        
        return kernel1, kernel2, kernel3, kernel4

class CrossAttentionGenerator(nn.Module):

    def __init__(
        self, base_channels=8, attn_channels=32
    ):
        super(CrossAttentionGenerator, self).__init__()

        self.q = nn.Linear(base_channels, attn_channels)
        self.k = nn.Linear(base_channels, attn_channels)
        self.v = nn.Linear(base_channels, attn_channels)

        self.num_heads = 6
        head_dim = attn_channels // self.num_heads
        self.out_proj = nn.Linear(attn_channels, base_channels)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, content, style):
        B, N, C = content.shape
        q = self.q(content).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k(style).reshape(B, N, self.num_heads, C // self.num_heads)
        v = self.v(style).reshape(B, N, self.num_heads, C // self.num_heads)

        if memory_efficient_attention is not None and content.is_cuda:
            x = memory_efficient_attention(q, k, v)
        else:
            x = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
            ).transpose(1, 2)
        x = x.reshape(B, N, C)

        x = self.out_proj(x)
        return x

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x): 
        return torch.sigmoid(x) * x

class ColorFM_L(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = DinoVisionTransformer(
            img_size=256,
            patch_size=16,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            block_fn=partial(Block, attn_class=MemEffAttention),
            num_register_tokens=0
        )
        self.cross_attn = CrossAttentionGenerator(base_channels=384, attn_channels=384)
        self.param_gen = ParamGenerator()
        self.params = {}
        self.act = Swish()

    def forward(self, x, style, dual=False):
        """Transfer colors and optionally compute the reverse direction.

        Returns:
            A three-item tuple ``(forward_result, intermediate, reverse_result)``.
            The intermediate is the content image after the first residual
            transform and before the style-side inverse transform. The reverse
            result is ``None`` unless ``dual`` is True.
        """
        B, C, H, W = x.shape
        x_resize = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        style_resize = F.interpolate(style, size=(256, 256), mode='bilinear', align_corners=False)
        self.params = {}
        content_features = self.encoder.forward_features(x_resize)['x_norm_patchtokens']
        style_features = self.encoder.forward_features(style_resize)['x_norm_patchtokens']
        content_shift = self.cross_attn(content_features, style_features)
        style_shift = self.cross_attn(style_features, content_features)

        kernel1_content, kernel2_content, kernel3_content, kernel4_content = self.param_gen(content_features, content_shift)
        kernel1_style, kernel2_style, kernel3_style, kernel4_style = self.param_gen(style_features, style_shift)
        self.params["kernel1_content"] = kernel1_content
        self.params["kernel2_content"] = kernel2_content
        self.params["kernel3_content"] = kernel3_content
        self.params["kernel4_content"] = kernel4_content
        self.params["kernel1_style"] = kernel1_style
        self.params["kernel2_style"] = kernel2_style
        self.params["kernel3_style"] = kernel3_style
        self.params["kernel4_style"] = kernel4_style

        x_transform = x.view(B, C, -1).permute(0, 2, 1).contiguous()
        chunk_size = 1024 * 1024
        results = []
        mid_results = []

        for i in range(0, x_transform.shape[1], chunk_size):
            chunk = x_transform[:, i:i+chunk_size, :]
            t = torch.zeros(chunk.shape[:-1] + (1,), device=chunk.device, dtype=chunk.dtype)
            v = torch.cat([chunk, t], dim=-1)
            v = torch.bmm(v, kernel1_content)
            v = self.act(v)
            v = torch.bmm(v, kernel2_content)
            v = self.act(v)
            v = torch.bmm(v, kernel3_content)
            v = self.act(v)
            v = torch.bmm(v, kernel4_content)
            chunk = chunk + v

            mid_results.append(chunk)

            t = torch.ones(chunk.shape[:-1] + (1,), device=chunk.device, dtype=chunk.dtype)
            v = torch.cat([chunk, t], dim=-1)
            v = torch.bmm(v, kernel1_style)
            v = self.act(v)
            v = torch.bmm(v, kernel2_style)
            v = self.act(v)
            v = torch.bmm(v, kernel3_style)
            v = self.act(v)
            v = torch.bmm(v, kernel4_style)
            chunk = chunk - v
            results.append(chunk)

        x_transform = torch.cat(results, dim=1)
        x_transform = x_transform.permute(0, 2, 1).contiguous()
        x_transform = x_transform.view(B, C, H, W)

        mid_transform = torch.cat(mid_results, dim=1)
        mid_transform = mid_transform.permute(0, 2, 1).contiguous()
        mid_transform = mid_transform.view(B, C, H, W)

        reverse_transform = None
        if dual:
            B, C, H, W = style.shape
            style_transform = style.view(B, C, H * W).permute(0, 2, 1).contiguous()
            chunk_size = 1024 * 1024
            results = []

            for i in range(0, style_transform.shape[1], chunk_size):
                chunk = style_transform[:, i:i+chunk_size, :]
                t = torch.zeros(chunk.shape[:-1] + (1,), device=chunk.device, dtype=chunk.dtype)
                v = torch.cat([chunk, t], dim=-1)
                v = torch.bmm(v, kernel1_style)
                v = self.act(v)
                v = torch.bmm(v, kernel2_style)
                v = self.act(v)
                v = torch.bmm(v, kernel3_style)
                v = self.act(v)
                v = torch.bmm(v, kernel4_style)
                chunk = chunk + v

                t = torch.ones(chunk.shape[:-1] + (1,), device=chunk.device, dtype=chunk.dtype)
                v = torch.cat([chunk, t], dim=-1)
                v = torch.bmm(v, kernel1_content)
                v = self.act(v)
                v = torch.bmm(v, kernel2_content)
                v = self.act(v)
                v = torch.bmm(v, kernel3_content)
                v = self.act(v)
                v = torch.bmm(v, kernel4_content)
                chunk = chunk - v
                results.append(chunk)

            reverse_transform = torch.cat(results, dim=1)
            reverse_transform = reverse_transform.permute(0, 2, 1).contiguous()
            reverse_transform = reverse_transform.view(B, C, H, W)

        return x_transform, mid_transform, reverse_transform

    def get_params(self):
        return self.params
