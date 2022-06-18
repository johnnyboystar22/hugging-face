# coding=utf-8
# Copyright 2022 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Omnivore model."""


from functools import lru_cache, reduce
from operator import mul
from typing import Optional

import numpy as np
import torch
import torch.utils.checkpoint
import torch.utils.checkpoint as checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import functional as F

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_omnivore import OmnivoreConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "OmnivoreConfig"
_FEAT_EXTRACTOR_FOR_DOC = "OmniverseFeatureExtractor"

# Base docstring
_CHECKPOINT_FOR_DOC = "anugunj/omnivore"
_EXPECTED_OUTPUT_SHAPE = [1, 768, 7, 7]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "anugunj/omnivore"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

OMNIVORE_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "anugunj/omnivore",
    # See all Omnivore models at https://huggingface.co/models?filter=omnivore
]


def drop_path(input, drop_prob=0.0, training=False, scale_by_keep=True):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = input.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return input * random_tensor


class OmnivoreDropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor):
        return drop_path(x, self.drop_prob, self.training)


class OmnivoreIm2Video(nn.Module):
    """Convert Image into a trivial video"""

    def forward(self, pixel_values):
        if pixel_values.ndim == 4:
            return pixel_values.unsqueeze(2)
        elif pixel_values.ndim == 5:
            return pixel_values
        else:
            raise ValueError(f"Dimension incorrect {pixel_values.shape}")


class OmnivoreSwinMLPLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout_rate=0.0, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.activation = act_layer()
        self.linear2 = nn.Linear(hidden_features, out_features)
        self.drop_out = nn.Dropout(dropout_rate)

    def forward(self, hidden_state):
        hidden_state = self.linear1(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.drop_out(hidden_state)
        hidden_state = self.linear2(hidden_state)
        hidden_state = self.drop_out(hidden_state)
        return hidden_state


def window_partition(input_feature, window_size):
    batch_size, D, height, width, channels = input_feature.shape
    input_feature = input_feature.view(
        batch_size,
        D // window_size[0],
        window_size[0],
        height // window_size[1],
        window_size[1],
        width // window_size[2],
        window_size[2],
        channels,
    )
    windows = input_feature.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), channels)
    return windows


def window_partition_image(input_feature, window_size):
    batch_size, height, width, channels = input_feature.shape
    input_feature = input_feature.view(
        batch_size, height // window_size[1], window_size[1], width // window_size[2], window_size[2], channels
    )
    windows = input_feature.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[1], window_size[2], channels)
    return windows


def window_reverse(windows, windows_size, batch_size, D, height, width):
    input_feature = windows.view(
        batch_size,
        D // windows_size[0],
        height // windows_size[1],
        width // windows_size[2],
        windows_size[0],
        windows_size[1],
        windows_size[2],
        -1,
    )
    input_feature = input_feature.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(batch_size, D, height, width, -1)
    return input_feature


def get_window_size(input_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(input_size)):
        if input_size[i] <= window_size[i]:
            use_window_size[i] = input_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class OmnivoreSwinAttention(nn.Module):
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=False,
        qk_scale=None,
        attention_dropout_rate=0.0,
        projection_dropout_rate=0.0,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                num_heads,
            )
        )

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.queries_keys_values = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attention_dropout = nn.Dropout(attention_dropout_rate)
        self.projection = nn.Linear(dim, dim)
        self.projection_dropout = nn.Dropout(projection_dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hidden_state, attention_mask=None):
        batch_size, seq_len, channels = hidden_state.shape
        queries_keys_values = (
            self.queries_keys_values(hidden_state)
            .reshape(batch_size, seq_len, 3, self.num_heads, channels // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        queries, keys, values = queries_keys_values[0], queries_keys_values[1], queries_keys_values[2]

        queries = queries * self.scale
        attention = queries @ keys.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:seq_len, :seq_len].reshape(-1)
        ].reshape(seq_len, seq_len, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attention = attention + relative_position_bias.unsqueeze(0)

        if attention_mask is not None:
            nW = attention_mask.shape[0]
            attention = attention.view(
                batch_size // nW, nW, self.num_heads, seq_len, seq_len
            ) + attention_mask.unsqueeze(1).unsqueeze(0)
            attention = attention.view(-1, self.num_heads, seq_len, seq_len)
            attention = self.softmax(attention)
        else:
            attention = self.softmax(attention)

        attention = self.attention_dropout(attention)

        hidden_state = (attention @ values).transpose(1, 2).reshape(batch_size, seq_len, channels)
        hidden_state = self.projection(hidden_state)
        hidden_state = self.projection_dropout(hidden_state)
        return hidden_state


class OmnivoreSwinLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=(2, 7, 7),
        shift_size=(0, 0, 0),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
        drop_path_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attention = OmnivoreSwinAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attention_dropout_rate=attention_dropout_rate,
            projection_dropout_rate=dropout_rate,
        )

        self.drop_path = OmnivoreDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = OmnivoreSwinMLPLayer(
            in_features=dim, hidden_features=mlp_hidden_dim, dropout_rate=dropout_rate, act_layer=act_layer
        )

    def forward_before(self, hidden_state, attention_mask):
        batch_size, D, height, width, channels = hidden_state.shape
        window_size, shift_size = get_window_size((D, height, width), self.window_size, self.shift_size)

        hidden_state = self.norm1(hidden_state)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - height % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - width % window_size[2]) % window_size[2]
        hidden_state = F.pad(hidden_state, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = hidden_state.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_hidden_state = torch.roll(
                hidden_state, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3)
            )
            attention_mask = attention_mask
        else:
            shifted_hidden_state = hidden_state
            attention_mask = None
        # partition windows
        hidden_state_windows = window_partition(shifted_hidden_state, window_size)
        # W-MSA/SW-MSA
        attention_windows = self.attention(hidden_state_windows, attention_mask=attention_mask)
        # merge windows
        attention_windows = attention_windows.view(-1, *(window_size + (channels,)))
        shifted_hidden_state = window_reverse(attention_windows, window_size, batch_size, Dp, Hp, Wp)
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            hidden_state = torch.roll(
                shifted_hidden_state, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3)
            )
        else:
            hidden_state = shifted_hidden_state

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            hidden_state = hidden_state[:, :D, :height, :width, :].contiguous()
        return hidden_state

    def forward_after(self, hidden_state):
        hidden_state = self.norm2(hidden_state)
        hidden_state = self.mlp(hidden_state)
        hidden_state = self.drop_path(hidden_state)
        return hidden_state

    def forward(self, hidden_state, mask_matrix, use_checkpoint=False):
        shortcut = hidden_state
        if use_checkpoint:
            hidden_state = checkpoint.checkpoint(self.forward_before, hidden_state, mask_matrix)
        else:
            hidden_state = self.forward_before(hidden_state, mask_matrix)
        hidden_state = shortcut + self.drop_path(hidden_state)

        if use_checkpoint:
            hidden_state = hidden_state + checkpoint.checkpoint(self.forward_after, hidden_state)
        else:
            hidden_state = hidden_state + self.forward_after(hidden_state)

        return hidden_state


class OmnivoreSwinPatchMerging(nn.Module):
    """
    Args:
    Patch Merging Layer
        dim (`int`): Number of input channels. norm_layer (`nn.Module`, *optional*): Normalization layer. Default:
        `nn.LayerNorm`
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, hidden_state, height=None, width=None):
        if height is None:
            batch_size, D, height, width, channels = hidden_state.shape

        # padding
        pad_input = (height % 2 == 1) or (width % 2 == 1)
        if pad_input:
            hidden_state = F.pad(hidden_state, (0, 0, 0, width % 2, 0, height % 2))

        hidden_state0 = hidden_state[:, :, 0::2, 0::2, :]
        hidden_state1 = hidden_state[:, :, 1::2, 0::2, :]
        hidden_state2 = hidden_state[:, :, 0::2, 1::2, :]
        hidden_state3 = hidden_state[:, :, 1::2, 1::2, :]
        hidden_state = torch.cat([hidden_state0, hidden_state1, hidden_state2, hidden_state3], -1)

        hidden_state = self.norm(hidden_state)
        hidden_state = self.reduction(hidden_state)

        return hidden_state


@lru_cache()
def compute_mask(D, height, width, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, height, width, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in (
        slice(-window_size[0]),
        slice(-window_size[0], -shift_size[0]),
        slice(-shift_size[0], None),
    ):
        for h in (
            slice(-window_size[1]),
            slice(-window_size[1], -shift_size[1]),
            slice(-shift_size[1], None),
        ):
            for w in (
                slice(-window_size[2]),
                slice(-window_size[2], -shift_size[2]),
                slice(-shift_size[2], None),
            ):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attention_mask = attention_mask.masked_fill(attention_mask != 0, float(-100.0)).masked_fill(
        attention_mask == 0, float(0.0)
    )
    return attention_mask


class OmnivoreSwinStage(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size=(1, 7, 7),
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth

        # build layers
        self.layers = nn.ModuleList(
            [
                OmnivoreSwinLayer(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                    drop_path_rate=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def forward(self, hidden_state, use_checkpoint=False, height=None, width=None, use_seg=False):
        if use_seg:
            return self.forward_seg(hidden_state, height, width)
        batch_size, channels, D, height, width = hidden_state.shape
        window_size, shift_size = get_window_size((D, height, width), self.window_size, self.shift_size)
        hidden_state = hidden_state.permute(0, 2, 3, 4, 1)

        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(height / window_size[1])) * window_size[1]
        Wp = int(np.ceil(width / window_size[2])) * window_size[2]

        attention_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, hidden_state.device)

        for layer in self.layers:
            hidden_state = layer(hidden_state, attention_mask, use_checkpoint=use_checkpoint)
        hidden_state = hidden_state.view(batch_size, D, height, width, -1)

        if self.downsample is not None:
            hidden_state = self.downsample(hidden_state)

        hidden_state = hidden_state.permute(0, 4, 1, 2, 3)

        return hidden_state

    def forward_seg(self, hidden_state, height, width):

        Hp = int(np.ceil(height / self.window_size[1])) * self.window_size[1]
        Wp = int(np.ceil(width / self.window_size[2])) * self.window_size[2]
        img_mask = torch.zeros((1, Hp, Wp, 1), device=hidden_state.device)  # 1 Hp Wp 1
        h_slices = (
            slice(0, -self.window_size[1]),
            slice(-self.window_size[1], -self.shift_size[1]),
            slice(-self.shift_size[1], None),
        )
        w_slices = (
            slice(0, -self.window_size[2]),
            slice(-self.window_size[2], -self.shift_size[2]),
            slice(-self.shift_size[2], None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition_image(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size[1] * self.window_size[2])
        attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attention_mask = attention_mask.masked_fill(attention_mask != 0, float(-100.0)).masked_fill(
            attention_mask == 0, float(0.0)
        )

        for layer in self.layers:
            layer.height, layer.width = height, width
            if hidden_state.ndim == 4:
                batch_size, D, channels, seq_len = hidden_state.shape
                assert seq_len == height * width, "input feature has wrong size"
                hidden_state = hidden_state.reshape(batch_size, D, channels, height, width)
                hidden_state = hidden_state.permute(0, 1, 3, 4, 2)
            assert hidden_state.shape[2] == height
            assert hidden_state.shape[3] == width
            hidden_state = layer(hidden_state, attention_mask)
        if self.downsample is not None:
            x_down = self.downsample(hidden_state, height, width)
            Wh, Ww = (height + 1) // 2, (width + 1) // 2
            return hidden_state, height, width, x_down, Wh, Ww
        else:
            return hidden_state, height, width, hidden_state, height, width


class OmnivoreSwinPatchEmbeddings(nn.Module):
    """Video to Patch Embedding"""

    def __init__(
        self,
        patch_size=(2, 4, 4),
        input_channels=3,
        embed_dim=96,
        norm_layer=None,
        additional_variable_channels=None,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.input_channels = input_channels
        self.embed_dim = embed_dim
        self.additional_variable_channels = additional_variable_channels

        self.projection = nn.Conv3d(input_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        if additional_variable_channels:
            # we create var_proj separately from proj
            # this makes it convenient to ignore var_proj on downstream tasks
            # where we only use RGB
            self.var_projection = [
                nn.Conv3d(x, embed_dim, kernel_size=patch_size, stride=patch_size)
                for x in additional_variable_channels
            ]
            self.var_projection = nn.ModuleList(self.var_projection)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def run_variable_channel_forward(self, hidden_state):
        sidx = 0
        out = None
        for idx in range(len(self.additional_variable_channels)):
            eidx = sidx + self.additional_variable_channels[idx]
            c_out = self.var_projection[idx](hidden_state[:, sidx:eidx, ...])
            if idx == 0:
                out = c_out
            else:
                out += c_out
            sidx = eidx
        return out

    def forward(self, hidden_state):
        _, _, D, height, width = hidden_state.size()
        if width % self.patch_size[2] != 0:
            hidden_state = F.pad(hidden_state, (0, self.patch_size[2] - width % self.patch_size[2]))
        if height % self.patch_size[1] != 0:
            hidden_state = F.pad(hidden_state, (0, 0, 0, self.patch_size[1] - height % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            hidden_state = F.pad(hidden_state, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        if self.additional_variable_channels:
            hidden_state_rgb = hidden_state[:, :3, ...]
            hidden_state_rem = hidden_state[:, 3:, ...]
            hidden_state_rgb = self.projection(hidden_state_rgb)
            if hidden_state.shape[1] > 3:
                hidden_state_rem = self.run_variable_channel_forward(hidden_state_rem)
                hidden_state = hidden_state_rgb + hidden_state_rem
            else:
                hidden_state = hidden_state_rgb
        else:
            hidden_state = self.projection(hidden_state)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = hidden_state.size(2), hidden_state.size(3), hidden_state.size(4)
            hidden_state = hidden_state.flatten(2).transpose(1, 2)
            hidden_state = self.norm(hidden_state)
            hidden_state = hidden_state.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return hidden_state


class OmnivoreSwinTrunk(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.im2vid = OmnivoreIm2Video()
        self.num_stages = len(self.config.depths)
        self.patch_size = self.config.patch_size
        self.input_channels = self.config.input_channels
        self.embed_dim = self.config.embed_dim
        self.depths = self.config.depths
        self.num_heads = self.config.num_heads
        self.window_size = self.config.window_size
        self.mlp_ratio = self.config.mlp_ratio
        self.qkv_bias = self.config.qkv_bias
        self.qk_scale = self.config.qk_scale
        self.dropout_rate = self.config.dropout_rate
        self.attention_dropout_rate = self.config.attention_dropout_rate
        self.drop_path_rate = self.config.drop_path_rate
        self.norm_layer = nn.LayerNorm
        self.patch_norm = self.config.patch_norm
        self.frozen_stages = self.config.frozen_stages
        self.depth_patch_embed_separate_params = True
        self.depth_mode = self.config.depth_mode
        depth_chans = None
        assert self.input_channels == 3, "Only 3 channels supported"

        # split image into non-overlapping patches
        self.patch_embed = OmnivoreSwinPatchEmbeddings(
            patch_size=self.patch_size,
            input_channels=self.input_channels,
            embed_dim=self.embed_dim,
            norm_layer=self.norm_layer if self.patch_norm else None,
        )

        if self.depth_mode is not None:
            msg = f"Using depth mode {self.depth_mode}"
            logger.info(msg)
            assert self.depth_mode in ["separate_d_tokens", "summed_rgb_d_tokens", "rgbd"]
            if self.depth_mode in ["separate_d_tokens", "summed_rgb_d_tokens"]:
                depth_chans = 1
                assert self.depth_patch_embed_separate_params, "separate tokenization needs separate parameters"
                if self.depth_mode == "separate_d_tokens":
                    raise NotImplementedError()
            else:
                assert self.depth_mode == "rgbd"
                depth_chans = 4

            self.depth_patch_embed_separate_params = self.depth_patch_embed_separate_params

            if self.depth_patch_embed_separate_params:
                self.depth_patch_embed = OmnivoreSwinPatchEmbeddings(
                    patch_size=self.patch_size,
                    input_channels=depth_chans,
                    embed_dim=self.embed_dim,
                    norm_layer=self.norm_layer if self.patch_norm else None,
                )
            else:
                del self.patch_embed
                assert depth_chans == 4
                logger.info("Certain channels of patch projection may not be used in forward pass")
                logger.info("Make sure config.DISTRIBUTED.FIND_UNUSED_PARAMETERS is set to True")
                self.patch_embed = OmnivoreSwinPatchEmbeddings(
                    patch_size=self.patch_size,
                    input_channels=3,
                    embed_dim=self.embed_dim,
                    additional_variable_channels=[1],
                    norm_layer=self.norm_layer if self.patch_norm else None,
                )

        self.pos_drop = nn.Dropout(p=self.dropout_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))
        ]  # stochastic depth decay rule

        # build stages
        self.stages = nn.ModuleList()
        for stage in range(self.num_stages):
            stage_module = OmnivoreSwinStage(
                dim=int(self.embed_dim * 2**stage),
                depth=self.depths[stage],
                num_heads=self.num_heads[stage],
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                drop_path_rate=dpr[sum(self.depths[:stage]) : sum(self.depths[: stage + 1])],
                norm_layer=self.norm_layer,
                downsample=OmnivoreSwinPatchMerging if stage < self.num_stages - 1 else None,
            )
            self.stages.append(stage_module)

        self.num_features = int(self.embed_dim * 2 ** (self.num_stages - 1))
        self.norm = self.norm_layer(self.num_features)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def _apply_norm(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x

    def forward_intermediate_features(self, stage_outputs, out_feat_keys):
        out_features = []
        for key in out_feat_keys:
            if key.startswith("stage"):
                rep = "stage"
            elif key.startswith("interim"):
                rep = "interim"
            else:
                raise ValueError(f"Invalid key {key}")
            idx = int(key.replace(rep, ""))
            feat = stage_outputs[idx]
            if rep == "stage":
                feat = self._apply_norm(feat)
            out_features.append(feat)
        return out_features

    def get_patch_embedding(self, hidden_state):
        assert hidden_state.ndim == 5
        has_depth = hidden_state.shape[1] == 4

        if has_depth:
            if self.depth_mode in ["summed_rgb_d_tokens"]:
                hidden_state_rgb = hidden_state[:, :3, ...]
                hidden_state_d = hidden_state[:, 3:, ...]
                hidden_state_d = self.depth_patch_embed(hidden_state_d)
                hidden_state_rgb = self.patch_embed(hidden_state_rgb)
                # sum the two sets of tokens
                hidden_state = hidden_state_rgb + hidden_state_d
            elif self.depth_mode == "rgbd":
                if self.depth_patch_embed_separate_params:
                    hidden_state = self.depth_patch_embed(hidden_state)
                else:
                    hidden_state = self.patch_embed(hidden_state)
            else:
                logger.info("Depth mode %s not supported" % self.depth_mode)
                raise NotImplementedError()
        else:
            hidden_state = self.patch_embed(hidden_state)
        return hidden_state

    def forward(
        self, hidden_state, out_feat_keys=None, use_checkpoint=False, output_hidden_states=False, return_dict=True
    ):
        all_hidden_states = () if output_hidden_states else None
        hidden_state = self.im2vid(hidden_state)
        hidden_state = self.get_patch_embedding(hidden_state)
        hidden_state = self.pos_drop(hidden_state)

        stage_outputs = []

        for stage in self.stages:
            hidden_state = stage(hidden_state.contiguous(), use_checkpoint=use_checkpoint)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)
            stage_outputs.append(hidden_state)

        if out_feat_keys is not None and len(out_feat_keys) > 0:
            final_hidden_state = self.forward_intermediate_features(stage_outputs, out_feat_keys)
        else:
            hidden_state = self._apply_norm(hidden_state)
            # Mean over the spatiotemporal dimensions
            hidden_state = torch.mean(hidden_state, [-3, -2, -1])

            final_hidden_state = hidden_state

        if not return_dict:
            return tuple(v for v in [final_hidden_state, all_hidden_states] if v is not None)
        return BaseModelOutputWithNoAttention(last_hidden_state=final_hidden_state, hidden_states=all_hidden_states)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(OmnivoreSwinTrunk, self).train(mode)
        self._freeze_stages()


class OmnivoreImageHead(nn.Module):
    def __init__(self, in_features=1024, out_features=1000, bias=True):
        super().__init__()
        self.image_head = nn.Linear(in_features, out_features, bias)

    def forward(self, hidden_state):
        logits = self.image_head(hidden_state)
        return logits


class OmnivoreVideoHead(nn.Module):
    def __init__(self, in_features=1024, out_features=400, bias=True):
        super().__init__()
        self.video_head = nn.Linear(in_features, out_features, bias)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, hidden_state):
        logits = self.video_head(hidden_state)
        logits = self.dropout(logits)
        return logits


class OmnivoreRGBDHead(nn.Module):
    def __init__(self, in_features=1024, out_features=19, bias=True):
        super().__init__()
        self.rgbd_head = nn.Linear(in_features, out_features, bias)

    def forward(self, hidden_state):
        logits = self.rgbd_head(hidden_state)
        return logits


class OmnivorePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = OmnivoreConfig
    base_model_prefix = "omnivore"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, OmnivoreModel):
            module.gradient_checkpointing = value


OMNIVORE_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`OmnivoreConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

OMNIVORE_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoFeatureExtractor`]. See
            [`AutoFeatureExtractor.__call__`] for details.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Omnivore model outputting raw features without any specific head on top.",
    OMNIVORE_START_DOCSTRING,
)
class OmnivoreModel(OmnivorePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.trunk = OmnivoreSwinTrunk(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(OMNIVORE_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        outputs = self.trunk(pixel_values)
        last_hidden_state = outputs[0]
        # global average pooling, (N, C, D, H, W) -> (N, C)
        pooled_output = last_hidden_state.mean([-1])

        if not return_dict:
            return (last_hidden_state, pooled_output) + outputs[1:]

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=outputs.hidden_states,
        )


@add_start_docstrings(
    """
    Omnivore Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    OMNIVORE_START_DOCSTRING,
)
class OmnivoreForVisionClassification(OmnivorePreTrainedModel):
    # TODO Change Name
    def __init__(self, config):
        super().__init__(config)

        self.num_image_labels = config.num_image_labels or config.num_labels
        self.num_video_labels = config.num_video_labels or config.num_labels
        self.num_rgbd_labels = config.num_rgbd_labels or config.num_labels
        self.omnivore = OmnivoreModel(config)
        self.image_classifier = OmnivoreImageHead(config.embed_dim * 8, self.num_image_labels)
        self.rgbd_classifier = OmnivoreRGBDHead(config.embed_dim * 8, self.num_rgbd_labels)
        self.video_classifier = OmnivoreVideoHead(config.embed_dim * 8, self.num_video_labels)
        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(OMNIVORE_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        pixel_input_type: str = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        pixel_input_type (`str`):
            Which classification head to use for the classification of given pixel_values
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Example:

        ```python
        >>> from transformers import OmnivoreFeatureExtractor, OmnivoreForImageClassification
        >>> import torch
        >>> from datasets import load_dataset

        >>> dataset = load_dataset("huggingface/cats-image")
        >>> image = dataset["test"]["image"][0]

        >>> feature_extractor = OmnivoreFeatureExtractor.from_pretrained("anugunj/omnivore-swinT")
        >>> model = OmnivoreForImageClassification.from_pretrained("anugunj/omnivore-swinT")

        >>> inputs = feature_extractor(image, return_tensors="pt")

        >>> logits = model(**inputs).logits

        >>> # model predicts one of the 1000 ImageNet classes
        >>> predicted_label = logits.argmax(-1).item()
        >>> print(model.config.id2label[predicted_label])
        tabby, tabby cat
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.omnivore(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]

        logits = None
        if pixel_input_type == "image":
            logits = self.image_classifier(sequence_output)

        if pixel_input_type == "video":
            logits = self.video_classifier(sequence_output)

        if pixel_input_type == "rgbd":
            logits = self.rgbd_classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
