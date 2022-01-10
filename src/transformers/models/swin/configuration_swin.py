# coding=utf-8
# Copyright temp-authors and The HuggingFace Inc. team. All rights reserved.
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
""" Swin model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from torch import nn


logger = logging.get_logger(__name__)

SWIN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "swin-base": "https://huggingface.co/swin-base/resolve/main/config.json",
    # See all Swin models at https://huggingface.co/models?filter=swin
}


class SwinConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~SwinModel`].
    It is used to instantiate an Swin model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the Swin [swin-base](https://huggingface.co/swin-base) architecture.

    Configuration objects inherit from  [`PretrainedConfig`] and can be used
    to control the model outputs. Read the documentation from  [`PretrainedConfig`]
    for more information.


    Args:
        image_size (`int`, *optional*, defaults to 224):
            Size (resolution) of the input images.
        patch_size (`int`, *optional*, defaults to 4):
            Size of patches.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in input images.
        num_labels (`int`, *optional*, defaults to 1000):
            Number of classes for classification head
        embed_dim (`int`, *optional*, defaults to 96):
            Dimensionality of patch embedding.
        depths (`tuple(int)`, *optional*, defaults to (2, 2, 6, 2)):
            Depth of each layer in the Transformer encoder.
        num_heads (`tuple(int)`, *optional*, defaults to (3, 6, 12, 24)):
            Number of attention heads in each layer of the Transformer encoder.
        window_size (`int`, *optional*, defaults to 7):
            Size of windows.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            Ratio of MLP hidden dimesionality to embedding dimensionality.
        qkv_bias (`bool`, *optional*, defaults to True):
            Whether or not learnable bias should be added to query, key, value
        drop_rate (`float`, *optional*, defaults to 0.0):
            Dropout rate.
        attn_drop_rate (`float`, *optional*, defaults to 0.0):
            Attention dropout rate.
        drop_path_rate (`float`, *optional*, defaults to 0.1):
            Stochastic depth rate.
        norm_layer (`nn.Module`, *optional*, defaults to nn.LayerNorm):
            Normalization layer.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        ape (`bool`, *optional*, defaults to False):
            Whether or not to add absolute position embedding to the patch embedding.
        patch_norm (`bool`, *optional*, defaults to True):
            Whether or not to add normalization after patch embedding.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.

        Example:

    ```python
    >>> from transformers import SwinModel, SwinConfig

    >>> # Initializing a Swin swin-base style configuration
    >>> configuration = SwinConfig()

    >>> # Initializing a model from the swin-base style configuration
    >>> model = SwinModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
"""
    model_type = "swin"

    def __init__(
        self,
        image_size=224, 
        patch_size=4, 
        num_channels=3, 
        num_labels=1000,
        embed_dim=96, 
        depths=(2, 2, 6, 2), 
        num_heads=(3, 6, 12, 24),
        window_size=7, 
        mlp_ratio=4., 
        qkv_bias=True,
        drop_rate=0., 
        attn_drop_rate=0., 
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        hidden_act="gelu",
        ape=False, 
        patch_norm=True,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_labels = num_labels
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.hidden_act = hidden_act
        self.ape = ape
        self.path_norm = patch_norm
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range