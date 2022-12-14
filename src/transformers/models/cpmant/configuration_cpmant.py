# coding=utf-8
# Copyright 2022 The OpenBMB Team and The HuggingFace Inc. team. All rights reserved.
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
""" CPMAnt model configuration"""

from typing import List, Optional, Tuple

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

CPMANT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "cpm-ant-10b": "https://huggingface.co/cpm-ant-10b/resolve/main/config.json",
    # See all CPMAnt models at https://huggingface.co/models?filter=cpmant
}


class CPMAntConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~CPMAntModel`]. It is used to instantiate an
    CPMAnt model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the CPMAnt
    [cpm-ant-10b](https://huggingface.co/cpm-ant-10b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 30720):
            Vocabulary size of the CPMAnt model. Defines the number of different tokens that can be represented by the
            `input` passed when calling [`~CPMAntModel`].
        dim_model (`int`, *optional*, defaults to 4096):
            Dimension of the encoder layers.
        num_heads (`int`, *optional*, defaults to 32):
            Number of attention heads in the Transformer encoder.
        dim_head (`int`, *optional*, defaults to 128):
            Dimension of attention heads for each attention layer in the Transformer encoder.
        dim_ff (`int`, *optional*, defaults to 10240):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_layers (`int`, *optional*, defaults to 48):
            Number of layers of the Transformer encoder.
        dropout_p (`float`, *optional*, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder.
        position_bias_num_buckets (`int`, *optional*, defaults to 512):
            The number of position_bias buckets.
        position_bias_max_distance (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        half (`bool`, *optional*, defaults to False):
            The dtype of tensor.
        prompt_types (`int`, *optional*, defaults to 32):
            The type of prompt.
        prompt_length (`int`, *optional*, defaults to 32):
            The length of prompt.
        segment_types (`int`, *optional*, defaults to 32):
            The type of segment.
        mask_modules (`List[Tuple[bool, bool]]`, *optional*, defaults to None):
            Determine whether the module should be masked.

        Example:

    ```python
    >>> from transformers import CPMAntModel, CPMAntConfig

    >>> # Initializing a CPMAnt cpm-ant-10b style configuration
    >>> configuration = CPMAntConfig()

    >>> # Initializing a model from the cpm-ant-10b style configuration
    >>> model = CPMAntModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "cpmant"

    def __init__(
        self,
        vocab_size=30720,
        dim_model=4096,
        num_heads=32,
        dim_head=128,
        dim_ff=10240,
        num_layers=48,
        dropout_p=0.0,
        position_bias_num_buckets=512,
        position_bias_max_distance=2048,
        eps=1e-6,
        half: bool = False,
        prompt_types: int = 32,
        prompt_length: int = 32,
        segment_types: int = 32,
        mask_modules: Optional[List[Tuple[bool, bool]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prompt_types = prompt_types
        self.prompt_length = prompt_length
        self.segment_types = segment_types
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.num_layers = num_layers
        self.position_bias_num_buckets = position_bias_num_buckets
        self.position_bias_max_distance = position_bias_max_distance
        self.dropout_p = dropout_p
        self.eps = eps
        self.vocab_size = vocab_size
        self.mask_modules = mask_modules
