# coding=utf-8
# Copyright 2022, HuggingFace Inc.
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
""" Switch Transformers model configuration"""
from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

GPTSAN_JAPANESE_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "tanreinama/GPTSAN-2.8B-spout_is_uniform": (
        "https://huggingface.co/tanreinama/GPTSAN-2.8B-spout_is_uniform/resolve/main/config.json"
    ),
}


class GPTSANJapaneseConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GPTSANJapaneseModel`]. It is used to instantiate
    a GPTSANJapanese model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the GPTSANJapanese
    [tanreinama/GPTSAN-2.8B-spout_is_uniform](https://huggingface.co/tanreinama/GPTSAN-2.8B-spout_is_uniform)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Arguments:
        vocab_size (`int`, *optional*, defaults to 36000):
            Vocabulary size of the GPTSANJapanese model. Defines the number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`GPTSANJapaneseModel`].
        num_contexts (`int`, *optional*, defaults to 1280):
            Size of the maximum tokens.
        d_model (`int`, *optional*, defaults to 1024):
            Size of the encoder layers and the pooler layer.
        d_ff (`int`, *optional*, defaults to 8192):
            Size of the intermediate feed forward layer in each `SwitchTransformersBlock`.
        d_ext (`int`, *optional*, defaults to 4096):
            Size of the intermediate feed forward layer in each Extra-layers.
        d_spout (`int`, *optional*, defaults to 128):
            Size of the `spout` vector.
        num_switch_layers (`int`, *optional*, defaults to 10):
            Number of layers in the Switch Transformer layer.
        num_ext_layers (`int`, *optional*, defaults to 0):
            Number of layers in the Extra-layers.
        num_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_experts (`int`, *optional*, defaults to 16):
            Number of experts for each SwitchTransformer layer.
        expert_capacity (`int`, *optional*, defaults to 128):
            Number of tokens that can be stored in each expert. If set to 1, the model will behave like a regular
            Transformer.
        dropout_rate (`float`, *optional*, defaults to 0.0):
            The ratio for all dropout layers.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        router_bias (`bool`, *optional*, defaults to `False`):
            Whether to add a bias to the router.
        router_jitter_noise (`float`, *optional*, defaults to 1e-2):
            Amount of noise to add to the router.
        router_dtype (`str`, *optional*, default to `"float32"`):
            The `dtype` used for the routers. It is preferable to keep the `dtype` to `"float32"` as specified in the
            *selective precision* discussion in [the paper](https://arxiv.org/abs/2101.03961).
        router_ignore_padding_tokens (`bool`, *optional*, defaults to `False`):
            Whether to ignore padding tokens when routing.
        output_hidden_states (`bool`, *optional*, default to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        use_cache (`bool`, *optional*, default to `False`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_router_logits (`bool`, *optional*, default to `False`):
            Whether or not to return the router logits of all experts.
    """
    model_type = "gptsan-japanese"
    keys_to_ignore_at_inference = [
        "past_key_values",
    ]
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        vocab_size=36000,
        num_contexts=1280,
        d_model=1024,
        d_ff=8192,
        d_ext=4096,
        d_spout=128,
        num_switch_layers=10,
        num_ext_layers=0,
        num_heads=16,
        num_experts=16,
        expert_capacity=128,
        dropout_rate=0.0,
        layer_norm_epsilon=1e-5,
        router_bias=False,
        router_jitter_noise=0.0,
        router_dtype="float32",
        router_ignore_padding_tokens=False,
        output_hidden_states=False,
        output_attentions=False,
        initializer_factor=0.002,
        output_router_logits=False,
        top_k=120,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.num_contexts = num_contexts
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_ext = d_ext
        self.d_spout = d_spout
        self.num_switch_layers = num_switch_layers
        self.num_ext_layers = num_ext_layers
        self.num_layers = num_switch_layers + num_ext_layers
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.router_bias = router_bias
        self.router_jitter_noise = router_jitter_noise
        self.router_dtype = router_dtype
        self.router_ignore_padding_tokens = router_ignore_padding_tokens
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions
        self.initializer_factor = initializer_factor
        self.output_router_logits = output_router_logits

        kwargs["eos_token_id"] = vocab_size - 1
        kwargs["separator_token_id"] = vocab_size - 2
        kwargs["unk_token_id"] = vocab_size - 4
        kwargs["pad_token_id"] = vocab_size - 5
        kwargs["mask_token_id"] = vocab_size - 6
        kwargs["bos_token_id"] = vocab_size - 7
        kwargs["is_encoder_decoder"] = False
        kwargs["use_cache"] = True
        kwargs["do_sample"] = True
        kwargs["top_k"] = top_k
        super().__init__(
            **kwargs,
        )
