# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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

"""Convert SwitchTransformersX checkpoints from the original repository to JAX/FLAX model."""

import argparse
import re

from t5x import checkpoints
from transformers import SwitchTransformersForConditionalGeneration, SwitchTransformersConfig
from transformers.modeling_flax_pytorch_utils import load_flax_weights_in_pytorch_model

from flax.traverse_util import flatten_dict, unflatten_dict

MOE_LAYER_NAME_MAPPING = {
    "/attention/": "/0/SelfAttention/",
    "/self_attention/": "/0/SelfAttention/",
    "/encoder_decoder_attention/": "/1/EncDecAttention/",
    "value": "v",
    "query": "q",
    "key": "k",
    "out": "o",
    "pre_self_attention_layer_norm": "0/layer_norm",
    "pre_cross_attention_layer_norm": "1/layer_norm",
    "pre_attention_layer_norm": "1/layer_norm",
    "token_embedder": "shared",
    "encoder_norm": "final_layer_norm",
    "decoder_norm": "final_layer_norm",
    "relpos_bias/rel_embedding": "block/0/layer/0/SelfAttention/relative_attention_bias/weight",
    "router/router_weights/w/": "router/classifier/",
    "roer/roer_weights/w/": "router/classifier/",
}


def rename_keys(s_dict):
    # 1. in HF T5, we have block.{x}.layer.{y}. which corresponds to layer.{x} in
    # the original model
    keys = list(s_dict.keys())
    for key in keys:
        layer_to_block_of_layer = r".*/layers_(\d+)"
        new_key = key
        if re.match(layer_to_block_of_layer, key):
            new_key = re.sub(r"layers_(\d+)", r"block/\1/layer", new_key)
            # s_dict[new_key] = s_dict.pop(key)

        layer_to_block_of_layer = r"(encoder|decoder)\/"

        if re.match(layer_to_block_of_layer, key):
            groups = re.match(layer_to_block_of_layer, new_key).groups()
            if groups[0] == "encoder":
                new_key = re.sub(r"/mlp/", r"/1/mlp/", new_key)
                new_key = re.sub(r"/pre_mlp_layer_norm/", r"/0/layer_norm/", new_key)

            elif groups[0] == "decoder":
                new_key = re.sub(r"/mlp/", r"/2/mlp/", new_key)
                new_key = re.sub(r"/pre_mlp_layer_norm/", r"/1/layer_norm/", new_key)

        # 2. Convert other classic mappings
        for old_key, temp_key in MOE_LAYER_NAME_MAPPING.items():
            if old_key in new_key:
                new_key = new_key.replace(old_key, temp_key)
                

        print(f"{key} -> {new_key}")
        s_dict[new_key] = s_dict.pop(key)

    s_dict["encoder/block/0/layer/0/SelfAttention/relative_attention_bias/weight"] = s_dict["encoder/block/0/layer/0/SelfAttention/relative_attention_bias/weight"].T
    s_dict["decoder/block/0/layer/0/SelfAttention/relative_attention_bias/weight"] = s_dict["decoder/block/0/layer/0/SelfAttention/relative_attention_bias/weight"].T

        
    # 3. Take extra care of the EXPERTS layer
    for key in list(s_dict.keys()):
        if "expert" in key:          
            num_experts = s_dict[key].shape[0]
            expert_weihts = s_dict[key]
            for idx in range(num_experts):
                s_dict[key.replace("expert/", f"experts/expert_{idx}/")] = expert_weihts[idx]
            s_dict.pop(key)
    return s_dict

def convert_flax_checkpoint_to_pytorch(flax_params, pt_model):
    # Flatten Flax param dict, rename it and unflatten it
    params = flatten_dict(flax_params, sep="/")
    params = rename_keys(params)
    params = unflatten_dict(params, sep="/")

    # Load the flax params in the PT model
    load_flax_weights_in_pytorch_model(pt_model, params)
    return pt_model



def convert_switch_transformersx_checkpoint_to_flax(
    switch_transformersx_checkpoint_path, config_name, pytorch_dump_path
):
    config = SwitchTransformersConfig.from_pretrained(config_name)
    pt_model = SwitchTransformersForConditionalGeneration(config=config)
    flax_params = checkpoints.load_t5x_checkpoint(switch_transformersx_checkpoint_path)

    pt_model = convert_flax_checkpoint_to_pytorch(flax_params['target'], pt_model)

    print(f"Save PyTorch model to {pytorch_dump_path}")
    pt_model.save_pretrained(pytorch_dump_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--switch_t5x_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path the TX5 checkpoint.",
    )
    parser.add_argument(
        "--config_name", default=None, type=str, required=True, help="Config name of SwitchTransformers model."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output FLAX model."
    )
    args = parser.parse_args()
    convert_switch_transformersx_checkpoint_to_flax(
        args.switch_t5x_checkpoint_path, args.config_name, args.pytorch_dump_folder_path
    )