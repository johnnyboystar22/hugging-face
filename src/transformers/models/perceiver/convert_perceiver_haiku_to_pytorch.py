# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
"""Convert Perceiver checkpoints originally implemented in Haiku."""


import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import haiku as hk
from transformers import (
    PerceiverBasicDecoder,
    PerceiverConfig,
    PerceiverModel,
    PerceiverTextPostprocessor,
    PerceiverTextPreprocessor,
    PerceiverTokenizer,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def to_string(inputs: np.ndarray) -> str:
    inputs_no_special = inputs[inputs >= 6] - 6
    decoded_bytes = inputs_no_special.astype(np.uint8).tobytes()
    return decoded_bytes.decode("utf-8", errors="replace")


def prepare_dummy_inputs():
    # here we prepare a dummy input for MLM
    def to_int(inputs: str) -> np.ndarray:
        if isinstance(inputs, str):
            inputs = inputs.encode("utf-8")
        encoded = np.frombuffer(inputs, np.uint8).astype(np.int32)
        encoded = encoded + 6
        return encoded.astype(np.int32)

    def pad(max_sequence_length: int, inputs, input_mask):
        input_len = inputs.shape[1]
        assert input_len <= max_sequence_length
        pad_len = max_sequence_length - input_len
        padded_inputs = np.pad(inputs, pad_width=((0, 0), (0, pad_len)), constant_values=0)
        padded_mask = np.pad(input_mask, pad_width=((0, 0), (0, pad_len)), constant_values=0)
        return padded_inputs, padded_mask

    input_str = "This is an incomplete sentence where some words are missing."
    input_tokens = to_int(input_str)

    # Mask " missing.". Note that the model performs much better if the masked chunk
    # starts with a space.
    input_tokens[51:60] = 3
    inputs = input_tokens[None]
    input_mask = np.ones_like(inputs)
    inputs, input_mask = pad(2048, inputs, input_mask)
    inputs = torch.from_numpy(inputs)
    input_mask = torch.from_numpy(input_mask)

    return inputs, input_mask


def rename_keys(state_dict):
    for name in list(state_dict):
        param = state_dict.pop(name)

        # rename latent embeddings
        name = name.replace("perceiver_encoder/~/trainable_position_encoding/pos_embs", "embeddings.latents")

        # rename decoder queries
        name = name.replace(
            "basic_decoder/~/trainable_position_encoding/pos_embs", "decoder.output_position_encodings.weight"
        )

        # rename embedding decoder bias
        name = name.replace("embedding_decoder/bias", "output_postprocessor.bias")

        # rename preprocessor embeddings
        name = name.replace("embed/embeddings", "input_preprocessor.embeddings.weight")
        name = name.replace("trainable_position_encoding/pos_embs", "input_preprocessor.position_embeddings.weight")

        # rename prefixes
        if name.startswith("perceiver_encoder/~/"):
            if "self_attention" in name:
                suffix = "self_attends."
            else:
                suffix = ""
            name = name.replace("perceiver_encoder/~/", "encoder." + suffix)
        if name.startswith("basic_decoder/cross_attention/"):
            name = name.replace("basic_decoder/cross_attention/", "decoder.decoding_cross_attention.")
        # rename layernorm parameters
        if "offset" in name:
            name = name.replace("offset", "bias")
        if "scale" in name:
            name = name.replace("scale", "weight")
        # in HuggingFace, the layernorm in between attention + MLP is just called "layernorm"
        # rename layernorm in between attention + MLP of cross-attention
        if "cross_attention" in name and "layer_norm_2" in name:
            name = name.replace("layer_norm_2", "layernorm")
        # rename layernorm in between attention + MLP of self-attention
        if "self_attention" in name and "layer_norm_1" in name:
            name = name.replace("layer_norm_1", "layernorm")

        # in HuggingFace, the layernorms for queries + keys are called "layernorm1" and "layernorm2"
        if "cross_attention" in name and "layer_norm_1" in name:
            name = name.replace("layer_norm_1", "attention.self.layernorm2")
        if "cross_attention" in name and "layer_norm" in name:
            name = name.replace("layer_norm", "attention.self.layernorm1")
        if "self_attention" in name and "layer_norm" in name:
            name = name.replace("layer_norm", "attention.self.layernorm1")

        # rename special characters by dots
        name = name.replace("-", ".")
        name = name.replace("/", ".")
        # rename keys, queries, values and output of attention layers
        if ("cross_attention" in name or "self_attention" in name) and "mlp" not in name:
            if "linear.b" in name:
                name = name.replace("linear.b", "self.query.bias")
            if "linear.w" in name:
                name = name.replace("linear.w", "self.query.weight")
            if "linear_1.b" in name:
                name = name.replace("linear_1.b", "self.key.bias")
            if "linear_1.w" in name:
                name = name.replace("linear_1.w", "self.key.weight")
            if "linear_2.b" in name:
                name = name.replace("linear_2.b", "self.value.bias")
            if "linear_2.w" in name:
                name = name.replace("linear_2.w", "self.value.weight")
            if "linear_3.b" in name:
                name = name.replace("linear_3.b", "output.dense.bias")
            if "linear_3.w" in name:
                name = name.replace("linear_3.w", "output.dense.weight")
        if "self_attention_" in name:
            name = name.replace("self_attention_", "")
        if "self_attention" in name:
            name = name.replace("self_attention", "0")
        # rename dense layers of 2-layer MLP
        if "mlp" in name:
            if "linear.b" in name:
                name = name.replace("linear.b", "dense1.bias")
            if "linear.w" in name:
                name = name.replace("linear.w", "dense1.weight")
            if "linear_1.b" in name:
                name = name.replace("linear_1.b", "dense2.bias")
            if "linear_1.w" in name:
                name = name.replace("linear_1.w", "dense2.weight")

        # finally, TRANSPOSE if kernel and not embedding layer, and set value
        if name[-6:] == "weight" and "input_preprocessor" not in name and "output_position_encodings" not in name:
            param = np.transpose(param)

        # preprocessor embeddings need special treatment
        state_dict[name] = torch.from_numpy(param)


@torch.no_grad()
def convert_perceiver_checkpoint(pickle_file, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our Perceiver structure.
    """

    # load parameters as FlatMapping data structure
    with open(pickle_file, "rb") as f:
        params = pickle.loads(f.read())

    # turn into initial state dict
    state_dict = dict()
    for scope_name, parameters in hk.data_structures.to_mutable_dict(params).items():
        for param_name, param in parameters.items():
            state_dict[scope_name + "/" + param_name] = param

    # rename keys
    rename_keys(state_dict)

    # load HuggingFace model
    config = PerceiverConfig()
    preprocessor = PerceiverTextPreprocessor(config)
    decoder = PerceiverBasicDecoder(
        config,
        output_num_channels=1280,
        output_index_dims=2048,
        qk_channels=8 * 32,
        v_channels=config.d_model,
        num_heads=8,
        use_query_residual=False,
        final_project=False,
    )
    postprocessor = PerceiverTextPostprocessor(config)
    model = PerceiverModel(
        config, input_preprocessor=preprocessor, decoder=decoder, output_postprocessor=postprocessor
    )
    model.eval()

    # load weights
    model.load_state_dict(state_dict)

    # forward pass on dummy input
    inputs, input_mask = prepare_dummy_inputs()

    outputs = model(inputs=inputs, attention_mask=input_mask)

    # verify outputs
    print("Shape of outputs:", outputs.shape)
    print("First elements of outputs:", outputs[0, :3, :3])

    masked_tokens_predictions = outputs[0, 51:60].argmax(dim=-1)
    print("Greedy predictions:")
    print(masked_tokens_predictions)
    print()
    print("Predicted string:")
    print(to_string(masked_tokens_predictions.numpy()))

    # Finally, save files
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--pickle_file",
        type=str,
        default=None,
        required=True,
        help="Path to local pickle file of a Perceiver checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the output PyTorch model directory, provided as a string.",
    )

    args = parser.parse_args()
    convert_perceiver_checkpoint(args.pickle_file, args.pytorch_dump_folder_path)
