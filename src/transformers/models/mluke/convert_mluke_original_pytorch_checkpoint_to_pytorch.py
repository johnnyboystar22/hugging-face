# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
"""Convert LUKE checkpoint."""

import argparse
import json
import os

import torch

from transformers import LukeConfig, LukeModel, XLMRobertaTokenizer, MLukeTokenizer
from transformers.tokenization_utils_base import AddedToken


@torch.no_grad()
def convert_luke_checkpoint(checkpoint_path, metadata_path, entity_vocab_path, pytorch_dump_folder_path, model_size):
    # Load configuration defined in the metadata file
    with open(metadata_path) as metadata_file:
        metadata = json.load(metadata_file)
    config = LukeConfig(use_entity_aware_attention=True, **metadata["model_config"])

    # Load in the weights from the checkpoint_path
    state_dict = torch.load(checkpoint_path, map_location="cpu")["module"]

    # Load the entity vocab file
    entity_vocab = load_original_entity_vocab(entity_vocab_path)
    # add an entry for [MASK2]
    entity_vocab["[MASK2]"] = max(entity_vocab.values()) + 1
    config.entity_vocab_size += 1

    tokenizer = XLMRobertaTokenizer.from_pretrained(metadata["model_config"]["bert_model_name"])

    # Add special tokens to the token vocabulary for downstream tasks
    entity_token_1 = AddedToken("<ent>", lstrip=False, rstrip=False)
    entity_token_2 = AddedToken("<ent2>", lstrip=False, rstrip=False)
    tokenizer.add_special_tokens(dict(additional_special_tokens=[entity_token_1, entity_token_2]))
    config.vocab_size += 2

    print(f"Saving tokenizer to {pytorch_dump_folder_path}")
    tokenizer.save_pretrained(pytorch_dump_folder_path)
    with open(os.path.join(pytorch_dump_folder_path, "tokenizer_config.json"), "r") as f:
        tokenizer_config = json.load(f)
    tokenizer_config["tokenizer_class"] = "MLukeTokenizer"
    with open(os.path.join(pytorch_dump_folder_path, "tokenizer_config.json"), "w") as f:


        json.dump(tokenizer_config, f)

    with open(os.path.join(pytorch_dump_folder_path, MLukeTokenizer.vocab_files_names["entity_vocab_file"]), "w") as f:
        json.dump(entity_vocab, f)

    tokenizer = MLukeTokenizer.from_pretrained(pytorch_dump_folder_path)

    # Initialize the embeddings of the special tokens
    word_emb = state_dict["embeddings.word_embeddings.weight"]
    ent_emb = word_emb[tokenizer.convert_tokens_to_ids(["@"])[0]].unsqueeze(0)
    ent2_emb = word_emb[tokenizer.convert_tokens_to_ids(["#"])[0]].unsqueeze(0)
    state_dict["embeddings.word_embeddings.weight"] = torch.cat([word_emb, ent_emb, ent2_emb])

    # Initialize the query layers of the entity-aware self-attention mechanism
    for layer_index in range(config.num_hidden_layers):
        for matrix_name in ["query.weight", "query.bias"]:
            prefix = f"encoder.layer.{layer_index}.attention.self."
            state_dict[prefix + "w2e_" + matrix_name] = state_dict[prefix + matrix_name]
            state_dict[prefix + "e2w_" + matrix_name] = state_dict[prefix + matrix_name]
            state_dict[prefix + "e2e_" + matrix_name] = state_dict[prefix + matrix_name]

    # Initialize the embedding of the [MASK2] entity using that of the [MASK] entity for downstream tasks
    entity_emb = state_dict["entity_embeddings.entity_embeddings.weight"]
    entity_mask_emb = entity_emb[entity_vocab["[MASK]"]].unsqueeze(0)
    state_dict["entity_embeddings.entity_embeddings.weight"] = torch.cat([entity_emb, entity_mask_emb])

    model = LukeModel(config=config).eval()

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if not (
        all(
            key.startswith("entity_predictions") or key.startswith("lm_head") or key == "embeddings.position_ids"
            for key in unexpected_keys
        )
    ):
        raise ValueError(
            f"Unexpected keys {', '.join([key for key in unexpected_keys if not (key.startswith('entity_predictions') or key.startswith('lm_head'))])}"
        )

    # Check outputs
    tokenizer = MLukeTokenizer.from_pretrained(pytorch_dump_folder_path, task="entity_classification")

    text = "ISO 639-3 uses the code fas for the dialects spoken across Iran and アフガニスタン (Afghanistan)."
    span = (0, 9)
    encoding = tokenizer(text, entity_spans=[span], return_tensors="pt")

    outputs = model(**encoding)

    # Verify word hidden states
    if model_size == "large":
        raise NotImplementedError
    else:  # base
        expected_shape = torch.Size((1, 45, 768))
        expected_slice = torch.tensor([[0.0892, 0.0596, -0.2819], [0.0134, 0.1199, 0.0573], [-0.0169, 0.0927, 0.0644]])

    if not (outputs.last_hidden_state.shape == expected_shape):
        raise ValueError(
            f"Outputs.last_hidden_state.shape is {outputs.last_hidden_state.shape}, Expected shape is {expected_shape}"
        )
    if not torch.allclose(outputs.last_hidden_state[0, :3, :3], expected_slice, atol=1e-4):
        raise ValueError

    # Verify entity hidden states
    if model_size == "large":
        raise NotImplementedError
    else:  # base
        expected_shape = torch.Size((1, 1, 768))
        expected_slice = torch.tensor([[-0.1482, 0.0609, 0.0322]])

    if not (outputs.entity_last_hidden_state.shape != expected_shape):
        raise ValueError(
            f"Outputs.entity_last_hidden_state.shape is {outputs.entity_last_hidden_state.shape}, Expected shape is {expected_shape}"
        )
    if not torch.allclose(outputs.entity_last_hidden_state[0, :3, :3], expected_slice, atol=1e-4):
        raise ValueError

    # Finally, save our PyTorch model and tokenizer
    print("Saving PyTorch model to {}".format(pytorch_dump_folder_path))
    model.save_pretrained(pytorch_dump_folder_path)


def load_original_entity_vocab(entity_vocab_path):
    SPECIAL_TOKENS = ["[MASK]", "[PAD]", "[UNK]"]

    data = [json.loads(line) for line in open(entity_vocab_path)]

    new_mapping = {}
    for entry in data:
        entity_id = entry["id"]
        for entity_name, language in entry["entities"]:
            if entity_name in SPECIAL_TOKENS:
                new_mapping[entity_name] = entity_id
                break
            new_entity_name = f"{language}:{entity_name}"
            new_mapping[new_entity_name] = entity_id
    return new_mapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--checkpoint_path", type=str, help="Path to a pytorch_model.bin file.")
    parser.add_argument(
        "--metadata_path", default=None, type=str, help="Path to a metadata.json file, defining the configuration."
    )
    parser.add_argument(
        "--entity_vocab_path",
        default=None,
        type=str,
        help="Path to an entity_vocab.tsv file, containing the entity vocabulary.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to where to dump the output PyTorch model."
    )
    parser.add_argument(
        "--model_size", default="base", type=str, choices=["base", "large"], help="Size of the model to be converted."
    )
    args = parser.parse_args()
    convert_luke_checkpoint(
        args.checkpoint_path,
        args.metadata_path,
        args.entity_vocab_path,
        args.pytorch_dump_folder_path,
        args.model_size,
    )
