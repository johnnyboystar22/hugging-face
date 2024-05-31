# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Idefics2 model."""

import copy
import gc
import unittest
from io import BytesIO

import requests

from transformers import (
    AutoProcessor,
    Idefics2Config,
    Idefics2ForConditionalGeneration,
    Idefics2ForSequenceClassification,
    Idefics2Model,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import require_bitsandbytes, require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch
else:
    is_torch_greater_or_equal_than_2_0 = False

if is_vision_available():
    from PIL import Image


class Idefics2VisionText2TextModelTester:
    def __init__(
        self,
        parent,
        is_training=True,
        batch_size=2,
        num_images=2,
        seq_length=10,
        vision_config={
            "image_size": 12,
            "patch_size": 12,
            "num_channels": 3,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 32,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "initializer_range": 0.02,
        },
        perceiver_config={
            "hidden_act": "silu",
            "resampler_n_latents": 2,
            "resampler_depth": 2,
            "resampler_n_heads": 2,
            "num_key_value_heads": 1,
            "resampler_head_dim": 12,
            "attention_dropout": 0.0,
        },
        text_config={
            "vocab_size": 100,
            "hidden_size": 64,
            "intermediate_size": 56,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "hidden_act": "silu",
            "max_position_embeddings": 256,
            "initializer_range": 0.02,
            "rms_norm_eps": 1e-6,
            "pad_token_id": 0,  # None in the original configuration_mistral, we set it to the unk_token_id
            "bos_token_id": 1,
            "eos_token_id": 2,
            "image_token_id": 32_001,
            "tie_word_embeddings": False,
            "rope_theta": 10000.0,
            "sliding_window": 32,
            "attention_dropout": 0.0,
        },
        use_cache=False,
        tie_word_embeddings=False,
        image_token_id=99,
        type_sequence_label_size=2,
    ):
        self.parent = parent
        self.is_training = is_training
        self.batch_size = batch_size
        self.num_images = num_images
        self.num_channels = 3
        self.seq_length = seq_length
        self.use_cache = use_cache
        self.image_token_id = image_token_id
        self.tie_word_embeddings = tie_word_embeddings
        # Hack - add properties here so use common tests
        self.vocab_size = text_config["vocab_size"]
        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.hidden_size = text_config["hidden_size"]
        self.type_sequence_label_size = type_sequence_label_size

        self.vision_config = vision_config
        self.perceiver_config = perceiver_config
        self.text_config = text_config

    def get_config(self):
        return Idefics2Config(
            use_cache=self.use_cache,
            image_token_id=self.image_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
            vision_config=self.vision_config,
            perceiver_config=self.perceiver_config,
            text_config=self.text_config,
            vocab_size=self.vocab_size,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.num_images,
                self.vision_config["num_channels"],
                self.vision_config["image_size"],
                self.vision_config["image_size"],
            ]
        )
        config = self.get_config()

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 2) + 1

        # For simplicity just set the last n tokens to the image token
        n_image_tokens_per_batch = self.num_images * self.perceiver_config["resampler_n_latents"]
        input_ids[:, -n_image_tokens_per_batch:] = self.image_token_id
        attention_mask = input_ids.ne(1).to(torch_device)
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class Idefics2ModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Model tester for `Idefics2`.
    """

    all_model_classes = (Idefics2Model,) if is_torch_available() else ()
    fx_compatible = False
    test_torchscript = False
    test_pruning = False
    test_resize_embeddings = True
    test_head_masking = False

    def setUp(self):
        self.model_tester = Idefics2VisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Idefics2Config, has_text_modality=False)

    @unittest.skip("input_embeds cannot be passed in without input_ids")
    def test_inputs_embeds():
        pass

    @unittest.skip("input_embeds cannot be passed in without input_ids")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip("Model does not support padding right")
    def test_flash_attn_2_generate_padding_right(self):
        pass

    @unittest.skip("Model does not support padding right")
    def test_flash_attn_2_inference_padding_right(self):
        pass

    # We need to override as we need to prepare such that the image token is the last token
    def test_resize_tokens_embeddings(self):
        (original_config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = model_class(config)
            model.to(torch_device)

            if self.model_tester.is_training is False:
                model.eval()

            model_vocab_size = config.text_config.vocab_size
            # Retrieve the embeddings and clone theme
            model_embed = model.resize_token_embeddings(model_vocab_size)
            cloned_embeddings = model_embed.weight.clone()

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size + 10)
            self.assertEqual(model.config.text_config.vocab_size, model_vocab_size + 10)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] + 10)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size - 15)
            self.assertEqual(model.config.text_config.vocab_size, model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] - 15)

            # Ignore copy
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            # Input ids should be clamped to the maximum size of the vocabulary - 1 and the image token should be the last token
            inputs_dict["input_ids"].clamp_(max=model_vocab_size - 15 - 2)
            n_images = self.model_tester.num_images * self.model_tester.perceiver_config["resampler_n_latents"]
            model.image_token_id = model_vocab_size - 15 - 1
            inputs_dict["input_ids"][:, -n_images:] = model.image_token_id

            # make sure that decoder_input_ids are resized as well
            if "decoder_input_ids" in inputs_dict:
                inputs_dict["decoder_input_ids"].clamp_(max=model_vocab_size - 15 - 1)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that adding and removing tokens has not modified the first part of the embedding matrix.
            models_equal = True
            for p1, p2 in zip(cloned_embeddings, model_embed.weight):
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)

            config = copy.deepcopy(original_config)
            model = model_class(config)
            model.to(torch_device)

            model_vocab_size = config.text_config.vocab_size
            model.resize_token_embeddings(model_vocab_size + 10, pad_to_multiple_of=1)
            self.assertTrue(model.config.text_config.vocab_size + 10, model_vocab_size)

            model_embed = model.resize_token_embeddings(model_vocab_size, pad_to_multiple_of=64)
            self.assertTrue(model_embed.weight.shape[0] // 64, 0)

            self.assertTrue(model_embed.weight.shape[0], model.config.text_config.vocab_size)
            self.assertTrue(model.config.text_config.vocab_size, model.vocab_size)

            model_embed = model.resize_token_embeddings(model_vocab_size + 13, pad_to_multiple_of=64)
            self.assertTrue(model_embed.weight.shape[0] // 64, 0)

            # Check that resizing a model to a multiple of pad_to_multiple leads to a model of exactly that size
            target_dimension = 128
            model_embed = model.resize_token_embeddings(target_dimension, pad_to_multiple_of=64)
            self.assertTrue(model_embed.weight.shape[0], target_dimension)

            with self.assertRaisesRegex(
                ValueError,
                "Asking to pad the embedding matrix to a multiple of `1.3`, which is not and integer. Please make sure to pass an integer",
            ):
                model.resize_token_embeddings(model_vocab_size, pad_to_multiple_of=1.3)

    # We need to override as we need to prepare such that the image token is the last token
    def test_resize_embeddings_untied(self):
        (original_config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()

        original_config.tie_word_embeddings = False

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = model_class(config).to(torch_device)

            # if no output embeddings -> leave test
            if model.get_output_embeddings() is None:
                continue

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_vocab_size = config.text_config.vocab_size
            model.resize_token_embeddings(model_vocab_size + 10)
            self.assertEqual(model.config.text_config.vocab_size, model_vocab_size + 10)
            output_embeds = model.get_output_embeddings()
            self.assertEqual(output_embeds.weight.shape[0], model_vocab_size + 10)
            # Check bias if present
            if output_embeds.bias is not None:
                self.assertEqual(output_embeds.bias.shape[0], model_vocab_size + 10)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model.resize_token_embeddings(model_vocab_size - 15)
            self.assertEqual(model.config.text_config.vocab_size, model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            output_embeds = model.get_output_embeddings()
            self.assertEqual(output_embeds.weight.shape[0], model_vocab_size - 15)
            # Check bias if present
            if output_embeds.bias is not None:
                self.assertEqual(output_embeds.bias.shape[0], model_vocab_size - 15)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            # Input ids should be clamped to the maximum size of the vocabulary - 1 and the image token should be the last token
            inputs_dict["input_ids"].clamp_(max=model_vocab_size - 15 - 2)
            n_images = self.model_tester.num_images * self.model_tester.perceiver_config["resampler_n_latents"]
            model.image_token_id = model_vocab_size - 15 - 1
            inputs_dict["input_ids"][:, -n_images:] = model.image_token_id

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

    def test_idefics2_sequence_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = Idefics2ForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        input_dict["labels"] = sequence_labels
        result = model(**input_dict)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, config.num_labels))


@require_torch
class Idefics2ForConditionalGenerationModelTest(GenerationTesterMixin, ModelTesterMixin, unittest.TestCase):
    """
    Model tester for `Idefics2ForConditionalGeneration`.
    """

    all_model_classes = (
        (Idefics2ForConditionalGeneration, Idefics2ForSequenceClassification) if is_torch_available() else ()
    )
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = True
    test_head_masking = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = Idefics2VisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Idefics2Config, has_text_modality=False)

    @unittest.skip("input_embeds cannot be passed in without input_ids")
    def test_inputs_embeds():
        pass

    @unittest.skip("Model does not support padding right")
    def test_flash_attn_2_generate_padding_right(self):
        pass

    @unittest.skip("Model does not support padding right")
    def test_flash_attn_2_inference_padding_right(self):
        pass

    # We need to override as we need to prepare such that the image token is the last token
    def test_resize_tokens_embeddings(self):
        (original_config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = model_class(config)
            model.to(torch_device)

            model_vocab_size = config.text_config.vocab_size
            # Retrieve the embeddings and clone theme
            model_embed = model.resize_token_embeddings(model_vocab_size)
            cloned_embeddings = model_embed.weight.clone()

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size + 10)
            self.assertEqual(model.config.text_config.vocab_size, model_vocab_size + 10)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] + 10)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size - 15)
            self.assertEqual(model.config.text_config.vocab_size, model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] - 15)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            # Input ids should be clamped to the maximum size of the vocabulary - 1 and the image token should be the last token
            inputs_dict["input_ids"].clamp_(max=model_vocab_size - 15 - 2)
            n_images = self.model_tester.num_images * self.model_tester.perceiver_config["resampler_n_latents"]
            model.model.image_token_id = model_vocab_size - 15 - 1
            inputs_dict["input_ids"][:, -n_images:] = model.model.image_token_id

            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that adding and removing tokens has not modified the first part of the embedding matrix.
            models_equal = True
            for p1, p2 in zip(cloned_embeddings, model_embed.weight):
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)

            config = copy.deepcopy(original_config)
            model = model_class(config)
            model.to(torch_device)

            model_vocab_size = config.text_config.vocab_size
            model.resize_token_embeddings(model_vocab_size + 10, pad_to_multiple_of=1)
            self.assertTrue(model.config.text_config.vocab_size + 10, model_vocab_size)

            model_embed = model.resize_token_embeddings(model_vocab_size, pad_to_multiple_of=64)
            self.assertTrue(model_embed.weight.shape[0] // 64, 0)

            self.assertTrue(model_embed.weight.shape[0], model.config.text_config.vocab_size)
            self.assertTrue(model.config.text_config.vocab_size, model.vocab_size)

            model_embed = model.resize_token_embeddings(model_vocab_size + 13, pad_to_multiple_of=64)
            self.assertTrue(model_embed.weight.shape[0] // 64, 0)

            # Check that resizing a model to a multiple of pad_to_multiple leads to a model of exactly that size
            target_dimension = 128
            model_embed = model.resize_token_embeddings(target_dimension, pad_to_multiple_of=64)
            self.assertTrue(model_embed.weight.shape[0], target_dimension)

            with self.assertRaisesRegex(
                ValueError,
                "Asking to pad the embedding matrix to a multiple of `1.3`, which is not and integer. Please make sure to pass an integer",
            ):
                model.resize_token_embeddings(model_vocab_size, pad_to_multiple_of=1.3)

    # We need to override as we need to prepare such that the image token is the last token
    def test_resize_embeddings_untied(self):
        (original_config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()

        original_config.tie_word_embeddings = False

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = model_class(config).to(torch_device)

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_vocab_size = config.text_config.vocab_size
            model.resize_token_embeddings(model_vocab_size + 10)
            self.assertEqual(model.config.text_config.vocab_size, model_vocab_size + 10)
            output_embeds = model.get_output_embeddings()
            self.assertEqual(output_embeds.weight.shape[0], model_vocab_size + 10)
            # Check bias if present
            if output_embeds.bias is not None:
                self.assertEqual(output_embeds.bias.shape[0], model_vocab_size + 10)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model.resize_token_embeddings(model_vocab_size - 15)
            self.assertEqual(model.config.text_config.vocab_size, model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            output_embeds = model.get_output_embeddings()
            self.assertEqual(output_embeds.weight.shape[0], model_vocab_size - 15)
            # Check bias if present
            if output_embeds.bias is not None:
                self.assertEqual(output_embeds.bias.shape[0], model_vocab_size - 15)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            # Input ids should be clamped to the maximum size of the vocabulary - 1 and the image token should be the last token
            inputs_dict["input_ids"].clamp_(max=model_vocab_size - 15 - 2)
            n_images = self.model_tester.num_images * self.model_tester.perceiver_config["resampler_n_latents"]
            model.model.image_token_id = model_vocab_size - 15 - 1
            inputs_dict["input_ids"][:, -n_images:] = model.model.image_token_id

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))


@require_torch
class Idefics2ForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b-base")
        self.image1 = Image.open(
            BytesIO(
                requests.get(
                    "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
                ).content
            )
        )
        self.image2 = Image.open(
            BytesIO(requests.get("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg").content)
        )
        self.image3 = Image.open(
            BytesIO(
                requests.get(
                    "https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg"
                ).content
            )
        )

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    @slow
    def test_integration_test(self):
        model = Idefics2ForConditionalGeneration.from_pretrained(
            "HuggingFaceM4/idefics2-8b-base",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.to(torch_device)

        # Create inputs
        text = "<image>In this image, we see"
        images = self.image1
        inputs = self.processor(text=text, images=images, return_tensors="pt", padding=True)
        inputs.to(torch_device)

        generated_ids = model.generate(**inputs, max_new_tokens=10)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Batch affects generated text. Single batch output: ['In this image, we see the Statue of Liberty in the foreground and']
        expected_generated_text = "In this image, we see the Statue of Liberty, the New York City"
        self.assertEqual(generated_texts[0], expected_generated_text)

    @slow
    @require_bitsandbytes
    def test_integration_test_4bit(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model = Idefics2ForConditionalGeneration.from_pretrained(
            "HuggingFaceM4/idefics2-8b-base", load_in_4bit=True, device_map="auto"
        )

        # Create pixel inputs
        text = ["<image>In this image, we see", "bla, bla <image><image>"]
        images = [[self.image1], [self.image2, self.image3]]
        inputs = self.processor(text=text, images=images, padding=True, return_tensors="pt")

        generated_ids = model.generate(**inputs, max_new_tokens=10)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        expected_generated_text = "In this image, we see the Statue of Liberty, the Hudson River,"
        self.assertEqual(generated_texts[0], expected_generated_text)
