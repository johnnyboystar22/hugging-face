# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import inspect
import unittest
from typing import List

from datasets import load_dataset

from transformers.models.superglue.configuration_superglue import SuperGlueConfig
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    import torch

    from transformers import SuperGlueForKeypointMatching

if is_vision_available():
    from transformers import AutoImageProcessor


class SuperGlueModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        image_width=80,
        image_height=60,
        keypoint_detector_config=None,
        descriptor_dim: int = 64,
        keypoint_encoder_sizes: List[int] = [32, 64],
        gnn_layers_types: List[str] = ["self", "cross"] * 2,
        num_heads: int = 4,
        sinkhorn_iterations: int = 100,
        matching_threshold: float = 0.2,
    ):
        if keypoint_detector_config is None:
            keypoint_detector_config = {
                "encoder_hidden_sizes": [32, 64],
                "decoder_hidden_size": 64,
                "keypoint_decoder_dim": 65,
                "descriptor_decoder_dim": 64,
                "keypoint_threshold": 0.005,
                "max_keypoints": 256,
                "nms_radius": 4,
                "border_removal_distance": 4,
            }
        self.parent = parent
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height

        self.keypoint_detector_config = keypoint_detector_config
        self.descriptor_dim = descriptor_dim
        self.keypoint_encoder_sizes = keypoint_encoder_sizes
        self.gnn_layers_types = gnn_layers_types
        self.num_heads = num_heads
        self.sinkhorn_iterations = sinkhorn_iterations
        self.matching_threshold = matching_threshold

    def prepare_config_and_inputs(self):
        # SuperGlue expects a grayscale image as input
        pixel_values = floats_tensor([self.batch_size, 2, 3, self.image_height, self.image_width])
        config = self.get_config()
        return config, pixel_values

    def get_config(self):
        return SuperGlueConfig(
            keypoint_detector_config=self.keypoint_detector_config,
            descriptor_dim=self.descriptor_dim,
            keypoint_encoder_sizes=self.keypoint_encoder_sizes,
            gnn_layers_types=self.gnn_layers_types,
            num_heads=self.num_heads,
            sinkhorn_iterations=self.sinkhorn_iterations,
            matching_threshold=self.matching_threshold,
        )

    def create_and_check_model(self, config, pixel_values):
        model = SuperGlueForKeypointMatching(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        maximum_num_matches = result.mask.shape[-1]
        self.parent.assertEqual(
            result.keypoints.shape,
            (self.batch_size, 2, maximum_num_matches, 2),
        )
        self.parent.assertEqual(
            result.matches.shape,
            (self.batch_size, 2, maximum_num_matches),
        )
        self.parent.assertEqual(
            result.matching_scores.shape,
            (self.batch_size, 2, maximum_num_matches),
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class SuperGlueModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (SuperGlueForKeypointMatching,) if is_torch_available() else ()
    all_generative_model_classes = () if is_torch_available() else ()

    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    has_attentions = True

    def setUp(self):
        self.model_tester = SuperGlueModelTester(self)
        self.config_tester = ConfigTester(self, config_class=SuperGlueConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    @unittest.skip(reason="SuperGlueForKeypointMatching does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="SuperGlueForKeypointMatching does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="SuperGlueForKeypointMatching does not use feedforward chunking")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="SuperGlueForKeypointMatching is not trainable")
    def test_training(self):
        pass

    @unittest.skip(reason="SuperGlueForKeypointMatching is not trainable")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="SuperGlueForKeypointMatching is not trainable")
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(reason="SuperGlueForKeypointMatching is not trainable")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="SuperGlue does not output any loss term in the forward pass")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states
            maximum_num_matches = outputs.mask.shape[-1]

            hidden_states_sizes = (
                self.model_tester.keypoint_encoder_sizes
                + [self.model_tester.descriptor_dim]
                + [self.model_tester.descriptor_dim, self.model_tester.descriptor_dim * 2]
                * len(self.model_tester.gnn_layers_types)
                + [self.model_tester.descriptor_dim] * 2
            )

            for i, hidden_states_size in enumerate(hidden_states_sizes):
                self.assertListEqual(
                    list(hidden_states[i].shape[-2:]),
                    [maximum_num_matches, hidden_states_size],
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_attention_outputs(self):
        def check_attention_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            attentions = outputs.attentions
            maximum_num_matches = outputs.mask.shape[-1]

            expected_attention_shape = [self.model_tester.num_heads, maximum_num_matches, maximum_num_matches]

            for i, attention in enumerate(attentions):
                self.assertListEqual(
                    list(attention.shape[-3:]),
                    expected_attention_shape,
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            check_attention_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True

            check_attention_output(inputs_dict, config, model_class)

    @slow
    def test_model_from_pretrained(self):
        from_pretrained_ids = ["stevenbucaille/superglue_indoor", "stevenbucaille/superglue_outdoor"]
        for model_name in from_pretrained_ids:
            model = SuperGlueForKeypointMatching.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_forward_labels_should_be_none(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                model_inputs = self._prepare_for_class(inputs_dict, model_class)
                # Provide an arbitrary sized Tensor as labels to model inputs
                model_inputs["labels"] = torch.rand((128, 128))

                with self.assertRaises(ValueError) as cm:
                    model(**model_inputs)
                self.assertEqual(ValueError, cm.exception.__class__)


def prepare_imgs():
    dataset = load_dataset("stevenbucaille/image_matching_fixtures", split="train")
    image1 = dataset[0]["image"]
    image2 = dataset[1]["image"]
    image3 = dataset[2]["image"]
    return [[image3, image1], [image3, image2]]


@require_torch
@require_vision
class SuperGlueModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return (
            AutoImageProcessor.from_pretrained("stevenbucaille/superglue_outdoor") if is_vision_available() else None
        )

    @slow
    def test_inference(self):
        model = SuperGlueForKeypointMatching.from_pretrained(
            "stevenbucaille/superglue_outdoor", matching_threshold=0.2
        ).to(torch_device)
        preprocessor = self.default_image_processor
        images = prepare_imgs()
        inputs = preprocessor(images=images, return_tensors="pt").to(torch_device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

        expected_number_keypoints_image0 = 559
        expected_number_keypoints_image1 = 592
        expected_number_keypoints_image2 = 865
        expected_max_number_keypoints = max(
            [expected_number_keypoints_image0, expected_number_keypoints_image1, expected_number_keypoints_image2]
        )
        expected_matches_shape = torch.Size((len(images), 2, expected_max_number_keypoints))
        expected_matching_scores_shape = torch.Size((len(images), 2, expected_max_number_keypoints))

        # Check output shapes
        self.assertEqual(outputs.matches.shape, expected_matches_shape)
        self.assertEqual(outputs.matching_scores.shape, expected_matching_scores_shape)

        expected_matches_values = torch.tensor([-1, 0, 2, 4, -1, 3, 6, -1, -1, -1], dtype=torch.int32).to(torch_device)
        expected_matching_scores_values = torch.tensor(
            [1.1161e-5, 9.8031e-1, 8.8953e-1, 9.6738e-1, 0, 9.767e-1, 8.1111e-1, 2.1811e-2, 9.7602e-4, 1.0968e-3]
        ).to(torch_device)

        predicted_matches_values = outputs.matches[0, 0, :10]
        predicted_matching_scores_values = outputs.matching_scores[0, 0, :10]

        self.assertTrue(torch.allclose(predicted_matches_values, expected_matches_values, atol=1e-4))

        self.assertTrue(torch.allclose(predicted_matching_scores_values, expected_matching_scores_values, atol=1e-4))

        expected_number_of_matches = 144
        predicted_number_of_matches = torch.sum(outputs.matches[0][0] != -1).item()
        self.assertEqual(predicted_number_of_matches, expected_number_of_matches)
