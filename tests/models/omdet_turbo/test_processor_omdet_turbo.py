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

import tempfile

import numpy as np
import pytest

from transformers import AutoProcessor, CLIPTokenizerFast, OmDetTurboProcessor
from transformers.testing_utils import TestCasePlus, require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available


IMAGE_MEAN = [123.675, 116.28, 103.53]
IMAGE_STD = [58.395, 57.12, 57.375]

if is_torch_available():
    import torch

    from transformers.models.omdet_turbo.modeling_omdet_turbo import OmDetTurboObjectDetectionOutput

if is_vision_available():
    from PIL import Image

    from transformers import DetrImageProcessor


@require_torch
@require_vision
class OmDetTurboProcessorTest(TestCasePlus):
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        super().setUp()

        self.checkpoint_path = self.get_auto_remove_tmp_dir()

        image_processor = DetrImageProcessor(
            return_tensors="pt",
            size=[640, 640],
            do_rescale=False,
            image_mean=IMAGE_MEAN,
            image_std=IMAGE_STD,
            do_pad=False,
        )
        tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

        processor = OmDetTurboProcessor(image_processor, tokenizer)

        processor.save_pretrained(self.checkpoint_path)

        self.input_keys = ["tasks", "labels", "pixel_values"]
        self.text_input_keys = ["input_ids", "attention_mask"]

        self.batch_size = 5
        self.num_queries = 5
        self.embed_dim = 3

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.checkpoint_path, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.checkpoint_path, **kwargs).image_processor

    def prepare_image_inputs(self):
        """This function prepares a list of PIL images, or a list of numpy arrays if one specifies numpify=True,
        or a list of PyTorch tensors if one specifies torchify=True.
        """

        image_inputs = [np.random.randint(255, size=(3, 30, 400), dtype=np.uint8)]

        image_inputs = [Image.fromarray(np.moveaxis(x, 0, -1)) for x in image_inputs]

        return image_inputs

    def get_fake_omdet_turbo_output(self):
        torch.manual_seed(42)
        return OmDetTurboObjectDetectionOutput(
            decoder_bboxes=torch.rand(self.batch_size, self.num_queries, 4),
            decoder_cls=torch.rand(self.batch_size, self.num_queries, self.embed_dim),
        )

    def get_fake_omdet_turbo_labels(self):
        input_labels = [f"label{i}" for i in range(self.num_queries)]
        return [input_labels] * self.batch_size

    def test_post_process_grounded_object_detection(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = OmDetTurboProcessor(tokenizer=tokenizer, image_processor=image_processor)

        omdet_turbo_output = self.get_fake_omdet_turbo_output()
        omdet_turbo_labels = self.get_fake_omdet_turbo_labels()

        post_processed = processor.post_process_grounded_object_detection(
            omdet_turbo_output, omdet_turbo_labels, target_sizes=[(30, 400) for _ in range(self.batch_size)]
        )

        self.assertEqual(len(post_processed), self.batch_size)
        self.assertEqual(list(post_processed[0].keys()), ["pred_boxes", "scores", "pred_classes"])
        self.assertEqual(post_processed[0]["pred_boxes"].shape, (self.num_queries, 4))
        self.assertEqual(post_processed[0]["scores"].shape, (self.num_queries,))
        expected_scores = torch.tensor([0.7310, 0.6579, 0.6513, 0.6444, 0.6252])
        self.assertTrue(torch.allclose(post_processed[0]["scores"], expected_scores, atol=1e-4))

        expected_box_slice = torch.tensor([14.9657, 141.2052, 30.0000, 312.9670])
        self.assertTrue(torch.allclose(post_processed[0]["pred_boxes"][0], expected_box_slice, atol=1e-4))

    def test_save_load_pretrained_additional_features(self):
        processor = OmDetTurboProcessor(tokenizer=self.get_tokenizer(), image_processor=self.get_image_processor())
        processor.save_pretrained(self.checkpoint_path)

        tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")
        image_processor_add_kwargs = self.get_image_processor(do_normalize=False, padding_value=1.0)

        processor = OmDetTurboProcessor.from_pretrained(
            self.checkpoint_path, bos_token="(BOS)", eos_token="(EOS)", do_normalize=False, padding_value=1.0
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, CLIPTokenizerFast)

        self.assertEqual(processor.image_processor.to_json_string(), image_processor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.image_processor, DetrImageProcessor)

    def test_image_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = OmDetTurboProcessor(tokenizer=tokenizer, image_processor=image_processor)

        image_input = self.prepare_image_inputs()

        input_image_proc = image_processor(image_input, return_tensors="np")
        input_processor = processor(images=image_input, return_tensors="np")

        for key in input_image_proc.keys():
            self.assertAlmostEqual(input_image_proc[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_tokenizer(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = OmDetTurboProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "lower newer"

        encoded_processor = processor(tasks=input_str)["tasks"]

        encoded_tok = tokenizer(input_str, padding="max_length", truncation=True, max_length=77)

        for key in encoded_tok.keys():
            self.assertListEqual(encoded_tok[key], encoded_processor[key])

    def test_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = OmDetTurboProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_tasks = "task"
        input_labels = ["label1", "label2"]
        image_input = self.prepare_image_inputs()

        input_processor = processor(images=image_input, tasks=input_tasks, labels=input_labels, return_tensors="pt")

        assert torch.is_tensor(input_processor["pixel_values"])
        for key in self.text_input_keys:
            assert torch.is_tensor(input_processor["tasks"][key])
            for _, label in input_processor["labels"].items():
                assert torch.is_tensor(label[key])
        # test if it raises when no input is passed
        with pytest.raises(ValueError):
            processor()

    def test_tokenizer_decode(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = OmDetTurboProcessor(tokenizer=tokenizer, image_processor=image_processor)

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)

    def test_model_input_names(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = OmDetTurboProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_tasks = "task"
        input_labels = ["label1", "label2"]
        image_input = self.prepare_image_inputs()

        inputs = processor(images=image_input, tasks=input_tasks, labels=input_labels, return_tensors="pt")

        self.assertListEqual(list(inputs.keys()), self.input_keys)
