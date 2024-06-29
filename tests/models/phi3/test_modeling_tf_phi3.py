# coding=utf-8
# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
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

"""Testing suite for the PyTorch Phi-3 model."""

import unittest
import numpy as np
from parameterized import parameterized

from transformers import Phi3Config, is_tf_available, set_seed
from transformers.testing_utils import (
    require_read_token,
    require_tf,
    slow,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_tf_available():
    import tensorflow as tf

    from transformers import TFPhi3ForCausalLM, TFPhi3ForSequenceClassification, TFPhi3Model


class Phi3ModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.pad_token_id = pad_token_id
        self.scope = scope

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.prepare_config_and_inputs
    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = tf.linalg.band_part(tf.ones((self.batch_size, self.seq_length)), -1, 0)

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return Phi3Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
        )

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_model with Llama->Phi3
    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = TFPhi3Model(config=config)
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_model_as_decoder(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.add_cross_attention = True
        model = TFPhi3Model(config)
        result = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        result = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
        )
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_for_causal_lm with Llama->Phi3
   
    def create_and_check_for_causal_lm(
            self,
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        ):
            model = TFPhi3ForCausalLM(config=config)
            result = model(input_ids, attention_mask=input_mask, labels=token_labels)
    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_decoder_model_past_large_inputs with Llama->Phi3

    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.is_decoder = True
        config.add_cross_attention = True
        model = TFPhi3ForCausalLM(config=config)

        # first forward pass
        outputs = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
        )

        past_key_values = outputs.past_key_values

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and
        next_input_ids = tf.concat([input_ids, next_tokens], axis=-1)
        next_attention_mask = tf.concat([input_mask, next_mask], axis=-1)

        output_from_no_past = model(
            next_input_ids,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=True,
        )["hidden_states"][0]
        output_from_past = model(
            next_tokens,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )["hidden_states"][0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(tf.reduce_all(tf.abs(output_from_past_slice - output_from_no_past_slice) <= 1e-3))


    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.prepare_config_and_inputs_for_common
    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_tf
class TFPhi3ModelTest(TFModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (TFPhi3Model, TFPhi3ForCausalLM, TFPhi3ForSequenceClassification) if is_tf_available() else ()
    )
    all_generative_model_classes = (TFPhi3ForCausalLM,) if is_tf_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": TFPhi3Model,
            "text-classification": TFPhi3ForSequenceClassification,
            "text-generation": TFPhi3ForCausalLM,
            "zero-shot": TFPhi3ForSequenceClassification,
        }
        if is_tf_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    test_onnx = False
    # Need to remove 0.9 in `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    # model_split_percents = [0.5, 0.6]

    # TODO (ydshieh): Check this. See https://app.circleci.com/pipelines/github/huggingface/transformers/79292/workflows/fa2ba644-8953-44a6-8f67-ccd69ca6a476/jobs/1012905
    def is_pipeline_test_to_skip(
        self, pipeline_test_casse_name, config_class, model_architecture, tokenizer_name, processor_name
    ):
        return True

    def setUp(self):
        self.model_tester = Phi3ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Phi3Config, hidden_size=37)

    def test_config(self):
       self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    # def test_model_various_embeddings(self):
    #     config_and_inputs = self.model_tester.prepare_config_and_inputs()
    #     for type in ["absolute", "relative_key", "relative_key_query"]:
    #         config_and_inputs[0].position_embedding_type = type
    #         self.model_tester.create_and_check_model(*config_and_inputs)

    def test_Phi3_sequence_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        print(config)
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = tf.cast(tf.not_equal(input_ids, 1), tf.int32)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = TFPhi3ForSequenceClassification(config)
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_Phi3_sequence_classification_model_for_single_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "single_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = tf.cast(tf.not_equal(input_ids, 1), tf.int32)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = TFPhi3ForSequenceClassification(config)
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_Phi3_sequence_classification_model_for_multi_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "multi_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = tf.cast(tf.not_equal(input_ids, 1), tf.int32)
        sequence_labels = ids_tensor(
            [self.model_tester.batch_size, config.num_labels], self.model_tester.type_sequence_label_size
        )
        model = TFPhi3ForSequenceClassification(config)
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))


    @parameterized.expand([("su",), ("yarn",)])
    def test_model_rope_scaling_from_config(self, scaling_type):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        short_input = ids_tensor([1, 10], config.vocab_size)
        long_input = ids_tensor([1, int(config.max_position_embeddings * 1.5)], config.vocab_size)

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        original_model = TFPhi3Model(config)
        original_model(short_input)
        original_short_output = original_model(short_input)['last_hidden_state']
        original_long_output = original_model(long_input)['last_hidden_state']

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        n_factors = config.hidden_size // config.num_attention_heads // 2
        config.rope_scaling = {
            "type": scaling_type,
            "short_factor": [5.0 for _ in range(n_factors)],
            "long_factor": [5.0 for _ in range(n_factors)],
        }
        scaled_model = TFPhi3Model(config)
        scaled_model(short_input)
        scaled_short_output = scaled_model(short_input)['last_hidden_state']
        scaled_long_output = scaled_model(long_input)['last_hidden_state']

        # Scaling changes the RoPE embeddings, both for the short and long outputs
        self.assertFalse(np.allclose(original_short_output, scaled_short_output, atol=1e-5))
        self.assertFalse(np.allclose(original_long_output, scaled_long_output, atol=1e-5))

        


    # @unittest.skip("TODO @gante fix this for Llama")
    # @parameterized.expand([(1, False), (1, True), (4, False)])
    # def test_new_cache_format(self, num_beams, do_sample):
    #     pass

    # @unittest.skip("Gemma buffers include complex numbers, which breaks this test")
    # def test_save_load_fast_init_from_base(self):
    #     pass

    # @unittest.skip("Gemma uses GQA on all models so the KV cache is a non standard format")
    # def test_past_key_values_format(self):
    #     pass

    # @unittest.skip("TFGemma does not support caching")
    # def test_xla_generate_contrastive(self):
    #     pass

    # @unittest.skip("TFGemma does not support caching")
    # def test_xla_generate_fast(self):
    #     pass

    # @unittest.skip("TFGemma does not support caching")
    # def test_xla_generate_slow(self):
    #     pass

@slow
class Phi3IntegrationTest(unittest.TestCase):
    def test_model_phi3_mini_4k_instruct_logits(self):
        input_ids = {
            "input_ids": tf.convert_to_tensor(
                [[1212, 318, 281, 1672, 2643, 290, 428, 318, 257, 1332]], dtype=tf.int32
            )
        }

        model = TFPhi3ForCausalLM.from_pretrained("microsoft/phi-3-mini-4k-instruct")
        model.compile()  # Necessary for setting the model in evaluation mode

        output = model(input_ids).logits

        EXPECTED_OUTPUT = tf.convert_to_tensor([[ 0.9979, -1.9449, -2.5613, -2.2110, -0.9323, -2.2726, -3.2468, -2.0122,-1.0021, -1.2764, -1.0876, -1.2358,  3.9385,  6.2152, -0.3695, -2.3285,-1.2907, -1.8238, -1.9941, -2.2098, -0.6923, -1.6793, -1.1660, -2.0469,-0.7369, -1.4101, -1.4091, -3.1694, -1.8383, -1.1952],[ 3.0525,  1.9178,  3.7016,  0.9263,  0.3397,  1.9584,  2.1347,  0.3482, 1.3773,  0.2153,  0.2798,  0.8360,  9.0936, 11.4944, -0.3575, -0.9442,-0.1246,  1.3869,  0.9846,  1.7243,  0.9150,  1.0823,  0.4313,  1.5742, 0.2566, -0.1401, -1.3019,  0.4967,  0.6941,  0.7214]], dtype=tf.float32)

        self.assertTrue(tf.reduce_all(tf.abs(EXPECTED_OUTPUT - output[0, :2, :30]) < 1e-4))

    def test_phi3_mini_4k_instruct_generation(self):
        model = TFPhi3ForCausalLM.from_pretrained("microsoft/phi-3-mini-4k-instruct")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-4k-instruct")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user.",
            },
            {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
        ]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="tf")

        outputs = model.generate(inputs["input_ids"], max_new_tokens=32)
        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        EXPECTED_OUTPUT = [
            "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user. Can you provide ways to eat combinations of bananas and dragonfruits? Absolutely! Bananas and dragonfruits are both delicious fruits that can be combined in various ways to create tasty and nutritious dishes. Here are some ideas:\n\n1."
        ]

        self.assertListEqual(output_text, EXPECTED_OUTPUT)

    def test_model_phi3_mini_128k_instruct_logits(self):
        input_ids = {
            "input_ids": tf.convert_to_tensor(
                [[1212, 318, 281, 1672, 2643, 290, 428, 318, 257, 1332]], dtype=tf.int32
            )
        }

        model = TFPhi3ForCausalLM.from_pretrained("microsoft/phi-3-mini-128k-instruct")
        model.compile()  # Necessary for setting the model in evaluation mode

        output = model(input_ids).logits

        EXPECTED_OUTPUT = tf.convert_to_tensor([[ 1.8478, -0.5709, -1.6792, -1.2133, -0.7809, -0.8817, -2.0969, -1.1191,-0.7731, -1.0483, -0.5961, -1.3067,  3.1325,  6.9442, -0.4803, -0.9154,-1.3085, -1.0822, -1.1433, -0.7660, -0.8531, -0.9150, -0.6179, -1.6153,-0.2239, -1.3207, -1.1187, -2.4795, -1.4733, -0.4931],[ 3.5839,  2.4722,  3.7130,  1.2032,  0.7356,  2.7777,  2.5256,  0.9157, 1.6431,  0.3533,  0.5100,  1.3512,  8.9873, 10.9815,  0.3530,  0.1473, 0.2051,  1.8553,  1.5988,  2.2268,  1.1897,  1.2829,  0.7894,  1.8895, 0.7666,  0.4122, -0.9316,  0.9936,  1.2722,  0.8263]], dtype=tf.float32)

        self.assertTrue(tf.reduce_all(tf.abs(EXPECTED_OUTPUT - output[0, :2, :30]) < 1e-4))

    def test_phi3_mini_128k_instruct_generation(self):
        model = TFPhi3ForCausalLM.from_pretrained("microsoft/phi-3-mini-128k-instruct")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-128k-instruct")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user.",
            },
            {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
        ]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="tf")

        outputs = model.generate(inputs["input_ids"], max_new_tokens=32)
        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        EXPECTED_OUTPUT = [
            "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user. Can you provide ways to eat combinations of bananas and dragonfruits? Certainly! Bananas and dragonfruits can be combined in various delicious and healthy ways. Here are some ideas:\n\n1."
        ]

        self.assertListEqual(output_text, EXPECTED_OUTPUT)