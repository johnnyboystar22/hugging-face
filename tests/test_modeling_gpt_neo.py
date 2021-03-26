# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch GPTNeo model. """


import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from .test_configuration_common import ConfigTester
from .test_generation_utils import GenerationTesterMixin
from .test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import (
        GPT_NEO_PRETRAINED_MODEL_ARCHIVE_LIST,
        GPTNeoConfig,
        GPTNeoForCausalLM,
        GPTNeoModel,
        GPTNeoTokenizer,
    )


class GPTNeoModelTester:
    def __init__(
        self,
        parent,
        batch_size=14,
        seq_length=7,
        is_training=True,
        use_token_type_ids=True,
        use_input_mask=True,
        use_labels=True,
        use_mc_token_ids=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=4,
        attn_layers=("global", "local", "global", "local"),
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        window_size=7,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_token_type_ids = use_token_type_ids
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.use_mc_token_ids = use_mc_token_ids
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.window_size = window_size
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = None
        self.bos_token_id = vocab_size - 1
        self.eos_token_id = vocab_size - 1
        self.pad_token_id = vocab_size - 1
        self.chunk_length = window_size
        self.attn_layers = attn_layers

    def get_large_model_config(self):
        return GPTNeoConfig.from_pretrained("gpt_neo")

    def prepare_config_and_inputs(self, gradient_checkpointing=False):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        mc_token_ids = None
        if self.use_mc_token_ids:
            mc_token_ids = ids_tensor([self.batch_size, self.num_choices], self.seq_length)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = GPTNeoConfig(
            vocab_size=self.vocab_size,
            n_embd=self.hidden_size,
            n_layer=self.num_hidden_layers,
            n_head=self.num_attention_heads,
            # intermediate_size=self.intermediate_size,
            # hidden_act=self.hidden_act,
            # hidden_dropout_prob=self.hidden_dropout_prob,
            # attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            n_positions=self.max_position_embeddings,
            n_ctx=self.max_position_embeddings,
            # type_vocab_size=self.type_vocab_size,
            # initializer_range=self.initializer_range,
            use_cache=not gradient_checkpointing,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            gradient_checkpointing=gradient_checkpointing,
            window_size=self.window_size,
            attn_layers=self.attn_layers,
        )

        head_mask = ids_tensor([self.num_hidden_layers, self.num_attention_heads], 2)

        return (
            config,
            input_ids,
            input_mask,
            head_mask,
            token_type_ids,
            mc_token_ids,
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def prepare_config_and_inputs_for_decoder(self):
        (
            config,
            input_ids,
            input_mask,
            head_mask,
            token_type_ids,
            mc_token_ids,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = self.prepare_config_and_inputs()

        encoder_hidden_states = floats_tensor([self.batch_size, self.seq_length, self.hidden_size])
        encoder_attention_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

        return (
            config,
            input_ids,
            input_mask,
            head_mask,
            token_type_ids,
            sequence_labels,
            token_labels,
            choice_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        )

    def create_and_check_gpt_neo_model(self, config, input_ids, input_mask, head_mask, token_type_ids, *args):
        model = GPTNeoModel(config=config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids, token_type_ids=token_type_ids, head_mask=head_mask)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        # past_key_values is not implemented
        # self.parent.assertEqual(len(result.past_key_values), config.n_layer)

    def create_and_check_lm_head_model(self, config, input_ids, input_mask, head_mask, token_type_ids, *args):
        model = GPTNeoForCausalLM(config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids, token_type_ids=token_type_ids, labels=input_ids)
        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_forward_and_backwards(self, config, input_ids, input_mask, head_mask, token_type_ids, *args):
        model = GPTNeoForCausalLM(config)
        model.to(torch_device)

        result = model(input_ids, token_type_ids=token_type_ids, labels=input_ids)
        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))
        result.loss.backward()

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()

        (
            config,
            input_ids,
            input_mask,
            head_mask,
            token_type_ids,
            mc_token_ids,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "head_mask": head_mask,
        }

        return config, inputs_dict


@require_torch
class GPTNeoModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):

    all_model_classes = (GPTNeoModel, GPTNeoForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (GPTNeoForCausalLM,) if is_torch_available() else ()
    all_parallelizable_model_classes = (GPTNeoForCausalLM,) if is_torch_available() else ()
    test_missing_keys = False
    test_pruning = False
    test_model_parallel = True

    # special case for DoubleHeads model
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        return inputs_dict

    def setUp(self):
        self.model_tester = GPTNeoModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GPTNeoConfig, n_embd=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_gpt_neo_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gpt_neo_model(*config_and_inputs)

    def test_gpt_neo_model_past(self):
        pass

    def test_gpt_neo_model_att_mask_past(self):
        pass

    def test_gpt_neo_model_past_large_inputs(self):
        pass

    def test_gpt_neo_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

    def test_gpt_neo_gradient_checkpointing(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs(gradient_checkpointing=True)
        self.model_tester.create_and_check_forward_and_backwards(*config_and_inputs)

    def _get_local_attn_seq_len(self, seq_len, window_size):
        block_length = window_size
        while seq_len % block_length != 0:
            block_length -= 1
        partial_size = window_size % block_length
        return window_size + partial_size

    def test_attention_outputs(self):
        pass
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = getattr(self.model_tester, "seq_length", None)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)
        chunk_length = getattr(self.model_tester, "chunk_length", None)
        if chunk_length is not None and hasattr(self.model_tester, "num_hashes"):
            encoder_seq_length = encoder_seq_length * self.model_tester.num_hashes

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            encoder_key_length = self._get_local_attn_seq_len(seq_len, chunk_length)
            self.assertListEqual(
                list(attentions[-1].shape[-4:]),
                [self.model_tester.num_attention_heads, encoder_seq_length, chunk_length, encoder_key_length],
            )

            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            if hasattr(self.model_tester, "num_hidden_states_types"):
                added_hidden_states = self.model_tester.num_hidden_states_types
            elif self.is_encoder_decoder:
                added_hidden_states = 2
            else:
                added_hidden_states = 1
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions

            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            if chunk_length is not None:
                self.assertListEqual(
                    list(self_attentions[0].shape[-4:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, chunk_length, encoder_key_length],
                )
            else:
                self.assertListEqual(
                    list(self_attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
                )

    def _check_attentions_for_generate(
        self, batch_size, attentions, min_length, max_length, config, use_cache=False, num_beam_groups=1
    ):
        # self.assertIsInstance(attentions, tuple)
        # self.assertListEqual(
        #     [isinstance(iter_attentions, tuple) for iter_attentions in attentions], [True] * len(attentions)
        # )
        # self.assertEqual(len(attentions), (max_length - min_length) * num_beam_groups)

        # for idx, iter_attentions in enumerate(attentions):
        #     tgt_len = min_length + idx if not use_cache else 1
        #     src_len = min_length + idx

        #     expected_shape = (
        #         batch_size * num_beam_groups,
        #         config.num_attention_heads,
        #         tgt_len,
        #         src_len,
        #     )
        #     # check attn size
        #     self.assertListEqual(
        #         [layer_attention.shape for layer_attention in iter_attentions], [expected_shape] * len(iter_attentions)
        #     )
        pass

    # caching is not yet implemented
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    # batch generation is not supported yet
    @slow
    def test_batch_generation(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        for model_name in GPT_NEO_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = GPTNeoModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_torch
class GPTNeoModelLanguageGenerationTest(unittest.TestCase):
    @slow
    def test_lm_generate_gpt_neo(self):
        for checkpointing in [True, False]:
            model = GPTNeoForCausalLM.from_pretrained("gpt_neo", gradient_checkpointing=checkpointing)
            model.to(torch_device)
            input_ids = torch.tensor([[464, 3290]], dtype=torch.long, device=torch_device)  # The dog
            expected_output_ids = [
                464,
                3290,
                12,
                3380,
                4866,
                286,
                262,
                1492,
                11,
                543,
                318,
                257,
                4947,
                286,
                27126,
                416,
                262,
                2739,
                1772,
                11,
            ]  # The dog-eared copy of the book, which is a collection of essays by the late author,
            output_ids = model.generate(input_ids, do_sample=False)
            self.assertListEqual(output_ids[0].tolist(), expected_output_ids)

    @slow
    def test_gpt_neo_sample(self):
        tokenizer = GPTNeoTokenizer.from_pretrained("gpt_neo")
        model = GPTNeoForCausalLM.from_pretrained("gpt_neo")
        model.to(torch_device)

        torch.manual_seed(0)
        tokenized = tokenizer("Today is a nice day and", return_tensors="pt", return_token_type_ids=True)
        input_ids = tokenized.input_ids.to(torch_device)
        output_ids = model.generate(input_ids, do_sample=True)
        output_str = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        EXPECTED_OUTPUT_STR = "Today is a nice day and if you don’t get the memo here is what you can"
        self.assertEqual(output_str, EXPECTED_OUTPUT_STR)
