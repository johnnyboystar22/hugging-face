# coding=utf-8
# Copyright 2022 Microsoft Research, Inc. and The HuggingFace Inc. team. All rights reserved.
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
""" Tensorflow ResNet model."""

import math
from typing import Dict, Optional, Tuple, Union

import tensorflow as tf

from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithNoAttention,
    TFBaseModelOutputWithPoolingAndNoAttention,
    TFImageClassifierOutputWithNoAttention,
)
from ...modeling_tf_utils import TFPreTrainedModel, TFSequenceClassificationLoss, keras_serializable, unpack_inputs
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_resnet import ResNetConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "ResNetConfig"
_FEAT_EXTRACTOR_FOR_DOC = "AutoFeatureExtractor"

# Base docstring
_CHECKPOINT_FOR_DOC = "microsoft/resnet-50"
_EXPECTED_OUTPUT_SHAPE = [1, 2048, 7, 7]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "microsoft/resnet-50"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tiger cat"

TF_RESNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/resnet-50",
    # See all resnet models at https://huggingface.co/models?filter=resnet
]


class IdentityLayer(tf.keras.layers.Layer):
    """Helper class to give identity a layer API."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def call(self, x: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        return x


class TFResNetConvLayer(tf.keras.layers.Layer):
    def __init__(
        self, out_channels: int, kernel_size: int = 3, stride: int = 1, activation: str = "relu", **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.pad_value = kernel_size // 2
        self.conv = tf.keras.layers.Conv2D(
            out_channels, kernel_size=kernel_size, strides=stride, padding="valid", use_bias=False, name="convolution"
        )
        # Use same default momentum and epsilon as PyTorch equivalent
        self.normalization = tf.keras.layers.BatchNormalization(
            axis=1, epsilon=1e-5, momentum=0.1, name="normalization"
        )
        self.activation = ACT2FN[activation] if activation is not None else IdentityLayer()

    def convolution(self, hidden_state: tf.Tensor) -> tf.Tensor:
        # Pad to match that done in the PyTorch Conv2D model
        height_pad = width_pad = (self.pad_value, self.pad_value)
        hidden_state = tf.pad(hidden_state, [(0, 0), (0, 0), height_pad, width_pad])
        # B, C, H, W -> B, H, W, C
        hidden_state = tf.transpose(hidden_state, (0, 2, 3, 1))
        hidden_state = self.conv(hidden_state)
        # B, H, W, C -> B, C, H, W
        hidden_state = tf.transpose(hidden_state, (0, 3, 1, 2))
        return hidden_state

    def call(self, hidden_state: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_state = self.convolution(hidden_state)
        hidden_state = self.normalization(hidden_state, training=training)
        hidden_state = self.activation(hidden_state)
        return hidden_state


class TFResNetEmbeddings(tf.keras.layers.Layer):
    """
    ResNet Embeddings (stem) composed of a single aggressive convolution.
    """

    def __init__(self, config: ResNetConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.embedder = TFResNetConvLayer(
            config.embedding_size,
            kernel_size=7,
            stride=2,
            activation=config.hidden_act,
            name="embedder",
        )
        self.pooler = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="valid", name="pooler")

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_state = inputs
        hidden_state = self.embedder(hidden_state)
        hidden_state = tf.pad(hidden_state, [[0, 0], [0, 0], [1, 1], [1, 1]])
        # B, C, H, W -> B, H, W, C
        hidden_state = tf.transpose(hidden_state, (0, 2, 3, 1))
        hidden_state = self.pooler(hidden_state)
        # B, H, W, C -> B, C, H, W
        hidden_state = tf.transpose(hidden_state, (0, 3, 1, 2))
        return hidden_state


class TFResNetShortCut(tf.keras.layers.Layer):
    """
    ResNet shortcut, used to project the residual features to the correct size. If needed, it is also used to
    downsample the input using `stride=2`.
    """

    def __init__(self, out_channels: int, stride: int = 2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.convolution = tf.keras.layers.Conv2D(
            out_channels, kernel_size=1, strides=stride, use_bias=False, name="convolution"
        )
        # Use same default momentum and epsilon as PyTorch equivalent
        self.normalization = tf.keras.layers.BatchNormalization(
            axis=1, epsilon=1e-5, momentum=0.1, name="normalization"
        )

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_state = x
        # B, C, H, W -> B, H, W, C
        hidden_state = tf.transpose(hidden_state, (0, 2, 3, 1))
        hidden_state = self.convolution(hidden_state)
        # B, H, W, C -> B, C, H, W
        hidden_state = tf.transpose(hidden_state, (0, 3, 1, 2))
        hidden_state = self.normalization(hidden_state, training=training)
        return hidden_state


class TFResNetBasicLayer(tf.keras.layers.Layer):
    """
    A classic ResNet's residual layer composed by a two `3x3` convolutions.
    """

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, activation: str = "relu", **kwargs
    ) -> None:
        super().__init__(**kwargs)
        should_apply_shortcut = in_channels != out_channels or stride != 1
        self.conv1 = TFResNetConvLayer(out_channels, stride=stride)
        self.conv2 = TFResNetConvLayer(out_channels, activation=None)
        self.shortcut = (
            TFResNetShortCut(out_channels, stride=stride, name="shortcut")
            if should_apply_shortcut
            else IdentityLayer(name="shortcut")
        )
        self.activation = ACT2FN[activation]

    def call(self, hidden_state: tf.Tensor, training: bool = False) -> tf.Tensor:
        residual = hidden_state
        hidden_state = self.conv1(hidden_state, training=training)
        hidden_state = self.conv2(hidden_state, training=training)
        residual = self.shortcut(residual, training=training)
        hidden_state += residual
        hidden_state = self.activation(hidden_state)
        return hidden_state


class TFResNetBottleNeckLayer(tf.keras.layers.Layer):
    """
    A classic ResNet's bottleneck layer composed by a three `3x3` convolutions.

    The first `1x1` convolution reduces the input by a factor of `reduction` in order to make the second `3x3`
    convolution faster. The last `1x1` convolution remap the reduced features to `out_channels`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation: str = "relu",
        reduction: int = 4,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        should_apply_shortcut = in_channels != out_channels or stride != 1
        reduces_channels = out_channels // reduction
        self.conv0 = TFResNetConvLayer(reduces_channels, kernel_size=1, name="layer.0")
        self.conv1 = TFResNetConvLayer(reduces_channels, stride=stride, name="layer.1")
        self.conv2 = TFResNetConvLayer(out_channels, kernel_size=1, activation=None, name="layer.2")
        self.shortcut = (
            TFResNetShortCut(out_channels, stride=stride, name="shortcut")
            if should_apply_shortcut
            else IdentityLayer(name="shortcut")
        )
        self.activation = ACT2FN[activation]

    def call(self, hidden_state: tf.Tensor, training: bool = False) -> tf.Tensor:
        residual = hidden_state
        hidden_state = self.conv0(hidden_state, training=training)
        hidden_state = self.conv1(hidden_state, training=training)
        hidden_state = self.conv2(hidden_state, training=training)
        residual = self.shortcut(residual, training=training)
        hidden_state += residual
        hidden_state = self.activation(hidden_state)
        return hidden_state


class TFResNetStage(tf.keras.layers.Layer):
    """
    A ResNet stage composed by stacked layers.
    """

    def __init__(
        self, config: ResNetConfig, in_channels: int, out_channels: int, stride: int = 2, depth: int = 2, **kwargs
    ) -> None:
        super().__init__(**kwargs)

        layer = TFResNetBottleNeckLayer if config.layer_type == "bottleneck" else TFResNetBasicLayer

        layers = [layer(in_channels, out_channels, stride=stride, activation=config.hidden_act, name="layers.0")]
        layers += [
            layer(out_channels, out_channels, activation=config.hidden_act, name=f"layers.{i + 1}")
            for i in range(depth - 1)
        ]
        self.stage_layers = layers

    def call(self, hidden_state: tf.Tensor, training: bool = False) -> tf.Tensor:
        for layer in self.stage_layers:
            hidden_state = layer(hidden_state, training=training)
        return hidden_state


class TFResNetEncoder(tf.keras.layers.Layer):
    def __init__(self, config: ResNetConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        # based on `downsample_in_first_stage` the first layer of the first stage may or may not downsample the input
        self.stages = [
            TFResNetStage(
                config,
                config.embedding_size,
                config.hidden_sizes[0],
                stride=2 if config.downsample_in_first_stage else 1,
                depth=config.depths[0],
                name="stages.0",
            )
        ]
        for i, (in_channels, out_channels, depth) in enumerate(
            zip(config.hidden_sizes, config.hidden_sizes[1:], config.depths[1:])
        ):
            self.stages.append(TFResNetStage(config, in_channels, out_channels, depth=depth, name=f"stages.{i + 1}"))

    def call(
        self,
        hidden_state: tf.Tensor,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        training: bool = False,
    ) -> TFBaseModelOutputWithNoAttention:
        hidden_states = () if output_hidden_states else None

        for stage_module in self.stages:
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state,)

            hidden_state = stage_module(hidden_state, training=training)

        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states] if v is not None)

        return TFBaseModelOutputWithNoAttention(
            last_hidden_state=hidden_state,
            hidden_states=hidden_states,
        )


class TFResNetPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ResNetConfig
    base_model_prefix = "resnet"
    main_input_name = "pixel_values"

    @property
    def dummy_inputs(self) -> Dict[str, tf.Tensor]:
        """
        Dummy inputs to build the network. Returns:
            `Dict[str, tf.Tensor]`: The dummy inputs.
        """
        VISION_DUMMY_INPUTS = tf.random.uniform(
            shape=(3, self.config.num_channels, self.config.image_size, self.config.image_size), dtype=tf.float32
        )
        return {"pixel_values": tf.constant(VISION_DUMMY_INPUTS)}


RESNET_START_DOCSTRING = r"""
    This model is a Tensorflow
    [tf.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) sub-class. Use it as a
    regular Tensorflow Module and refer to the Tensorflow documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ResNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""


RESNET_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoFeatureExtractor`]. See
            [`AutoFeatureExtractor.__call__`] for details.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# Copied from:
# https://gist.github.com/Rocketknight1/efc47242914788def0144b341b1ad638
class TFAdaptiveAvgPool2D(tf.keras.layers.Layer):
    def __init__(self, output_dims: Tuple[int, int], input_ordering: str = "NHWC", **kwargs):
        super().__init__(**kwargs)
        self.output_dims = output_dims
        self.input_ordering = input_ordering
        if input_ordering not in ("NCHW", "NHWC"):
            raise ValueError("Unrecognized input_ordering, should be 'NCHW' or 'NHWC'!")
        self.h_axis = input_ordering.index("H")
        self.w_axis = input_ordering.index("W")

    def pseudo_1d_pool(self, inputs: tf.Tensor, h_pooling: bool):
        # Figure out which axis we're pooling on
        if h_pooling:
            axis = self.h_axis
            output_dim = self.output_dims[0]
        else:
            axis = self.w_axis
            output_dim = self.output_dims[1]
        input_dim = inputs.shape[axis]

        # Figure out the potential pooling windows
        # This is the key idea - the torch op will always use only two
        # consecutive pooling window sizes, like 3 and 4. Therefore,
        # if we pool with both possible sizes, we simply need to gather
        # the 'correct' pool at each position to reimplement the torch op.
        small_window = math.ceil(input_dim / output_dim)
        big_window = small_window + 1
        if h_pooling:
            output_dim = self.output_dims[0]
            small_window_shape = (small_window, 1)
            big_window_shape = (big_window, 1)
        else:
            output_dim = self.output_dims[1]
            small_window_shape = (1, small_window)
            big_window_shape = (1, big_window)

        # For integer resizes, we can take a very quick shortcut
        if input_dim % output_dim == 0:
            return tf.nn.avg_pool2d(
                inputs,
                ksize=small_window_shape,
                strides=small_window_shape,
                padding="VALID",
                data_format=self.input_ordering,
            )

        # For non-integer resizes, we pool with both possible window sizes and concatenate them
        small_pool = tf.nn.avg_pool2d(
            inputs, ksize=small_window_shape, strides=1, padding="VALID", data_format=self.input_ordering
        )
        big_pool = tf.nn.avg_pool2d(
            inputs, ksize=big_window_shape, strides=1, padding="VALID", data_format=self.input_ordering
        )
        both_pool = tf.concat([small_pool, big_pool], axis=axis)

        # We compute vectors of the start and end positions for each pooling window
        # Each (start, end) pair here corresponds to a single output position
        window_starts = tf.math.floor((tf.range(output_dim, dtype=tf.float32) * input_dim) / output_dim)
        window_starts = tf.cast(window_starts, tf.int64)
        window_ends = tf.math.ceil((tf.range(1, output_dim + 1, dtype=tf.float32) * input_dim) / output_dim)
        window_ends = tf.cast(window_ends, tf.int64)

        # pool_selector is a boolean array of shape (output_dim,) where 1 indicates that output position
        # has a big receptive field and 0 indicates that that output position has a small receptive field
        pool_selector = tf.cast(window_ends - window_starts - small_window, tf.bool)

        # Since we concatenated the small and big pools, we need to do a bit of
        # pointer arithmetic to get the indices of the big pools
        small_indices = window_starts
        big_indices = window_starts + small_pool.shape[axis]

        # Finally, we use the pool_selector to generate a list of indices, one per output position
        gather_indices = tf.where(pool_selector, big_indices, small_indices)

        # Gathering from those indices yields the final, correct pooling
        return tf.gather(both_pool, gather_indices, axis=axis)

    def call(self, inputs: tf.Tensor):
        if self.input_ordering == "NHWC":
            input_shape = inputs.shape[1:3]
        else:
            input_shape = inputs.shape[2:]

        if input_shape[0] % self.output_dims[0] == 0 and input_shape[1] % self.output_dims[1] == 0:
            # If we're resizing by an integer factor on both dimensions, we can take
            # a very quick shortcut.
            h_resize = int(input_shape[0] // self.output_dims[0])
            w_resize = int(input_shape[1] // self.output_dims[1])
            return tf.nn.avg_pool2d(
                inputs,
                ksize=(h_resize, w_resize),
                strides=(h_resize, w_resize),
                padding="VALID",
                data_format=self.input_ordering,
            )
        else:
            # If we can't take the shortcut, we do a 1D pool on each axis
            h_pooled = self.pseudo_1d_pool(inputs, h_pooling=True)
            return self.pseudo_1d_pool(h_pooled, h_pooling=False)


@keras_serializable
class TFResNetMainLayer(tf.keras.layers.Layer):
    config_class = ResNetConfig

    def __init__(self, config: ResNetConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.embedder = TFResNetEmbeddings(config, name="embedder")
        self.encoder = TFResNetEncoder(config, name="encoder")
        self.pooler = TFAdaptiveAvgPool2D((1, 1), input_ordering="NCHW")

    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor], TFBaseModelOutputWithPoolingAndNoAttention]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        embedding_output = self.embedder(pixel_values, training=training)

        encoder_outputs = self.encoder(
            embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training
        )

        last_hidden_state = encoder_outputs[0]

        pooled_output = self.pooler(last_hidden_state)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return TFBaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


@add_start_docstrings(
    "The bare ResNet model outputting raw features without any specific head on top.",
    RESNET_START_DOCSTRING,
)
class TFResNetModel(TFResNetPreTrainedModel):
    def __init__(self, config: ResNetConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.resnet = TFResNetMainLayer(config=config, name="resnet")

    @add_start_docstrings_to_model_forward(RESNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor], TFBaseModelOutputWithPoolingAndNoAttention]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        resnet_outputs = self.resnet(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        return resnet_outputs


@add_start_docstrings(
    """
    ResNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    RESNET_START_DOCSTRING,
)
class TFResNetForImageClassification(TFResNetPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: ResNetConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.num_labels = config.num_labels
        self.resnet = TFResNetMainLayer(config, name="resnet")
        # classification head
        self.classifier_layer = (
            tf.keras.layers.Dense(config.num_labels, name="classifier.1")
            if config.num_labels > 0
            else IdentityLayer(name="classifier.1")
        )

    def classifier(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.keras.layers.Flatten()(x)
        logits = self.classifier_layer(x)
        return logits

    @add_start_docstrings_to_model_forward(RESNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor = None,
        labels: tf.Tensor = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor], TFImageClassifierOutputWithNoAttention]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.resnet(
            pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training
        )

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        logits = self.classifier(pooled_output)

        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output

        return TFImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)
