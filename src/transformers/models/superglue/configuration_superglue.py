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
from typing import List

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING

logger = logging.get_logger(__name__)

SUPERPOINT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "sbucaille/superglue": "https://huggingface.co/sbucaille/superpoint/blob/main/config.json"
}


class SuperGlueConfig(PretrainedConfig):
    #TODO add documentation

    model_type = "superglue"

    def __init__(
            self,
            keypoint_detector_config=None,
            descriptor_dim: int = 256,
            keypoint_encoder_sizes: List[int] = [32, 64, 128, 256],
            gnn_layers_types: List[str] = ['self', 'cross'] * 9,
            num_heads: int = 4,
            sinkhorn_iterations: int = 100,
            matching_threshold: float = 0.2,
            model_version: str = "indoor",
            **kwargs,
    ):

        # Check whether all gnn_layers_types are either 'self' or 'cross'
        if not all([layer_type in ['self', 'cross'] for layer_type in gnn_layers_types]):
            raise ValueError("All gnn_layers_types must be either 'self' or 'cross'")

        if model_version != "indoor" and model_version != "outdoor":
            raise ValueError("model_version must be either 'indoor' or 'outdoor'")

        if descriptor_dim % num_heads != 0:
            raise ValueError("descriptor_dim % num_heads is different from zero")

        self.descriptor_dim = descriptor_dim
        self.keypoint_encoder_sizes = keypoint_encoder_sizes
        self.gnn_layers_types = gnn_layers_types
        self.num_heads = num_heads
        self.sinkhorn_iterations = sinkhorn_iterations
        self.matching_threshold = matching_threshold
        self.model_version = model_version

        if isinstance(keypoint_detector_config, dict):
            keypoint_detector_config["model_type"] = (
                keypoint_detector_config["model_type"] if "model_type" in keypoint_detector_config else "superpoint"
            )
            keypoint_detector_config = CONFIG_MAPPING[keypoint_detector_config["model_type"]](
                **keypoint_detector_config)
        if keypoint_detector_config is None:
            keypoint_detector_config = CONFIG_MAPPING["superpoint"]()

        self.keypoint_detector_config = keypoint_detector_config

        super().__init__(**kwargs)
