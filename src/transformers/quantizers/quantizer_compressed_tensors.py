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
from ..utils import is_torch_available, logging
from ..utils.quantization_config import QuantizationConfigMixin
from .base import HfQuantizer


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class CompressedTensorsHfQuantizer(HfQuantizer):
    """
    Quantizer for the compressed_tensors package.  Loads and restores models to
    quantized state with compressed_tensors
    """

    requires_calibration = False
    # requires_parameters_quantization = True
    required_packages = ["compressed_tensors"]

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)

        from compressed_tensors.compressors import ModelCompressor

        self.compressor = ModelCompressor.from_compression_config(quantization_config)

    def validate_environment(self, *args, **kwargs):
        # check torch and compressed_tensors are available, let ImportError raise otherwise
        import torch  # noqa
        from compressed_tensors.compressors import ModelCompressor  # noqa

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            torch_dtype = torch.float16
        elif torch_dtype != torch.float16:
            logger.info(
                "We suggest you to set `torch_dtype=torch.float16` for better efficiency with compressed_tensors."
            )
        return torch_dtype

    def _process_model_before_weight_loading(self, model, **kwargs):
        if self.quantization_config.quantization_config is not None:
            from compressed_tensors.quantization import apply_quantization_config

            apply_quantization_config(model, self.quantization_config.quantization_config)

    def _process_model_after_weight_loading(self, model, resolved_archive_file, **kwargs):
        self.compressor.decompress(model_path=resolved_archive_file, model=model)

    @property
    def is_trainable(self):
        return False

    @property
    def is_serializable(self):
        return False
