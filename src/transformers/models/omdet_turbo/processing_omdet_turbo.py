# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""
Processor class for OmDet-Turbo.
"""

from typing import List, Tuple, Union


try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack


from ...image_transforms import center_to_corners_format
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PreTokenizedInput, TextInput
from ...utils import (
    TensorType,
    is_torch_available,
    is_torchvision_available,
)


if is_torch_available():
    import torch

if is_torchvision_available():
    from torchvision.ops.boxes import batched_nms


def clip_boxes(box, box_size: Tuple[int, int]):
    """
    Clip the boxes by limiting x coordinates to the range [0, width]
    and y coordinates to the range [0, height].

    Args:
        box (Tensor): The box to be clipped.
        box_size (height, width): The clipping box's size.
    """
    assert torch.isfinite(box).all(), "Box tensor contains infinite or NaN!"
    height, width = box_size
    x1 = box[:, 0].clamp(min=0, max=width)
    y1 = box[:, 1].clamp(min=0, max=height)
    x2 = box[:, 2].clamp(min=0, max=width)
    y2 = box[:, 3].clamp(min=0, max=height)
    box = torch.stack((x1, y1, x2, y2), dim=-1)

    return box


def compute_score(boxes):
    num_classes = boxes.shape[2]
    proposal_num = boxes.shape[1]
    scores = torch.sigmoid(boxes)
    classes = torch.arange(num_classes, device=boxes.device).unsqueeze(0).repeat(proposal_num, 1).flatten(0, 1)
    return scores, classes


class OmDetTurboProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": True,
            "padding": "max_length",
            "truncation": True,
            "max_length": 77,
            "stride": 0,
            "return_overflowing_tokens": False,
            "return_special_tokens_mask": False,
            "return_offsets_mapping": False,
            "return_token_type_ids": False,
            "return_length": False,
            "verbose": True,
        },
        "images_kwargs": {},
    }


class OmDetTurboProcessor(ProcessorMixin):
    r"""
    Constructs a OmDet-Turbo processor which wraps a Deformable DETR image processor and an AutoTokenizer into a
    single processor.

    [`OmDetTurboProcessor`] offers all the functionalities of [`DetrImageProcessor`] and
    [`AutoTokenizer`]. See the docstring of [`~OmDetTurboProcessor.__call__`] and [`~OmDetTurboProcessor.decode`]
    for more information.

    Args:
        image_processor (`DetrImageProcessor`):
            An instance of [`DetrImageProcessor`]. The image processor is a required input.
        tokenizer (`AutoTokenizer`):
            An instance of ['PreTrainedTokenizer`]. The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "DetrImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[str, List[str], TextInput, PreTokenizedInput] = None,
        classes: Union[List[str], List[List[str]]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[OmDetTurboProcessorKwargs],
    ) -> BatchEncoding:
        """
        This method uses [`DetrImageProcessor.__call__`] method to prepare image(s) for the model, and
        [`CLIPTokenizerFast.__call__`] to prepare text for the model.

        Please refer to the docstring of the above two methods for more information.
        """
        if images is None or text is None or classes is None:
            raise ValueError("You have to specify `images`, `text` and `classes`.")

        if isinstance(text, str):
            text = [text]
        if isinstance(classes[0], str):
            classes = [classes]

        # error when using `tokenizer_init_kwargs=self.tokenizer.init_kwargs` in _merge_kwargs` as
        # some init_kwargs are not defined in the forward method of the tokenizer e.g "padding_side"
        output_kwargs = self._merge_kwargs(
            OmDetTurboProcessorKwargs,
            tokenizer_init_kwargs=OmDetTurboProcessorKwargs._defaults["text_kwargs"],
            # tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        encoding_image_processor = self.image_processor(images, **output_kwargs["images_kwargs"])
        tasks_encoding = self.tokenizer(text=text, **output_kwargs["text_kwargs"])

        classes_structure = torch.tensor([len(class_single) for class_single in classes], dtype=torch.long)
        classes_flattened = [class_single for class_batch in classes for class_single in class_batch]
        classes_encoding = self.tokenizer(text=classes_flattened, **output_kwargs["text_kwargs"])
        classes_encoding.update({"structure": classes_structure})

        encoding = BatchEncoding({"tasks": tasks_encoding, "classes": classes_encoding})
        encoding.update(encoding_image_processor)

        return encoding

    # Copied from transformers.models.blip.processing_blip.BlipProcessor.batch_decode with BertTokenizerFast->PreTrainedTokenizer
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.blip.processing_blip.BlipProcessor.decode with BertTokenizerFast->PreTrainedTokenizer
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def post_process_boxes_for_image(
        self,
        boxes,
        scores,
        predicted_classes,
        classes: List[str],
        image_size: Tuple[int, int],
        num_classes: int,
        score_threshold: float,
        nms_threshold: float,
        max_num_det: int = None,
    ) -> dict:
        """
        Filter predicted results using given thresholds and NMS.
        Args:
            boxes (Tensor): A Tensor of predicted class-specific or class-agnostic
                boxes for the image. Shape : (num_queries, max_num_classes_in_batch * 4) if doing
                class-specific regression, or (num_queries, 4) if doing class-agnostic
                regression.
                This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
            scores (Tensor): A Tensor of predicted class scores for the image.
                Shape : (num_queries, max_num_classes_in_batch + 1)
                This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
            predicted_classes (Tensor): A Tensor of predicted classes for the image.
                Shape : (num_queries * (max_num_classes_in_batch + 1),)
            classes (List[str]): The input classes names.
            image_size (tuple): A tuple of (height, width) for the image.
            num_classes (int): The number of classes given for this image.
            score_threshold (float): Only return detections with a confidence score exceeding this
                threshold.
            nms_threshold (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
            max_num_det (int, optional): The maximum number of detections to return. Default is None.
        Returns:
            dict: A dictionary the following keys:
                "boxes" (Tensor): A tensor of shape (num_filtered_objects, 4), containing the predicted boxes in (x1, y1, x2, y2) format.
                "scores" (Tensor): A tensor of shape (num_filtered_objects,), containing the predicted confidence scores for each detection.
                "classes" (List[str]): A list of strings, where each string is the predicted class for the
                    corresponding detection
        """
        proposal_num = len(boxes) if max_num_det is None else max_num_det
        print("boxes", boxes.shape)
        print("scores", scores.shape)
        print("predicted_classes", predicted_classes.shape)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(proposal_num, sorted=False)
        classes_per_image = predicted_classes[topk_indices]
        box_pred_per_image = boxes.view(-1, 1, 4).repeat(1, num_classes, 1).view(-1, 4)
        box_pred_per_image = box_pred_per_image[topk_indices]

        # Score filtering
        box_pred_per_image = center_to_corners_format(box_pred_per_image)
        box_pred_per_image = box_pred_per_image * torch.tensor(image_size[::-1]).repeat(2).to(
            box_pred_per_image.device
        )
        filter_mask = scores_per_image > score_threshold  # R x K
        score_keep = filter_mask.nonzero(as_tuple=False).view(-1)
        box_pred_per_image = box_pred_per_image[score_keep]
        scores_per_image = scores_per_image[score_keep]
        classes_per_image = classes_per_image[score_keep]

        filter_classes_mask = classes_per_image < len(classes)
        classes_keep = filter_classes_mask.nonzero(as_tuple=False).view(-1)
        box_pred_per_image = box_pred_per_image[classes_keep]
        scores_per_image = scores_per_image[classes_keep]
        classes_per_image = classes_per_image[classes_keep]

        # NMS
        keep = batched_nms(box_pred_per_image, scores_per_image, classes_per_image, nms_threshold)
        box_pred_per_image = box_pred_per_image[keep]
        scores_per_image = scores_per_image[keep]
        classes_per_image = classes_per_image[keep]
        classes_per_image = [classes[i] for i in classes_per_image]

        # create an instance
        result = {}
        result["boxes"] = clip_boxes(box_pred_per_image, image_size)
        result["scores"] = scores_per_image
        result["classes"] = classes_per_image

        print("result bbox", result["boxes"].shape)
        print("result scores", result["scores"].shape)
        # print("result classes", len(result["classes"]))
        return result

    def post_process_grounded_object_detection(
        self,
        outputs,
        classes: List[str],
        score_threshold: float = 0.3,
        nms_threshold: float = 0.5,
        target_sizes: Union[TensorType, List[Tuple]] = None,
        max_num_det=None,
    ):
        """
        Converts the raw output of [`OmDetTurboForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format and get the associated text class.

        Args:
            outputs ([`OmDetTurboObjectDetectionOutput`]):
                Raw outputs of the model.
            classes (list[str]): The input classes names.
            score_threshold (float): Only return detections with a confidence score exceeding this
                threshold.
            nms_threshold (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                `(width, height)` of each image in the batch. If unset, predictions will not be resized.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, classes and boxes for an image
            in the batch as predicted by the model.
        """
        boxes_logits = outputs.decoder_coord_logits
        scores_logits = outputs.decoder_class_logits
        scores, predicted_classes = compute_score(scores_logits)
        num_classes = scores_logits.shape[2]
        results = []
        for scores_img, box_per_img, image_size, class_names in zip(scores, boxes_logits, target_sizes, classes):
            results.append(
                self.post_process_boxes_for_image(
                    box_per_img,
                    scores_img,
                    predicted_classes,
                    class_names,
                    image_size,
                    num_classes,
                    score_threshold=score_threshold,
                    nms_threshold=nms_threshold,
                    max_num_det=max_num_det,
                )
            )

        return results
