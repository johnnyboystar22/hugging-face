from ...generation.configuration_utils import GenerationConfig


class ChameleonGenerationConfig(GenerationConfig):
    """Generation Config for [Chameleon](https://huggingface.co/docs/transformers/model_doc/chameleon)

    Args:
        multimodal_generation_mode (`Literal["text-only", "image-only", "interleaved-text-image", "unrestricted"]`, *optional*, defaults to `None`):
            Chameleon can generate text, images, or both in an interleaved manner. However, only text generation is
            supported by the official model checkpoint. This flag enables the other modes for use with finetuned versions
            of the model such as [Anole](https://arxiv.org/abs/2407.06135).
            - If set to `"unrestricted"`, the logits are left as-is.
            - If set to `"text-only"`, logits for image tokens will be masked out during generation.
            - If set to `"image-only"`, logits for non-image tokens will be masked out during generation.
            - For `"interleaved-text-image"`, Chameleon implements a finite state machine to dynamically switch between text and image modalities.
                Here, we simply use logits processors that exclusively allow image tokens to be generated within a relative window after the
                begin image token and disallow them elsewhere.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.multimodal_generation_mode = kwargs.pop("multimodal_generation_mode", "text-only")
