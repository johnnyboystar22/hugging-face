from typing import List

import torch


class ChameleonImageOnlyLogitsProcessor:
    """Helper logits processor for Chameleon that masks non-image tokens."""

    def __init__(
        self,
        eos_token_id: int,
        boi_token_id: int,
        eoi_token_id: int,
        image_token_ids: List[int],
        image_seq_length: int,
        begin_index: int,
        max_new_tokens: int,
    ):
        """
        Mask the logits of non-image tokens to prevent them from being generated.

        Args
            eos_token_id (`int`): The end of sequence token id.
            boi_token_id (`int`): The start image marker token id.
            eoi_token_id (`int`): The end image marker token id.
            image_token_ids (`List[int]`): The image token ids, not including the start
                and end image marker tokens.
            image_seq_length (`int`): The length of the image sequence.
            begin_index (`int`): The index at which generation starts.
            max_new_tokens (`int`): The maximum number of tokens that can be generated.
        """
        self.eos_token_id = eos_token_id
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.image_token_ids = image_token_ids
        self.image_seq_length = image_seq_length
        self.begin_index = begin_index
        self.max_new_tokens = max_new_tokens

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Masks non-image tokens

        Args:
            input_ids (`torch.Tensor` of shape `(batch, sequence_length)`):
                The input token ids.
            scores (`torch.Tensor` of shape `(batch, vocab_size)`):
                The logits to be biased.

        Returns:
            `torch.Tensor` of shape `(batch, vocab_size)`:  The biased scores.
        """
        mask = torch.full_like(scores, torch.finfo(scores.dtype).min)
        idx = input_ids.shape[1] - self.begin_index
        if idx % (self.image_seq_length + 2) == 0:
            # Only generate a boi token if there is enough space for the rest of the image tokens
            if idx + self.image_seq_length + 1 < self.max_new_tokens:
                mask[:, self.boi_token_id] = scores[:, self.boi_token_id]
            mask[:, self.eos_token_id] = scores[:, self.eos_token_id]
        elif idx % (self.image_seq_length + 2) == self.image_seq_length + 1:
            mask[:, self.eoi_token_id] = scores[:, self.eoi_token_id]
        else:
            mask[:, self.image_token_ids] = scores[:, self.image_token_ids]
        return mask
