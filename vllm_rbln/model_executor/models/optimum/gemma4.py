# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from vllm.logger import init_logger
from vllm.model_executor.models.gemma4_mm import (
    Gemma4AudioInputs,
    Gemma4DummyInputsBuilder,
    Gemma4ImageInputs,
    Gemma4ImagePixelInputs,
    Gemma4MultiModalProcessor,
    Gemma4ProcessingInfo,
    Gemma4VideoInputs,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from .base import ModelInputForRBLN
from .gemma3 import (
    PAD_TOKEN_ID,
    RBLNOptimumGemma3ForConditionalGeneration,
)

logger = init_logger(__name__)


class RBLNGemma4MultiModalProcessor(Gemma4MultiModalProcessor):
    """Left-pads ``prompt_token_ids`` so image blocks align to prefill chunk
    boundaries."""

    def apply(self, *args, **kwargs):
        output = super().apply(*args, **kwargs)
        output["prompt_token_ids"] = self._pad_image_boundaries(
            output["prompt_token_ids"]
        )
        return output

    def _pad_image_boundaries(self, prompt_ids: list[int]) -> list[int]:
        """Left-pad ``prompt_token_ids`` so each image block aligns to a prefill
        chunk boundary.

        The padding only inflates the prompt length vLLM sees, so vLLM allocates
        enough KV-cache blocks. The actual chunk-aligned attention is handled
        inside optimum-rbln, which infers the padding length from these
        left-padded tokens.
        """
        token_type_ids = (
            torch.tensor(prompt_ids) == self.info.get_hf_processor().image_token_id
        )
        # FIXME: hardcoded prefill chunk size
        IMAGE_PREFILL_CHUNK_SIZE = 512
        # Find image block start positions. Unlike Gemma3 (fixed image_seq_length
        # soft tokens per image), Gemma4 emits a dynamic number of soft tokens per
        # image, so we detect the start of each contiguous image-token run (a True
        # whose predecessor is False) instead of assuming a fixed block length.
        # Shift right by one so each position holds "was the previous token an
        # image token?", then a block start is "image now AND not image before":
        #   token_type_ids        [F F T T T F F T T F]
        #   prev_is_image         [F F F T T T F F T T]   (shifted right)
        #   now & ~prev (starts)  [. . S . . . . S . .]   -> image_starts = [2, 7]
        prev_is_image = torch.cat(
            [token_type_ids.new_zeros(size=(1,)), token_type_ids[:-1]]
        )
        image_starts = torch.where(token_type_ids & ~prev_is_image)[0]
        # TEMP: variable-length image blocks are not fully implemented yet, so for
        # now every image is assumed to emit a fixed number of soft tokens. Detect
        # each block's end (mirror of the start logic: shift left so each position
        # holds "is the next token an image token?", a block end is "image now AND
        # not image next") and assert the fixed length. Remove this guard once the
        # dynamic per-image token count above is supported.
        EXPECTED_IMAGE_TOKENS = 280
        next_is_image = torch.cat(
            [token_type_ids[1:], token_type_ids.new_zeros(size=(1,))]
        )
        image_ends = torch.where(token_type_ids & ~next_is_image)[0]
        block_lengths = image_ends - image_starts + 1
        assert torch.all(block_lengths < EXPECTED_IMAGE_TOKENS), (
            f"Expected each image block to be less than {EXPECTED_IMAGE_TOKENS} tokens, "
            f"got {block_lengths.tolist()}"
        )
        # Pad BOTH before and after each image block so no prefill chunk mixes
        # image tokens with text tokens. Only aligning the image start (the
        # earlier "all PAD at front" approach) leaves the chunk's tail filled
        # with the text that follows the image — violating the no-mixing rule.
        #
        #   - pre-pad  : align block start to chunk boundary
        #                → image block always opens a fresh chunk
        #   - post-pad : align position right AFTER the block to chunk boundary
        #                → following text (e.g. eoi, next prompt) always opens
        #                  a fresh chunk
        #
        # `(-len(out)) % chunk` gives the distance UP to the next multiple
        # (0 when already aligned). Long image blocks (> chunk) span multiple
        # chunks naturally; only the tail of the last chunk needs post-pad.
        # With chunk=512:
        #   cur_len=530 → (-530) % 512 = 494   # bump to next boundary 1024
        #   cur_len=512 → 0                    # already aligned
        #   cur_len=0   → 0                    # already aligned
        #   cur_len=1   → 511                  # 511 pads → reach 512
        #
        # Only the cumulative total `padded_seq_len` is left-padded onto the
        # prompt at the end, so we don't materialise a list of the would-be
        # padded prompt — a single running `cur_len` counter is enough to
        # compute each pre/post-pad amount.
        # Two cursors run side-by-side over different coordinate systems:
        #
        #   original prompt (cursor):  [text 26][image 273][text ...]
        #                               0       26         299
        #
        #   padded prompt (cur_len):   [text 26][PAD 486][image 273][PAD 239][text ...]
        #                               0       26       512        785      1024
        #
        # - cursor : position in the ORIGINAL prompt_ids (no PAD); used to
        #            slice the next text segment and jump past each image.
        # - cur_len: length of the HYPOTHETICAL padded prompt so far (includes
        #            PAD); used to compute each pre/post-pad amount.
        # Their difference grows by (pre_pad + post_pad) every iteration; at
        # the end, `cur_len - cursor == padded_seq_len`.
        starts = image_starts.tolist()
        ends = image_ends.tolist()
        cur_len = 0  # running length of the (hypothetical) padded prompt
        cursor = 0  # running position in the original prompt
        padded_seq_len = 0  # total PAD tokens to prepend
        for s, e in zip(starts, ends):
            # text / markers (e.g. boi) before this image block
            cur_len += s - cursor
            # pre-pad → image block starts on chunk boundary
            pre_pad = (-cur_len) % IMAGE_PREFILL_CHUNK_SIZE
            cur_len += pre_pad
            padded_seq_len += pre_pad
            # image block itself
            cur_len += e - s + 1
            # post-pad → next position (eoi / text) starts on chunk boundary
            post_pad = (-cur_len) % IMAGE_PREFILL_CHUNK_SIZE
            cur_len += post_pad
            padded_seq_len += post_pad
            cursor = e + 1
        # Left-pad to inflate the prompt length vLLM sees → vLLM allocates
        # enough KV-cache blocks. The real chunk-aligned pre/post-pad runs
        # inside optimum-rbln at attention time (gemma4_runtime_utils.py).
        return [PAD_TOKEN_ID] * padded_seq_len + prompt_ids


@MULTIMODAL_REGISTRY.register_processor(
    RBLNGemma4MultiModalProcessor,
    info=Gemma4ProcessingInfo,
    dummy_inputs=Gemma4DummyInputsBuilder,
)
class RBLNOptimumGemma4ForConditionalGeneration(
    RBLNOptimumGemma3ForConditionalGeneration
):
    def _build_prefill_embeds(
        self, model_input: ModelInputForRBLN, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # FIXME It should be delivered from runner
        # FIXME It should allow video_token_ids, audio_token_ids as well.
        # https://github.com/huggingface/transformers/blob/0588858f54c8c79d28497d3ad6eac3417b716c49/src/transformers/processing_utils.py#L897
        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == self.model.config.image_token_id] = 1
        pixel_values, image_position_ids = self.get_image_values(model_input)
        inputs_embeds = self.model._preprocess_prefill(
            input_ids,
            None,
            pixel_values,
            image_position_ids=image_position_ids,
        )
        return inputs_embeds, mm_token_type_ids

    def get_image_values(
        self, model_input: ModelInputForRBLN
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not model_input.multi_modal_kwargs:
            return None, None

        multimodal_inputs = self._parse_and_validate_multimodal_inputs(
            **model_input.multi_modal_kwargs
        )
        image_input = multimodal_inputs.get("image")
        if not isinstance(image_input, Gemma4ImagePixelInputs):
            return None, None

        return image_input["pixel_values"], image_input["pixel_position_ids"]

    def _parse_and_validate_multimodal_inputs(
        self, **kwargs: object
    ) -> dict[str, Gemma4ImageInputs | Gemma4AudioInputs | Gemma4VideoInputs | None]:
        mm_input_by_modality = {}
        for input_key in list(kwargs):
            if (
                input_key in ("pixel_values", "image_embeds")
                and "image" not in mm_input_by_modality
            ):
                mm_input_by_modality["image"] = self._parse_and_validate_image_input(
                    **kwargs
                )
            # NOTE: Other multimodal types (audio, video) will be implemented later.

        return mm_input_by_modality

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Gemma4ImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        pixel_position_ids = kwargs.pop("pixel_position_ids", None)
        image_embeds = kwargs.pop("image_embeds", None)
        assert image_embeds is None, "Gemma4 does not support image_embeds."
        if pixel_values is None:
            return None
        return Gemma4ImagePixelInputs(
            pixel_values=pixel_values,
            pixel_position_ids=pixel_position_ids,
        )
