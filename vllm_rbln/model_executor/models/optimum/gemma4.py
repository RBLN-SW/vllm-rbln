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
from vllm.config import VllmConfig
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
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.interfaces_base import VllmModelForTextGeneration
from vllm.multimodal import MULTIMODAL_REGISTRY

from .base import ModelInputForRBLN, version_error
from .model_base import RBLNOptimumDecoderMixin, RBLNOptimumModelBase
from .optimum_attention import HybridAttentionImageManager, HybridAttentionImageStrategy

logger = init_logger(__name__)

PAD_TOKEN_ID = 0


class RBLNGemma4MultiModalProcessor(Gemma4MultiModalProcessor):
    def _pad_for_gemma4(self, prompt_ids: list[int]):
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
            f"Expected each image block to be {EXPECTED_IMAGE_TOKENS} tokens, "
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
        #   len(out)=530 → (-530) % 512 = 494   # bump to next boundary 1024
        #   len(out)=512 → 0                    # already aligned
        #   len(out)=0   → 0                    # already aligned
        #   len(out)=1   → 511                  # 511 pads → reach 512
        starts = image_starts.tolist()
        ends = image_ends.tolist()
        out: list[int] = []
        cursor = 0
        for s, e in zip(starts, ends):
            # text / markers (e.g. boi) before this image block
            out.extend(prompt_ids[cursor:s])
            print("@@@ pre-pad", (-len(out)) % IMAGE_PREFILL_CHUNK_SIZE)
            # pre-pad → image block starts on chunk boundary
            out.extend([PAD_TOKEN_ID] * ((-len(out)) % IMAGE_PREFILL_CHUNK_SIZE))
            # image block itself
            out.extend(prompt_ids[s : e + 1])
            print("@@@ post-pad", (-len(out)) % IMAGE_PREFILL_CHUNK_SIZE)
            # post-pad → next position (eoi / text) starts on chunk boundary
            out.extend([PAD_TOKEN_ID] * ((-len(out)) % IMAGE_PREFILL_CHUNK_SIZE))
            cursor = e + 1
        # trailing text after the last image block
        out.extend(prompt_ids[cursor:])
        print("@@@ total pad", len(out) - len(prompt_ids))
        return out

    def apply(self, *args, **kwargs):
        # NOTE: Check if padding works correctly
        output = super().apply(*args, **kwargs)
        prompt_ids = self._pad_for_gemma4(output["prompt_token_ids"])

        output["prompt_token_ids"] = prompt_ids

        return output


@MULTIMODAL_REGISTRY.register_processor(
    RBLNGemma4MultiModalProcessor,
    info=Gemma4ProcessingInfo,
    dummy_inputs=Gemma4DummyInputsBuilder,
)
class RBLNOptimumGemma4ForConditionalGeneration(
    RBLNOptimumModelBase,
    RBLNOptimumDecoderMixin,
    VllmModelForTextGeneration,
    SupportsMultiModal,
):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(vllm_config=vllm_config)
        # NOTE:
        # model_config.vocab_size != tokenizer.vocab_size in Gemma3
        assert self.kv_block_adapter is not None
        self.setup_decoder_mixin(
            attn_impl=self.attn_impl,
            vocab_size=self.model_config.get_vocab_size,
            use_multiple_decoder=getattr(
                self.model.rbln_config.language_model,
                "use_multiple_decoder",
                False,
            ),
            default_batch_size=self.scheduler_config.max_num_seqs,
            decoder_batch_sizes=self.model.rbln_config.language_model.decoder_batch_sizes,
            num_blocks=self.kv_block_adapter._estimated_num_blocks(),
        )
        self.strategy = HybridAttentionImageStrategy(PAD_TOKEN_ID)
        self.attention_manager: HybridAttentionImageManager = (
            HybridAttentionImageManager(self.strategy)
        )

    def forward(self, model_input: ModelInputForRBLN, **kwargs) -> torch.Tensor:
        input_ids = model_input.input_tokens
        position_ids = model_input.input_positions
        block_tables = model_input.block_tables

        is_prompt = model_input.is_prompt

        finished_requests_ids = model_input.finished_requests_ids
        running_requests_ids = model_input.running_requests_ids
        request_nums = input_ids.shape[0]

        # In prefill phase, the length of list must be 1
        sliding_window_table_ids, padded_cache_lengths, attention_masks = (
            self.attention_manager.get(
                is_prompt,
                self.decoder_batch_size,
                running_requests_ids,
                finished_requests_ids,
                input_ids=input_ids,
            )
        )

        kwargs = self.preprocess_for_decoder(
            is_prompt, block_tables, input_ids, position_ids
        )

        # [prefill] the length of the padded cache is calculated
        # during the forward pass and stored in self.sliding_window_table.
        # [decode] `cache_position` and `position_ids` are distinguished
        # due to the padding space reserved for the sliding window.
        cache_position = kwargs.pop("cache_position")
        input_ids = kwargs.pop("input_ids")
        block_tables = kwargs.pop("block_tables")

        if is_prompt:
            inputs_embeds = None
            prefill_batch_idx = sliding_window_table_ids[0]
            local_block_table_id = torch.tensor([prefill_batch_idx], dtype=torch.int16)
            # FIXME It should be delivered from runner
            # FIXME It should allow video_token_ids, audio_token_ids as well.
            # https://github.com/huggingface/transformers/blob/0588858f54c8c79d28497d3ad6eac3417b716c49/src/transformers/processing_utils.py#L897
            mm_token_type_ids = torch.zeros_like(input_ids)
            mm_token_type_ids[input_ids == self.model.config.image_token_id] = 1
            pixel_values, image_position_ids = self.get_image_values(model_input)
            inputs_embeds = self.model._preprocess_prefill(
                input_ids,
                inputs_embeds,
                pixel_values,
                image_position_ids=image_position_ids,
            )
            if self.model.language_model.prefill_decoder is None:
                raise version_error
            assert attention_masks is not None
            attention_mask = attention_masks[0]
            output = self.model.language_model.prefill_decoder(
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                local_block_tables=local_block_table_id,
                block_tables=block_tables,
                token_type_ids=mm_token_type_ids,
            )
            logits = output.logits
            updated_padded_cache_length = output.padded_cache_lengths

            updated_attention_mask = output.attention_mask

            assert len(running_requests_ids) == 1
            self.attention_manager.add(
                running_requests_id=running_requests_ids[0],
                local_table_id=sliding_window_table_ids[0],
                pad_len=updated_padded_cache_length,
                attention_mask=updated_attention_mask,
            )
        else:
            if self.model.language_model.decoders is None:
                raise ValueError("Decoders is None")
            padded_batch_size = kwargs.pop("padded_batch_size", self.decoder_batch_size)
            self.model.language_model.decoder = self.model.language_model.decoders[
                padded_batch_size
            ]
            (
                local_block_table_id,
                cache_position,
                position_ids,
                attention_mask,
            ) = self.attention_manager.preprocess(
                sliding_window_table_ids,
                cache_position,
                request_nums,
                padded_batch_size,
                pad_lens=padded_cache_lengths,
                attention_masks=attention_masks,
            )
            attention_mask = self.attention_manager.update(
                running_requests_ids,
                attention_mask,
                cache_position,
            )
            logits = self.model.language_model.decoder(
                input_ids=input_ids,
                cache_position=cache_position,
                block_tables=block_tables,
                local_block_tables=local_block_table_id,
                position_ids=position_ids,  # FIXME duplicated?
                attention_mask=attention_mask,
            ).logits

        if not is_prompt:
            logits = logits[:request_nums]
        return logits

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
            # FIXME Not implemented yet.
            # if (
            #     input_key == "pixel_values_videos"
            #     and "video" not in mm_input_by_modality
            # ):
            #     mm_input_by_modality["video"] = self._parse_and_validate_video_input(
            #         **kwargs
            #     )
            # if (
            #     input_key == "input_features_padded"
            #     and "audio" not in mm_input_by_modality
            # ):
            #     mm_input_by_modality["audio"] = self._parse_and_validate_audio_input(
            #         **kwargs
            #     )

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
