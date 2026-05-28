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
from vllm.multimodal.inputs import PlaceholderRange
from vllm.multimodal.processing import (
    ProcessorInputs,
    TimingContext,
)

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
        # Pad both BEFORE and AFTER each image block so no prefill chunk mixes
        # image tokens with text tokens:
        #   - pre-pad  : align block start    to a multiple of chunk_size
        #                (image block always starts a fresh chunk)
        #   - post-pad : align position after block to a multiple of chunk_size
        #                (following text always starts a fresh chunk)
        # `(-len(out)) % chunk` gives the distance UP to the next multiple
        # (0 when already aligned). Long image blocks (> chunk) span multiple
        # chunks naturally; only the tail of the last chunk needs post-pad.
        starts = image_starts.tolist()
        ends = image_ends.tolist()
        out: list[int] = []
        cursor = 0
        # Per-block layout in the padded prompt, used to rebuild PlaceholderRange
        # so downstream `is_embed` reflects which slots are image-token positions
        # (1) vs PAD fillers (-1) inserted for chunk alignment, both pre and post.
        block_info: list[
            tuple[int, int, int, int]
        ] = []  # (range_start, pre_pad, block_length, post_pad)
        for s, e in zip(starts, ends):
            # text / markers (e.g. boi) before this image block
            out.extend(prompt_ids[cursor:s])
            # range_start: where pre-pad begins (= placeholder range offset so
            # that pre-pad slots are inside the range, marked with -1).
            range_start = len(out)
            # pre-pad → image block starts on chunk boundary
            pre_pad = (-len(out)) % IMAGE_PREFILL_CHUNK_SIZE
            out.extend([PAD_TOKEN_ID] * pre_pad)
            block_length = e - s + 1
            # image block itself
            out.extend(prompt_ids[s : e + 1])
            # post-pad → next position (eoi/text) starts on chunk boundary
            post_pad = (-len(out)) % IMAGE_PREFILL_CHUNK_SIZE
            out.extend([PAD_TOKEN_ID] * post_pad)
            block_info.append((range_start, pre_pad, block_length, post_pad))
            cursor = e + 1
        # trailing text after the last image block
        out.extend(prompt_ids[cursor:])
        print("@@@ image_starts", image_starts)
        print("@@@ added pad tokens", len(out) - len(prompt_ids))
        print(
            "@@@ block_info (range_start, pre_pad, block_length, post_pad)",
            block_info,
        )
        return out, block_info

    def apply(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ):
        output = super().apply(inputs, timing_ctx)
        padded_prompt_ids, block_info = self._pad_for_gemma4(output["prompt_token_ids"])
        output["prompt_token_ids"] = padded_prompt_ids

        # Rebuild image PlaceholderRanges to reflect the inserted padding:
        #   - offset   : start of the pre-pad slot (just after the previous
        #                text), so both pre-pad and post-pad live inside the
        #                range
        #   - length   : pre_pad + block_length + post_pad
        #   - is_embed : int tensor of
        #                  [-1]*pre_pad + [1]*block_length + [-1]*post_pad
        #                ( 1 = vision-tower embedding slot,
        #                 -1 = PAD filler inserted for chunk alignment,
        #                  0 = text — never appears inside this range)
        # Downstream model code reads `is_embed` to know exactly which slots
        # receive image embeddings vs which are PADs.
        image_ranges = (
            output["mm_placeholders"].get("image")
            if output.get("mm_placeholders")
            else None
        )
        if image_ranges:
            assert len(image_ranges) == len(block_info), (
                f"placeholder/block count mismatch: "
                f"{len(image_ranges)} vs {len(block_info)}"
            )
            new_ranges = []
            for range_start, pre_pad, block_length, post_pad in block_info:
                is_embed = torch.cat(
                    [
                        # FIXME embed value should be boolean type.
                        torch.full((pre_pad,), -1, dtype=torch.int64),
                        torch.ones(block_length, dtype=torch.int64),
                        torch.full((post_pad,), -1, dtype=torch.int64),
                    ]
                )
                new_ranges.append(
                    PlaceholderRange(
                        offset=range_start,
                        length=pre_pad + block_length + post_pad,
                        is_embed=is_embed,
                    )
                )
            output["mm_placeholders"]["image"] = new_ranges

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
    # Opt-in flag read by the runner to build the per-position `is_embed`
    # label mask (1=image, -1=PAD, 0=text) from MultiModalFeatureSpec and
    # pass it through ModelInputForRBLN. Only Gemma-family models need this.
    requires_is_embed: bool = True

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
            if model_input.is_embed is not None:
                assert model_input.is_embed.shape == input_ids.shape, (
                    f"is_embed shape {tuple(model_input.is_embed.shape)} "
                    f"!= input_ids shape {tuple(input_ids.shape)}"
                )
                mm_token_type_ids = model_input.is_embed
            else:
                # text-only data
                mm_token_type_ids = torch.zeros_like(input_ids)
            pixel_values, image_position_ids = self.get_image_values(model_input)
            print("@@@ input_ids", input_ids.shape, input_ids)
            print("@@@ mm_token_type_ids", mm_token_type_ids)
            print("@@@ count", (mm_token_type_ids == -1).sum().item())
            print("@@@ is_embed (from runner)", model_input.is_embed)
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
            print("@@@ attention_mask", attention_mask)
            output = self.model.language_model.prefill_decoder(
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                local_block_tables=local_block_table_id,
                block_tables=block_tables,
                token_type_ids=mm_token_type_ids,
            )
            logits = output.logits
            updated_padded_cache_length = output.padded_cache_lengths
            print("@@@ updated_padded_cache_length", updated_padded_cache_length)
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
            print("@@@ [decode] padded_cache_lengths", padded_cache_lengths)
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
            print("@@@ position_ids", position_ids)
            print("@@@ input_ids", input_ids)
            print("@@@ cache_position", cache_position)
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
