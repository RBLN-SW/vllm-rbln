# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from vllm.config.scheduler import SchedulerConfig
from vllm.v1.engine.input_processor import InputProcessor

from run_pdd_once import _build_decode_cmd
from vllm_rbln import utils as rbln_utils
from vllm_rbln.v1.worker.rbln_model_runner import (
    RBLNModelRunner,
    _get_warmup_prompt_tokens,
)


def test_max_model_len_512_reserves_one_sample_token():
    assert _get_warmup_prompt_tokens(512, 512, 1) == 511


def test_max_model_len_512_without_sampling_uses_full_prompt():
    assert _get_warmup_prompt_tokens(512, 512, 0) == 512


def test_max_model_len_one_with_sampling_fails_clearly():
    with pytest.raises(ValueError, match="requires at least one prompt token"):
        _get_warmup_prompt_tokens(1, 1, 1)


def test_shorter_warmup_prompt_is_unchanged():
    assert _get_warmup_prompt_tokens(128, 512, 1) == 128


def test_warm_up_model_passes_clamped_prefill_length_to_dummy_request():
    class StopAfterPrefill(Exception):
        pass

    captured: dict[str, object] = {}

    def capture_dummy_request(**kwargs):
        captured.update(kwargs)
        raise StopAfterPrefill

    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=[object()]),
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=True,
            max_num_batched_tokens=512,
        ),
        model_config=SimpleNamespace(max_model_len=512),
        is_pooling_model=False,
        _add_dummy_requests=capture_dummy_request,
    )

    with pytest.raises(StopAfterPrefill):
        RBLNModelRunner.warm_up_model.__wrapped__(runner)

    assert captured["total_tokens"] == 511
    assert captured["sampling_params"] is not None


def test_sampling_prompt_preserves_512_prefill_compile_input_shape():
    prompt_tokens = _get_warmup_prompt_tokens(512, 512, 1)
    logical_input = torch.arange(prompt_tokens).view(1, -1)

    compile_input = rbln_utils.pad(logical_input, -1, target_len=512)

    assert logical_input.shape == (1, 511)
    assert compile_input.shape == (1, 512)


def test_actual_513_token_request_is_still_rejected():
    processor = SimpleNamespace(
        skip_prompt_length_check=False,
        model_config=SimpleNamespace(
            max_model_len=512,
            runner_type="generate",
        ),
        mm_encoder_cache_size=0,
        supports_mm_inputs=False,
    )

    with pytest.raises(
        ValueError,
        match=r"decoder prompt \(length 513\).*maximum model length of 512",
    ):
        InputProcessor._validate_prompt_len(processor, 513, "decoder")


def test_bookkeeping_still_rejects_513_tokens_for_max_model_len_512():
    input_batch = SimpleNamespace(
        num_reqs=1,
        generators={},
        req_ids=["request-0"],
        req_id_to_index={"request-0": 0},
        num_tokens_no_spec=np.array([512]),
    )
    runner = SimpleNamespace(
        input_batch=input_batch,
        discard_request_mask=SimpleNamespace(np=np.array([False])),
        use_async_scheduling=False,
        max_model_len=512,
        _to_list=lambda sampled: sampled.tolist(),
    )
    sampler_output = SimpleNamespace(
        sampled_token_ids=torch.tensor([[42]]),
        logprobs_tensors=None,
    )

    with (
        patch(
            "vllm_rbln.v1.worker.rbln_model_runner.envs."
            "VLLM_COMPUTE_NANS_IN_LOGITS",
            False,
        ),
        pytest.raises(
            AssertionError,
            match=r"Total number of tokens: 513 > max_model_len: 512",
        ),
    ):
        RBLNModelRunner._bookkeeping_sync(
            runner,
            SimpleNamespace(),
            sampler_output,
            None,
            torch.empty(0),
            512,
            None,
        )


def test_warmup_clamp_preserves_scheduler_compile_hash():
    scheduler = SimpleNamespace(max_num_batched_tokens=512)
    hash_before = SchedulerConfig.compute_hash(scheduler)

    assert _get_warmup_prompt_tokens(
        scheduler.max_num_batched_tokens,
        max_model_len=512,
        warmup_sample_tokens=1,
    ) == 511

    assert scheduler.max_num_batched_tokens == 512
    assert SchedulerConfig.compute_hash(scheduler) == hash_before
    assert SchedulerConfig.compute_hash(
        SimpleNamespace(max_num_batched_tokens=511)
    ) != hash_before


def test_pdd_decode_command_keeps_512_model_and_block_contract():
    command = _build_decode_cmd(
        model="/models/Qwen3-0.6B",
        served_model_name="Qwen3-0.6B",
        block_size=512,
        max_model_len=512,
        max_num_seqs=1,
    )

    assert command[command.index("--block-size") + 1] == "512"
    assert command[command.index("--max-model-len") + 1] == "512"
    assert "--max-num-batched-tokens" not in command
