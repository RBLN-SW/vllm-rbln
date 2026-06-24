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

from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.spec_decode.eagle import EagleProposer

import vllm_rbln.v1.spec_decode.eagle as eagle_module
from vllm_rbln.v1.spec_decode.eagle import RBLNEagleProposer

DEVICE = torch.device(current_platform.device_type)


class FakeEagleModel:
    """Draft model double.

    Forward returns (last_hidden, hidden) with distinct offsets so the two
    streams are tell-apart; compute_logits records the hidden states it gets.
    """

    def __init__(self):
        self.compute_logits_inputs: list[torch.Tensor] = []

    def __call__(self, *, input_ids, positions, hidden_states, inputs_embeds):
        return hidden_states + 1000, hidden_states + 2000

    def compute_logits(self, sample_hidden_states: torch.Tensor) -> torch.Tensor:
        self.compute_logits_inputs.append(sample_hidden_states.clone())
        return sample_hidden_states + 1


class FakeBackupNextTokenIds:
    def __init__(self, size: int):
        self.np = np.zeros(size, dtype=np.int32)
        self.gpu = torch.zeros(size, dtype=torch.int32, device=DEVICE)

    def copy_to_gpu(self, num_reqs: int) -> None:
        self.gpu[:num_reqs] = torch.from_numpy(self.np[:num_reqs])


class FakeRequestState:
    def __init__(self, token_base: int):
        self.token_base = token_base

    def get_token_id(self, seq_len: int) -> int:
        return self.token_base + seq_len


class FakeEagle3Model(Eagle3LlamaForCausalLM):
    """eagle3 model double; subclasses the real type to pass the isinstance gate.

    combine_hidden_states returns a fixed tensor so its output shape can be
    controlled independently of the input.
    """

    def __init__(self, combined: torch.Tensor):
        self.combined = combined

    def combine_hidden_states(self, target_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.combined


class FakeMetadataBuilder:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def build(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(kwargs=kwargs)


class FakeAttentionGroup:
    def __init__(self, layer_names: list[str], builder: FakeMetadataBuilder):
        self.layer_names = layer_names
        self._builder = builder

    def get_metadata_builder(self) -> FakeMetadataBuilder:
        return self._builder


def make_fake_runner(
    *,
    batch_bucket: int = 4,
    num_reqs: int = 2,
    is_prefill: bool = False,
    is_intermediate_chunked_prefill: bool = False,
):
    """Minimal runner fake exposing only what _build/_dummy/propose paths read."""
    return SimpleNamespace(
        bucketing_manager=SimpleNamespace(
            find_decode_batch_bucket=lambda n: batch_bucket
        ),
        input_batch=SimpleNamespace(
            num_reqs=num_reqs,
            block_table=[
                SimpleNamespace(
                    get_cpu_tensor=lambda: torch.zeros((8, 4), dtype=torch.int32)
                )
            ],
        ),
        kv_caches=[],
        kv_cache_bases=[],
        kv_cache_view_infos=[],
        compile_context=object(),
        is_prefill=is_prefill,
        is_intermediate_chunked_prefill=is_intermediate_chunked_prefill,
        # _build_dummy_attn_metadata only consumes the cumsum (arange ignored).
        _get_cumsum_and_arange=lambda num_tokens, cumsum_dtype=None: (
            np.cumsum(num_tokens, dtype=cumsum_dtype),
            None,
        ),
    )


def make_common_attn_metadata(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table_tensor: torch.Tensor | None = None,
) -> CommonAttentionMetadata:
    total_num_tokens = int(query_start_loc[-1].item())
    if block_table_tensor is None:
        block_table_tensor = torch.zeros((seq_lens.shape[0], 4), dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=seq_lens,
        _seq_lens_cpu=seq_lens.cpu(),
        _num_computed_tokens_cpu=torch.zeros_like(seq_lens),
        num_reqs=seq_lens.shape[0],
        num_actual_tokens=total_num_tokens,
        max_query_len=int((query_start_loc[1:] - query_start_loc[:-1]).max().item()),
        max_seq_len=int(seq_lens.max().item()),
        block_table_tensor=block_table_tensor,
        slot_mapping=torch.arange(total_num_tokens, dtype=torch.int64),
        causal=True,
        dcp_local_seq_lens=None,
    )


def make_eagle_stub(
    *,
    max_num_tokens: int = 32,
    hidden_size: int = 4,
    num_speculative_tokens: int = 1,
    method: str = "eagle",
    max_model_len: int = 128,
    enforce_eager: bool = True,
    runner: object | None = None,
) -> RBLNEagleProposer:
    """Build a proposer without running the heavy base __init__.

    Only the buffers/attributes the method under test reads are populated; each
    test layers on whatever else it needs (model, draft_attn_groups, ...).
    """
    stub = object.__new__(RBLNEagleProposer)
    stub.device = DEVICE
    stub.hidden_size = hidden_size
    stub.method = method
    stub.max_num_tokens = max_num_tokens
    stub.max_model_len = max_model_len
    stub.num_speculative_tokens = num_speculative_tokens
    stub.input_ids = torch.full((max_num_tokens,), -1, dtype=torch.int32, device=DEVICE)
    stub.positions = torch.full((max_num_tokens,), -1, dtype=torch.int64, device=DEVICE)
    stub.hidden_states = torch.zeros(
        max_num_tokens, hidden_size, dtype=torch.float32, device=DEVICE
    )
    stub.arange = torch.arange(max_num_tokens + 1, dtype=torch.int32, device=DEVICE)
    stub.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(enforce_eager=enforce_eager)
    )
    stub.runner = (
        runner if runner is not None else SimpleNamespace(compile_context=object())
    )
    stub.allowed_attn_types = None
    return stub


@pytest.fixture(autouse=True)
def neutralize_collaborators(monkeypatch):
    """Stub the heavy forward-context / kv-cache plumbing for orchestration."""
    monkeypatch.setattr(
        eagle_module, "set_forward_context", lambda *a, **k: nullcontext()
    )
    monkeypatch.setattr(eagle_module, "attach_kv_cache_bindings", lambda *a, **k: None)
    monkeypatch.setattr(
        eagle_module, "build_kv_cache_forward_context_kwargs", lambda *a, **k: {}
    )


# ---------------------------------------------------------------------------
# _preprocess
# ---------------------------------------------------------------------------


# Verifies the prefill branch views the whole buffer into [B, L(, H)] shapes
# without slicing or padding.
def test_preprocess_prefill_views_full_buffers():
    stub = make_eagle_stub(max_num_tokens=8, hidden_size=4)
    stub.input_ids[:] = torch.arange(8, dtype=torch.int32)
    stub.positions[:] = torch.arange(8, dtype=torch.int64)
    stub.hidden_states[:] = torch.arange(32, dtype=torch.float32).view(8, 4)

    input_ids, positions, hidden_states, token_indices = RBLNEagleProposer._preprocess(
        stub,
        num_reqs=2,
        num_reqs_padded=2,
        num_input_tokens=8,
        token_indices_to_sample=torch.tensor([1, 3], dtype=torch.int32),
        is_prefill=True,
    )

    assert input_ids.shape == (2, 4)
    assert positions.shape == (2, 4)
    assert hidden_states.shape == (2, 4, 4)
    torch.testing.assert_close(input_ids, torch.arange(8, dtype=torch.int32).view(2, 4))
    torch.testing.assert_close(token_indices, torch.tensor([1, 3], dtype=torch.int32))


# Verifies the decode branch slices to the active tokens, then pads the batch
# dimension up to the decode bucket while leaving the active rows intact.
def test_preprocess_decode_slices_and_pads_to_bucket():
    stub = make_eagle_stub(max_num_tokens=8, hidden_size=4)
    stub.input_ids[:4] = torch.tensor([10, 11, 20, 21], dtype=torch.int32)
    stub.positions[:4] = torch.tensor([4, 5, 6, 7], dtype=torch.int64)
    stub.hidden_states[:4] = torch.arange(16, dtype=torch.float32).view(4, 4)

    input_ids, positions, hidden_states, token_indices = RBLNEagleProposer._preprocess(
        stub,
        num_reqs=2,
        num_reqs_padded=4,
        num_input_tokens=4,
        token_indices_to_sample=torch.tensor([1, 3], dtype=torch.int32),
        is_prefill=False,
    )

    assert input_ids.shape == (4, 2)
    assert positions.shape == (4, 2)
    assert hidden_states.shape == (4, 2, 4)
    torch.testing.assert_close(
        input_ids[:2], torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
    )
    torch.testing.assert_close(input_ids[2:], torch.zeros((2, 2), dtype=torch.int32))
    # token indices are zero-padded up to the bucket.
    torch.testing.assert_close(
        token_indices, torch.tensor([1, 3, 0, 0], dtype=torch.int32)
    )


# Verifies token indices stay None when none are requested.
def test_preprocess_passes_through_absent_token_indices():
    stub = make_eagle_stub(max_num_tokens=8, hidden_size=4)

    *_, token_indices = RBLNEagleProposer._preprocess(
        stub,
        num_reqs=2,
        num_reqs_padded=4,
        num_input_tokens=4,
        token_indices_to_sample=None,
        is_prefill=False,
    )

    assert token_indices is None


# ---------------------------------------------------------------------------
# set_inputs_first_pass
# ---------------------------------------------------------------------------


# Verifies the first pass shifts target tokens left, scatters the sampled
# next-tokens at the inferred indices, and copies positions/hidden states.
def test_set_inputs_first_pass_shifts_tokens_and_infers_default_indices():
    stub = make_eagle_stub(max_num_tokens=8, hidden_size=4)
    stub.needs_extra_input_slots = False
    stub._set_positions = lambda n, p: stub.positions[:n].copy_(p)
    cad = make_common_attn_metadata(
        query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
        seq_lens=torch.tensor([8, 9], dtype=torch.int32),
    )
    target_positions = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    target_hidden_states = torch.arange(20, dtype=torch.float32).view(5, 4)

    num_tokens, token_indices = RBLNEagleProposer.set_inputs_first_pass(
        stub,
        target_token_ids=torch.tensor([10, 11, 20, 21, 22], dtype=torch.int32),
        next_token_ids=torch.tensor([99, 88], dtype=torch.int32),
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        token_indices_to_sample=None,
        cad=cad,
    )

    assert num_tokens == 5
    # default indices = query_start_loc[1:] - 1 = [2, 5] - 1
    torch.testing.assert_close(token_indices, torch.tensor([1, 4], dtype=torch.int32))
    # shifted target[1:] then next-token scatter at indices 1 and 4
    torch.testing.assert_close(
        stub.input_ids[:5], torch.tensor([11, 99, 21, 22, 88], dtype=torch.int32)
    )
    torch.testing.assert_close(stub.positions[:5], target_positions)
    torch.testing.assert_close(stub.hidden_states[:5], target_hidden_states)


# Verifies explicit sample indices are used verbatim instead of being inferred.
def test_set_inputs_first_pass_uses_explicit_indices():
    stub = make_eagle_stub(max_num_tokens=8, hidden_size=4)
    stub.needs_extra_input_slots = False
    stub._set_positions = lambda n, p: stub.positions[:n].copy_(p)
    cad = make_common_attn_metadata(
        query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
        seq_lens=torch.tensor([8, 9], dtype=torch.int32),
    )
    explicit = torch.tensor([0, 3], dtype=torch.int64)

    num_tokens, token_indices = RBLNEagleProposer.set_inputs_first_pass(
        stub,
        target_token_ids=torch.tensor([10, 11, 20, 21, 22], dtype=torch.int32),
        next_token_ids=torch.tensor([99, 88], dtype=torch.int32),
        target_positions=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64),
        target_hidden_states=torch.zeros((5, 4), dtype=torch.float32),
        token_indices_to_sample=explicit,
        cad=cad,
    )

    assert num_tokens == 5
    assert token_indices is explicit
    # input_ids[:4] = [11, 20, 21, 22]; scatter 99,88 at indices 0 and 3
    torch.testing.assert_close(
        stub.input_ids[:4], torch.tensor([99, 20, 21, 88], dtype=torch.int32)
    )


# Verifies extra input slots (parallel drafting) are explicitly unsupported.
def test_set_inputs_first_pass_rejects_extra_input_slots():
    stub = make_eagle_stub()
    stub.needs_extra_input_slots = True
    cad = make_common_attn_metadata(
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        seq_lens=torch.tensor([5, 6], dtype=torch.int32),
    )

    with pytest.raises(NotImplementedError, match="extra input slots"):
        RBLNEagleProposer.set_inputs_first_pass(
            stub,
            target_token_ids=torch.tensor([1, 2, 3, 4], dtype=torch.int32),
            next_token_ids=torch.tensor([10, 11], dtype=torch.int32),
            target_positions=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
            target_hidden_states=torch.zeros((4, 4), dtype=torch.float32),
            token_indices_to_sample=None,
            cad=cad,
        )


# ---------------------------------------------------------------------------
# _build_dummy_attn_metadata
# ---------------------------------------------------------------------------


# Verifies the dummy metadata derives query_start_loc from a cumulative sum and
# reports the expected per-request shapes.
def test_build_dummy_attn_metadata_computes_cumsum_and_shapes():
    stub = make_eagle_stub(max_num_tokens=32)
    stub.runner = make_fake_runner()

    cad = RBLNEagleProposer._build_dummy_attn_metadata(
        stub, num_reqs=3, num_tokens_per_req=2
    )

    torch.testing.assert_close(
        cad.query_start_loc, torch.tensor([0, 2, 4, 6], dtype=torch.int32)
    )
    torch.testing.assert_close(cad.seq_lens, torch.tensor([2, 2, 2], dtype=torch.int32))
    assert cad.num_reqs == 3
    assert cad.num_actual_tokens == 6
    assert cad.max_query_len == 2
    assert cad.max_seq_len == 2
    assert cad.block_table_tensor.shape[0] == 3


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


# Verifies multimodal draft inputs are rejected.
def test_init_rejects_multimodal_inputs(monkeypatch):
    def fake_super_init(self, vllm_config, device, runner):
        self.supports_mm_inputs = True

    monkeypatch.setattr(EagleProposer, "__init__", fake_super_init)

    with pytest.raises(NotImplementedError):
        RBLNEagleProposer(object(), DEVICE, object())


# Verifies the runner is stored when multimodal inputs are absent.
def test_init_stores_runner(monkeypatch):
    def fake_super_init(self, vllm_config, device, runner):
        self.supports_mm_inputs = False

    monkeypatch.setattr(EagleProposer, "__init__", fake_super_init)
    runner = object()

    proposer = RBLNEagleProposer(object(), DEVICE, runner)

    assert proposer.runner is runner


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------


def _install_model(monkeypatch, stub, model):
    monkeypatch.setattr(
        EagleProposer,
        "load_model",
        lambda self, target_model: setattr(self, "model", model),
    )


# Verifies either eager trigger installs the direct wrapper and skips compile.
@pytest.mark.parametrize(
    ("enforce_eager", "compile_model"),
    [
        (True, True),  # enforce_eager arm
        (False, False),  # VLLM_RBLN_COMPILE_MODEL arm
    ],
)
def test_load_model_uses_eager_wrapper(monkeypatch, enforce_eager, compile_model):
    stub = make_eagle_stub(enforce_eager=enforce_eager)
    _install_model(monkeypatch, stub, FakeEagleModel())
    compile_mock = Mock(side_effect=AssertionError("compile must not run"))
    monkeypatch.setattr(eagle_module, "compile", compile_mock)
    monkeypatch.setattr(eagle_module.envs, "VLLM_RBLN_COMPILE_MODEL", compile_model)

    RBLNEagleProposer.load_model(stub, target_model=object())

    compile_mock.assert_not_called()
    assert callable(stub.model_executable)


# Verifies the compile path forwards the proposer's contract args (notably the
# compile context sourced from the runner) and tracks the strict-mode flag.
@pytest.mark.parametrize("strict", [True, False])
def test_load_model_compiles_wrapper(monkeypatch, strict):
    runner = SimpleNamespace(compile_context=object())
    stub = make_eagle_stub(enforce_eager=False, runner=runner)
    _install_model(monkeypatch, stub, FakeEagleModel())
    compiled_sentinel = object()
    process_group_sentinel = object()
    captured: dict[str, object] = {}

    def fake_compile(target, **kwargs):
        captured["target"] = target
        captured.update(kwargs)
        return compiled_sentinel

    monkeypatch.setattr(eagle_module, "compile", fake_compile)
    monkeypatch.setattr(
        eagle_module, "build_process_group_dict", lambda: process_group_sentinel
    )
    monkeypatch.setattr(eagle_module.envs, "VLLM_RBLN_COMPILE_MODEL", True)
    monkeypatch.setattr(eagle_module.envs, "VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK", 8)
    monkeypatch.setattr(eagle_module.envs, "VLLM_RBLN_COMPILE_STRICT_MODE", strict)

    RBLNEagleProposer.load_model(stub, target_model=object())

    assert stub.model_executable is compiled_sentinel
    assert captured["dynamic"] is False
    assert captured["fullgraph"] is True
    assert captured["compile_context"] is runner.compile_context
    assert captured["tensor_parallel_size"] == 8
    assert captured["process_group_dict"] is process_group_sentinel
    assert captured["guard_filter_fn"] is torch.compiler.keep_tensor_guards_unsafe
    assert captured["mode"] == ("strict" if strict else "")


# Verifies the wrapper gathers the last-token hidden states when indices are
# given, and feeds the full last-hidden stream when they are not.
@pytest.mark.parametrize("with_indices", [True, False])
def test_load_model_wrapper_composition(monkeypatch, with_indices):
    stub = make_eagle_stub(hidden_size=4, enforce_eager=True)
    model = FakeEagleModel()
    _install_model(monkeypatch, stub, model)
    monkeypatch.setattr(eagle_module.envs, "VLLM_RBLN_COMPILE_MODEL", True)

    RBLNEagleProposer.load_model(stub, target_model=object())

    hidden = torch.arange(24, dtype=torch.float32).view(2, 3, 4)
    indices = torch.tensor([0, 5]) if with_indices else None

    out_hidden, logits = stub.model_executable(
        input_ids=torch.zeros((2, 3), dtype=torch.int32),
        positions=torch.zeros((2, 3), dtype=torch.int64),
        hidden_states=hidden,
        last_token_indices=indices,
    )

    # Returned hidden stream is the second forward output, flattened to [-1, H].
    torch.testing.assert_close(out_hidden, (hidden + 2000).view(-1, 4))

    last_hidden_flat = (hidden + 1000).view(-1, 4)
    expected_sample = last_hidden_flat[indices] if with_indices else last_hidden_flat
    torch.testing.assert_close(model.compute_logits_inputs[0], expected_sample)
    torch.testing.assert_close(logits, expected_sample + 1)


# ---------------------------------------------------------------------------
# delegation to spec-decode utils
# ---------------------------------------------------------------------------


# Verifies backup next-tokens are sourced from request state at the current
# seq_len and the sliced tensors are handed to eagle_prepare_next_token_padded.
def test_prepare_next_token_ids_padded_builds_backup_and_delegates(monkeypatch):
    stub = make_eagle_stub()
    stub.backup_next_token_ids = FakeBackupNextTokenIds(size=8)
    sentinel = (object(), object())
    captured: dict[str, object] = {}

    def fake_util(sampled, discard, backup, vocab):
        captured.update(sampled=sampled, discard=discard, backup=backup, vocab=vocab)
        return sentinel

    monkeypatch.setattr(eagle_module, "eagle_prepare_next_token_padded", fake_util)
    cad = make_common_attn_metadata(
        query_start_loc=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        seq_lens=torch.tensor([5, 6, 7], dtype=torch.int32),
    )
    requests = {f"req-{i}": FakeRequestState(100) for i in range(3)}
    gpu_input_batch = SimpleNamespace(
        num_reqs=3, req_ids=["req-0", "req-1", "req-2"], vocab_size=50
    )
    sampled = torch.tensor([[5, -1], [6, 7], [-1, -1]], dtype=torch.int32)
    discard = torch.tensor([False, True, False])

    out = RBLNEagleProposer.prepare_next_token_ids_padded(
        stub,
        common_attn_metadata=cad,
        sampled_token_ids=sampled,
        requests=requests,
        gpu_input_batch=gpu_input_batch,
        discard_request_mask=discard,
    )

    assert out is sentinel
    # backup token = token_base(100) + seq_len -> [105, 106, 107]
    torch.testing.assert_close(
        captured["backup"], torch.tensor([105, 106, 107], dtype=torch.int32)
    )
    torch.testing.assert_close(captured["sampled"], sampled)
    torch.testing.assert_close(captured["discard"], discard)
    assert captured["vocab"] == 50


# Verifies the spec-decode common metadata is rebuilt with derived lengths and
# delegates index computation to eagle_prepare_inputs_padded.
def test_prepare_inputs_padded_builds_spec_metadata_and_delegates(monkeypatch):
    stub = make_eagle_stub()
    tis_sentinel, rej_sentinel = object(), object()
    captured: dict[str, object] = {}

    def fake_util(cu_num_draft_tokens, valid_count, query_start_loc):
        captured.update(cu=cu_num_draft_tokens, valid=valid_count, qsl=query_start_loc)
        return tis_sentinel, rej_sentinel

    monkeypatch.setattr(eagle_module, "eagle_prepare_inputs_padded", fake_util)
    cad = make_common_attn_metadata(
        query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
        seq_lens=torch.tensor([8, 9], dtype=torch.int32),
    )
    spec_md = SimpleNamespace(
        cu_num_draft_tokens=torch.tensor([1, 2], dtype=torch.int32)
    )
    valid_count = torch.tensor([1, 1], dtype=torch.int32)

    spec_cad, tis, rej = RBLNEagleProposer.prepare_inputs_padded(
        stub,
        common_attn_metadata=cad,
        spec_decode_metadata=spec_md,
        valid_sampled_tokens_count=valid_count,
    )

    assert tis is tis_sentinel
    assert rej is rej_sentinel
    assert spec_cad.num_actual_tokens == 5  # query_start_loc_cpu[-1]
    assert spec_cad.max_query_len == 3  # max of per-req query lens [2, 3]
    assert spec_cad.max_seq_len == 9
    assert spec_cad.causal is True
    torch.testing.assert_close(captured["cu"], spec_md.cu_num_draft_tokens)
    torch.testing.assert_close(captured["valid"], valid_count)


# ---------------------------------------------------------------------------
# propose
# ---------------------------------------------------------------------------


def _prime_for_first_pass(stub):
    """Wire the buffers/hooks propose needs to run its first draft pass."""
    stub.needs_extra_input_slots = False
    stub._set_positions = lambda n, p: stub.positions[:n].copy_(p)
    builder = FakeMetadataBuilder()
    stub.draft_attn_groups = [FakeAttentionGroup(["draft.layer"], builder)]
    return builder


# Verifies eagle3 rejects a combined hidden-state with the wrong final dim,
# before any draft pass runs.
def test_propose_eagle3_rejects_wrong_combined_hidden_size():
    stub = make_eagle_stub(method="eagle3", hidden_size=4)
    stub.model = FakeEagle3Model(torch.zeros((8, 5)))  # last dim 5 != hidden_size
    cad = make_common_attn_metadata(
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        seq_lens=torch.tensor([10, 11], dtype=torch.int32),
    )

    with pytest.raises(AssertionError):
        RBLNEagleProposer.propose(
            stub,
            target_token_ids=torch.tensor([10, 11, 20, 21], dtype=torch.int32),
            target_positions=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
            target_hidden_states=torch.zeros((8, 8), dtype=torch.float32),
            next_token_ids=torch.tensor([30, 31], dtype=torch.int32),
            token_indices_to_sample=torch.tensor([1, 3], dtype=torch.int32),
            common_attn_metadata=cad,
        )


# Verifies an intermediate chunked-prefill pass updates the draft KV cache but
# returns zero draft tokens shaped [num_reqs, num_speculative_tokens].
def test_propose_returns_zeros_for_intermediate_chunked_prefill():
    stub = make_eagle_stub(num_speculative_tokens=2)
    builder = _prime_for_first_pass(stub)
    stub.runner = make_fake_runner(
        is_prefill=True, is_intermediate_chunked_prefill=True
    )

    def model_executable(
        *, input_ids, positions, hidden_states, inputs_embeds, last_token_indices
    ):
        return hidden_states.view(-1, stub.hidden_size), torch.zeros((2, 3))

    stub.model_executable = model_executable
    cad = make_common_attn_metadata(
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        seq_lens=torch.tensor([10, 11], dtype=torch.int32),
    )

    output = RBLNEagleProposer.propose(
        stub,
        target_token_ids=torch.tensor([10, 11, 20, 21], dtype=torch.int32),
        target_positions=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        target_hidden_states=torch.arange(16, dtype=torch.float32).view(4, 4),
        next_token_ids=torch.tensor([30, 31], dtype=torch.int32),
        token_indices_to_sample=torch.tensor([1, 3], dtype=torch.int32),
        common_attn_metadata=cad,
    )

    assert len(builder.calls) == 1
    assert output.shape == (2, 2)
    torch.testing.assert_close(output, torch.zeros((2, 2), dtype=torch.int64))


# Verifies the single-step decode path pads inputs to the decode bucket and
# returns the per-request argmax as [num_reqs, 1].
def test_propose_single_step_decode_returns_argmax():
    stub = make_eagle_stub(num_speculative_tokens=1, hidden_size=4)
    builder = _prime_for_first_pass(stub)
    stub.runner = make_fake_runner(batch_bucket=4, num_reqs=2, is_prefill=False)
    captured: dict[str, object] = {}

    def model_executable(
        *, input_ids, positions, hidden_states, inputs_embeds, last_token_indices
    ):
        captured["input_shape"] = input_ids.shape
        captured["last_token_indices"] = last_token_indices
        logits = torch.tensor(
            [[0.0, 5.0, 1.0], [0.0, 1.0, 7.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        return hidden_states.view(-1, stub.hidden_size), logits

    stub.model_executable = model_executable
    cad = make_common_attn_metadata(
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        seq_lens=torch.tensor([10, 11], dtype=torch.int32),
    )

    output = RBLNEagleProposer.propose(
        stub,
        target_token_ids=torch.tensor([10, 11, 20, 21], dtype=torch.int32),
        target_positions=torch.tensor([4, 5, 6, 7], dtype=torch.int64),
        target_hidden_states=torch.arange(16, dtype=torch.float32).view(4, 4),
        next_token_ids=torch.tensor([30, 31], dtype=torch.int32),
        token_indices_to_sample=torch.tensor([1, 3], dtype=torch.int32),
        common_attn_metadata=cad,
    )

    # decode: [2, 2] reshaped then padded up to the bucket batch of 4
    assert captured["input_shape"] == (4, 2)
    torch.testing.assert_close(
        captured["last_token_indices"], torch.tensor([1, 3, 0, 0], dtype=torch.int32)
    )
    assert builder.calls[0]["is_prefill"] is False
    assert builder.calls[0]["batch_pad"] == 4
    # logits[:2].argmax(-1) = [1, 2] -> [[1], [2]]
    assert output.shape == (2, 1)
    torch.testing.assert_close(output, torch.tensor([[1], [2]], dtype=torch.int64))


# Verifies the multi-step decode loop feeds each step's sampled draft tokens
# into the next step and stacks stepwise argmax results in draft-token order.
def test_propose_multistep_decode_feeds_previous_draft_and_stacks_tokens():
    stub = make_eagle_stub(num_speculative_tokens=2, hidden_size=4)
    builder = _prime_for_first_pass(stub)
    stub.runner = make_fake_runner(batch_bucket=4, num_reqs=2, is_prefill=False)
    calls: list[dict[str, object]] = []
    logits_by_call = [
        torch.tensor(
            [
                [0.0, 9.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 8.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                [0.0, 1.0, 7.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 6.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    ]

    def model_executable(
        *,
        input_ids,
        positions,
        hidden_states,
        inputs_embeds,
        last_token_indices,
    ):
        calls.append(
            {
                "input_ids": input_ids.clone(),
                "last_token_indices": (
                    None if last_token_indices is None else last_token_indices.clone()
                ),
            }
        )
        return (
            hidden_states.reshape(-1, stub.hidden_size),
            logits_by_call[len(calls) - 1],
        )

    stub.model_executable = model_executable
    cad = make_common_attn_metadata(
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        seq_lens=torch.tensor([10, 11], dtype=torch.int32),
    )

    output = RBLNEagleProposer.propose(
        stub,
        target_token_ids=torch.tensor([10, 11, 20, 21], dtype=torch.int32),
        target_positions=torch.tensor([4, 5, 6, 7], dtype=torch.int64),
        target_hidden_states=torch.arange(16, dtype=torch.float32).view(4, 4),
        next_token_ids=torch.tensor([30, 31], dtype=torch.int32),
        token_indices_to_sample=torch.tensor([1, 3], dtype=torch.int32),
        common_attn_metadata=cad,
    )

    assert len(calls) == 2
    assert len(builder.calls) == 2
    assert builder.calls[0]["is_prefill"] is False
    assert builder.calls[1]["is_prefill"] is False
    assert builder.calls[0]["batch_pad"] == 4
    assert builder.calls[1]["batch_pad"] == 4
    torch.testing.assert_close(
        calls[0]["last_token_indices"], torch.tensor([1, 3, 0, 0], dtype=torch.int32)
    )
    assert calls[1]["last_token_indices"] is None
    second_input_ids = cast(torch.Tensor, calls[1]["input_ids"])
    torch.testing.assert_close(
        second_input_ids[:2, 0], torch.tensor([1, 3], dtype=torch.int32)
    )
    assert output.shape == (2, 2)
    torch.testing.assert_close(
        output,
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
    )


# Verifies multi-token EAGLE rejects unsupported attention metadata types before
# entering the autoregressive draft loop.
def test_propose_multistep_rejects_unsupported_attention_metadata_type():
    stub = make_eagle_stub(num_speculative_tokens=2, hidden_size=4)
    _prime_for_first_pass(stub)
    stub.runner = make_fake_runner(batch_bucket=4, num_reqs=2, is_prefill=False)
    stub.allowed_attn_types = str

    def model_executable(
        *,
        input_ids,
        positions,
        hidden_states,
        inputs_embeds,
        last_token_indices,
    ):
        logits = torch.tensor(
            [
                [0.0, 5.0, 1.0],
                [0.0, 1.0, 7.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        return hidden_states.reshape(-1, stub.hidden_size), logits

    stub.model_executable = model_executable
    cad = make_common_attn_metadata(
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        seq_lens=torch.tensor([10, 11], dtype=torch.int32),
    )

    with pytest.raises(ValueError, match="Unsupported attention metadata type"):
        RBLNEagleProposer.propose(
            stub,
            target_token_ids=torch.tensor([10, 11, 20, 21], dtype=torch.int32),
            target_positions=torch.tensor([4, 5, 6, 7], dtype=torch.int64),
            target_hidden_states=torch.arange(16, dtype=torch.float32).view(4, 4),
            next_token_ids=torch.tensor([30, 31], dtype=torch.int32),
            token_indices_to_sample=torch.tensor([1, 3], dtype=torch.int32),
            common_attn_metadata=cad,
        )


# ---------------------------------------------------------------------------
# dummy_run
# ---------------------------------------------------------------------------


# Verifies single-step warmup runs exactly one draft forward pass.
def test_dummy_run_single_step_runs_one_forward():
    stub = make_eagle_stub(num_speculative_tokens=1)
    stub.runner = make_fake_runner()
    builder = FakeMetadataBuilder()
    stub.draft_attn_groups = [FakeAttentionGroup(["draft.layer"], builder)]
    calls: list[dict[str, object]] = []

    def model_executable(**kwargs):
        calls.append(kwargs)
        return torch.zeros(1), torch.zeros(1)

    stub.model_executable = model_executable

    RBLNEagleProposer.dummy_run(stub, num_reqs=2, num_tokens_per_req=1, is_prefill=True)

    assert len(calls) == 1
    assert len(builder.calls) == 1
    assert builder.calls[0]["is_prefill"] is True


# Verifies multi-step warmup runs a second decode pass with the bucket pad.
def test_dummy_run_multistep_runs_second_decode_pass():
    stub = make_eagle_stub(num_speculative_tokens=2)
    stub.runner = make_fake_runner(batch_bucket=4)
    builder = FakeMetadataBuilder()
    stub.draft_attn_groups = [FakeAttentionGroup(["draft.layer"], builder)]
    calls: list[dict[str, object]] = []

    def model_executable(**kwargs):
        calls.append(kwargs)
        return torch.zeros(1), torch.zeros(1)

    stub.model_executable = model_executable

    RBLNEagleProposer.dummy_run(
        stub, num_reqs=2, num_tokens_per_req=1, is_prefill=False
    )

    assert len(calls) == 2
    assert len(builder.calls) == 2
    # The second build is the decode step rebuilt at the decode bucket pad.
    assert builder.calls[1]["is_prefill"] is False
    assert builder.calls[1]["batch_pad"] == 4
