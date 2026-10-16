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

from contextlib import ExitStack
from functools import wraps

import torch
from torch import nn

from optimum.rbln.transformers.models.decoderonly.decoderonly_architecture import (
    DecoderOnlyLayer,
)
from optimum.rbln.transformers.models.gpt_oss.gpt_oss_architecture import (
    RBLNGptOssExperts,
    RBLNGptOssLayer,
)
from vllm_rbln import envs


def with_decoder_regions(compile_fn):
    @wraps(compile_fn)
    def compile(cls, model, *args, **kwargs):
        if not envs.VLLM_RBLN_NESTED_COMPILE_REGION:
            return compile_fn(cls, model, *args, **kwargs)

        region_classes = {}
        with ExitStack() as restore:
            for layer in list(model.modules()):
                if not isinstance(layer, DecoderOnlyLayer):
                    continue
                if isinstance(layer, RBLNGptOssLayer):
                    experts = layer.mlp.experts
                    layer.mlp.experts = RegionExperts(experts)
                    restore.callback(setattr, layer.mlp, "experts", experts)
                original_cls = type(layer)
                forward = layer.forward.__func__
                key = (original_cls, forward)
                if key not in region_classes:
                    # Instance-bound forwards change export's parameter paths.
                    # A class method preserves the original module hierarchy.
                    region_classes[key] = type(
                        original_cls.__name__,
                        (original_cls,),
                        {"forward": torch.compiler.nested_compile_region(forward)},
                    )
                if "forward" in layer.__dict__:
                    restore.callback(setattr, layer, "forward", layer.forward)
                    del layer.forward
                restore.callback(setattr, layer, "__class__", original_cls)
                layer.__class__ = region_classes[key]
            if not region_classes:
                raise ValueError(
                    "No Optimum DecoderOnlyLayer found for decoder regions."
                )
            kwargs["model_trace_method"] = "export_strict"
            return compile_fn(cls, model, *args, **kwargs)

    return compile


class RegionExperts(RBLNGptOssExperts):
    def __init__(self, experts):
        nn.Module.__init__(self)
        for name in (
            "intermediate_size",
            "num_experts",
            "hidden_size",
            "alpha",
            "limit",
            "top_k",
        ):
            setattr(self, name, getattr(experts, name))
        for suffix in ("blocks", "scales", "bias"):
            gate = getattr(experts, f"gate_proj_{suffix}")
            up = getattr(experts, f"up_proj_{suffix}")
            # Lift one storage per gate/up pair; create the aliased views inside
            # the decoder region, where Dynamo permits them.
            fused = gate.as_strided(
                (gate.shape[0], gate.shape[1] * 2, *gate.shape[2:]),
                (gate.stride(0), gate.stride(1) // 2, *gate.stride()[2:]),
            )
            assert fused[:, 1::2].data_ptr() == up.data_ptr()
            assert fused[:, 1::2].stride() == up.stride()
            self.register_buffer(f"gate_up_proj_{suffix}", fused)
            self.register_buffer(
                f"down_proj_{suffix}", getattr(experts, f"down_proj_{suffix}")
            )

    @property
    def gate_proj_blocks(self):
        return self.gate_up_proj_blocks[:, ::2]

    @property
    def up_proj_blocks(self):
        return self.gate_up_proj_blocks[:, 1::2]

    @property
    def gate_proj_scales(self):
        return self.gate_up_proj_scales[:, ::2]

    @property
    def up_proj_scales(self):
        return self.gate_up_proj_scales[:, 1::2]

    @property
    def gate_proj_bias(self):
        return self.gate_up_proj_bias[:, ::2]

    @property
    def up_proj_bias(self):
        return self.gate_up_proj_bias[:, 1::2]
