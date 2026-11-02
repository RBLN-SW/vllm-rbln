# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

from vllm_rbln.patches.axk2.config import AXK2Config


class AXK3Config(AXK2Config):
    """A.X K3: the AXK2 config plus Gated Residual and DSA index-share.

    The extra keys (``gated_residual*``, ``index_topk_freq``,
    ``index_skip_topk_offset``) reach the instance through the
    ``PretrainedConfig`` kwargs; only their defaults and the index-share rule
    live here. The rule is the checkpoint's own ``configuration_axk3.py``.
    """

    model_type = "axk3"

    # Assigned by the frozen ``AXK2Config.__init__``, which mypy does not check;
    # declared here so the single-group reset below type-checks.
    n_group: int | None
    topk_group: int | None

    def __init__(
        self,
        gated_residual: bool = False,
        gated_residual_num_streams: int = 4,
        gated_residual_rank: int = 16,
        gated_residual_first_layer: int = 1,
        gated_residual_final: str = "gated_read",
        index_topk_freq: int | None = None,
        index_skip_topk_offset: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gated_residual = gated_residual
        self.gated_residual_num_streams = gated_residual_num_streams
        self.gated_residual_rank = gated_residual_rank
        self.gated_residual_first_layer = gated_residual_first_layer
        self.gated_residual_final = gated_residual_final
        self.index_topk_freq = int(index_topk_freq or 1)
        self.index_skip_topk_offset = int(index_skip_topk_offset or 0)
        # One expert group is a plain top-k (modeling_axk3.py skips group
        # routing for n_group <= 1). Leaving it set would send the router down
        # the grouped path, whose single-group mask scatter RBLN cannot lower.
        if self.n_group is not None and self.n_group <= 1:
            self.n_group = None
            self.topk_group = None

    def is_index_skip_layer(self, layer_idx: int) -> bool:
        """Whether 0-indexed ``layer_idx`` reuses another layer's DSA top-k."""
        if not hasattr(self, "index_topk") or self.index_topk_freq <= 1:
            return False
        offset = max(self.index_skip_topk_offset, 1)
        return (max(layer_idx + 1 - offset, 0) % self.index_topk_freq) != 0

    def index_source_layer(self, layer_idx: int) -> int:
        """0-indexed layer whose top-k ``layer_idx`` uses (itself if it computes)."""
        if not self.is_index_skip_layer(layer_idx):
            return layer_idx
        offset = max(self.index_skip_topk_offset, 1)
        layer_number = layer_idx + 1
        return layer_number - ((layer_number - offset) % self.index_topk_freq) - 1


__all__ = ["AXK3Config"]
