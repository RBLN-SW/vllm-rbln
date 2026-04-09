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


class TransferStats:
    """Lifetime accumulator for D2H / H2D transfer timing and byte counts."""

    __slots__ = (
        "label",
        "calls",
        "total_bytes",
        "t_total",
        "t_dma",
        "t_scatter_gather",
    )

    def __init__(self, label: str):
        self.label = label
        self.calls = 0
        self.total_bytes = 0
        self.t_total = 0.0
        self.t_dma = 0.0
        self.t_scatter_gather = 0.0

    def record(
        self,
        nbytes: int,
        t_total: float,
        t_dma: float,
        t_scatter_gather: float,
    ) -> None:
        self.calls += 1
        self.total_bytes += nbytes
        self.t_total += t_total
        self.t_dma += t_dma
        self.t_scatter_gather += t_scatter_gather

    def get_avg_total_ms(self) -> float:
        return (self.t_total / self.calls * 1000) if self.calls > 0 else 0.0

    def get_avg_dma_ms(self) -> float:
        return (self.t_dma / self.calls * 1000) if self.calls > 0 else 0.0

    def get_avg_scatter_gather_ms(self) -> float:
        return (self.t_scatter_gather / self.calls * 1000) if self.calls > 0 else 0.0

    def get_throughput_gbps(self) -> float:
        return (self.total_bytes / self.t_total / 1e9) if self.t_total > 0 else 0.0

    def get_total_mb(self) -> float:
        return self.total_bytes / 1e6
