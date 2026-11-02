# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Receive sub-block KV cache events from an offline RBLN engine over ZeroMQ.

Generation publishes BlockStored events with block_size=128. Resetting the
prefix cache and generating again publishes AllBlocksCleared and BlockStored.
The script runs the engine and subscriber together on a local ephemeral port.
See docs/sub_block_prefix_caching.md for sub-block caching device requirements.
"""

import argparse
from time import monotonic, sleep

import msgspec
import zmq
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.config import KVEventsConfig
from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
)
from vllm.utils.network_utils import get_open_port

from vllm_rbln.config import RBLNConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    args = parser.parse_args()

    sub_block_size = 128
    topic = "kv-events"
    decoder = msgspec.msgpack.Decoder(type=KVEventBatch)
    with zmq.Context() as context, context.socket(zmq.SUB) as subscriber:
        subscriber.setsockopt(zmq.LINGER, 0)
        subscriber.setsockopt_string(zmq.SUBSCRIBE, topic)
        port = get_open_port()
        subscriber.connect(f"tcp://127.0.0.1:{port}")
        llm = LLM(
            model=args.model,
            model_impl="vllm",
            block_size=1024,
            max_num_batched_tokens=512,
            max_num_seqs=1,
            enable_prefix_caching=True,
            additional_config=RBLNConfig(
                enable_sub_block_cache=True, sub_block_size=sub_block_size
            ),
            kv_events_config=KVEventsConfig(
                enable_kv_cache_events=True,
                publisher="zmq",
                endpoint=f"tcp://*:{port}",
                topic=topic,
            ),
        )
        token_ids = llm.get_tokenizer().encode(
            "A robot may not injure a human being " * 50
        )
        assert sub_block_size < len(token_ids) < 1024, (
            "The prompt must span sub-blocks while fitting within one KV block."
        )
        prompt = TokensPrompt(prompt_token_ids=token_ids)
        sampling_params = SamplingParams(temperature=0.0, max_tokens=1)

        # PUB/SUB subscriptions propagate asynchronously before the first event.
        sleep(1)
        for phase in ("generation", "after_reset"):
            if phase == "after_reset" and not llm.reset_prefix_cache():
                raise RuntimeError("Failed to reset the prefix cache")
            # Reset events are published during the next scheduler output update.
            llm.generate([prompt], sampling_params)
            print(f"\nKV cache events ({phase}):")
            saw_stored = False
            saw_cleared = phase == "generation"
            deadline = monotonic() + 5
            while not (saw_stored and saw_cleared):
                timeout_ms = max(0, int((deadline - monotonic()) * 1000))
                if not subscriber.poll(timeout_ms):
                    raise TimeoutError(
                        f"Expected KV cache events not received: {phase}"
                    )
                _, _, payload = subscriber.recv_multipart()
                batch = decoder.decode(payload)
                for event in batch.events:
                    if isinstance(event, BlockStored):
                        assert event.block_size == sub_block_size, (
                            "Stored events must use sub-block granularity."
                        )
                        if saw_cleared:
                            saw_stored = True
                        print(
                            f"  BlockStored: block_size={event.block_size}, "
                            f"blocks={len(event.block_hashes)}, "
                            f"group_idx={event.group_idx}"
                        )
                    elif isinstance(event, BlockRemoved):
                        print(f"  BlockRemoved: blocks={len(event.block_hashes)}")
                    elif isinstance(event, AllBlocksCleared):
                        saw_cleared = True
                        saw_stored = False
                        print("  AllBlocksCleared")
        print("KV cache events: PASS; sub-block storage and cache reset observed.")


if __name__ == "__main__":
    main()
