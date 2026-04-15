# SPDX-License-Identifier: Apache-2.0
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

"""
EC disaggregated inference example (programmatic, single-script).

Spawns N producers (vision encoder, 1 device each) and 1 consumer
(decoder, multi-device) as separate AsyncLLMEngine instances in child
processes. The consumer automatically pulls encoder caches from any
producer via NIXL.

For a serving-based setup, see:
  - serve_ec_producer.sh / serve_ec_consumer.sh  (vllm serve)
  - client_ec_disaggregated.py                   (OpenAI-compatible client)

Usage:
  # 8 producers (device 0-7) + 1 consumer (device 8-15):
  python run_qwen_vl_disaggregated.py --num_producers 8 --num_input_prompt 8

  # Legacy single-producer mode:
  python run_qwen_vl_disaggregated.py --num_producers 1
"""

import asyncio
import multiprocessing as mp
import os
import time

import fire
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.config import ECTransferConfig


def generate_prompts_image(batch_size: int, model_id: str):
    dataset = load_dataset(
        "lmms-lab/llava-bench-in-the-wild", split="train"
    ).shuffle(seed=42)
    processor = AutoProcessor.from_pretrained(model_id, padding_side="left")
    messages = [
        [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant."
                        "Answer the each question based on the image.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": dataset[i]["image"]},
                    {"type": "text", "text": dataset[i]["question"]},
                ],
            },
        ]
        for i in range(batch_size)
    ]

    texts = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    arr_image_inputs = []
    for i in range(batch_size):
        image_inputs, _ = process_vision_info(messages[i])
        arr_image_inputs.append(image_inputs)

    return [
        {
            "prompt": text,
            "multi_modal_data": {"image": image_inputs},
            "mm_processor_kwargs": {
                "min_pixels": 1024 * 14 * 14,
                "max_pixels": 5120 * 14 * 14,
                "padding": True,
            },
        }
        for text, image_inputs in zip(texts, arr_image_inputs)
    ]


async def _generate(engine, tokenizer, request_id, request):
    t0 = time.perf_counter()
    results_generator = engine.generate(
        request,
        SamplingParams(
            temperature=0,
            ignore_eos=False,
            skip_special_tokens=True,
            stop_token_ids=[tokenizer.eos_token_id],
            max_tokens=200,
        ),
        str(request_id),
    )
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    final_output._elapsed = time.perf_counter() - t0
    return final_output


def _make_producer_ec_config(nixl_host: str, nixl_port: int):
    """EC config for a single producer (listens on its own side-channel port)."""
    return ECTransferConfig(
        ec_connector="RblnECNixlConnector",
        ec_role="ec_producer",
        ec_buffer_device="cpu",
        ec_connector_extra_config={
            "side_channel_host": nixl_host,
            "side_channel_port": nixl_port,
        },
    )


def _make_consumer_ec_config(
    nixl_host: str, nixl_base_port: int, num_producers: int
):
    """EC config for the consumer (connects to all producer side-channels)."""
    producer_endpoints = [
        {"host": nixl_host, "port": nixl_base_port + i}
        for i in range(num_producers)
    ]
    return ECTransferConfig(
        ec_connector="RblnECNixlConnector",
        ec_role="ec_consumer",
        ec_buffer_device="cpu",
        ec_connector_extra_config={
            "producer_endpoints": producer_endpoints,
        },
    )


# ---------------------------------------------------------------------------
# Engine subprocess
# ---------------------------------------------------------------------------

def _engine_proc(
    model_id: str,
    max_model_len: int | None,
    ec_transfer_config: ECTransferConfig,
    req_queue: "mp.Queue",
    res_queue: "mp.Queue",
    rbln_devices: str | None = None,
):
    if rbln_devices is not None:
        os.environ["RBLN_DEVICES"] = rbln_devices
    asyncio.run(
        _engine_proc_async(
            model_id, max_model_len, ec_transfer_config,
            req_queue, res_queue,
        )
    )


async def _engine_proc_async(
    model_id, max_model_len, ec_transfer_config,
    req_queue, res_queue,
):
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=model_id,
        max_model_len=max_model_len,
        ec_transfer_config=ec_transfer_config,
    ))
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    loop = asyncio.get_event_loop()
    while True:
        request_id, request = await loop.run_in_executor(None, req_queue.get)
        if request is None:
            break
        output = await _generate(engine, tokenizer, request_id, request)
        elapsed = getattr(output, "_elapsed", None)
        num_tokens = len(output.outputs[0].token_ids)
        res_queue.put((request_id, output.outputs[0].text, elapsed, num_tokens))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    num_input_prompt: int = 1,
    model_id: str = "Qwen2-VL-7B-Instruct",
    max_model_len: int | None = None,
    ec_nixl_host: str = "127.0.0.1",
    ec_nixl_base_port: int = 15100,
    num_producers: int = 8,
    producer_devices: str = "0,1,2,3,4,5,6,7",
    consumer_devices: str = "8,9,10,11,12,13,14,15",
):
    ctx = mp.get_context("spawn")

    # Each producer gets 1 device and its own side-channel port.
    producer_device_list = producer_devices.split(",")
    if len(producer_device_list) < num_producers:
        raise ValueError(
            f"Need at least {num_producers} producer devices, "
            f"got {len(producer_device_list)}"
        )

    producers = []
    producer_queues = []  # (req_queue, res_queue) per producer
    for p_idx in range(num_producers):
        req_q, res_q = ctx.Queue(), ctx.Queue()
        producer_queues.append((req_q, res_q))
        ec_cfg = _make_producer_ec_config(
            ec_nixl_host, ec_nixl_base_port + p_idx
        )
        proc = ctx.Process(
            target=_engine_proc,
            args=(model_id, max_model_len, ec_cfg, req_q, res_q,
                  producer_device_list[p_idx]),
        )
        producers.append(proc)

    consumer_req, consumer_res = ctx.Queue(), ctx.Queue()
    consumer_ec_cfg = _make_consumer_ec_config(
        ec_nixl_host, ec_nixl_base_port, num_producers
    )
    consumer = ctx.Process(
        target=_engine_proc,
        args=(model_id, max_model_len, consumer_ec_cfg,
              consumer_req, consumer_res, consumer_devices),
    )

    for p in producers:
        p.start()
    consumer.start()

    inputs = generate_prompts_image(num_input_prompt, model_id)

    # Round-robin requests across producers; send all to consumer.
    for i, req in enumerate(inputs):
        p_idx = i % num_producers
        producer_queues[p_idx][0].put((i, req))
        consumer_req.put((i, req))

    # Collect results from consumer.
    consumer_results = {}
    for _ in inputs:
        req_id, text, elapsed, num_tokens = consumer_res.get()
        consumer_results[req_id] = (text, elapsed, num_tokens)

    # Collect results from producers.
    producer_results = {}
    for i in range(len(inputs)):
        p_idx = i % num_producers
        req_id, text, elapsed, num_tokens = producer_queues[p_idx][1].get()
        producer_results[req_id] = (text, elapsed, num_tokens)

    # Shutdown.
    for p_idx in range(num_producers):
        producer_queues[p_idx][0].put((None, None))
    consumer_req.put((None, None))
    for p in producers:
        p.join(timeout=30)
    consumer.join(timeout=30)

    for i in range(len(inputs)):
        c_text, c_elapsed, c_tokens = consumer_results[i]
        p_text, p_elapsed, p_tokens = producer_results[i]
        print(f"==================== Output {i} ==============================")
        print(c_text)
        print(f"--- producer: {p_elapsed:.2f}s | {p_tokens} tokens")
        print(f"--- consumer: {c_elapsed:.2f}s | {c_tokens} tokens | "
              f"{c_tokens / c_elapsed:.1f} tok/s")
        print("===============================================================\n")


if __name__ == "__main__":
    fire.Fire(main)
