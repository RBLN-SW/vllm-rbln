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

import asyncio
import logging
import multiprocessing as mp

import fire
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.config import ECTransferConfig


def _setup_logging(role: str = "main") -> logging.Logger:
    """Return a named logger with its own StreamHandler.

    Using a named logger (not root) with an explicit handler ensures logs are
    visible even after vllm has already configured the root logger (which would
    make a subsequent basicConfig call a no-op).
    """
    logger = logging.getLogger(f"ec_disagg.{role}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            f"[%(asctime)s] [{role}] %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # don't double-print via root logger
    return logger

# If the video is too long
# set `VLLM_ENGINE_ITERATION_TIMEOUT_S` to a higher timeout value.
VIDEO_URLS = [
    "https://duguang-labelling.oss-cn-shanghai.aliyuncs.com/qiansun/video_ocr/videos/50221078283.mp4",
    "https://cdn.pixabay.com/video/2022/04/18/114413-701051082_large.mp4",
    "https://videos.pexels.com/video-files/855282/855282-hd_1280_720_25fps.mp4",
]


def generate_prompts_video(batch_size: int, model_id: str):
    processor = AutoProcessor.from_pretrained(model_id, padding_side="left")
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": VIDEO_URLS[i],
                    },
                    {"type": "text", "text": "Describe this video."},
                ],
            },
        ]
        for i in range(batch_size)
    ]

    texts = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    arr_image_inputs = []
    arr_video_inputs = []
    arr_video_kwargs = []
    for i in range(batch_size):
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages[i],
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        arr_image_inputs.append(image_inputs)
        arr_video_inputs.append(video_inputs)
        arr_video_kwargs.append(video_kwargs)

    return [
        {
            "prompt": text,
            "multi_modal_data": {
                "video": video_inputs,
            },
            "mm_processor_kwargs": {
                "min_pixels": 1024 * 14 * 14,
                "max_pixels": 5120 * 14 * 14,
                **video_kwargs,
            },
        }
        for text, image_inputs, video_inputs, video_kwargs in zip(
            texts, arr_image_inputs, arr_video_inputs, arr_video_kwargs
        )
    ]


def generate_prompts_image(batch_size: int, model_id: str):
    dataset = load_dataset("lmms-lab/llava-bench-in-the-wild", split="train").shuffle(
        seed=42
    )
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
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    arr_image_inputs = []

    for i in range(batch_size):
        image_inputs, _ = process_vision_info(
            messages[i],
        )
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


def generate_prompts_wo_processing(batch_size: int, model_id: str):
    dataset = load_dataset("lmms-lab/llava-bench-in-the-wild", split="train").shuffle(
        seed=42
    )
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
    images = [[dataset[i]["image"]] for i in range(batch_size)]

    texts = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    return [
        {
            "prompt": text,
            "multi_modal_data": {"image": image},
            "mm_processor_kwargs": {
                "min_pixels": 1024 * 14 * 14,
                "max_pixels": 5120 * 14 * 14,
            },
        }
        for text, image in zip(texts, images)
    ]


async def generate(engine: AsyncLLMEngine, tokenizer, request_id, request):
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
    return final_output


# ---------------------------------------------------------------------------
# Auto-disaggregation: spawn producer + consumer in a single invocation
# ---------------------------------------------------------------------------

def _engine_loop(
    model_id: str,
    max_model_len: int | None,
    ec_connector: str,
    ec_role: str,
    request_queue: "mp.Queue",
    result_queue: "mp.Queue",
    ec_nixl_host: str = "127.0.0.1",
    ec_nixl_port: int = 15100,
):
    """Subprocess entry point: runs an AsyncLLMEngine serving requests from a queue."""
    logger = _setup_logging(ec_role)
    logger.info("Process started (pid=%d)", mp.current_process().pid)
    logger.info("model_id=%s  max_model_len=%s  ec_connector=%s",
                model_id, max_model_len, ec_connector)
    asyncio.run(
        _engine_loop_async(
            model_id, max_model_len, ec_connector, ec_role,
            request_queue, result_queue, logger,
            ec_nixl_host=ec_nixl_host, ec_nixl_port=ec_nixl_port,
        )
    )


async def _engine_loop_async(
    model_id: str,
    max_model_len: int | None,
    ec_connector: str,
    ec_role: str,
    request_queue: "mp.Queue",
    result_queue: "mp.Queue",
    logger: logging.Logger,
    ec_nixl_host: str = "127.0.0.1",
    ec_nixl_port: int = 15100,
):
    ec_transfer_config = ECTransferConfig(
        ec_connector=ec_connector,
        ec_role=ec_role,
        ec_buffer_device="cpu",
        ec_connector_extra_config={
            "side_channel_host": ec_nixl_host,
            "side_channel_port": ec_nixl_port,
        },
    )
    engine_args = AsyncEngineArgs(
        model=model_id,
        max_model_len=max_model_len,
        ec_transfer_config=ec_transfer_config,
    )
    logger.info("Initializing AsyncLLMEngine ...")
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    logger.info("Engine ready. Waiting for requests ...")

    loop = asyncio.get_event_loop()
    while True:
        request_id, request = await loop.run_in_executor(None, request_queue.get)
        if request is None:
            logger.info("Received shutdown signal. Exiting.")
            break
        logger.info("Received request_id=%s", request_id)
        output = await generate(engine, tokenizer, request_id, request)
        logger.info("Completed request_id=%s", request_id)
        result_queue.put((request_id, output))


def _run_disaggregated(
    num_input_prompt: int,
    model_id: str,
    max_model_len: int | None,
    ec_connector: str,
    ec_nixl_host: str = "127.0.0.1",
    ec_nixl_port: int = 15100,
):
    """Spawn a producer and a consumer process, coordinate requests between them.

    Flow per request batch:
      1. Send all requests to producer  → vision encoder runs, registers with NIXL
      2. Wait for all producer results  → cache is guaranteed to be registered
      3. Send all requests to consumer  → pulls cache via NIXL, runs prefill_decoder + decoder
      4. Print consumer outputs
    """
    logger = _setup_logging("main")
    logger.info("Auto-disaggregation mode: spawning producer + consumer processes")
    logger.info("ec_connector=%s", ec_connector)

    ctx = mp.get_context("spawn")
    producer_req_q = ctx.Queue()
    producer_res_q = ctx.Queue()
    consumer_req_q = ctx.Queue()
    consumer_res_q = ctx.Queue()

    producer_proc = ctx.Process(
        target=_engine_loop,
        args=(model_id, max_model_len, ec_connector, "ec_producer",
              producer_req_q, producer_res_q),
        kwargs={"ec_nixl_host": ec_nixl_host, "ec_nixl_port": ec_nixl_port},
        daemon=False,
    )
    consumer_proc = ctx.Process(
        target=_engine_loop,
        args=(model_id, max_model_len, ec_connector, "ec_consumer",
              consumer_req_q, consumer_res_q),
        kwargs={"ec_nixl_host": ec_nixl_host, "ec_nixl_port": ec_nixl_port},
        daemon=False,
    )

    producer_proc.start()
    logger.info("Producer process spawned (pid=%d)", producer_proc.pid)
    consumer_proc.start()
    logger.info("Consumer process spawned (pid=%d)", consumer_proc.pid)

    inputs = generate_prompts_image(num_input_prompt, model_id)
    # inputs = generate_prompts_video(num_input_prompt, model_id)
    # inputs = generate_prompts_wo_processing(num_input_prompt, model_id)

    logger.info(
        "Prepared %d prompt(s). Processing sequentially (producer → consumer per request).",
        len(inputs),
    )

    # Process each request sequentially: producer encodes, then consumer decodes.
    # This ensures the encoder cache is registered via NIXL before the consumer pulls it.
    results = {}
    for request_id, request in enumerate(inputs):
        logger.info("[request %d] Sending to producer (vision encoding) ...", request_id)
        producer_req_q.put((request_id, request))
        req_id, _ = producer_res_q.get()
        logger.info("[request %d] Producer done → cache registered via NIXL", req_id)

        logger.info("[request %d] Sending to consumer (prefill + decode) ...", request_id)
        consumer_req_q.put((request_id, request))
        req_id, output = consumer_res_q.get()
        logger.info("[request %d] Consumer done.", req_id)
        results[req_id] = output

    logger.info("All requests complete. Shutting down subprocesses ...")

    # Shutdown subprocesses
    producer_req_q.put((None, None))
    consumer_req_q.put((None, None))
    producer_proc.join(timeout=30)
    consumer_proc.join(timeout=30)
    logger.info("All processes exited. Printing results.")

    for i in range(len(inputs)):
        output = results[i].outputs[0].text
        print(f"===================== Output {i} ==============================")
        print(output)
        print("===============================================================\n")


def entry_point(
    num_input_prompt: int = 1,
    # NOTE: This example supports Qwen2-VL, Qwen2.5-VL, and Qwen3-VL.
    model_id: str = "/qwen2_5-vl-7b-32k-b4-kv16k",
    # max_model_len must be a multiple of kvcache_partition_len when using flash_attn.
    max_model_len: int | None = None,
    # NIXL side-channel settings
    ec_nixl_host: str = "127.0.0.1",
    ec_nixl_port: int = 15100,
):

    _run_disaggregated(
        num_input_prompt=num_input_prompt,
        model_id=model_id,
        max_model_len=max_model_len,
        ec_connector="RblnECNixlConnector",
        ec_nixl_host=ec_nixl_host,
        ec_nixl_port=ec_nixl_port,
    )

if __name__ == "__main__":
    fire.Fire(entry_point)
