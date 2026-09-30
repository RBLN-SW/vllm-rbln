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

r"""Call the OpenAI-compatible Chat Completion API, optionally with streaming.

Start the server in one terminal:
    vllm serve meta-llama/Llama-3.2-1B-Instruct --model-impl vllm \
        --block-size 1024 --max-num-batched-tokens 512 --max-num-seqs 1
Run the client in another terminal:
    python examples/vllm/basic/openai_chat_completion_client.py
    python examples/vllm/basic/openai_chat_completion_client.py --stream

The client connects to localhost:8000; the model must have a chat template.
The client uses the first model returned by the server's models endpoint.
"""

import argparse

from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stream", action="store_true", help="Enable streaming response"
    )
    args = parser.parse_args()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {
            "role": "assistant",
            "content": "The Los Angeles Dodgers won the World Series in 2020.",
        },
        {"role": "user", "content": "Where was it played?"},
    ]

    with OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1") as client:
        models = client.models.list()
        model = models.data[0].id
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            stream=args.stream,
        )

        print("-" * 50)
        print("Chat completion results:")
        if args.stream:
            for chunk in chat_completion:
                print(chunk)
        else:
            print(chat_completion)
        print("-" * 50)


if __name__ == "__main__":
    main()
