# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The inputs and score() calls follow upstream vLLM's Qwen3-VL-Reranker example:
# https://github.com/vllm-project/vllm/blob/main/examples/pooling/score/vision_reranker_offline.py

import fire
from huggingface_hub import hf_hub_download
from vllm import LLM
from vllm.multimodal.utils import encode_image_url, fetch_image

query = "A woman playing with her dog on a beach at sunset."
document = (
    "A woman shares a joyful moment with her golden retriever on a sun-drenched "
    "beach at sunset, as the dog offers its paw in a heartwarming display of "
    "companionship and trust."
)
image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
documents = [
    {
        "type": "text",
        "text": document,
    },
    {
        "type": "image_url",
        "image_url": {"url": image_url},
    },
    {
        "type": "image_url",
        "image_url": {"url": encode_image_url(fetch_image(image_url))},
    },
]


def main(
    max_model_len: int = 8192,
    vit_max_seq_len: int = 16384,
    model: str = "Qwen/Qwen3-VL-Reranker-2B",
):
    llm = LLM(
        model=model,
        runner="pooling",
        block_size=4096,
        max_model_len=max_model_len,
        # optimum-rbln requires the vision encoder's patch budget per image.
        additional_config={
            "optimum_overrides": {"visual": {"max_seq_len": vit_max_seq_len}}
        },
        # The same overrides as Qwen3-Reranker; see
        # examples/optimum/pooling_models/qwen3_reranker_score_offline.py.
        hf_overrides={
            "architectures": ["Qwen3VLForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        },
    )

    # The checkpoint ships its judge prompt as a separate template that renders
    # the `query` and `document` roles `score()` passes, images included.
    template_path = hf_hub_download(model, "additional_chat_templates/reranker.jinja")
    with open(template_path) as f:
        chat_template = f.read()

    print("Query: string & Document: string")
    outputs = llm.score(query, document, chat_template=chat_template)
    print("Relevance scores:", [output.outputs.score for output in outputs])

    print("Query: string & Document: text")
    outputs = llm.score(query, {"content": [documents[0]]}, chat_template=chat_template)
    print("Relevance scores:", [output.outputs.score for output in outputs])

    print("Query: string & Document: image url")
    outputs = llm.score(query, {"content": [documents[1]]}, chat_template=chat_template)
    print("Relevance scores:", [output.outputs.score for output in outputs])

    print("Query: string & Document: image base64")
    outputs = llm.score(query, {"content": [documents[2]]}, chat_template=chat_template)
    print("Relevance scores:", [output.outputs.score for output in outputs])

    print("Query: string & Document: text + image url")
    outputs = llm.score(
        query,
        {"content": [documents[0], documents[1]]},
        chat_template=chat_template,
    )
    print("Relevance scores:", [output.outputs.score for output in outputs])

    print("Query: string & Document: list")
    outputs = llm.score(
        query,
        [
            document,
            {"content": [documents[0]]},
            {"content": [documents[1]]},
            {"content": [documents[0], documents[1]]},
        ],
        chat_template=chat_template,
    )
    print("Relevance scores:", [output.outputs.score for output in outputs])


if __name__ == "__main__":
    fire.Fire(main)
