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

import fire
from huggingface_hub import hf_hub_download
from vllm import LLM

IMAGE_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
QUERY = "A woman playing with her dog on a beach at sunset."
DOCUMENTS = [
    {"content": [{"type": "image_url", "image_url": {"url": IMAGE_URL}}]},
    {"content": [{"type": "text", "text": "A woman walks her dog along the shore."}]},
    # Deliberately irrelevant, to show the scores separate.
    {"content": [{"type": "text", "text": "City skyline at night."}]},
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

    outputs = llm.score(QUERY, DOCUMENTS, chat_template=chat_template)
    print(f"scores: {[output.outputs.score for output in outputs]}")


if __name__ == "__main__":
    fire.Fire(main)
