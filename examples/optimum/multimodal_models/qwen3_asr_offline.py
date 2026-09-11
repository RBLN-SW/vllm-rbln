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

import fire
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor.models.whisper import ISO639_1_SUPPORTED_LANGS

# Qwen3-ASR prompt format, matching upstream vLLM's
# Qwen3ASRForConditionalGeneration.get_generation_prompt:
#
#   user: <|audio_start|><|audio_pad|><|audio_end|>
#   assistant: [language {Lang}<asr_text>]   # only when language is forced
#
# The model outputs "language {Lang}<asr_text>{transcription}".
AUDIO_PLACEHOLDER = "<|audio_start|><|audio_pad|><|audio_end|>"
ASR_TEXT_TAG = "<asr_text>"


def build_prompt(language: str | None) -> str:
    prompt = f"<|im_start|>user\n{AUDIO_PLACEHOLDER}<|im_end|>\n<|im_start|>assistant\n"
    if language is not None:
        # Qwen3-ASR expects the full language name (e.g. "English"), not the ISO code.
        full_lang_name = ISO639_1_SUPPORTED_LANGS.get(language, language)
        prompt += f"language {full_lang_name}{ASR_TEXT_TAG}"
    return prompt


def post_process(text: str) -> str:
    if ASR_TEXT_TAG not in text:
        return text.strip()
    return text.split(ASR_TEXT_TAG, 1)[1].strip()


def generate_prompts(
    batch_size: int,
    model: str,
    language: str | None,
):
    from datasets import load_dataset

    dataset = load_dataset(
        "distil-whisper/librispeech_asr-noise",
        "test-pub-noise",
        streaming=True,
        split="40",
    )
    dataset = dataset.take(batch_size)

    tokenizer = AutoTokenizer.from_pretrained(model)
    prompt_token_ids = tokenizer.encode(build_prompt(language))

    messages = []
    for item in dataset:
        messages.append(
            {
                "prompt_token_ids": prompt_token_ids,
                "multi_modal_data": {
                    "audio": (item["audio"]["array"], item["audio"]["sampling_rate"])
                },
            }
        )

    return messages


def main(
    num_input_prompt: int = 12,
    model: str = "Qwen/Qwen3-ASR-0.6B-hf",
    max_num_seqs: int = 4,
    language: str | None = "en",
    max_tokens: int = 448,
):
    """Offline transcription with Qwen3-ASR.

    Args:
        language: ISO 639-1 code to force the output language (e.g. "en", "ko").
            Pass None to let the model detect the language.
    """
    inputs = generate_prompts(num_input_prompt, model, language)

    llm = LLM(
        model=model,
        limit_mm_per_prompt={"audio": 1},
        max_num_seqs=max_num_seqs,
        block_size=4096,
        max_model_len=8192,
    )

    sampling_params = SamplingParams(
        temperature=0,
        ignore_eos=False,
        skip_special_tokens=True,
        max_tokens=max_tokens,
    )

    results = llm.generate(inputs, sampling_params)

    for i, result in enumerate(results):
        raw = result.outputs[0].text
        print(f"===================== Output {i} ==============================")
        print(f"[raw]  {raw}")
        print(f"[text] {post_process(raw)}")
        print("===============================================================\n")


if __name__ == "__main__":
    fire.Fire(main)
