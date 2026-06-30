# Copyright 2025 Rebellions Inc. All rights reserved.
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

from vllm import LLM

model_name = "Qwen/Qwen3-Reranker-0.6B"

# What is the difference between the official original version and one
# that has been converted into a sequence classification model?
# Qwen3-Reranker is a language model that doing reranker by using the
# logits of "no" and "yes" tokens.
# It needs to computing 151669 tokens logits, making this method extremely
# inefficient, not to mention incompatible with the vllm score API.
# A method for converting the original model into a sequence classification
# model was proposed. See：https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/discussions/3
# Models converted offline using this method can not only be more efficient
# and support the vllm score API, but also make the init parameters more
# concise, for example.
# llm = LLM(model="tomaarsen/Qwen3-Reranker-0.6B-seq-cls", runner="pooling")

# If you want to load the official original version, the init parameters are
# as follows.


def get_llm() -> LLM:
    """Initializes and returns the LLM model for Qwen3-Reranker."""
    return LLM(
        model=model_name,
        runner="pooling",
        max_model_len=2024,
        block_size=512,
        enable_chunked_prefill=True,
        max_num_batched_tokens=128,
        max_num_seqs=4,
        hf_overrides={
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        },
    )


def main():
    # Why do we need hf_overrides for the official original version:
    # vllm converts it to Qwen3ForSequenceClassification when loaded for
    # better performance.
    # - Firstly, we need using
    #   `"architectures": ["Qwen3ForSequenceClassification"],`
    #   to manually route to Qwen3ForSequenceClassification.
    # - Then, we will extract the vector corresponding to classifier_from_token
    #   from lm_head using `"classifier_from_token": ["no", "yes"]`.
    # - Third, we will convert these two vectors into one vector.
    #   The use of conversion logic is controlled by
    #   `using "is_original_qwen3_reranker": True`.

    # Please use the query_template and document_template to format the query
    # and document for better reranker results.

    prefix = (
        "<|im_start|>system\nJudge whether the Document meets the "
        + "requirements based on the Query and the Instruct provided. "
        + 'Note that the answer can only be "yes" or "no".<|im_end|>\n'
        + "<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
    document_template = "<Document>: {doc}{suffix}"

    instruction = "Given a query, retrieve relevant passages that answer the query"

    queries = [
        "What is the capital of China?",
        "Explain gravity",
        "Who is the president of the United States?",
    ]

    documents = [
        "Gravity is a force that attracts two bodies towards each other. "
        + "It gives weight to physical objects and is responsible for the "
        + "movement of planets around the sun.",
        "The capital of China is Beijing.",
        "The president of the United States is Donald Trump.",
    ]

    queries = [
        query_template.format(prefix=prefix, instruction=instruction, query=query)
        for query in queries
    ]
    documents = [document_template.format(doc=doc, suffix=suffix) for doc in documents]

    llm = get_llm()
    outputs = llm.score(queries, documents)

    print([output.outputs.score for output in outputs])
    # [6.2320450524566695e-06, 4.1464179957984015e-05, 0.9628150463104248]


if __name__ == "__main__":
    main()
