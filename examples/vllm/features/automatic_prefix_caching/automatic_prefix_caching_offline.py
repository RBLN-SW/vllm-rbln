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

"""Reuse complete KV blocks with sub-block caching disabled on RBLN.

See docs/sub_block_prefix_caching.md for the distinction and configuration.
"""

import argparse

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.tokenizers import TokenizerLike

from vllm_rbln.config import RBLNConfig

LONG_PROMPT = (
    "You are a helpful assistant in recognizes the content of tables in markdown "
    "format. Here is a table as follows.\n"
    "# Table\n"
    "\n"
    "| ID  | Name          | Age | Occupation    | Country       | Email         "
    "         | Phone Number   | Address                       |\n"
    "|-----|---------------|-----|---------------|---------------|"
    "------------------------|----------------|------------------------------|\n"
    "| 1   | John Doe      | 29  | Engineer      | USA           | "
    "john.doe@example.com   | 555-1234       | 123 Elm St, Springfield, IL  |\n"
    "| 2   | Jane Smith    | 34  | Doctor        | Canada        | "
    "jane.smith@example.com | 555-5678       | 456 Oak St, Toronto, ON      |\n"
    "| 3   | Alice Johnson | 27  | Teacher       | UK            | "
    "alice.j@example.com    | 555-8765       | 789 Pine St, London, UK      |\n"
    "| 4   | Bob Brown     | 45  | Artist        | Australia     | "
    "bob.b@example.com      | 555-4321       | 321 Maple St, Sydney, NSW    |\n"
    "| 5   | Carol White   | 31  | Scientist     | New Zealand   | "
    "carol.w@example.com    | 555-6789       | 654 Birch St, Wellington, NZ |\n"
    "| 6   | Dave Green    | 28  | Lawyer        | Ireland       | "
    "dave.g@example.com     | 555-3456       | 987 Cedar St, Dublin, IE     |\n"
    "| 7   | Emma Black    | 40  | Musician      | USA           | "
    "emma.b@example.com     | 555-1111       | 246 Ash St, New York, NY     |\n"
    "| 8   | Frank Blue    | 37  | Chef          | Canada        | "
    "frank.b@example.com    | 555-2222       | 135 Spruce St, Vancouver, BC |\n"
    "| 9   | Grace Yellow  | 50  | Engineer      | UK            | "
    "grace.y@example.com    | 555-3333       | 864 Fir St, Manchester, UK   |\n"
    "| 10  | Henry Violet  | 32  | Artist        | Australia     | "
    "henry.v@example.com    | 555-4444       | 753 Willow St, Melbourne, VIC|\n"
    "| 11  | Irene Orange  | 26  | Scientist     | New Zealand   | "
    "irene.o@example.com    | 555-5555       | 912 Poplar St, Auckland, NZ  |\n"
    "| 12  | Jack Indigo   | 38  | Teacher       | Ireland       | "
    "jack.i@example.com     | 555-6666       | 159 Elm St, Cork, IE         |\n"
    "| 13  | Karen Red     | 41  | Lawyer        | USA           | "
    "karen.r@example.com    | 555-7777       | 357 Cedar St, Boston, MA     |\n"
    "| 14  | Leo Brown     | 30  | Chef          | Canada        | "
    "leo.b@example.com      | 555-8888       | 246 Oak St, Calgary, AB      |\n"
    "| 15  | Mia Green     | 33  | Musician      | UK            | "
    "mia.g@example.com      | 555-9999       | 975 Pine St, Edinburgh, UK   |\n"
    "| 16  | Noah Yellow   | 29  | Doctor        | Australia     | "
    "noah.y@example.com     | 555-0000       | 864 Birch St, Brisbane, QLD  |\n"
    "| 17  | Olivia Blue   | 35  | Engineer      | New Zealand   | "
    "olivia.b@example.com   | 555-1212       | 753 Maple St, Hamilton, NZ   |\n"
    "| 18  | Peter Black   | 42  | Artist        | Ireland       | "
    "peter.b@example.com    | 555-3434       | 912 Fir St, Limerick, IE     |\n"
    "| 19  | Quinn White   | 28  | Scientist     | USA           | "
    "quinn.w@example.com    | 555-5656       | 159 Willow St, Seattle, WA   |\n"
    "| 20  | Rachel Red    | 31  | Teacher       | Canada        | "
    "rachel.r@example.com   | 555-7878       | 357 Poplar St, Ottawa, ON    |\n"
    "| 21  | Steve Green   | 44  | Lawyer        | UK            | "
    "steve.g@example.com    | 555-9090       | 753 Elm St, Birmingham, UK   |\n"
    "| 22  | Tina Blue     | 36  | Musician      | Australia     | "
    "tina.b@example.com     | 555-1213       | 864 Cedar St, Perth, WA      |\n"
    "| 23  | Umar Black    | 39  | Chef          | New Zealand   | "
    "umar.b@example.com     | 555-3435       | 975 Spruce St, Christchurch, NZ|\n"
    "| 24  | Victor Yellow | 43  | Engineer      | Ireland       | "
    "victor.y@example.com   | 555-5657       | 246 Willow St, Galway, IE    |\n"
    "| 25  | Wendy Orange  | 27  | Artist        | USA           | "
    "wendy.o@example.com    | 555-7879       | 135 Elm St, Denver, CO       |\n"
    "| 26  | Xavier Green  | 34  | Scientist     | Canada        | "
    "xavier.g@example.com   | 555-9091       | 357 Oak St, Montreal, QC     |\n"
    "| 27  | Yara Red      | 41  | Teacher       | UK            | "
    "yara.r@example.com     | 555-1214       | 975 Pine St, Leeds, UK       |\n"
    "| 28  | Zack Blue     | 30  | Lawyer        | Australia     | "
    "zack.b@example.com     | 555-3436       | 135 Birch St, Adelaide, SA   |\n"
    "| 29  | Amy White     | 33  | Musician      | New Zealand   | "
    "amy.w@example.com      | 555-5658       | 159 Maple St, Wellington, NZ |\n"
    "| 30  | Ben Black     | 38  | Chef          | Ireland       | "
    "ben.b@example.com      | 555-7870       | 246 Fir St, Waterford, IE    |\n"
)


def make_prompt(tokenizer: TokenizerLike, question: str) -> TokensPrompt:
    messages = [
        {
            "role": "system",
            "content": "Answer the question using the table. Return only the answer.",
        },
        {"role": "user", "content": LONG_PROMPT + f"\nQuestion: {question}"},
    ]
    prompt_token_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False
    )
    return TokensPrompt(prompt_token_ids=prompt_token_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    args = parser.parse_args()

    block_size = 1024
    llm = LLM(
        model=args.model,
        model_impl="vllm",
        block_size=block_size,
        max_num_batched_tokens=512,
        enable_prefix_caching=True,
        additional_config=RBLNConfig(enable_sub_block_cache=False),
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=64)
    tokenizer = llm.get_tokenizer()
    questions = [
        "What is the age of John Doe?",
        "What is the age of Zack Blue?",
        "What is the occupation of Jane Smith?",
        "Which country does Alice Johnson live in?",
    ]

    # Complete each request before submitting the next so its KV is cached.
    cached_tokens = []
    for question in questions:
        prompt = make_prompt(tokenizer, question)
        output = llm.generate([prompt], sampling_params)[0]
        cached_tokens.append(output.num_cached_tokens)
        print("-" * 60)
        print(f"Question: {question}")
        print(f"Generated text: {output.outputs[0].text!r}")
        print(f"Cached prompt tokens: {output.num_cached_tokens}")

    assert cached_tokens[0] == 0, "The first request should have a cold cache."
    assert all(
        count >= block_size and count % block_size == 0 for count in cached_tokens[1:]
    ), "Later requests should reuse complete KV blocks."


if __name__ == "__main__":
    main()
