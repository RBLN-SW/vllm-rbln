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

"""Constrain generated text with choice, regex, JSON schema, and grammar on RBLN."""

import argparse
from enum import Enum

from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

MAX_TOKENS = 100


class CarType(str, Enum):
    sedan = "sedan"
    suv = "SUV"
    truck = "Truck"
    coupe = "Coupe"


class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        model_impl="vllm",
        block_size=1024,
        max_num_batched_tokens=512,
        additional_config={"use_custom_sampler": False},
    )

    simplified_sql_grammar = """
root ::= select_statement
select_statement ::= "SELECT " column " from " table " where " condition
column ::= "col_1 " | "col_2 "
table ::= "table_1 " | "table_2 "
condition ::= column "= " number
number ::= "1 " | "2 "
"""
    examples = [
        (
            "Structured outputs by Choice",
            "Classify this sentiment: vLLM is wonderful!",
            SamplingParams(
                structured_outputs=StructuredOutputsParams(
                    choice=["Positive", "Negative"]
                ),
            ),
        ),
        (
            "Structured outputs by Regex",
            "Generate an email address for Alan Turing, who works in Enigma."
            "End in .com and new line. Example result:"
            "alan.turing@enigma.com\n",
            SamplingParams(
                structured_outputs=StructuredOutputsParams(regex=r"\w+@\w+\.com\n"),
                stop=["\n"],
                max_tokens=MAX_TOKENS,
            ),
        ),
        (
            "Structured outputs by JSON",
            "Generate a JSON with the brand, model and car_type of "
            "the most iconic car from the 90's",
            SamplingParams(
                structured_outputs=StructuredOutputsParams(
                    json=CarDescription.model_json_schema()
                ),
                max_tokens=MAX_TOKENS,
            ),
        ),
        (
            "Structured outputs by Grammar",
            "Generate an SQL query to show the 'username' and 'email' "
            "from the 'users' table.",
            SamplingParams(
                structured_outputs=StructuredOutputsParams(
                    grammar=simplified_sql_grammar
                ),
                max_tokens=MAX_TOKENS,
            ),
        ),
    ]

    for title, prompt, sampling_params in examples:
        outputs = llm.generate(prompt, sampling_params=sampling_params)
        output = outputs[0].outputs[0].text
        print(f"{'-' * 50}\n{title}: {output}\n{'-' * 50}")


if __name__ == "__main__":
    main()
