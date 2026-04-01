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

"""End-to-end tests for structured output decoding.

Tests choice, regex, json, grammar, and structural_tag modes to ensure
the model generates outputs conforming to the specified constraints.
"""

from __future__ import annotations

import json
import re
from enum import Enum

import pytest
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"


@pytest.fixture(scope="module")
def llm(monkeypatch_module):
    """Module-scoped LLM instance shared across all structured output tests."""
    monkeypatch_module.setenv("VLLM_RBLN_USE_VLLM_MODEL", "1")
    monkeypatch_module.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
    return LLM(
        model=MODEL_ID,
        max_model_len=4096,
        max_num_seqs=4,
        block_size=1024,
        max_num_batched_tokens=128,
        enable_chunked_prefill=True,
    )


class TestChoiceStructuredOutput:
    """Test choice-based structured output."""

    def test_sentiment_classification(self, llm):
        choices = ["Positive", "Negative"]
        outputs = llm.generate(
            prompts="Classify this sentiment: vLLM is wonderful!",
            sampling_params=SamplingParams(
                structured_outputs=StructuredOutputsParams(choice=choices)
            ),
        )
        assert outputs[0].outputs[0].text in choices


class TestRegexStructuredOutput:
    """Test regex-constrained structured output."""

    def test_email_format(self, llm):
        outputs = llm.generate(
            prompts=(
                "Generate an example email address for Alan Turing, "
                "who works in Enigma. End in .com and new line. "
                "Example result: alan.turing@enigma.com\n"
            ),
            sampling_params=SamplingParams(
                structured_outputs=StructuredOutputsParams(regex=r"\w+@\w+\.com\n")
            ),
        )
        text = outputs[0].outputs[0].text
        assert re.fullmatch(r"\w+@\w+\.com\n", text), (
            f"Output does not match regex: {text!r}"
        )


class TestJsonStructuredOutput:
    """Test JSON schema-constrained structured output."""

    def test_car_description(self, llm):
        class CarType(str, Enum):
            sedan = "sedan"
            suv = "SUV"
            truck = "Truck"
            coupe = "Coupe"

        class CarDescription(BaseModel):
            brand: str
            model: str
            car_type: CarType

        outputs = llm.generate(
            prompts=(
                "Generate a JSON with the brand, model and car_type "
                "of the most iconic car from the 90's"
            ),
            sampling_params=SamplingParams(
                structured_outputs=StructuredOutputsParams(
                    json=CarDescription.model_json_schema()
                )
            ),
        )
        text = outputs[0].outputs[0].text
        parsed = json.loads(text)
        # Validate it conforms to the schema
        car = CarDescription(**parsed)
        assert isinstance(car.brand, str)
        assert isinstance(car.model, str)
        assert car.car_type in CarType


class TestGrammarStructuredOutput:
    """Test EBNF grammar-constrained structured output."""

    def test_sql_query(self, llm):
        simplified_sql_grammar = """
            root ::= select_statement

            select_statement ::= "SELECT " column " from " table " where " condition

            column ::= "col_1 " | "col_2 "

            table ::= "table_1 " | "table_2 "

            condition ::= column "= " number

            number ::= "1 " | "2 "
        """
        outputs = llm.generate(
            prompts=(
                "Generate a SQL query to show the 'username' and 'email' "
                "from the 'users' table."
            ),
            sampling_params=SamplingParams(
                structured_outputs=StructuredOutputsParams(
                    grammar=simplified_sql_grammar
                )
            ),
        )
        text = outputs[0].outputs[0].text
        assert text.startswith("SELECT "), f"Expected SQL SELECT, got: {text!r}"
        assert " from " in text
        assert " where " in text


class TestStructuralTagStructuredOutput:
    """Test structural_tag-constrained structured output."""

    def test_function_call_tag(self, llm):
        structural_tag_obj = {
            "type": "structural_tag",
            "structures": [
                {
                    "begin": "<function=get_weather>",
                    "schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                    "end": "</function>",
                }
            ],
            "triggers": ["<function="],
        }

        prompt = """You have access to the following function to retrieve the weather:
{
    "name": "get_weather",
    "parameters": {
        "city": {
            "param_type": "string",
            "description": "The city to get the weather for",
            "required": true
        }
    }
}

If you choose to call a function ONLY reply in the following format:
<function=get_weather>{parameters}</function>
where parameters is a JSON dict.

Example:
<function=get_weather>{"city": "Boston"}</function>

What is the weather in New York City?
"""
        outputs = llm.generate(
            prompts=prompt,
            sampling_params=SamplingParams(
                structured_outputs=StructuredOutputsParams(
                    structural_tag=json.dumps(structural_tag_obj)
                )
            ),
        )
        text = outputs[0].outputs[0].text

        # Verify the function call tags are present
        assert "<function=get_weather>" in text, (
            f"Expected function call tag in output: {text!r}"
        )
        assert "</function>" in text, (
            f"Expected closing function tag in output: {text!r}"
        )

        # Extract and validate JSON between tags
        match = re.search(r"<function=get_weather>(.*?)</function>", text, re.DOTALL)
        assert match is not None, f"Could not extract function call from: {text!r}"
        params = json.loads(match.group(1))
        assert "city" in params
