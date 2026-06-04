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
from transformers import GenerationConfig
from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from .utils import managed_llm

MODEL_ID = "Qwen/Qwen3-1.7B"

LLM_KWARGS = {
    "model": MODEL_ID,
    "max_model_len": 4096,
    "max_num_seqs": 4,
    "block_size": 1024,
    "max_num_batched_tokens": 128,
    "enable_chunked_prefill": True,
}

ENV = {
    "VLLM_RBLN_USE_VLLM_MODEL": "1",
    "VLLM_DISABLE_COMPILE_CACHE": "1",
    "VLLM_RBLN_SAMPLER": "1",
}


@pytest.fixture(scope="module", params=["greedy", "random"])
def sampling_kwargs(request: pytest.FixtureRequest) -> dict:
    """Sampling kwargs for greedy (temp=0) and random (from model gen config)."""
    if request.param == "greedy":
        return {"temperature": 0.0}
    gen_cfg = GenerationConfig.from_pretrained(MODEL_ID)
    return {
        "temperature": gen_cfg.temperature,
        "top_p": gen_cfg.top_p,
        "top_k": gen_cfg.top_k,
    }


def test_choice_sentiment_classification(
    monkeypatch: pytest.MonkeyPatch, sampling_kwargs: dict
) -> None:
    choices = ["Positive", "Negative"]
    with managed_llm(monkeypatch, ENV, **LLM_KWARGS) as llm:
        outputs = llm.generate(
            prompts="Classify this sentiment: vLLM is wonderful!",
            sampling_params=SamplingParams(
                **sampling_kwargs,
                structured_outputs=StructuredOutputsParams(choice=choices),
            ),
        )
    assert outputs[0].outputs[0].text in choices


def test_regex_email_format(
    monkeypatch: pytest.MonkeyPatch, sampling_kwargs: dict
) -> None:
    with managed_llm(monkeypatch, ENV, **LLM_KWARGS) as llm:
        outputs = llm.generate(
            prompts=(
                "Generate an example email address for Alan Turing, "
                "who works in Enigma. End in .com and new line. "
                "Example result: alan.turing@enigma.com\n"
            ),
            sampling_params=SamplingParams(
                **sampling_kwargs,
                structured_outputs=StructuredOutputsParams(regex=r"\w+@\w+\.com\n?"),
            ),
        )
    text = outputs[0].outputs[0].text
    assert re.fullmatch(r"\w+@\w+\.com\n?", text), (
        f"Output does not match regex: {text!r}"
    )


def test_json_car_description(
    monkeypatch: pytest.MonkeyPatch, sampling_kwargs: dict
) -> None:
    class CarType(str, Enum):
        sedan = "sedan"
        suv = "SUV"
        truck = "Truck"
        coupe = "Coupe"

    class CarDescription(BaseModel):
        brand: str
        model: str
        car_type: CarType

    with managed_llm(monkeypatch, ENV, **LLM_KWARGS) as llm:
        outputs = llm.generate(
            prompts=(
                "Generate a JSON with the brand, model and car_type "
                "of the most iconic car from the 90's"
            ),
            sampling_params=SamplingParams(
                **sampling_kwargs,
                max_tokens=32,
                structured_outputs=StructuredOutputsParams(
                    json=CarDescription.model_json_schema()
                ),
            ),
        )
    text = outputs[0].outputs[0].text
    parsed = json.loads(text)
    car = CarDescription(**parsed)
    assert isinstance(car.brand, str)
    assert isinstance(car.model, str)
    assert car.car_type in CarType


def test_grammar_sql_query(
    monkeypatch: pytest.MonkeyPatch, sampling_kwargs: dict
) -> None:
    simplified_sql_grammar = """
        root ::= select_statement

        select_statement ::= "SELECT " column " from " table " where " condition

        column ::= "col_1 " | "col_2 "

        table ::= "table_1 " | "table_2 "

        condition ::= column "= " number

        number ::= "1 " | "2 "
    """
    with managed_llm(monkeypatch, ENV, **LLM_KWARGS) as llm:
        outputs = llm.generate(
            prompts=(
                "Generate a SQL query to show the 'username' and 'email' "
                "from the 'users' table."
            ),
            sampling_params=SamplingParams(
                **sampling_kwargs,
                structured_outputs=StructuredOutputsParams(
                    grammar=simplified_sql_grammar
                ),
            ),
        )
    text = outputs[0].outputs[0].text
    assert text.startswith("SELECT "), f"Expected SQL SELECT, got: {text!r}"
    assert " from " in text
    assert " where " in text


def test_structural_tag_function_call(
    monkeypatch: pytest.MonkeyPatch, sampling_kwargs: dict
) -> None:
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
    with managed_llm(monkeypatch, ENV, **LLM_KWARGS) as llm:
        outputs = llm.generate(
            prompts=prompt,
            sampling_params=SamplingParams(
                **sampling_kwargs,
                structured_outputs=StructuredOutputsParams(
                    structural_tag=json.dumps(structural_tag_obj)
                ),
            ),
        )
    text = outputs[0].outputs[0].text

    assert "<function=get_weather>" in text, (
        f"Expected function call tag in output: {text!r}"
    )
    assert "</function>" in text, f"Expected closing function tag in output: {text!r}"

    match = re.search(r"<function=get_weather>(.*?)</function>", text, re.DOTALL)
    assert match is not None, f"Could not extract function call from: {text!r}"
    params = json.loads(match.group(1))
    assert "city" in params
