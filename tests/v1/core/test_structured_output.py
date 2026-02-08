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

import json

from .utils import create_requests, create_scheduler


def test_structured_output():
    scheduler = create_scheduler()
    structured_output_manager = scheduler.structured_output_manager
    # 1. schedule the requests
    # 2. Execute the scheduleroutput => pass
    # 3. get_grammar_bitmask

    sample_json_schema = {"a": "int", "b": {"c": "str", "d": "float"}}
    str_sample_json_schema = json.dumps(sample_json_schema)
    requests = create_requests(num_requests=2,
                               sample_json_schema=str_sample_json_schema)
    for request in requests:
        scheduler.add_request(request)
        structured_output_manager.grammar_init(request)
        # The grammar might not yet be compiled, so we wait for it
        while not request.structured_output_request._check_grammar_completion(
        ):
            continue

    # Prefill 1st request
    scheduler_output = scheduler.schedule()
    grammar_bitmask = scheduler.get_grammar_bitmask(scheduler_output)
    assert len(grammar_bitmask.structured_output_request_ids) == 1
    assert grammar_bitmask.grammar_bitmask.shape[0] == 1
    # Prefill 2nd request
    scheduler_output = scheduler.schedule()
    grammar_bitmask = scheduler.get_grammar_bitmask(scheduler_output)
    assert len(grammar_bitmask.structured_output_request_ids) == 1
    assert grammar_bitmask.grammar_bitmask.shape[0] == 1
    # Decode step for both requests
    scheduler_output = scheduler.schedule()
    grammar_bitmask = scheduler.get_grammar_bitmask(scheduler_output)
    assert len(grammar_bitmask.structured_output_request_ids) == 2
    assert grammar_bitmask.grammar_bitmask.shape[0] == 2
