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

# TODO: drop entries as the underlying issues are resolved.
# - models/test_basic_models_correctness.py: slow and redundant with
#   the coverage suite; drop once the nightly budget accommodates it.
# - models/test_model_coverage_single.py: EXAONE-3.5-2.4B fails with
#   ImportError: RopeParameters (transformers version mismatch) and
#   Mistral-7B OOMs the runner container (exit 137) mid-run.
# - v1/lora/: LoRA e2e suite isn't yet stable under the nightly FSW
#   matrix.
# Pooling model tests under models/ continue to run.
collect_ignore = [
    "models/test_basic_models_correctness.py",
    "models/test_model_coverage_single.py",
    "v1/lora",
]
