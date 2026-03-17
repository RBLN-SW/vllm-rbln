"""Python startup shim for vLLM subprocess compatibility."""

import transformers.configuration_utils as configuration_utils

if not hasattr(configuration_utils, "ALLOWED_LAYER_TYPES") and hasattr(
    configuration_utils, "ALLOWED_ATTENTION_LAYER_TYPES"
):
    configuration_utils.ALLOWED_LAYER_TYPES = (
        configuration_utils.ALLOWED_ATTENTION_LAYER_TYPES
    )
