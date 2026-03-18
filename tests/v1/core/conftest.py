from unittest.mock import patch
import pytest

@pytest.fixture(autouse=True)
def skip_prepare_compile():
    with patch("vllm_rbln.utils.optimum.configuration.prepare_vllm_for_compile"):
        yield