import pytest

@pytest.fixture(autouse=True)
def set_npu_env_var(monkeypatch):
    monkeypatch.setenv("RBLN_FORCE_NPU_NAME", "RBLN-CA25")