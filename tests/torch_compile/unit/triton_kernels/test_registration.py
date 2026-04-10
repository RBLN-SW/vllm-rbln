import pytest
import torch

from .conftest import ALL_OPS


@pytest.mark.parametrize("op_name", ALL_OPS)
def test_triton_op_is_registered_in_torch_ops(op_name):
    ns = torch.ops.rbln_triton_ops
    assert hasattr(ns, op_name), f"{op_name} not found in rbln_triton_ops"
    op = getattr(ns, op_name)
    assert hasattr(op, "default"), f"{op_name}.default not found"
