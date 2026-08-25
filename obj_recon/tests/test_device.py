"""Tests for device resolution."""

import pytest
import torch

from obj_recon.device import resolve_torch_device, set_cuda_device_if_indexed


def test_resolve_auto_cuda_available(monkeypatch):
    """When CUDA is available, 'auto' resolves to 'cuda'."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    device = resolve_torch_device("auto")
    assert device.type == "cuda"


def test_resolve_auto_cuda_unavailable(monkeypatch):
    """When CUDA is not available, 'auto' resolves to 'cpu'."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device = resolve_torch_device("auto")
    assert device.type == "cpu"


def test_resolve_explicit_cpu():
    device = resolve_torch_device("cpu")
    assert device.type == "cpu"


def test_resolve_cuda_unavailable_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA device requested"):
        resolve_torch_device("cuda")


def test_set_cuda_device_no_index():
    """set_cuda_device_if_indexed is a no-op when device has no index."""
    set_cuda_device_if_indexed(torch.device("cuda"))  # type=0, index=None
