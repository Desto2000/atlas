"""
Device Utilities for ATLAS Architecture
=======================================

Utilities for managing device consistency across ATLAS components.
"""

import torch
from typing import Union, Optional, Dict, Any, List


def get_device_from_module(module: torch.nn.Module) -> torch.device:
    """Get the device of a PyTorch module."""
    try:
        return next(module.parameters()).device
    except StopIteration:
        # Module has no parameters, default to CPU
        return torch.device('cpu')


def ensure_tensor_device(tensor: torch.Tensor, target_device: torch.device) -> torch.Tensor:
    """Ensure tensor is on the target device."""
    if tensor.device != target_device:
        return tensor.to(target_device)
    return tensor


def sync_tensors_to_device(*tensors: torch.Tensor, device: torch.device) -> List[torch.Tensor]:
    """Synchronize multiple tensors to the same device."""
    return [ensure_tensor_device(tensor, device) for tensor in tensors]


def create_tensor_like_device(tensor: torch.Tensor,
                              *args,
                              device: Optional[torch.device] = None,
                              **kwargs) -> torch.Tensor:
    """Create a new tensor on the same device as the reference tensor."""
    target_device = device if device is not None else tensor.device

    if 'device' not in kwargs:
        kwargs['device'] = target_device

    if len(args) == 0:
        # Creating from kwargs only
        return torch.tensor(**kwargs)
    else:
        # Creating with positional args
        return torch.tensor(*args, **kwargs)


def device_consistent_operation(func):
    """Decorator to ensure device consistency in operations."""
    def wrapper(self, *args, **kwargs):
        if hasattr(self, 'device'):
            target_device = self.device
        else:
            target_device = get_device_from_module(self)

        # Move tensor arguments to correct device
        new_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                new_args.append(ensure_tensor_device(arg, target_device))
            else:
                new_args.append(arg)

        # Move tensor values in kwargs to correct device
        new_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                new_kwargs[key] = ensure_tensor_device(value, target_device)
            else:
                new_kwargs[key] = value

        return func(self, *new_args, **new_kwargs)

    return wrapper


class DeviceManager:
    """Centralized device management for ATLAS components."""

    def __init__(self, device: Union[str, torch.device] = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

    def to_device(self, *tensors: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Move tensors to managed device."""
        results = [tensor.to(self.device) for tensor in tensors]
        return results[0] if len(results) == 1 else results

    def create_tensor(self, *args, **kwargs) -> torch.Tensor:
        """Create tensor on managed device."""
        kwargs['device'] = self.device
        return torch.tensor(*args, **kwargs)

    def create_zeros(self, *shape, **kwargs) -> torch.Tensor:
        """Create zeros tensor on managed device."""
        kwargs['device'] = self.device
        return torch.zeros(*shape, **kwargs)

    def create_ones(self, *shape, **kwargs) -> torch.Tensor:
        """Create ones tensor on managed device."""
        kwargs['device'] = self.device
        return torch.ones(*shape, **kwargs)

    def create_randn(self, *shape, **kwargs) -> torch.Tensor:
        """Create random normal tensor on managed device."""
        kwargs['device'] = self.device
        return torch.randn(*shape, **kwargs)

    def create_randint(self, low, high, size, **kwargs) -> torch.Tensor:
        """Create random integer tensor on managed device."""
        kwargs['device'] = self.device
        return torch.randint(low, high, size, **kwargs)

    def create_eye(self, n, **kwargs) -> torch.Tensor:
        """Create identity matrix on managed device."""
        kwargs['device'] = self.device
        return torch.eye(n, **kwargs)