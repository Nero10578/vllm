# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU memory pooling for LoRA adapters to reduce memory usage."""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from vllm.config.lora import LoRAConfig
from vllm.logger import init_logger
from vllm.lora.lora_weights import LoRALayerWeights, PackedLoRALayerWeights
from vllm.utils import is_pin_memory_available

logger = init_logger(__name__)


class CPULoRAPool:
    """
    A memory pool for LoRA adapters on CPU that pre-allocates tensors
    to reduce memory overhead, similar to the GPU approach.
    """
    
    def __init__(
        self,
        max_cpu_loras: int,
        lora_config: LoRAConfig,
        pin_memory: bool = True,
    ):
        self.max_cpu_loras = max_cpu_loras
        self.lora_config = lora_config
        self.pin_memory = pin_memory and is_pin_memory_available()
        
        # Track allocated pools by tensor shape
        self.pools: Dict[Tuple[int, ...], torch.Tensor] = {}
        self.allocated_slots: Dict[int, Dict[str, Tuple[int, ...]]] = {}
        self.lora_id_to_slot: Dict[int, int] = {}
        self.free_slots: List[int] = list(range(max_cpu_loras))
        
    def _get_or_create_pool(
        self, 
        shape: Tuple[int, ...], 
        dtype: torch.dtype
    ) -> torch.Tensor:
        """Get or create a pooled tensor for the given shape and dtype."""
        key = (shape, dtype)
        if key not in self.pools:
            # Pre-allocate a tensor pool with space for all LoRAs
            pool_shape = (self.max_cpu_loras,) + shape
            self.pools[key] = torch.zeros(
                pool_shape, 
                dtype=dtype, 
                device="cpu",
                pin_memory=self.pin_memory
            )
            logger.debug(f"Created CPU pool for shape {shape} with {self.max_cpu_loras} slots")
        return self.pools[key]
    
    def allocate_slot(self, lora_id: int) -> Optional[int]:
        """Allocate a slot for a LoRA adapter."""
        if not self.free_slots:
            logger.warning("No free CPU LoRA slots available")
            return None
        
        slot = self.free_slots.pop(0)
        self.allocated_slots[lora_id] = {}
        self.lora_id_to_slot[lora_id] = slot
        logger.debug(f"Allocated CPU slot {slot} for LoRA {lora_id}")
        return slot
    
    def deallocate_slot(self, lora_id: int) -> None:
        """Deallocate a slot for a LoRA adapter."""
        if lora_id not in self.lora_id_to_slot:
            return
        
        slot = self.lora_id_to_slot[lora_id]
        
        # Clear the allocated tensors by setting them to zero
        for (shape, dtype), pool in self.pools.items():
            if slot < pool.shape[0]:
                pool[slot] = 0
        
        # Add the slot back to free slots
        self.free_slots.append(slot)
        self.free_slots.sort()  # Keep slots ordered
        
        del self.allocated_slots[lora_id]
        del self.lora_id_to_slot[lora_id]
        logger.debug(f"Deallocated CPU slot {slot} for LoRA {lora_id}")
    
    def store_tensor(
        self,
        lora_id: int,
        tensor: torch.Tensor,
        tensor_name: str
    ) -> Optional[torch.Tensor]:
        """Store a tensor in the pooled memory."""
        if lora_id not in self.lora_id_to_slot:
            slot = self.allocate_slot(lora_id)
            if slot is None:
                return None
        
        slot = self.lora_id_to_slot[lora_id]
        shape = tensor.shape
        dtype = tensor.dtype
        pool = self._get_or_create_pool(shape, dtype)
        
        # Store the tensor in the pool
        pool[slot] = tensor
        self.allocated_slots[lora_id][tensor_name] = (shape, dtype)
        
        # Return a view of the stored tensor
        return pool[slot]
    
    def get_tensor(
        self,
        lora_id: int,
        tensor_name: str
    ) -> Optional[torch.Tensor]:
        """Retrieve a tensor from the pooled memory."""
        if lora_id not in self.lora_id_to_slot:
            return None
        
        slot = self.lora_id_to_slot[lora_id]
        shape_dtype = self.allocated_slots[lora_id].get(tensor_name)
        if shape_dtype is None:
            return None
        
        shape, dtype = shape_dtype
        pool = self.pools.get((shape, dtype))
        if pool is None:
            return None
        
        return pool[slot]


class PooledLoRALayerWeights:
    """
    A wrapper for LoRALayerWeights that uses CPU memory pooling.
    """
    
    def __init__(
        self,
        original_weights: LoRALayerWeights,
        cpu_pool: CPULoRAPool,
        lora_id: int,
    ):
        self.original_weights = original_weights
        self.cpu_pool = cpu_pool
        self.lora_id = lora_id
        self._stored_tensors = {}
        
        # Store the tensors in the pool
        self._store_tensors()
    
    def _store_tensors(self):
        """Store all tensors in the CPU pool."""
        if self.original_weights.lora_a is not None:
            stored = self.cpu_pool.store_tensor(
                self.lora_id, 
                self.original_weights.lora_a, 
                "lora_a"
            )
            if stored is not None:
                self._stored_tensors["lora_a"] = stored
        
        if self.original_weights.lora_b is not None:
            stored = self.cpu_pool.store_tensor(
                self.lora_id, 
                self.original_weights.lora_b, 
                "lora_b"
            )
            if stored is not None:
                self._stored_tensors["lora_b"] = stored
        
        if self.original_weights.bias is not None:
            stored = self.cpu_pool.store_tensor(
                self.lora_id, 
                self.original_weights.bias, 
                "bias"
            )
            if stored is not None:
                self._stored_tensors["bias"] = stored
        
        if self.original_weights.embeddings_tensor is not None:
            stored = self.cpu_pool.store_tensor(
                self.lora_id, 
                self.original_weights.embeddings_tensor, 
                "embeddings_tensor"
            )
            if stored is not None:
                self._stored_tensors["embeddings_tensor"] = stored
    
    @property
    def lora_a(self) -> Optional[torch.Tensor]:
        """Get lora_a tensor from pool."""
        return self._stored_tensors.get("lora_a")
    
    @property
    def lora_b(self) -> Optional[torch.Tensor]:
        """Get lora_b tensor from pool."""
        return self._stored_tensors.get("lora_b")
    
    @property
    def bias(self) -> Optional[torch.Tensor]:
        """Get bias tensor from pool."""
        return self._stored_tensors.get("bias")
    
    @property
    def embeddings_tensor(self) -> Optional[torch.Tensor]:
        """Get embeddings_tensor from pool."""
        return self._stored_tensors.get("embeddings_tensor")
    
    def optimize(self):
        """Optimize the LoRA by merging the scaling into lora_b."""
        if self.original_weights.scaling == 1:
            return
        
        lora_b = self.lora_b
        if lora_b is not None:
            lora_b *= self.original_weights.scaling
            self.original_weights.scaling = 1
    
    def __getattr__(self, name):
        """Delegate attribute access to the original weights."""
        return getattr(self.original_weights, name)


class PooledPackedLoRALayerWeights:
    """
    A wrapper for PackedLoRALayerWeights that uses CPU memory pooling.
    """
    
    def __init__(
        self,
        original_weights: PackedLoRALayerWeights,
        cpu_pool: CPULoRAPool,
        lora_id: int,
    ):
        self.original_weights = original_weights
        self.cpu_pool = cpu_pool
        self.lora_id = lora_id
        self._stored_tensors = {}
        
        # Store the tensors in the pool
        self._store_tensors()
    
    def _store_tensors(self):
        """Store all tensors in the CPU pool."""
        # Handle packed LoRA tensors (lists)
        for i, lora_a in enumerate(self.original_weights.lora_a):
            if lora_a is not None:
                stored = self.cpu_pool.store_tensor(
                    self.lora_id, 
                    lora_a, 
                    f"lora_a_{i}"
                )
                if stored is not None:
                    self._stored_tensors[f"lora_a_{i}"] = stored
        
        for i, lora_b in enumerate(self.original_weights.lora_b):
            if lora_b is not None:
                stored = self.cpu_pool.store_tensor(
                    self.lora_id, 
                    lora_b, 
                    f"lora_b_{i}"
                )
                if stored is not None:
                    self._stored_tensors[f"lora_b_{i}"] = stored
        
        if self.original_weights.bias is not None:
            for i, bias in enumerate(self.original_weights.bias):
                if bias is not None:
                    stored = self.cpu_pool.store_tensor(
                        self.lora_id, 
                        bias, 
                        f"bias_{i}"
                    )
                    if stored is not None:
                        self._stored_tensors[f"bias_{i}"] = stored
    
    @property
    def lora_a(self) -> List[Optional[torch.Tensor]]:
        """Get lora_a tensors from pool."""
        result = []
        for i in range(len(self.original_weights.lora_a)):
            tensor = self._stored_tensors.get(f"lora_a_{i}")
            result.append(tensor)
        return result
    
    @property
    def lora_b(self) -> List[Optional[torch.Tensor]]:
        """Get lora_b tensors from pool."""
        result = []
        for i in range(len(self.original_weights.lora_b)):
            tensor = self._stored_tensors.get(f"lora_b_{i}")
            result.append(tensor)
        return result
    
    @property
    def bias(self) -> Optional[List[Optional[torch.Tensor]]]:
        """Get bias tensors from pool."""
        if self.original_weights.bias is None:
            return None
        
        result = []
        for i in range(len(self.original_weights.bias)):
            tensor = self._stored_tensors.get(f"bias_{i}")
            result.append(tensor)
        return result
    
    def optimize(self):
        """Optimize the LoRA by merging the scaling into lora_b."""
        for i in range(len(self.original_weights.lora_b)):
            if (self.original_weights.scaling[i] == 1 or 
                self.original_weights.lora_b[i] is None):
                continue
            
            lora_b = self._stored_tensors.get(f"lora_b_{i}")
            if lora_b is not None:
                lora_b *= self.original_weights.scaling[i]
                self.original_weights.scaling[i] = 1
    
    def __getattr__(self, name):
        """Delegate attribute access to the original weights."""
        return getattr(self.original_weights, name)