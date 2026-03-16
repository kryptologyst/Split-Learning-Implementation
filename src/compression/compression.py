"""
Model compression techniques for edge deployment.

This module implements quantization, pruning, and distillation techniques
optimized for split learning scenarios.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
import logging

from ..models.split_learning import SplitLearningClient, SplitLearningServer


class BaseCompression(ABC):
    """Abstract base class for model compression techniques."""
    
    def __init__(self, model: nn.Module, device: str = "cpu") -> None:
        """Initialize compression technique.
        
        Args:
            model: Model to compress
            device: Device to run on
        """
        self.model = model
        self.device = torch.device(device)
        self.compressed_model = None
        
    @abstractmethod
    def compress(self) -> nn.Module:
        """Apply compression to the model.
        
        Returns:
            Compressed model
        """
        pass
        
    @abstractmethod
    def get_compression_ratio(self) -> float:
        """Get compression ratio achieved.
        
        Returns:
            Compression ratio (original_size / compressed_size)
        """
        pass


class QuantizationCompression(BaseCompression):
    """Quantization-based compression for edge deployment."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        quantization_bits: int = 8,
        calibration_data: Optional[torch.Tensor] = None
    ) -> None:
        """Initialize quantization compression.
        
        Args:
            model: Model to quantize
            device: Device to run on
            quantization_bits: Number of bits for quantization
            calibration_data: Data for calibration
        """
        super().__init__(model, device)
        self.quantization_bits = quantization_bits
        self.calibration_data = calibration_data
        
    def compress(self) -> nn.Module:
        """Apply quantization to the model.
        
        Returns:
            Quantized model
        """
        # Dynamic quantization for simplicity
        self.compressed_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Conv2d, nn.Linear},
            dtype=torch.qint8
        )
        
        return self.compressed_model
        
    def get_compression_ratio(self) -> float:
        """Get quantization compression ratio.
        
        Returns:
            Compression ratio
        """
        if self.compressed_model is None:
            return 1.0
            
        # Estimate compression ratio
        original_params = sum(p.numel() for p in self.model.parameters())
        compressed_params = sum(p.numel() for p in self.compressed_model.parameters())
        
        # Quantization typically reduces size by 4x (float32 -> int8)
        return original_params / compressed_params * 4


class PruningCompression(BaseCompression):
    """Pruning-based compression for model efficiency."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        sparsity: float = 0.5,
        pruning_method: str = "magnitude"
    ) -> None:
        """Initialize pruning compression.
        
        Args:
            model: Model to prune
            device: Device to run on
            sparsity: Fraction of weights to prune
            pruning_method: Method for pruning ('magnitude', 'gradient')
        """
        super().__init__(model, device)
        self.sparsity = sparsity
        self.pruning_method = pruning_method
        
    def compress(self) -> nn.Module:
        """Apply pruning to the model.
        
        Returns:
            Pruned model
        """
        self.compressed_model = self.model
        
        # Apply magnitude-based pruning
        for module in self.compressed_model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                torch.nn.utils.prune.l1_unstructured(
                    module,
                    name='weight',
                    amount=self.sparsity
                )
                
        return self.compressed_model
        
    def get_compression_ratio(self) -> float:
        """Get pruning compression ratio.
        
        Returns:
            Compression ratio
        """
        if self.compressed_model is None:
            return 1.0
            
        # Calculate actual sparsity
        total_params = 0
        pruned_params = 0
        
        for module in self.compressed_model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if hasattr(module, 'weight_mask'):
                    total_params += module.weight.numel()
                    pruned_params += module.weight_mask.sum().item()
                    
        if total_params == 0:
            return 1.0
            
        actual_sparsity = 1 - (pruned_params / total_params)
        return 1 / (1 - actual_sparsity)


class DistillationCompression(BaseCompression):
    """Knowledge distillation for model compression."""
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        device: str = "cpu",
        temperature: float = 3.0,
        alpha: float = 0.7
    ) -> None:
        """Initialize distillation compression.
        
        Args:
            teacher_model: Large teacher model
            student_model: Small student model
            device: Device to run on
            temperature: Temperature for softmax
            alpha: Weight for distillation loss
        """
        super().__init__(student_model, device)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        
    def compress(self) -> nn.Module:
        """Apply distillation to create compressed model.
        
        Returns:
            Distilled student model
        """
        # Distillation is applied during training, not as post-processing
        self.compressed_model = self.model
        return self.compressed_model
        
    def distillation_loss(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Calculate distillation loss.
        
        Args:
            student_outputs: Student model outputs
            teacher_outputs: Teacher model outputs
            labels: Ground truth labels
            
        Returns:
            Distillation loss
        """
        # Soft targets from teacher
        soft_targets = F.softmax(teacher_outputs / self.temperature, dim=1)
        soft_prob = F.log_softmax(student_outputs / self.temperature, dim=1)
        
        # Distillation loss
        distillation_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean')
        
        # Hard targets
        hard_loss = F.cross_entropy(student_outputs, labels)
        
        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * hard_loss
        
        return total_loss
        
    def get_compression_ratio(self) -> float:
        """Get distillation compression ratio.
        
        Returns:
            Compression ratio
        """
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        student_params = sum(p.numel() for p in self.model.parameters())
        
        return teacher_params / student_params


class SplitLearningCompression:
    """Compression techniques specifically for split learning."""
    
    def __init__(
        self,
        client: SplitLearningClient,
        server: SplitLearningServer,
        device: str = "cpu"
    ) -> None:
        """Initialize split learning compression.
        
        Args:
            client: Client model
            server: Server model
            device: Device to run on
        """
        self.client = client
        self.server = server
        self.device = device
        
    def compress_client(
        self,
        compression_type: str = "quantization",
        **kwargs
    ) -> SplitLearningClient:
        """Compress client model for edge deployment.
        
        Args:
            compression_type: Type of compression to apply
            **kwargs: Additional compression parameters
            
        Returns:
            Compressed client model
        """
        if compression_type == "quantization":
            compressor = QuantizationCompression(
                self.client.model,
                device=str(self.device),
                **kwargs
            )
        elif compression_type == "pruning":
            compressor = PruningCompression(
                self.client.model,
                device=str(self.device),
                **kwargs
            )
        else:
            raise ValueError(f"Unknown compression type: {compression_type}")
            
        compressed_model = compressor.compress()
        
        # Create new client with compressed model
        compressed_client = SplitLearningClient(
            input_shape=self.client.input_shape,
            cut_layer=self.client.cut_layer,
            device=str(self.device)
        )
        compressed_client.model = compressed_model
        
        return compressed_client
        
    def compress_server(
        self,
        compression_type: str = "quantization",
        **kwargs
    ) -> SplitLearningServer:
        """Compress server model.
        
        Args:
            compression_type: Type of compression to apply
            **kwargs: Additional compression parameters
            
        Returns:
            Compressed server model
        """
        if compression_type == "quantization":
            compressor = QuantizationCompression(
                self.server.model,
                device=str(self.device),
                **kwargs
            )
        elif compression_type == "pruning":
            compressor = PruningCompression(
                self.server.model,
                device=str(self.device),
                **kwargs
            )
        else:
            raise ValueError(f"Unknown compression type: {compression_type}")
            
        compressed_model = compressor.compress()
        
        # Create new server with compressed model
        compressed_server = SplitLearningServer(
            input_shape=self.server.input_shape,
            num_classes=self.server.num_classes,
            hidden_size=self.server.hidden_size,
            device=str(self.device)
        )
        compressed_server.model = compressed_model
        
        return compressed_server
        
    def get_compression_metrics(self) -> Dict[str, float]:
        """Get compression metrics for both models.
        
        Returns:
            Dictionary of compression metrics
        """
        metrics = {}
        
        # Client compression metrics
        client_original_size = sum(p.numel() for p in self.client.model.parameters())
        metrics['client_original_size'] = client_original_size
        
        # Server compression metrics
        server_original_size = sum(p.numel() for p in self.server.model.parameters())
        metrics['server_original_size'] = server_original_size
        
        # Total model size
        total_size = client_original_size + server_original_size
        metrics['total_original_size'] = total_size
        
        return metrics


class EdgeOptimizedCompression:
    """Edge-specific compression optimizations."""
    
    def __init__(self, device_config: Dict[str, Any]) -> None:
        """Initialize edge-optimized compression.
        
        Args:
            device_config: Device-specific configuration
        """
        self.device_config = device_config
        
    def optimize_for_device(
        self,
        model: nn.Module,
        target_device: str = "raspberry_pi"
    ) -> nn.Module:
        """Optimize model for specific edge device.
        
        Args:
            model: Model to optimize
            target_device: Target edge device
            
        Returns:
            Optimized model
        """
        if target_device == "raspberry_pi":
            # Optimize for ARM CPU
            return self._optimize_for_arm(model)
        elif target_device == "jetson_nano":
            # Optimize for NVIDIA GPU
            return self._optimize_for_gpu(model)
        elif target_device == "mcu":
            # Optimize for microcontroller
            return self._optimize_for_mcu(model)
        else:
            return model
            
    def _optimize_for_arm(self, model: nn.Module) -> nn.Module:
        """Optimize model for ARM architecture."""
        # Apply aggressive quantization
        compressor = QuantizationCompression(model, quantization_bits=8)
        return compressor.compress()
        
    def _optimize_for_gpu(self, model: nn.Module) -> nn.Module:
        """Optimize model for GPU architecture."""
        # Apply moderate pruning
        compressor = PruningCompression(model, sparsity=0.3)
        return compressor.compress()
        
    def _optimize_for_mcu(self, model: nn.Module) -> nn.Module:
        """Optimize model for microcontroller."""
        # Apply aggressive pruning and quantization
        pruned_model = PruningCompression(model, sparsity=0.8).compress()
        quantized_model = QuantizationCompression(pruned_model, quantization_bits=4).compress()
        return quantized_model


def create_compression(
    compression_type: str = "quantization",
    **kwargs
) -> BaseCompression:
    """Factory function to create compression techniques.
    
    Args:
        compression_type: Type of compression ('quantization', 'pruning', 'distillation')
        **kwargs: Additional compression parameters
        
    Returns:
        Configured compression technique
    """
    if compression_type == "quantization":
        return QuantizationCompression(**kwargs)
    elif compression_type == "pruning":
        return PruningCompression(**kwargs)
    elif compression_type == "distillation":
        return DistillationCompression(**kwargs)
    else:
        raise ValueError(f"Unknown compression type: {compression_type}")
