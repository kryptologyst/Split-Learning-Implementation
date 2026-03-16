"""
Utility functions and helpers for split learning.

This module provides common utilities including seeding, device management,
logging, and other helper functions.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import numpy as np
import random
import os
import logging
import yaml
from pathlib import Path
import json
import time
from contextlib import contextmanager


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    """Get the best available device.
    
    Args:
        device: Device preference ('auto', 'cpu', 'cuda', 'mps')
        
    Returns:
        PyTorch device
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None
) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def create_directories(directories: List[str]) -> None:
    """Create directories if they don't exist.
    
    Args:
        directories: List of directory paths to create
    """
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def get_model_size(model: torch.nn.Module) -> float:
    """Get model size in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


@contextmanager
def timer(name: str = "Operation"):
    """Context manager for timing operations.
    
    Args:
        name: Name of the operation being timed
    """
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"{name} took {end_time - start_time:.4f} seconds")


def save_results(
    results: Dict[str, Any],
    filepath: str,
    format: str = "json"
) -> None:
    """Save results to file.
    
    Args:
        results: Results dictionary
        filepath: Path to save file
        format: File format ('json', 'yaml')
    """
    if format == "json":
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    elif format == "yaml":
        with open(filepath, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from file.
    
    Args:
        filepath: Path to results file
        
    Returns:
        Results dictionary
    """
    if filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            return json.load(f)
    elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def format_number(number: float, precision: int = 4) -> str:
    """Format number with appropriate precision.
    
    Args:
        number: Number to format
        precision: Number of decimal places
        
    Returns:
        Formatted number string
    """
    if abs(number) < 1e-6:
        return f"0.{'0' * precision}"
    elif abs(number) < 1e-3:
        return f"{number:.{precision}e}"
    else:
        return f"{number:.{precision}f}"


def calculate_compression_ratio(
    original_size: float,
    compressed_size: float
) -> float:
    """Calculate compression ratio.
    
    Args:
        original_size: Original size
        compressed_size: Compressed size
        
    Returns:
        Compression ratio
    """
    if compressed_size == 0:
        return float('inf')
    return original_size / compressed_size


def estimate_energy_consumption(
    model_size: float,
    inference_time: float,
    power_per_mb: float = 0.1  # Watts per MB
) -> float:
    """Estimate energy consumption for inference.
    
    Args:
        model_size: Model size in MB
        inference_time: Inference time in seconds
        power_per_mb: Power consumption per MB in Watts
        
    Returns:
        Energy consumption in Joules
    """
    power = model_size * power_per_mb
    energy = power * inference_time
    return energy


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
        
    Returns:
        True if valid, False otherwise
    """
    for key in required_keys:
        if key not in config:
            return False
    return True


def get_system_info() -> Dict[str, Any]:
    """Get system information.
    
    Returns:
        Dictionary of system information
    """
    import platform
    import psutil
    
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'mps_available': torch.backends.mps.is_available(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_name'] = torch.cuda.get_device_name(0)
        
    return info


def create_experiment_id() -> str:
    """Create unique experiment ID.
    
    Returns:
        Unique experiment ID string
    """
    import uuid
    return str(uuid.uuid4())[:8]


def log_system_info(logger: logging.Logger) -> None:
    """Log system information.
    
    Args:
        logger: Logger instance
    """
    info = get_system_info()
    logger.info("System Information:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")


def cleanup_gpu_memory() -> None:
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage.
    
    Returns:
        Dictionary of memory usage statistics
    """
    import psutil
    
    memory = psutil.virtual_memory()
    gpu_memory = {}
    
    if torch.cuda.is_available():
        gpu_memory = {
            'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
            'gpu_memory_max': torch.cuda.max_memory_allocated() / 1024**3     # GB
        }
    
    return {
        'cpu_memory_used': memory.used / 1024**3,      # GB
        'cpu_memory_available': memory.available / 1024**3,  # GB
        'cpu_memory_percent': memory.percent,
        **gpu_memory
    }
