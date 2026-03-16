"""
Split Learning Implementation for Edge AI & IoT

A modern implementation of split learning for collaborative training
between edge devices and cloud servers, optimized for privacy-preserving
distributed learning scenarios.
"""

__version__ = "0.1.0"
__author__ = "Edge AI Research Team"
__email__ = "research@example.com"

from .models.split_learning import (
    SplitLearningClient,
    SplitLearningServer,
    QuantizedSplitClient,
    PrunedSplitClient,
    create_split_models
)

from .data.datasets import (
    MNISTDataLoader,
    SyntheticDataLoader,
    StreamingDataLoader,
    EdgeDataLoader,
    create_data_loader
)

from .training.trainer import (
    SplitLearningTrainer,
    FederatedSplitTrainer,
    create_trainer
)

from .compression.compression import (
    QuantizationCompression,
    PruningCompression,
    DistillationCompression,
    SplitLearningCompression,
    EdgeOptimizedCompression,
    create_compression
)

from .evaluation.evaluator import (
    AccuracyEvaluator,
    EfficiencyEvaluator,
    SplitLearningEvaluator,
    EdgePerformanceEvaluator,
    ComprehensiveEvaluator,
    create_evaluator
)

from .utils.helpers import (
    set_seed,
    get_device,
    setup_logging,
    load_config,
    save_config,
    create_directories,
    count_parameters,
    get_model_size,
    timer,
    save_results,
    load_results,
    format_number,
    calculate_compression_ratio,
    estimate_energy_consumption,
    validate_config,
    get_system_info,
    create_experiment_id,
    log_system_info,
    cleanup_gpu_memory,
    get_memory_usage
)

__all__ = [
    # Models
    "SplitLearningClient",
    "SplitLearningServer", 
    "QuantizedSplitClient",
    "PrunedSplitClient",
    "create_split_models",
    
    # Data
    "MNISTDataLoader",
    "SyntheticDataLoader",
    "StreamingDataLoader",
    "EdgeDataLoader",
    "create_data_loader",
    
    # Training
    "SplitLearningTrainer",
    "FederatedSplitTrainer",
    "create_trainer",
    
    # Compression
    "QuantizationCompression",
    "PruningCompression",
    "DistillationCompression",
    "SplitLearningCompression",
    "EdgeOptimizedCompression",
    "create_compression",
    
    # Evaluation
    "AccuracyEvaluator",
    "EfficiencyEvaluator",
    "SplitLearningEvaluator",
    "EdgePerformanceEvaluator",
    "ComprehensiveEvaluator",
    "create_evaluator",
    
    # Utilities
    "set_seed",
    "get_device",
    "setup_logging",
    "load_config",
    "save_config",
    "create_directories",
    "count_parameters",
    "get_model_size",
    "timer",
    "save_results",
    "load_results",
    "format_number",
    "calculate_compression_ratio",
    "estimate_energy_consumption",
    "validate_config",
    "get_system_info",
    "create_experiment_id",
    "log_system_info",
    "cleanup_gpu_memory",
    "get_memory_usage"
]
