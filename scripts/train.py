#!/usr/bin/env python3
"""
Main training script for split learning implementation.

This script provides a command-line interface for training split learning models
with various configurations and compression techniques.
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import yaml
import logging
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.split_learning import create_split_models
from src.data.datasets import create_data_loader
from src.training.trainer import create_trainer
from src.compression.compression import SplitLearningCompression
from src.evaluation.evaluator import create_evaluator
from src.utils.helpers import (
    set_seed, get_device, setup_logging, load_config, 
    create_directories, log_system_info, timer
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train split learning models for edge AI & IoT"
    )
    
    # Model arguments
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="mnist",
        choices=["mnist", "synthetic", "streaming", "edge"],
        help="Dataset to use for training"
    )
    
    parser.add_argument(
        "--cut-layer",
        type=int,
        default=2,
        help="Number of layers in client model"
    )
    
    parser.add_argument(
        "--compression",
        type=str,
        default="none",
        choices=["none", "quantization", "pruning", "distillation"],
        help="Type of model compression"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    
    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run on"
    )
    
    # Configuration arguments
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--device-config",
        type=str,
        default="configs/device_config.yaml",
        help="Path to device configuration file"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save trained models"
    )
    
    # Logging arguments
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path"
    )
    
    # Reproducibility arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()


def load_configurations(args: argparse.Namespace) -> Dict[str, Any]:
    """Load and merge configurations."""
    # Load main config
    config = load_config(args.config)
    
    # Load device config
    device_config = load_config(args.device_config)
    
    # Override with command line arguments
    config["model"]["cut_layer"] = args.cut_layer
    config["model"]["compression_type"] = args.compression
    config["training"]["epochs"] = args.epochs
    config["training"]["batch_size"] = args.batch_size
    config["training"]["learning_rate"] = args.learning_rate
    config["data"]["dataset_name"] = args.dataset
    config["device"]["device"] = args.device
    config["logging"]["level"] = args.log_level
    config["logging"]["log_file"] = args.log_file
    
    return config, device_config


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configurations
    config, device_config = load_configurations(args)
    
    # Setup logging
    logger = setup_logging(
        log_level=config["logging"]["level"],
        log_file=config["logging"]["log_file"]
    )
    
    # Log system information
    log_system_info(logger)
    
    # Set random seed
    set_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")
    
    # Create output directories
    output_dir = Path(args.output_dir)
    create_directories([
        str(output_dir),
        str(output_dir / "models"),
        str(output_dir / "results"),
        str(output_dir / "assets")
    ])
    
    # Get device
    device = get_device(config["device"]["device"])
    logger.info(f"Using device: {device}")
    
    try:
        # Initialize models
        logger.info("Initializing models...")
        client, server = create_split_models(
            input_shape=tuple(config["model"]["input_shape"]),
            cut_layer=config["model"]["cut_layer"],
            num_classes=config["model"]["num_classes"],
            device=str(device),
            compression_type=config["model"]["compression_type"] if config["model"]["compression_type"] != "none" else None
        )
        
        logger.info(f"Client model parameters: {sum(p.numel() for p in client.get_parameters())}")
        logger.info(f"Server model parameters: {sum(p.numel() for p in server.get_parameters())}")
        
        # Load data
        logger.info("Loading data...")
        data_loader = create_data_loader(
            dataset_name=config["data"]["dataset_name"],
            batch_size=config["training"]["batch_size"],
            device=str(device),
            data_dir=config["data"]["data_dir"],
            num_workers=config["data"]["num_workers"]
        )
        train_loader, test_loader = data_loader.load_data()
        
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Test samples: {len(test_loader.dataset)}")
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = create_trainer(
            trainer_type="split",
            client=client,
            server=server,
            device=str(device),
            learning_rate=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"]
        )
        
        # Training
        logger.info("Starting training...")
        with timer("Training"):
            history = trainer.train(
                train_loader=train_loader,
                test_loader=test_loader,
                epochs=config["training"]["epochs"],
                verbose=True
            )
        
        # Evaluation
        logger.info("Evaluating model...")
        evaluator = create_evaluator(
            evaluator_type="comprehensive",
            client=client,
            server=server,
            device=str(device)
        )
        
        with timer("Evaluation"):
            metrics = evaluator.evaluate(test_loader)
        
        # Log results
        logger.info("Training Results:")
        logger.info(f"  Final Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Final Loss: {metrics['loss']:.4f}")
        logger.info(f"  Latency: {metrics['latency']:.4f}s")
        logger.info(f"  Communication Cost: {metrics['communication_cost']:.2f} bytes")
        
        # Save results
        results = {
            "config": config,
            "history": history,
            "metrics": metrics,
            "args": vars(args)
        }
        
        results_file = output_dir / "results" / "training_results.yaml"
        with open(results_file, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        logger.info(f"Results saved to {results_file}")
        
        # Save models
        if args.save_model:
            model_dir = output_dir / "models"
            torch.save(client.model.state_dict(), model_dir / "client_model.pth")
            torch.save(server.model.state_dict(), model_dir / "server_model.pth")
            logger.info(f"Models saved to {model_dir}")
        
        # Compression analysis
        if config["model"]["compression_type"] != "none":
            logger.info("Analyzing compression...")
            compression_analyzer = SplitLearningCompression(client, server, str(device))
            compression_metrics = compression_analyzer.get_compression_metrics()
            
            logger.info("Compression Metrics:")
            for key, value in compression_metrics.items():
                logger.info(f"  {key}: {value}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
