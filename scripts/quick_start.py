#!/usr/bin/env python3
"""
Quick start script for split learning implementation.

This script provides a simple way to run the split learning demo
with default configurations.
"""

import sys
import os
from pathlib import Path
import torch
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.split_learning import create_split_models
from src.data.datasets import create_data_loader
from src.training.trainer import create_trainer
from src.evaluation.evaluator import create_evaluator
from src.utils.helpers import set_seed, get_device, setup_logging


def main():
    """Quick start demo."""
    print("🚀 Split Learning Implementation - Quick Start")
    print("=" * 50)
    
    # Setup
    set_seed(42)
    device = get_device("auto")
    logger = setup_logging("INFO")
    
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    try:
        # Create models
        print("\n📦 Creating split learning models...")
        client, server = create_split_models(
            input_shape=(28, 28, 1),
            cut_layer=2,
            num_classes=10,
            device=str(device)
        )
        
        print(f"Client parameters: {sum(p.numel() for p in client.get_parameters()):,}")
        print(f"Server parameters: {sum(p.numel() for p in server.get_parameters()):,}")
        
        # Load data
        print("\n📊 Loading MNIST dataset...")
        data_loader = create_data_loader(
            dataset_name="mnist",
            batch_size=64,
            device=str(device)
        )
        train_loader, test_loader = data_loader.load_data()
        
        print(f"Training samples: {len(train_loader.dataset):,}")
        print(f"Test samples: {len(test_loader.dataset):,}")
        
        # Initialize trainer
        print("\n🎯 Initializing trainer...")
        trainer = create_trainer(
            trainer_type="split",
            client=client,
            server=server,
            device=str(device),
            learning_rate=0.001
        )
        
        # Train model
        print("\n🏋️ Training model (5 epochs)...")
        history = trainer.train(
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=5,
            verbose=True
        )
        
        # Evaluate model
        print("\n📈 Evaluating model...")
        evaluator = create_evaluator(
            evaluator_type="comprehensive",
            client=client,
            server=server,
            device=str(device)
        )
        
        metrics = evaluator.evaluate(test_loader)
        
        # Display results
        print("\n✅ Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  Latency: {metrics['latency']:.4f}s")
        print(f"  Communication Cost: {metrics['communication_cost']:.2f} bytes")
        
        print("\n🎉 Quick start completed successfully!")
        print("\nTo run the interactive demo:")
        print("  streamlit run demo/app.py")
        
        print("\nTo run full training:")
        print("  python scripts/train.py --epochs 10 --compression quantization")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        logger.exception("Quick start failed")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
