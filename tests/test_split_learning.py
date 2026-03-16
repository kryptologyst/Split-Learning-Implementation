#!/usr/bin/env python3
"""
Test script for split learning implementation.

This script provides unit tests for the split learning components.
"""

import unittest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.split_learning import (
    SplitLearningClient, SplitLearningServer, 
    QuantizedSplitClient, PrunedSplitClient,
    create_split_models
)
from src.data.datasets import (
    MNISTDataLoader, SyntheticDataLoader,
    create_data_loader
)
from src.training.trainer import (
    SplitLearningTrainer, FederatedSplitTrainer,
    create_trainer
)
from src.compression.compression import (
    QuantizationCompression, PruningCompression,
    SplitLearningCompression
)
from src.evaluation.evaluator import (
    AccuracyEvaluator, EfficiencyEvaluator,
    SplitLearningEvaluator, create_evaluator
)
from src.utils.helpers import set_seed, get_device


class TestSplitLearningModels(unittest.TestCase):
    """Test cases for split learning models."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_seed(42)
        self.device = get_device("cpu")
        self.input_shape = (28, 28, 1)
        self.batch_size = 4
        
    def test_client_model_creation(self):
        """Test client model creation."""
        client = SplitLearningClient(
            input_shape=self.input_shape,
            cut_layer=2,
            device=str(self.device)
        )
        
        # Test model creation
        self.assertIsNotNone(client.model)
        self.assertEqual(client.cut_layer, 2)
        
        # Test forward pass
        x = torch.randn(self.batch_size, 1, 28, 28)
        output = client.forward(x)
        
        # Check output shape
        expected_shape = client.get_output_shape()
        self.assertEqual(output.shape[1:], expected_shape)
        
    def test_server_model_creation(self):
        """Test server model creation."""
        client_output_shape = (32, 7, 7)  # Example shape
        server = SplitLearningServer(
            input_shape=client_output_shape,
            num_classes=10,
            device=str(self.device)
        )
        
        # Test model creation
        self.assertIsNotNone(server.model)
        self.assertEqual(server.num_classes, 10)
        
        # Test forward pass
        x = torch.randn(self.batch_size, *client_output_shape)
        output = server.forward(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 10))
        
    def test_split_models_integration(self):
        """Test client-server integration."""
        client, server = create_split_models(
            input_shape=self.input_shape,
            cut_layer=2,
            device=str(self.device)
        )
        
        # Test end-to-end forward pass
        x = torch.randn(self.batch_size, 1, 28, 28)
        
        client_output = client.forward(x)
        server_output = server.forward(client_output)
        
        # Check output shape
        self.assertEqual(server_output.shape, (self.batch_size, 10))
        
    def test_quantized_client(self):
        """Test quantized client model."""
        client = QuantizedSplitClient(
            input_shape=self.input_shape,
            cut_layer=2,
            device=str(self.device)
        )
        
        # Test forward pass
        x = torch.randn(self.batch_size, 1, 28, 28)
        output = client.forward(x)
        
        # Check that output is produced
        self.assertIsNotNone(output)
        
    def test_pruned_client(self):
        """Test pruned client model."""
        client = PrunedSplitClient(
            input_shape=self.input_shape,
            cut_layer=2,
            device=str(self.device),
            sparsity=0.5
        )
        
        # Test forward pass
        x = torch.randn(self.batch_size, 1, 28, 28)
        output = client.forward(x)
        
        # Check that output is produced
        self.assertIsNotNone(output)


class TestDataLoaders(unittest.TestCase):
    """Test cases for data loaders."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_seed(42)
        self.device = get_device("cpu")
        
    def test_mnist_loader(self):
        """Test MNIST data loader."""
        loader = MNISTDataLoader(
            batch_size=32,
            device=str(self.device)
        )
        
        train_loader, test_loader = loader.load_data()
        
        # Check that loaders are created
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(test_loader)
        
        # Check batch
        batch = next(iter(train_loader))
        self.assertEqual(len(batch), 2)  # data, labels
        
    def test_synthetic_loader(self):
        """Test synthetic data loader."""
        loader = SyntheticDataLoader(
            num_samples=1000,
            batch_size=32,
            device=str(self.device)
        )
        
        train_loader, test_loader = loader.load_data()
        
        # Check that loaders are created
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(test_loader)
        
        # Check batch
        batch = next(iter(train_loader))
        self.assertEqual(len(batch), 2)  # data, labels
        
    def test_data_loader_factory(self):
        """Test data loader factory function."""
        loader = create_data_loader(
            dataset_name="synthetic",
            batch_size=32,
            device=str(self.device)
        )
        
        train_loader, test_loader = loader.load_data()
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(test_loader)


class TestTrainers(unittest.TestCase):
    """Test cases for trainers."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_seed(42)
        self.device = get_device("cpu")
        
        # Create models
        self.client, self.server = create_split_models(
            input_shape=(28, 28, 1),
            cut_layer=2,
            device=str(self.device)
        )
        
        # Create data loader
        self.data_loader = create_data_loader(
            dataset_name="synthetic",
            batch_size=32,
            device=str(self.device)
        )
        self.train_loader, self.test_loader = self.data_loader.load_data()
        
    def test_split_learning_trainer(self):
        """Test split learning trainer."""
        trainer = SplitLearningTrainer(
            client=self.client,
            server=self.server,
            device=str(self.device)
        )
        
        # Test training for one epoch
        history = trainer.train(
            train_loader=self.train_loader,
            test_loader=self.test_loader,
            epochs=1,
            verbose=False
        )
        
        # Check that history is created
        self.assertIn('train_loss', history)
        self.assertIn('train_accuracy', history)
        
    def test_trainer_factory(self):
        """Test trainer factory function."""
        trainer = create_trainer(
            trainer_type="split",
            client=self.client,
            server=self.server,
            device=str(self.device)
        )
        
        self.assertIsInstance(trainer, SplitLearningTrainer)


class TestCompression(unittest.TestCase):
    """Test cases for compression techniques."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_seed(42)
        self.device = get_device("cpu")
        
        # Create models
        self.client, self.server = create_split_models(
            input_shape=(28, 28, 1),
            cut_layer=2,
            device=str(self.device)
        )
        
    def test_quantization_compression(self):
        """Test quantization compression."""
        compressor = QuantizationCompression(
            model=self.client.model,
            device=str(self.device)
        )
        
        compressed_model = compressor.compress()
        compression_ratio = compressor.get_compression_ratio()
        
        # Check that compression is applied
        self.assertIsNotNone(compressed_model)
        self.assertGreater(compression_ratio, 1.0)
        
    def test_pruning_compression(self):
        """Test pruning compression."""
        compressor = PruningCompression(
            model=self.client.model,
            device=str(self.device),
            sparsity=0.5
        )
        
        compressed_model = compressor.compress()
        compression_ratio = compressor.get_compression_ratio()
        
        # Check that compression is applied
        self.assertIsNotNone(compressed_model)
        self.assertGreater(compression_ratio, 1.0)
        
    def test_split_learning_compression(self):
        """Test split learning compression."""
        compression_analyzer = SplitLearningCompression(
            client=self.client,
            server=self.server,
            device=str(self.device)
        )
        
        # Test client compression
        compressed_client = compression_analyzer.compress_client(
            compression_type="quantization"
        )
        self.assertIsNotNone(compressed_client)
        
        # Test server compression
        compressed_server = compression_analyzer.compress_server(
            compression_type="pruning"
        )
        self.assertIsNotNone(compressed_server)
        
        # Test metrics
        metrics = compression_analyzer.get_compression_metrics()
        self.assertIn('client_original_size', metrics)
        self.assertIn('server_original_size', metrics)


class TestEvaluators(unittest.TestCase):
    """Test cases for evaluators."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_seed(42)
        self.device = get_device("cpu")
        
        # Create models
        self.client, self.server = create_split_models(
            input_shape=(28, 28, 1),
            cut_layer=2,
            device=str(self.device)
        )
        
        # Create data loader
        self.data_loader = create_data_loader(
            dataset_name="synthetic",
            batch_size=32,
            device=str(self.device)
        )
        self.train_loader, self.test_loader = self.data_loader.load_data()
        
    def test_accuracy_evaluator(self):
        """Test accuracy evaluator."""
        evaluator = AccuracyEvaluator(device=str(self.device))
        
        metrics = evaluator.evaluate(self.server.model, self.test_loader)
        
        # Check that metrics are calculated
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        
    def test_efficiency_evaluator(self):
        """Test efficiency evaluator."""
        evaluator = EfficiencyEvaluator(device=str(self.device))
        
        metrics = evaluator.evaluate(
            self.server.model, 
            self.test_loader,
            num_iterations=10  # Reduced for testing
        )
        
        # Check that metrics are calculated
        self.assertIn('latency_p50', metrics)
        self.assertIn('latency_p95', metrics)
        self.assertIn('throughput', metrics)
        self.assertIn('memory_usage', metrics)
        
    def test_split_learning_evaluator(self):
        """Test split learning evaluator."""
        evaluator = SplitLearningEvaluator(
            client=self.client,
            server=self.server,
            device=str(self.device)
        )
        
        metrics = evaluator.evaluate(self.test_loader)
        
        # Check that metrics are calculated
        self.assertIn('accuracy', metrics)
        self.assertIn('latency', metrics)
        self.assertIn('communication_cost', metrics)
        
    def test_evaluator_factory(self):
        """Test evaluator factory function."""
        evaluator = create_evaluator(
            evaluator_type="accuracy",
            device=str(self.device)
        )
        
        self.assertIsInstance(evaluator, AccuracyEvaluator)


class TestUtilities(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_seed_setting(self):
        """Test random seed setting."""
        set_seed(42)
        
        # Generate random numbers
        torch_rand = torch.rand(1).item()
        np_rand = np.random.rand()
        
        # Reset seed and generate again
        set_seed(42)
        torch_rand2 = torch.rand(1).item()
        np_rand2 = np.random.rand()
        
        # Check reproducibility
        self.assertEqual(torch_rand, torch_rand2)
        self.assertEqual(np_rand, np_rand2)
        
    def test_device_detection(self):
        """Test device detection."""
        device = get_device("auto")
        self.assertIsInstance(device, torch.device)
        
        device = get_device("cpu")
        self.assertEqual(device.type, "cpu")


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
