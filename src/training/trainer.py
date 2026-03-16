"""
Training utilities for split learning.

This module implements training loops, optimization strategies, and
communication protocols for split learning scenarios.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from abc import ABC, abstractmethod
import logging

from ..models.split_learning import SplitLearningClient, SplitLearningServer
from ..data.datasets import BaseDataLoader


class BaseTrainer(ABC):
    """Abstract base class for trainers."""
    
    def __init__(
        self,
        client: SplitLearningClient,
        server: SplitLearningServer,
        device: str = "cpu",
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4
    ) -> None:
        """Initialize trainer.
        
        Args:
            client: Client-side model
            server: Server-side model
            device: Device to train on
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
        """
        self.client = client
        self.server = server
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Move models to device
        self.client.to_device()
        self.server.to_device()
        
        # Initialize optimizers
        self.client_optimizer = optim.Adam(
            self.client.get_parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.server_optimizer = optim.Adam(
            self.server.get_parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics tracking
        self.train_losses = []
        self.train_accuracies = []
        self.communication_costs = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    @abstractmethod
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        pass
        
    def train(
        self,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Train the split learning model.
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader for evaluation
            epochs: Number of training epochs
            verbose: Whether to print training progress
            
        Returns:
            Dictionary of training history
        """
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
            'communication_cost': []
        }
        
        for epoch in range(epochs):
            # Train one epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Evaluate on test set if provided
            test_metrics = {}
            if test_loader is not None:
                test_metrics = self.evaluate(test_loader)
                
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            history['communication_cost'].append(train_metrics['communication_cost'])
            
            if test_metrics:
                history['test_loss'].append(test_metrics['loss'])
                history['test_accuracy'].append(test_metrics['accuracy'])
                
            # Log progress
            if verbose:
                self.logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.4f}"
                )
                if test_metrics:
                    self.logger.info(
                        f"Test Loss: {test_metrics['loss']:.4f}, "
                        f"Test Acc: {test_metrics['accuracy']:.4f}"
                    )
                    
        return history
        
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.client.model.eval()
        self.server.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                client_output = self.client.forward(data)
                output = self.server.forward(client_output)
                
                # Calculate loss
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                # Calculate accuracy
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }


class SplitLearningTrainer(BaseTrainer):
    """Trainer for split learning with communication simulation."""
    
    def __init__(
        self,
        client: SplitLearningClient,
        server: SplitLearningServer,
        device: str = "cpu",
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        simulate_communication: bool = True,
        communication_delay: float = 0.01
    ) -> None:
        """Initialize split learning trainer.
        
        Args:
            client: Client-side model
            server: Server-side model
            device: Device to train on
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            simulate_communication: Whether to simulate communication delays
            communication_delay: Simulated communication delay in seconds
        """
        super().__init__(client, server, device, learning_rate, weight_decay)
        self.simulate_communication = simulate_communication
        self.communication_delay = communication_delay
        
    def _simulate_communication(self, data_size: int) -> float:
        """Simulate communication cost and delay.
        
        Args:
            data_size: Size of data being transmitted
            
        Returns:
            Communication cost in bytes
        """
        if self.simulate_communication:
            # Simulate network delay
            time.sleep(self.communication_delay)
            
        # Calculate communication cost (bytes)
        communication_cost = data_size * 4  # Assuming float32
        
        return communication_cost
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch with split learning.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        self.client.model.train()
        self.server.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        total_communication_cost = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.client_optimizer.zero_grad()
            self.server_optimizer.zero_grad()
            
            # Forward pass through client
            client_output = self.client.forward(data)
            
            # Simulate communication
            communication_cost = self._simulate_communication(
                client_output.numel()
            )
            total_communication_cost += communication_cost
            
            # Forward pass through server
            output = self.server.forward(client_output)
            
            # Calculate loss
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            self.client_optimizer.step()
            self.server_optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        avg_communication_cost = total_communication_cost / len(train_loader)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'communication_cost': avg_communication_cost
        }


class FederatedSplitTrainer(BaseTrainer):
    """Trainer for federated split learning with multiple clients."""
    
    def __init__(
        self,
        clients: List[SplitLearningClient],
        server: SplitLearningServer,
        device: str = "cpu",
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        aggregation_rounds: int = 10,
        clients_per_round: int = 5
    ) -> None:
        """Initialize federated split learning trainer.
        
        Args:
            clients: List of client models
            server: Server model
            device: Device to train on
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            aggregation_rounds: Number of aggregation rounds
            clients_per_round: Number of clients per round
        """
        super().__init__(clients[0], server, device, learning_rate, weight_decay)
        self.clients = clients
        self.aggregation_rounds = aggregation_rounds
        self.clients_per_round = clients_per_round
        
        # Initialize client optimizers
        self.client_optimizers = [
            optim.Adam(client.get_parameters(), lr=learning_rate, weight_decay=weight_decay)
            for client in self.clients
        ]
        
    def _aggregate_client_weights(self) -> None:
        """Aggregate weights from all clients."""
        # Simple FedAvg aggregation
        aggregated_weights = {}
        
        # Collect weights from all clients
        for client in self.clients:
            for name, param in client.model.named_parameters():
                if name not in aggregated_weights:
                    aggregated_weights[name] = []
                aggregated_weights[name].append(param.data.clone())
                
        # Average the weights
        for name, weights_list in aggregated_weights.items():
            avg_weight = torch.stack(weights_list).mean(dim=0)
            
            # Update all clients with averaged weights
            for client in self.clients:
                for param_name, param in client.model.named_parameters():
                    if param_name == name:
                        param.data = avg_weight.clone()
                        break
                        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch with federated learning.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        # Select random clients for this round
        selected_clients = np.random.choice(
            len(self.clients),
            min(self.clients_per_round, len(self.clients)),
            replace=False
        )
        
        total_loss = 0.0
        total_accuracy = 0.0
        total_communication_cost = 0.0
        
        # Train selected clients
        for client_idx in selected_clients:
            client = self.clients[client_idx]
            optimizer = self.client_optimizers[client_idx]
            
            client.model.train()
            self.server.model.train()
            
            batch_loss = 0.0
            batch_correct = 0
            batch_total = 0
            batch_communication_cost = 0
            
            # Train client on a subset of data
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= len(train_loader) // len(selected_clients):
                    break
                    
                data, target = data.to(self.device), target.to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                self.server_optimizer.zero_grad()
                
                # Forward pass
                client_output = client.forward(data)
                
                # Simulate communication
                communication_cost = client_output.numel() * 4
                batch_communication_cost += communication_cost
                
                # Forward pass through server
                output = self.server.forward(client_output)
                
                # Calculate loss
                loss = self.criterion(output, target)
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                optimizer.step()
                self.server_optimizer.step()
                
                # Update metrics
                batch_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                batch_correct += pred.eq(target.view_as(pred)).sum().item()
                batch_total += target.size(0)
                
            # Aggregate metrics
            total_loss += batch_loss / (batch_idx + 1)
            total_accuracy += batch_correct / batch_total
            total_communication_cost += batch_communication_cost
            
        # Aggregate client weights
        self._aggregate_client_weights()
        
        # Average metrics across clients
        avg_loss = total_loss / len(selected_clients)
        avg_accuracy = total_accuracy / len(selected_clients)
        avg_communication_cost = total_communication_cost / len(selected_clients)
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'communication_cost': avg_communication_cost
        }


def create_trainer(
    trainer_type: str = "split",
    **kwargs
) -> BaseTrainer:
    """Factory function to create trainers.
    
    Args:
        trainer_type: Type of trainer ('split', 'federated')
        **kwargs: Additional arguments for trainer
        
    Returns:
        Configured trainer
    """
    if trainer_type == "split":
        return SplitLearningTrainer(**kwargs)
    elif trainer_type == "federated":
        return FederatedSplitTrainer(**kwargs)
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")
