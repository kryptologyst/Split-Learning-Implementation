"""
Evaluation metrics and performance analysis for split learning.

This module provides comprehensive evaluation including accuracy metrics,
efficiency metrics, and edge-specific performance analysis.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import numpy as np
import time
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass

from ..models.split_learning import SplitLearningClient, SplitLearningServer
from ..training.trainer import BaseTrainer


@dataclass
class PerformanceMetrics:
    """Data class for performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    latency_p50: float
    latency_p95: float
    throughput: float
    memory_usage: float
    communication_cost: float
    energy_consumption: float


class BaseEvaluator(ABC):
    """Abstract base class for evaluators."""
    
    def __init__(self, device: str = "cpu") -> None:
        """Initialize evaluator.
        
        Args:
            device: Device to run evaluation on
        """
        self.device = torch.device(device)
        self.logger = logging.getLogger(__name__)
        
    @abstractmethod
    def evaluate(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Evaluate model performance.
        
        Args:
            model: Model to evaluate
            data_loader: Data loader for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass


class AccuracyEvaluator(BaseEvaluator):
    """Evaluator for accuracy-based metrics."""
    
    def __init__(self, device: str = "cpu", num_classes: int = 10) -> None:
        """Initialize accuracy evaluator.
        
        Args:
            device: Device to run on
            num_classes: Number of classes
        """
        super().__init__(device)
        self.num_classes = num_classes
        
    def evaluate(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Evaluate model accuracy.
        
        Args:
            model: Model to evaluate
            data_loader: Data loader for evaluation
            
        Returns:
            Dictionary of accuracy metrics
        """
        model.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in data_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = model(data)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        # Calculate metrics
        accuracy = self._calculate_accuracy(all_predictions, all_labels)
        precision = self._calculate_precision(all_predictions, all_labels)
        recall = self._calculate_recall(all_predictions, all_labels)
        f1_score = self._calculate_f1_score(all_predictions, all_labels)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
        
    def _calculate_accuracy(self, predictions: List[int], labels: List[int]) -> float:
        """Calculate accuracy."""
        correct = sum(p == l for p, l in zip(predictions, labels))
        return correct / len(predictions)
        
    def _calculate_precision(self, predictions: List[int], labels: List[int]) -> float:
        """Calculate precision."""
        from sklearn.metrics import precision_score
        return precision_score(labels, predictions, average='weighted', zero_division=0)
        
    def _calculate_recall(self, predictions: List[int], labels: List[int]) -> float:
        """Calculate recall."""
        from sklearn.metrics import recall_score
        return recall_score(labels, predictions, average='weighted', zero_division=0)
        
    def _calculate_f1_score(self, predictions: List[int], labels: List[int]) -> float:
        """Calculate F1 score."""
        from sklearn.metrics import f1_score
        return f1_score(labels, predictions, average='weighted', zero_division=0)


class EfficiencyEvaluator(BaseEvaluator):
    """Evaluator for efficiency metrics."""
    
    def __init__(self, device: str = "cpu") -> None:
        """Initialize efficiency evaluator.
        
        Args:
            device: Device to run on
        """
        super().__init__(device)
        
    def evaluate(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        num_warmup: int = 10,
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """Evaluate model efficiency.
        
        Args:
            model: Model to evaluate
            data_loader: Data loader for evaluation
            num_warmup: Number of warmup iterations
            num_iterations: Number of timing iterations
            
        Returns:
            Dictionary of efficiency metrics
        """
        model.eval()
        
        # Get sample data
        sample_data = next(iter(data_loader))[0].to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(sample_data)
                
        # Timing measurements
        latencies = []
        memory_usage = []
        
        for i in range(num_iterations):
            # Measure memory before
            memory_before = psutil.virtual_memory().used
            
            # Time inference
            start_time = time.time()
            with torch.no_grad():
                _ = model(sample_data)
            end_time = time.time()
            
            # Measure memory after
            memory_after = psutil.virtual_memory().used
            
            latencies.append(end_time - start_time)
            memory_usage.append(memory_after - memory_before)
            
        # Calculate metrics
        latency_p50 = np.percentile(latencies, 50)
        latency_p95 = np.percentile(latencies, 95)
        throughput = 1.0 / np.mean(latencies)
        avg_memory_usage = np.mean(memory_usage)
        
        return {
            'latency_p50': latency_p50,
            'latency_p95': latency_p95,
            'throughput': throughput,
            'memory_usage': avg_memory_usage
        }


class SplitLearningEvaluator(BaseEvaluator):
    """Evaluator specifically for split learning scenarios."""
    
    def __init__(
        self,
        client: SplitLearningClient,
        server: SplitLearningServer,
        device: str = "cpu"
    ) -> None:
        """Initialize split learning evaluator.
        
        Args:
            client: Client model
            server: Server model
            device: Device to run on
        """
        super().__init__(device)
        self.client = client
        self.server = server
        
    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
        simulate_communication: bool = True
    ) -> Dict[str, float]:
        """Evaluate split learning performance.
        
        Args:
            data_loader: Data loader for evaluation
            simulate_communication: Whether to simulate communication
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.client.model.eval()
        self.server.model.eval()
        
        all_predictions = []
        all_labels = []
        latencies = []
        communication_costs = []
        
        with torch.no_grad():
            for data, labels in data_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                
                # Time client inference
                start_time = time.time()
                client_output = self.client.forward(data)
                client_time = time.time() - start_time
                
                # Simulate communication
                communication_cost = 0
                if simulate_communication:
                    communication_cost = client_output.numel() * 4  # float32 bytes
                    time.sleep(0.01)  # Simulate network delay
                    
                # Time server inference
                start_time = time.time()
                server_output = self.server.forward(client_output)
                server_time = time.time() - start_time
                
                # Collect metrics
                predictions = torch.argmax(server_output, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                latencies.append(client_time + server_time)
                communication_costs.append(communication_cost)
                
        # Calculate metrics
        accuracy = self._calculate_accuracy(all_predictions, all_labels)
        avg_latency = np.mean(latencies)
        avg_communication_cost = np.mean(communication_costs)
        
        return {
            'accuracy': accuracy,
            'latency': avg_latency,
            'communication_cost': avg_communication_cost,
            'client_latency': np.mean([l * 0.3 for l in latencies]),  # Estimate
            'server_latency': np.mean([l * 0.7 for l in latencies])  # Estimate
        }
        
    def _calculate_accuracy(self, predictions: List[int], labels: List[int]) -> float:
        """Calculate accuracy."""
        correct = sum(p == l for p, l in zip(predictions, labels))
        return correct / len(predictions)


class EdgePerformanceEvaluator(BaseEvaluator):
    """Evaluator for edge-specific performance metrics."""
    
    def __init__(self, device: str = "cpu") -> None:
        """Initialize edge performance evaluator.
        
        Args:
            device: Device to run on
        """
        super().__init__(device)
        
    def evaluate(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        power_limit: float = 5.0  # Watts
    ) -> Dict[str, float]:
        """Evaluate edge performance.
        
        Args:
            model: Model to evaluate
            data_loader: Data loader for evaluation
            power_limit: Power limit in watts
            
        Returns:
            Dictionary of edge performance metrics
        """
        model.eval()
        
        # Get sample data
        sample_data = next(iter(data_loader))[0].to(self.device)
        
        # Measure power consumption (simulated)
        power_consumption = self._estimate_power_consumption(model, sample_data)
        
        # Measure thermal performance (simulated)
        thermal_performance = self._estimate_thermal_performance(model, sample_data)
        
        # Measure battery life impact (simulated)
        battery_impact = self._estimate_battery_impact(power_consumption)
        
        return {
            'power_consumption': power_consumption,
            'thermal_performance': thermal_performance,
            'battery_impact': battery_impact,
            'power_efficiency': power_limit / power_consumption if power_consumption > 0 else 0
        }
        
    def _estimate_power_consumption(self, model: nn.Module, sample_data: torch.Tensor) -> float:
        """Estimate power consumption."""
        # Simple estimation based on model size and operations
        num_params = sum(p.numel() for p in model.parameters())
        num_ops = sample_data.numel() * num_params  # Rough estimate
        
        # Power consumption in watts (simulated)
        power_per_op = 1e-9  # 1 nW per operation
        return num_ops * power_per_op
        
    def _estimate_thermal_performance(self, model: nn.Module, sample_data: torch.Tensor) -> float:
        """Estimate thermal performance."""
        # Thermal performance as temperature rise in Celsius
        power_consumption = self._estimate_power_consumption(model, sample_data)
        thermal_resistance = 0.5  # C/W
        return power_consumption * thermal_resistance
        
    def _estimate_battery_impact(self, power_consumption: float) -> float:
        """Estimate battery life impact."""
        # Battery capacity in mAh (typical smartphone)
        battery_capacity = 3000  # mAh
        voltage = 3.7  # V
        
        # Battery life in hours
        battery_life = (battery_capacity * voltage) / (power_consumption * 1000)
        return battery_life


class ComprehensiveEvaluator:
    """Comprehensive evaluator combining all metrics."""
    
    def __init__(
        self,
        client: SplitLearningClient,
        server: SplitLearningServer,
        device: str = "cpu"
    ) -> None:
        """Initialize comprehensive evaluator.
        
        Args:
            client: Client model
            server: Server model
            device: Device to run on
        """
        self.client = client
        self.server = server
        self.device = device
        
        # Initialize sub-evaluators
        self.accuracy_evaluator = AccuracyEvaluator(device)
        self.efficiency_evaluator = EfficiencyEvaluator(device)
        self.split_evaluator = SplitLearningEvaluator(client, server, device)
        self.edge_evaluator = EdgePerformanceEvaluator(device)
        
    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
        include_edge_metrics: bool = True
    ) -> Dict[str, float]:
        """Comprehensive evaluation.
        
        Args:
            data_loader: Data loader for evaluation
            include_edge_metrics: Whether to include edge-specific metrics
            
        Returns:
            Dictionary of all evaluation metrics
        """
        metrics = {}
        
        # Accuracy metrics
        accuracy_metrics = self.accuracy_evaluator.evaluate(self.server.model, data_loader)
        metrics.update(accuracy_metrics)
        
        # Efficiency metrics
        efficiency_metrics = self.efficiency_evaluator.evaluate(self.server.model, data_loader)
        metrics.update(efficiency_metrics)
        
        # Split learning specific metrics
        split_metrics = self.split_evaluator.evaluate(data_loader)
        metrics.update(split_metrics)
        
        # Edge performance metrics
        if include_edge_metrics:
            edge_metrics = self.edge_evaluator.evaluate(self.client.model, data_loader)
            metrics.update(edge_metrics)
            
        return metrics
        
    def create_leaderboard(
        self,
        results: List[Dict[str, float]],
        model_names: List[str]
    ) -> pd.DataFrame:
        """Create performance leaderboard.
        
        Args:
            results: List of evaluation results
            model_names: Names of models
            
        Returns:
            Leaderboard DataFrame
        """
        import pandas as pd
        
        leaderboard_data = []
        for i, (result, name) in enumerate(zip(results, model_names)):
            row = {'Model': name}
            row.update(result)
            leaderboard_data.append(row)
            
        df = pd.DataFrame(leaderboard_data)
        
        # Sort by accuracy (primary metric)
        df = df.sort_values('accuracy', ascending=False)
        
        return df
        
    def plot_performance_comparison(
        self,
        results: List[Dict[str, float]],
        model_names: List[str],
        save_path: Optional[str] = None
    ) -> None:
        """Plot performance comparison.
        
        Args:
            results: List of evaluation results
            model_names: Names of models
            save_path: Path to save plot
        """
        import pandas as pd
        
        # Create DataFrame
        df = pd.DataFrame(results, index=model_names)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy comparison
        df['accuracy'].plot(kind='bar', ax=axes[0, 0], title='Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        
        # Latency comparison
        df['latency'].plot(kind='bar', ax=axes[0, 1], title='Latency')
        axes[0, 1].set_ylabel('Latency (s)')
        
        # Communication cost comparison
        df['communication_cost'].plot(kind='bar', ax=axes[1, 0], title='Communication Cost')
        axes[1, 0].set_ylabel('Bytes')
        
        # Memory usage comparison
        df['memory_usage'].plot(kind='bar', ax=axes[1, 1], title='Memory Usage')
        axes[1, 1].set_ylabel('Bytes')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def create_evaluator(
    evaluator_type: str = "comprehensive",
    **kwargs
) -> BaseEvaluator:
    """Factory function to create evaluators.
    
    Args:
        evaluator_type: Type of evaluator
        **kwargs: Additional evaluator parameters
        
    Returns:
        Configured evaluator
    """
    if evaluator_type == "accuracy":
        return AccuracyEvaluator(**kwargs)
    elif evaluator_type == "efficiency":
        return EfficiencyEvaluator(**kwargs)
    elif evaluator_type == "split":
        return SplitLearningEvaluator(**kwargs)
    elif evaluator_type == "edge":
        return EdgePerformanceEvaluator(**kwargs)
    elif evaluator_type == "comprehensive":
        return ComprehensiveEvaluator(**kwargs)
    else:
        raise ValueError(f"Unknown evaluator type: {evaluator_type}")
