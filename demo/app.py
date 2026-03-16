"""
Streamlit demo application for split learning.

This module provides an interactive web interface to demonstrate
split learning concepts and performance metrics.
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import pandas as pd
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.split_learning import create_split_models
from src.data.datasets import create_data_loader
from src.training.trainer import create_trainer
from src.compression.compression import SplitLearningCompression
from src.evaluation.evaluator import create_evaluator


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Split Learning Demo",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🧠 Split Learning Implementation Demo")
    st.markdown("**DISCLAIMER: This demo is for research and educational purposes only. NOT FOR SAFETY-CRITICAL DEPLOYMENT.**")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Model configuration
    st.sidebar.subheader("Model Settings")
    dataset_name = st.sidebar.selectbox(
        "Dataset",
        ["mnist", "synthetic"],
        help="Choose the dataset for training"
    )
    
    cut_layer = st.sidebar.slider(
        "Cut Layer",
        min_value=1,
        max_value=3,
        value=2,
        help="Number of layers in client model"
    )
    
    compression_type = st.sidebar.selectbox(
        "Compression Type",
        ["none", "quantization", "pruning"],
        help="Type of model compression"
    )
    
    # Training configuration
    st.sidebar.subheader("Training Settings")
    epochs = st.sidebar.slider(
        "Epochs",
        min_value=1,
        max_value=20,
        value=5,
        help="Number of training epochs"
    )
    
    batch_size = st.sidebar.slider(
        "Batch Size",
        min_value=16,
        max_value=128,
        value=64,
        help="Training batch size"
    )
    
    learning_rate = st.sidebar.slider(
        "Learning Rate",
        min_value=0.0001,
        max_value=0.01,
        value=0.001,
        step=0.0001,
        format="%.4f",
        help="Learning rate for optimization"
    )
    
    # Device configuration
    device = st.sidebar.selectbox(
        "Device",
        ["cpu", "cuda", "mps"],
        help="Device to run on"
    )
    
    # Main content
    if st.button("🚀 Start Training", type="primary"):
        run_training_demo(
            dataset_name=dataset_name,
            cut_layer=cut_layer,
            compression_type=compression_type,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device
        )
    
    # Show architecture diagram
    st.subheader("📊 Split Learning Architecture")
    show_architecture_diagram()
    
    # Show performance comparison
    st.subheader("📈 Performance Comparison")
    show_performance_comparison()


def run_training_demo(
    dataset_name: str,
    cut_layer: int,
    compression_type: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str
) -> None:
    """Run the training demo."""
    
    # Check device availability
    if device == "cuda" and not torch.cuda.is_available():
        st.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        st.warning("MPS not available, falling back to CPU")
        device = "cpu"
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize models
        status_text.text("Initializing models...")
        progress_bar.progress(10)
        
        client, server = create_split_models(
            input_shape=(28, 28, 1),
            cut_layer=cut_layer,
            num_classes=10,
            device=device,
            compression_type=compression_type if compression_type != "none" else None
        )
        
        # Load data
        status_text.text("Loading data...")
        progress_bar.progress(20)
        
        data_loader = create_data_loader(
            dataset_name=dataset_name,
            batch_size=batch_size,
            device=device
        )
        train_loader, test_loader = data_loader.load_data()
        
        # Initialize trainer
        status_text.text("Initializing trainer...")
        progress_bar.progress(30)
        
        trainer = create_trainer(
            trainer_type="split",
            client=client,
            server=server,
            device=device,
            learning_rate=learning_rate
        )
        
        # Training loop
        status_text.text("Training model...")
        progress_bar.progress(40)
        
        history = trainer.train(
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            verbose=False
        )
        
        # Evaluation
        status_text.text("Evaluating model...")
        progress_bar.progress(80)
        
        evaluator = create_evaluator(
            evaluator_type="comprehensive",
            client=client,
            server=server,
            device=device
        )
        
        metrics = evaluator.evaluate(test_loader)
        
        # Complete
        progress_bar.progress(100)
        status_text.text("Training completed!")
        
        # Display results
        display_training_results(history, metrics)
        
    except Exception as e:
        st.error(f"Error during training: {str(e)}")
        st.exception(e)


def display_training_results(history: Dict[str, List[float]], metrics: Dict[str, float]) -> None:
    """Display training results."""
    
    st.subheader("📊 Training Results")
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Final Accuracy",
            value=f"{metrics['accuracy']:.4f}",
            delta=f"{metrics['accuracy'] - history['train_accuracy'][0]:.4f}"
        )
    
    with col2:
        st.metric(
            label="Final Loss",
            value=f"{metrics['loss']:.4f}",
            delta=f"{metrics['loss'] - history['train_loss'][0]:.4f}"
        )
    
    with col3:
        st.metric(
            label="Latency (ms)",
            value=f"{metrics['latency'] * 1000:.2f}",
        )
    
    with col4:
        st.metric(
            label="Communication Cost (KB)",
            value=f"{metrics['communication_cost'] / 1024:.2f}",
        )
    
    # Training curves
    st.subheader("📈 Training Curves")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Training Loss', 'Training Accuracy', 'Test Loss', 'Test Accuracy'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    # Training loss
    fig.add_trace(
        go.Scatter(x=epochs_range, y=history['train_loss'], name='Train Loss', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Training accuracy
    fig.add_trace(
        go.Scatter(x=epochs_range, y=history['train_accuracy'], name='Train Accuracy', line=dict(color='green')),
        row=1, col=2
    )
    
    # Test loss
    if 'test_loss' in history:
        fig.add_trace(
            go.Scatter(x=epochs_range, y=history['test_loss'], name='Test Loss', line=dict(color='red')),
            row=2, col=1
        )
    
    # Test accuracy
    if 'test_accuracy' in history:
        fig.add_trace(
            go.Scatter(x=epochs_range, y=history['test_accuracy'], name='Test Accuracy', line=dict(color='orange')),
            row=2, col=2
        )
    
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics table
    st.subheader("📋 Performance Metrics")
    
    metrics_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Latency (ms)', 'Throughput (samples/s)', 'Memory Usage (MB)', 'Communication Cost (KB)'],
        'Value': [
            f"{metrics['accuracy']:.4f}",
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            f"{metrics['f1_score']:.4f}",
            f"{metrics['latency'] * 1000:.2f}",
            f"{metrics['throughput']:.2f}",
            f"{metrics['memory_usage'] / (1024 * 1024):.2f}",
            f"{metrics['communication_cost'] / 1024:.2f}"
        ]
    }
    
    df = pd.DataFrame(metrics_data)
    st.dataframe(df, use_container_width=True)


def show_architecture_diagram() -> None:
    """Show split learning architecture diagram."""
    
    # Create architecture diagram
    fig = go.Figure()
    
    # Client side
    fig.add_trace(go.Scatter(
        x=[1, 1, 1, 1],
        y=[1, 2, 3, 4],
        mode='markers+text',
        marker=dict(size=20, color='blue'),
        text=['Input', 'Conv2D', 'MaxPool', 'Cut Layer'],
        textposition='middle right',
        name='Client (Edge)'
    ))
    
    # Server side
    fig.add_trace(go.Scatter(
        x=[3, 3, 3, 3],
        y=[1, 2, 3, 4],
        mode='markers+text',
        marker=dict(size=20, color='red'),
        text=['Activations', 'Flatten', 'Dense', 'Output'],
        textposition='middle left',
        name='Server (Cloud)'
    ))
    
    # Communication arrows
    fig.add_annotation(
        x=2, y=2.5,
        text="Communication<br>Activations Only",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="green"
    )
    
    fig.update_layout(
        title="Split Learning Architecture",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=True,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def show_performance_comparison() -> None:
    """Show performance comparison charts."""
    
    # Sample performance data
    models = ['Baseline', 'Quantized', 'Pruned', 'Split Learning']
    accuracy = [0.95, 0.93, 0.92, 0.94]
    latency = [50, 25, 30, 40]
    memory = [100, 50, 60, 80]
    communication = [0, 0, 0, 20]
    
    # Create comparison chart
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accuracy', 'Latency (ms)', 'Memory Usage (MB)', 'Communication Cost (KB)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Accuracy
    fig.add_trace(
        go.Bar(x=models, y=accuracy, name='Accuracy', marker_color='green'),
        row=1, col=1
    )
    
    # Latency
    fig.add_trace(
        go.Bar(x=models, y=latency, name='Latency', marker_color='red'),
        row=1, col=2
    )
    
    # Memory
    fig.add_trace(
        go.Bar(x=models, y=memory, name='Memory', marker_color='blue'),
        row=2, col=1
    )
    
    # Communication
    fig.add_trace(
        go.Bar(x=models, y=communication, name='Communication', marker_color='orange'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
