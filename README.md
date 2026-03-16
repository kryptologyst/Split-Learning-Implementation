# Split Learning Implementation - Edge AI & IoT Project

**DISCLAIMER: This project is for research and educational purposes only. NOT FOR SAFETY-CRITICAL DEPLOYMENT.**

## Overview

This project implements a modern split learning framework for Edge AI & IoT applications. Split learning is a collaborative training method where a neural network is split between a client (edge device) and a server (cloud), enabling privacy-preserving distributed learning while offloading computation.

## Features

- **Split Learning Architecture**: Client-server model splitting with configurable cut layers
- **Model Compression**: Quantization, pruning, and distillation for edge deployment
- **Multiple Frameworks**: PyTorch and TensorFlow support with ONNX export
- **Edge Simulation**: Performance metrics and resource constraints simulation
- **Interactive Demo**: Streamlit-based demonstration of split learning concepts
- **Comprehensive Evaluation**: Accuracy and efficiency metrics with leaderboards

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Split-Learning-Implementation.git
cd Split-Learning-Implementation

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"

# For edge deployment
pip install -e ".[edge]"
```

### Basic Usage

```python
from src.models.split_learning import SplitLearningClient, SplitLearningServer
from src.data.datasets import MNISTDataLoader
from src.training.trainer import SplitLearningTrainer

# Initialize split learning components
client = SplitLearningClient(cut_layer=2)
server = SplitLearningServer(input_shape=(13, 13, 16))
trainer = SplitLearningTrainer(client, server)

# Load data
data_loader = MNISTDataLoader()
train_data, test_data = data_loader.load_data()

# Train the split model
trainer.train(train_data, epochs=10)

# Evaluate
accuracy = trainer.evaluate(test_data)
print(f"Test Accuracy: {accuracy:.4f}")
```

### Run Demo

```bash
streamlit run demo/app.py
```

## Project Structure

```
├── src/                    # Source code
│   ├── models/            # Model definitions
│   ├── data/              # Data loading and preprocessing
│   ├── training/          # Training loops and strategies
│   ├── compression/       # Model compression techniques
│   ├── communication/     # Client-server communication
│   ├── evaluation/        # Metrics and evaluation
│   └── utils/             # Utilities and helpers
├── configs/               # Configuration files
├── data/                  # Data storage
├── scripts/               # Utility scripts
├── tests/                 # Test suite
├── demo/                  # Interactive demo
├── assets/                # Generated artifacts
└── docs/                  # Documentation
```

## Supported Edge Targets

- **Raspberry Pi**: ARM64 with TensorFlow Lite
- **Jetson Nano**: NVIDIA GPU with TensorRT
- **Android/iOS**: Mobile deployment with CoreML
- **MCU**: Ultra-low power with quantized models

## Performance Metrics

The framework tracks both accuracy and efficiency metrics:

- **Model Quality**: Accuracy, F1-score, mAP
- **Efficiency**: Latency (p50/p95), throughput, memory usage
- **Communication**: Bandwidth usage, round-trip time
- **Edge Constraints**: Power consumption, thermal limits

## Configuration

All settings are configurable via YAML files in the `configs/` directory:

- `device_config.yaml`: Hardware-specific settings
- `model_config.yaml`: Model architecture parameters
- `training_config.yaml`: Training hyperparameters
- `compression_config.yaml`: Compression settings

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest`
5. Format code: `black src/ tests/`
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{split_learning_implementation,
  title={Split Learning Implementation for Edge AI \& IoT},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Split-Learning-Implementation}
}
```# Split-Learning-Implementation
