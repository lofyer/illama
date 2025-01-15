# iLLaMA2 - LLM Training and Inference Platform

A scalable platform for training and deploying Large Language Models across baremetal and cloud infrastructure.

## Features
- Multi-provider support (HuggingFace and ModelScope)
- Distributed training across multiple workers
- Support for both cloud and baremetal deployments
- Interactive Gradio web interface
- RESTful API for programmatic access
- Worker health monitoring and auto-scaling
- Batch inference support
- Advanced training options (gradient checkpointing, FP16)
- Experiment tracking with MLflow and Weights & Biases

## Directory Structure
```
illama2/
├── config/         # Configuration files
├── core/           # Core training and inference logic
├── data/          
│   ├── models/     # Model outputs and checkpoints
│   ├── training/   # Training datasets
│   ├── evaluation/ # Evaluation datasets
│   └── cache/      # Model and data cache
├── logs/           # Application logs
├── web/           # Web interface
├── workers/       # Worker management
└── api/           # RESTful API
```

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure your environment in `config/config.yaml`
   - Set model providers (HuggingFace/ModelScope)
   - Configure cloud credentials (if using cloud workers)
   - Set up experiment tracking (MLflow/W&B)
   - Adjust system parameters (GPU memory, CPU threads)

3. Run the platform:
   ```bash
   python web/app.py
   ```

## Components

### Model Provider
- Unified interface for HuggingFace and ModelScope models
- Automatic model caching and version management
- Model metadata and configuration handling

### Training
- Distributed training across multiple workers
- FP16 training support
- Gradient checkpointing for memory efficiency
- Configurable hyperparameters
- Training progress tracking
- Model checkpointing and export

### Inference
- Single and batch inference modes
- Advanced generation parameters (temperature, top_p, top_k)
- Inference caching for improved performance
- Stop sequence support

### Worker Management
- AWS SageMaker integration
- Baremetal worker support
- Auto-scaling based on workload
- Health monitoring and logging
- Resource optimization

### Web Interface
- Interactive model selection
- Real-time training monitoring
- Batch inference interface
- Worker management dashboard
- Job tracking and management

## Architecture
- Core: Contains the main logic for training and inference
- Workers: Manages cloud and baremetal compute resources
- API: RESTful endpoints for programmatic access
- Web: Gradio interface for interactive use

## Logging
- Configurable log levels and formats
- Separate log files for different components
- System metrics monitoring
- Error tracking and reporting

## Security
- AWS credentials management
- API key authentication
- Model access control
- Secure worker communication

## Contributing
Please see CONTRIBUTING.md for guidelines on contributing to this project.
