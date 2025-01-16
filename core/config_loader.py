import os
from pathlib import Path
import yaml
import logging
from typing import Dict, Any, Optional

class ConfigLoader:
    def __init__(self, config_path: str):
        """Initialize config loader with base path"""
        self.config_path = config_path
        self.config_dir = os.path.dirname(config_path)
        self.config = self._load_yaml(config_path)
        
        # Ensure required directories exist
        self.base_dir = Path(os.path.dirname(os.path.dirname(config_path)))
        self.log_dir = self.base_dir / "logs"
        self.data_dir = self.base_dir / "data"
        self.downloads_dir = self.base_dir / "downloads"
        
        # Create directories if they don't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / "platform.log"),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("illama")
    
    def get_config(self) -> Dict[str, Any]:
        """Get the full configuration"""
        return self.config
    
    def get_config_path(self) -> str:
        """Get the config file path"""
        return self.config_path
    
    def load_training_config(self) -> Dict[str, Any]:
        """Load training configuration"""
        training_config_path = os.path.join(self.config_dir, 'training_config.yaml')
        return self._load_yaml(training_config_path)
    
    def load_inference_config(self) -> Dict[str, Any]:
        """Load inference configuration"""
        inference_config_path = os.path.join(self.config_dir, 'inference_config.yaml')
        return self._load_yaml(inference_config_path)
    
    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """Load YAML file"""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config from {path}: {str(e)}")
            raise

class TrainingConfig:
    """Training configuration wrapper with type hints and validation"""
    def __init__(self, config: Dict[str, Any]):
        self.moe = config.get('moe', {})
        self.precision = config.get('precision', {})
        self.distributed_training = config.get('distributed_training', {})
        self.memory_optimization = config.get('memory_optimization', {})
        self.training_efficiency = config.get('training_efficiency', {})
        self.multi_token_prediction = config.get('multi_token_prediction', {})
        self.context = config.get('context', {})
        
    @property
    def is_moe_enabled(self) -> bool:
        return self.moe.get('enabled', False)
    
    @property
    def max_context_length(self) -> int:
        return self.context.get('max_length', 2048)

class InferenceConfig:
    """Inference configuration wrapper with type hints and validation"""
    def __init__(self, config: Dict[str, Any]):
        self.deployment = config.get('deployment', {})
        self.inference_optimization = config.get('inference_optimization', {})
        self.memory = config.get('memory', {})
        self.load_balancing = config.get('load_balancing', {})
        self.context = config.get('context', {})
        self.hardware = config.get('hardware', {})
    
    @property
    def prefill_gpus(self) -> int:
        return self.deployment.get('stages', {}).get('prefill', {}).get('gpus_per_unit', 32)
    
    @property
    def max_decode_gpus(self) -> int:
        return self.deployment.get('stages', {}).get('decode', {}).get('max_gpus', 320)
