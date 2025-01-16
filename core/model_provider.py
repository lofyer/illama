from typing import Dict, List, Optional
import os
from dataclasses import dataclass
from transformers import AutoModel, AutoTokenizer, AutoConfig
from modelscope import AutoModelForCausalLM, AutoTokenizer as MSTokenizer
from modelscope import snapshot_download
import torch
import logging
import yaml
from pathlib import Path

@dataclass
class ModelInfo:
    name: str
    provider: str  # "huggingface" or "modelscope"
    size: str
    description: str
    license: str
    tags: List[str]

class ModelProvider:
    def __init__(self, config_loader):
        self.config_loader = config_loader
        self.config = config_loader.get_config()
        self.model_cache_dir = self.config_loader.data_dir / "cache"
        self.logger = logging.getLogger("illama.model_provider")
        
        # Initialize providers
        self.providers = {
            'huggingface': self._load_from_huggingface,
            'modelscope': self._load_from_modelscope
        }
        
        # Default model mappings
        self.default_models = {
            'huggingface': [
                ModelInfo(
                    name="meta-llama/Llama-2-7b",
                    provider="huggingface",
                    size="7B",
                    description="Llama 2 base model",
                    license="llama2",
                    tags=["base", "llama2"]
                ),
                ModelInfo(
                    name="meta-llama/Llama-2-7b-chat",
                    provider="huggingface",
                    size="7B",
                    description="Llama 2 chat model",
                    license="llama2",
                    tags=["chat", "llama2"]
                )
            ],
            'modelscope': [
                ModelInfo(
                    name="deepseek-ai/deepseek-moe-16b-base",
                    provider="modelscope",
                    size="16B",
                    description="DeepSeek MoE base model",
                    license="deepseek",
                    tags=["base", "moe"]
                ),
                ModelInfo(
                    name="deepseek-ai/deepseek-moe-16b-chat",
                    provider="modelscope",
                    size="16B",
                    description="DeepSeek MoE chat model",
                    license="deepseek",
                    tags=["chat", "moe"]
                )
            ]
        }
    
    def list_models(self, provider: Optional[str] = None) -> List[ModelInfo]:
        """List available models from specified provider or all providers"""
        if provider:
            return self.default_models.get(provider, [])
        
        all_models = []
        for provider_models in self.default_models.values():
            all_models.extend(provider_models)
        return all_models
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get model info by name"""
        for models in self.default_models.values():
            for model in models:
                if model.name == model_name:
                    return model
        return None
    
    def load_model(self, model_name: str, **kwargs) -> tuple:
        """Load model and tokenizer from any supported provider"""
        model_info = self.get_model_info(model_name)
        if not model_info:
            raise ValueError(f"Model {model_name} not found")
            
        load_func = self.providers.get(model_info.provider)
        if not load_func:
            raise ValueError(f"Provider {model_info.provider} not supported")
            
        return load_func(model_name, **kwargs)
    
    def _load_from_huggingface(self, model_name: str, **kwargs) -> tuple:
        """Load model from Hugging Face"""
        try:
            config = AutoConfig.from_pretrained(
                model_name,
                cache_dir=self.model_cache_dir,
                trust_remote_code=True
            )
            
            model = AutoModel.from_pretrained(
                model_name,
                config=config,
                cache_dir=self.model_cache_dir,
                trust_remote_code=True,
                **kwargs
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.model_cache_dir,
                trust_remote_code=True
            )
            
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Failed to load model from Hugging Face: {str(e)}")
            raise
    
    def _load_from_modelscope(self, model_name: str, **kwargs) -> tuple:
        """Load model from ModelScope"""
        try:
            # Download model snapshot
            model_dir = snapshot_download(model_name, cache_dir=self.model_cache_dir)
            
            # Load model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                **kwargs
            )
            
            tokenizer = MSTokenizer.from_pretrained(
                model_dir,
                trust_remote_code=True
            )
            
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Failed to load model from ModelScope: {str(e)}")
            raise
    
    def download_model(self, model_name: str, force: bool = False) -> str:
        """Download model to cache without loading it"""
        model_info = self.get_model_info(model_name)
        if not model_info:
            raise ValueError(f"Model {model_name} not found")
            
        try:
            if model_info.provider == 'huggingface':
                # Download using Hugging Face
                AutoModel.from_pretrained(
                    model_name,
                    cache_dir=self.model_cache_dir,
                    trust_remote_code=True,
                    force_download=force
                )
            elif model_info.provider == 'modelscope':
                # Download using ModelScope
                snapshot_download(
                    model_name,
                    cache_dir=self.model_cache_dir,
                    force_download=force
                )
            else:
                raise ValueError(f"Provider {model_info.provider} not supported")
                
            return "Successfully downloaded model"
            
        except Exception as e:
            self.logger.error(f"Failed to download model: {str(e)}")
            raise
            
    def get_model_path(self, model_name: str) -> str:
        """Get local path of downloaded model"""
        model_info = self.get_model_info(model_name)
        if not model_info:
            raise ValueError(f"Model {model_name} not found")
            
        if model_info.provider == 'huggingface':
            # Get Hugging Face cache path
            return os.path.join(self.model_cache_dir, 'models--' + model_name.replace('/', '--'))
        elif model_info.provider == 'modelscope':
            # Get ModelScope cache path
            return os.path.join(self.model_cache_dir, model_name)
        else:
            raise ValueError(f"Provider {model_info.provider} not supported")
