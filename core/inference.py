from typing import Dict, List, Optional, Any
import torch
import logging
import os
from pathlib import Path
from transformers import PreTrainedModel, PreTrainedTokenizer
from .model_provider import ModelProvider
import yaml
from .config_loader import ConfigLoader, InferenceConfig
import time

class LLMInference:
    def __init__(self, config_loader: ConfigLoader):
        """Initialize LLM inference with DeepSeek optimizations"""
        self.config_loader = config_loader
        self.deepseek_config = self.config_loader.load_inference_config()
        self.model_provider = ModelProvider(self.config_loader)
        self.logger = logging.getLogger("illama.inference")
        
        # Create inference-specific directories
        self.cache_dir = self.config_loader.data_dir / "cache"
        self.outputs_dir = self.config_loader.data_dir / "outputs"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.tokenizer = None
        self._setup_inference_environment()
        
    def _setup_inference_environment(self):
        """Setup inference environment based on DeepSeek configuration"""
        deepseek_config = InferenceConfig(self.deepseek_config)
        
        # Set environment variables for deployment configuration
        os.environ['PREFILL_GPUS'] = str(deepseek_config.prefill_gpus)
        os.environ['MAX_DECODE_GPUS'] = str(deepseek_config.max_decode_gpus)
        
        # Configure memory optimization
        if self.deepseek_config['memory']['activation_quantization']['dtype'] == 'fp8':
            os.environ['ACTIVATION_DTYPE'] = 'fp8'
            
        # Setup communication backend
        if self.deepseek_config['memory']['communication']['backend'] == 'InfiniBand':
            os.environ['NCCL_IB_ENABLE'] = '1'
            
    def load_model(self, model_name: str) -> None:
        """Load model and tokenizer with DeepSeek optimizations"""
        try:
            device_map = self._get_optimal_device_map()
            self.model, self.tokenizer = self.model_provider.load_model(
                model_name,
                device_map=device_map,
                torch_dtype=self._get_optimal_dtype(),
                max_memory={i: f"{self.deepseek_config['hardware']['gpu_requirements']['prefill']['min_gpus']}GB" 
                           for i in range(torch.cuda.device_count())}
            )
            
            if self.deepseek_config['inference_optimization']['speculative_decoding']['enabled']:
                self._setup_speculative_decoding()
                
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
            
    def _get_optimal_device_map(self) -> str:
        """Get optimal device map based on DeepSeek configuration"""
        if self.deepseek_config['deployment']['stages']['prefill']['enabled']:
            return "sequential"  # Use sequential loading for pipeline parallel inference
        return "auto"
        
    def _get_optimal_dtype(self) -> torch.dtype:
        """Get optimal dtype based on DeepSeek configuration"""
        if self.deepseek_config['memory']['activation_quantization']['dtype'] == 'fp8':
            return torch.float8_e4m3fn  # Using FP8 format
        return torch.float16
        
    def _setup_speculative_decoding(self):
        """Setup speculative decoding if enabled"""
        if not hasattr(self.model, 'enable_speculative_decoding'):
            self.logger.warning("Model doesn't support speculative decoding")
            return
            
        self.model.enable_speculative_decoding(
            num_future_tokens=4,  # Typical value for MTP
            temperature=0.8
        )
        
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text with DeepSeek optimizations"""
        try:
            # Log generation request
            self.logger.info(f"Generating text for prompt: {prompt[:100]}...")
            
            if not self.model or not self.tokenizer:
                raise RuntimeError("Model and tokenizer must be loaded first")
                
            # Apply DeepSeek context length
            max_length = min(
                self.deepseek_config['context']['max_length'],
                self.model.config.max_position_embeddings
            )
            
            # Prepare inputs with dynamic shape optimization
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=max_length,
                truncation=True
            ).to(self.model.device)
            
            # Generate with optimized settings
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                **kwargs
            )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Save generation output
            output_file = self.outputs_dir / f"generation_{int(time.time())}.txt"
            with open(output_file, "w") as f:
                f.write(f"Prompt: {prompt}\n")
                f.write(f"Generated: {generated_text}\n")
            
            return {
                "generated_text": generated_text,
                "tokens_generated": len(outputs[0]) - len(inputs["input_ids"][0]),
                "context_length": len(inputs["input_ids"][0])
            }
            
        except Exception as e:
            self.logger.error(f"Generation failed: {str(e)}")
            raise
            
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> List[str]:
        """Generate text for multiple prompts"""
        try:
            # Log batch generation request
            self.logger.info(f"Processing batch generation with {len(prompts)} prompts")
            
            if not self.model or not self.tokenizer:
                raise RuntimeError("Model and tokenizer must be loaded first")
                
            results = []
            for prompt in prompts:
                result = self.generate(prompt, max_new_tokens, temperature, top_p, **kwargs)
                results.append(result["generated_text"])
            
            # Save batch outputs
            output_file = self.outputs_dir / f"batch_{int(time.time())}.txt"
            with open(output_file, "w") as f:
                for i, (prompt, result) in enumerate(zip(prompts, results)):
                    f.write(f"Prompt {i}: {prompt}\n")
                    f.write(f"Generated {i}: {result}\n\n")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch generation failed: {str(e)}")
            raise
            
    def get_model_info(self) -> Dict:
        """Get information about the currently loaded model"""
        if not self.model:
            return {"status": "No model loaded"}
            
        return {
            "name": self.model.config._name_or_path,
            "device": str(next(self.model.parameters()).device),
            "dtype": str(next(self.model.parameters()).dtype),
            "parameters": sum(p.numel() for p in self.model.parameters()),
        }
        
    def unload_model(self) -> None:
        """Unload the current model from memory"""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()
        return "Model unloaded successfully"
