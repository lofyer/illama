from typing import Dict, List, Optional
import torch
import logging
from transformers import PreTrainedModel, PreTrainedTokenizer
from .model_provider import ModelProvider
import yaml

class LLMInference:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = yaml.safe_load(open(config_path))
        self.model_provider = ModelProvider(config_path)
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.tokenizer = None
        
    def load_model(self, model_name: str) -> None:
        """Load model and tokenizer"""
        try:
            self.model, self.tokenizer = self.model_provider.load_model(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16
            )
            return "Model loaded successfully"
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
            
    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        num_return_sequences: int = 1
    ) -> List[str]:
        """Generate text from prompt"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model first.")
            
        try:
            # Get default values from config
            inference_config = self.config.get('inference', {})
            max_length = max_length or inference_config.get('max_length', 2048)
            temperature = temperature or inference_config.get('temperature', 0.7)
            top_p = top_p or inference_config.get('top_p', 0.95)
            top_k = top_k or inference_config.get('top_k', 50)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.model.device)
            
            # Generate
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            # Decode outputs
            generated_texts = []
            for output in outputs:
                text = self.tokenizer.decode(output, skip_special_tokens=True)
                generated_texts.append(text)
                
            return generated_texts
            
        except Exception as e:
            self.logger.error(f"Generation failed: {str(e)}")
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
