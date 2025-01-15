from typing import Dict, List, Optional, Union, Any
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    IntervalStrategy,
)
from datasets import load_dataset, Dataset
import os
import json
import yaml
import logging
from datetime import datetime
import wandb
import mlflow
from mlflow.models import infer_signature
import boto3
from botocore.exceptions import ClientError
import threading
import queue
from dataclasses import dataclass, asdict
import numpy as np
from accelerate import Accelerator
from torch.utils.data import DataLoader
import psutil
import GPUtil
from pathlib import Path
from .model_provider import ModelProvider

@dataclass
class TrainingMetrics:
    loss: float
    learning_rate: float
    epoch: float
    step: int
    train_runtime: float
    train_samples_per_second: float
    train_steps_per_second: float
    total_flos: float
    train_loss: float
    eval_loss: Optional[float] = None
    eval_perplexity: Optional[float] = None
    eval_runtime: Optional[float] = None
    eval_samples_per_second: Optional[float] = None
    eval_steps_per_second: Optional[float] = None
    
@dataclass
class TrainingConfig:
    model_name: str
    output_dir: str
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    warmup_steps: int = 500
    weight_decay: float = 0.01
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 1000
    max_steps: Optional[int] = None
    learning_rate: float = 5e-5
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    bf16: bool = False
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.0
    group_by_length: bool = True
    length_column_name: Optional[str] = None
    report_to: List[str] = None
    gradient_checkpointing: bool = False
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    use_deepspeed: bool = False
    deepspeed_config: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)

class LLMTrainer:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.model_provider = ModelProvider(config_path)
        self.setup_logging()
        self.setup_tracking()
        self.active_jobs = {}
        self.job_queues = {}
        self._setup_aws()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_tracking(self):
        """Setup experiment tracking with MLflow and Weights & Biases"""
        # Setup MLflow tracking
        mlflow_config = self.config.get('tracking', {}).get('mlflow', {})
        if mlflow_config.get('enabled', False):
            mlflow.set_tracking_uri(mlflow_config.get('tracking_uri', 'http://localhost:5000'))
            mlflow.set_experiment(mlflow_config.get('experiment_name', 'llm-training'))
            self.use_mlflow = True
        else:
            self.use_mlflow = False

        # Setup Weights & Biases tracking
        wandb_config = self.config.get('tracking', {}).get('wandb', {})
        if wandb_config.get('enabled', False):
            api_key = wandb_config.get('api_key', '')
            if api_key and len(api_key) == 40:  # Valid W&B API key is 40 characters
                try:
                    wandb.login(key=api_key)
                    self.use_wandb = True
                except Exception as e:
                    print(f"Failed to initialize W&B: {str(e)}")
                    self.use_wandb = False
            else:
                print("W&B API key is invalid or not provided. Disabling W&B tracking.")
                self.use_wandb = False
        else:
            self.use_wandb = False

    def _setup_aws(self):
        """Setup AWS client for SageMaker integration"""
        if self.config.get('aws', {}).get('enabled', False):
            self.aws_client = boto3.client(
                'sagemaker',
                aws_access_key_id=self.config['aws']['access_key'],
                aws_secret_access_key=self.config['aws']['secret_key'],
                region_name=self.config['aws']['region']
            )
        else:
            self.aws_client = None

    def _load_config(self) -> Dict:
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def prepare_dataset(
        self,
        data_path: str,
        tokenizer,
        validation_split: float = 0.1,
        max_length: int = 512,
        text_column: str = "text",
    ) -> tuple[Dataset, Optional[Dataset]]:
        """Prepare dataset for training"""
        # Load dataset based on file type
        if data_path.endswith('.json'):
            dataset = load_dataset('json', data_files=data_path)['train']
        elif data_path.endswith('.csv'):
            dataset = load_dataset('csv', data_files=data_path)['train']
        elif data_path.endswith('.txt'):
            with open(data_path, 'r') as f:
                texts = [line.strip() for line in f]
            dataset = Dataset.from_dict({"text": texts})
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        # Split dataset
        if validation_split > 0:
            dataset = dataset.train_test_split(
                test_size=validation_split,
                shuffle=True,
                seed=42
            )
            train_dataset = dataset['train']
            eval_dataset = dataset['test']
        else:
            train_dataset = dataset
            eval_dataset = None

        # Tokenize datasets
        def tokenize_function(examples):
            return tokenizer(
                examples[text_column],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_special_tokens_mask=True,
            )

        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names,
        )

        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=eval_dataset.column_names,
            )

        return train_dataset, eval_dataset

    def create_training_args(self, config: TrainingConfig) -> TrainingArguments:
        """Create training arguments from config"""
        return TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            warmup_steps=config.warmup_steps,
            weight_decay=config.weight_decay,
            logging_steps=config.logging_steps,
            evaluation_strategy="steps" if config.eval_steps else "no",
            eval_steps=config.eval_steps,
            save_steps=config.save_steps,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            fp16=config.fp16,
            bf16=config.bf16,
            learning_rate=config.learning_rate,
            max_steps=config.max_steps,
            optim=config.optim,
            lr_scheduler_type=config.lr_scheduler_type,
            max_grad_norm=config.max_grad_norm,
            warmup_ratio=config.warmup_ratio,
            group_by_length=config.group_by_length,
            length_column_name=config.length_column_name,
            report_to=config.report_to or ["none"],
            gradient_checkpointing=config.gradient_checkpointing,
        )

    def setup_tracking_experiment(
        self,
        job_id: str,
        config: TrainingConfig,
        metrics: Optional[Dict] = None
    ):
        """Setup experiment tracking for training job"""
        if self.config.get('tracking', {}).get('mlflow', {}).get('enabled', False):
            mlflow.set_experiment(job_id)
            mlflow.start_run()
            mlflow.log_params(config.to_dict())
            if metrics:
                mlflow.log_metrics(metrics)

        if self.config.get('tracking', {}).get('wandb', {}).get('enabled', False):
            wandb.init(
                project=self.config['tracking']['wandb']['project_name'],
                name=job_id,
                config=config.to_dict()
            )

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to tracking systems"""
        if self.config.get('tracking', {}).get('mlflow', {}).get('enabled', False):
            mlflow.log_metrics(metrics, step=step)

        if self.config.get('tracking', {}).get('wandb', {}).get('enabled', False):
            wandb.log(metrics, step=step)

    def get_system_metrics(self) -> Dict[str, float]:
        """Get system resource metrics"""
        metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
        }
        
        # Add GPU metrics if available
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                metrics.update({
                    f'gpu_{i}_memory_percent': gpu.memoryUtil * 100,
                    f'gpu_{i}_load_percent': gpu.load * 100,
                })
        except:
            pass
            
        return metrics

    def train(
        self,
        config: TrainingConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run training with the specified configuration"""
        # Generate unique job ID
        job_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Setup tracking
            self.setup_tracking_experiment(job_id, config)
            
            # Load model and tokenizer
            model, tokenizer = self.model_provider.load_model(
                config.model_name,
                gradient_checkpointing=config.gradient_checkpointing
            )
            
            # Create training arguments
            training_args = self.create_training_args(config)

            # Setup data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            )

            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            )

            # Start training
            train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            metrics = train_result.metrics

            # Log final metrics
            self.log_metrics(metrics)
            
            # Save model and tokenizer
            trainer.save_model()
            tokenizer.save_pretrained(config.output_dir)
            
            # Save model signature for MLflow
            if self.config.get('tracking', {}).get('mlflow', {}).get('enabled', False):
                signature = infer_signature(
                    train_dataset.select(range(min(5, len(train_dataset)))),
                    model(torch.tensor([[0]])).logits.detach().numpy()
                )
                mlflow.pytorch.log_model(
                    model,
                    "model",
                    signature=signature,
                    registered_model_name=f"{config.model_name}_finetuned"
                )

            # Update job status
            self.active_jobs[job_id] = {
                "status": "completed",
                "metrics": metrics,
                "config": config.to_dict(),
                "output_dir": config.output_dir,
            }

            return {
                "job_id": job_id,
                "status": "completed",
                "metrics": metrics,
                "output_dir": config.output_dir,
            }

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            self.active_jobs[job_id] = {
                "status": "failed",
                "error": str(e),
                "config": config.to_dict(),
            }
            raise

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a training job"""
        if job_id not in self.active_jobs:
            return {
                "status": "not_found",
                "message": f"Job {job_id} not found"
            }
        
        job_info = self.active_jobs[job_id].copy()
        
        # Add system metrics
        job_info["system_metrics"] = self.get_system_metrics()
        
        # Add AWS SageMaker status if applicable
        if self.aws_client and "sagemaker_job_name" in job_info:
            try:
                response = self.aws_client.describe_training_job(
                    TrainingJobName=job_info["sagemaker_job_name"]
                )
                job_info["sagemaker_status"] = response["TrainingJobStatus"]
            except ClientError as e:
                job_info["sagemaker_status"] = f"Error: {str(e)}"
        
        return job_info

    def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List training jobs with optional filtering"""
        jobs = list(self.active_jobs.items())
        
        if status:
            jobs = [(job_id, info) for job_id, info in jobs 
                   if info.get("status") == status]
        
        # Sort by job ID (timestamp) in descending order
        jobs.sort(reverse=True)
        
        # Apply pagination
        jobs = jobs[offset:offset + limit]
        
        return [{
            "job_id": job_id,
            **info,
            "system_metrics": self.get_system_metrics()
        } for job_id, info in jobs]

    def cancel_job(self, job_id: str) -> Dict[str, str]:
        """Cancel a training job"""
        if job_id not in self.active_jobs:
            return {
                "status": "not_found",
                "message": f"Job {job_id} not found"
            }
        
        job_info = self.active_jobs[job_id]
        
        # Cancel AWS SageMaker job if applicable
        if self.aws_client and "sagemaker_job_name" in job_info:
            try:
                self.aws_client.stop_training_job(
                    TrainingJobName=job_info["sagemaker_job_name"]
                )
            except ClientError as e:
                return {
                    "status": "error",
                    "message": f"Failed to cancel SageMaker job: {str(e)}"
                }
        
        job_info["status"] = "cancelled"
        return {
            "status": "cancelled",
            "message": f"Job {job_id} cancelled successfully"
        }

    def export_job_artifacts(
        self,
        job_id: str,
        export_dir: str,
        include_model: bool = True
    ) -> Dict[str, str]:
        """Export job artifacts including logs, metrics, and optionally the model"""
        if job_id not in self.active_jobs:
            return {
                "status": "not_found",
                "message": f"Job {job_id} not found"
            }
        
        job_info = self.active_jobs[job_id]
        export_path = Path(export_dir) / job_id
        export_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Export job metadata and metrics
            with open(export_path / "metadata.json", "w") as f:
                json.dump(job_info, f, indent=2)
            
            # Export training logs
            if os.path.exists(f"training_{job_id}.log"):
                os.system(f"cp training_{job_id}.log {export_path}/")
            
            # Export model if requested
            if include_model and job_info["status"] == "completed":
                model_path = Path(job_info["output_dir"])
                if model_path.exists():
                    os.system(f"cp -r {model_path} {export_path}/model")
            
            return {
                "status": "success",
                "export_path": str(export_path),
                "message": "Job artifacts exported successfully"
            }
        
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to export artifacts: {str(e)}"
            }
