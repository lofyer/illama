import os
import sys
import yaml
import gradio as gr
from typing import Dict, List, Optional
from datetime import datetime
import logging
import json
import torch
from pathlib import Path
import shutil

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Load config
config_path = os.path.join(project_root, "config/config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

# Setup logging
log_config = config['system']
log_dir = log_config['log_dir']
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(
    log_dir,
    f"app_{datetime.now().strftime(log_config['log_file_format'])}"
)

logging.basicConfig(
    level=getattr(logging, log_config['log_level']),
    format=log_config['log_format'],
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import core modules
from core.trainer import LLMTrainer, TrainingConfig
from core.inference import LLMInference
from core.model_provider import ModelProvider
from workers.cloud_worker import CloudWorker
from workers.baremetal_worker import BaremetalWorker

class LLMPlatform:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = yaml.safe_load(open(config_path))
        self.model_provider = ModelProvider(config_path)
        
        # Initialize workers if enabled
        self.aws_enabled = self.config.get('aws', {}).get('enabled', False)
        if self.aws_enabled:
            self.aws_manager = CloudWorker(config_path)
            
        self.baremetal_enabled = self.config.get('baremetal', {}).get('enabled', False)
        if self.baremetal_enabled:
            self.baremetal_manager = BaremetalWorker(config_path)
        
        # Initialize components
        self.inference = LLMInference(config_path)
        self.trainer = LLMTrainer(config_path)
        
        # Initialize tracking if enabled
        self._setup_tracking()
        
        # Initialize job management
        self.active_jobs = {}
        
    def _setup_tracking(self):
        """Setup experiment tracking"""
        tracking_config = self.config.get('tracking', {})
        
        # Setup MLflow
        mlflow_config = tracking_config.get('mlflow', {})
        if mlflow_config.get('enabled', False):
            import mlflow
            mlflow.set_tracking_uri(mlflow_config.get('tracking_uri'))
            mlflow.set_experiment(mlflow_config.get('experiment_name'))
        
        # Setup W&B
        wandb_config = tracking_config.get('wandb', {})
        if wandb_config.get('enabled', False):
            import wandb
            wandb.login(key=wandb_config.get('api_key'))
            
    def deploy_worker(self, worker_type: str, config: Dict) -> str:
        """Deploy a new worker"""
        if worker_type == "cloud" and self.aws_enabled:
            return self.aws_manager.deploy_worker(config)
        elif worker_type == "baremetal" and self.baremetal_enabled:
            return self.baremetal_manager.deploy_worker(config)
        else:
            return f"Worker type {worker_type} not enabled or invalid"
            
    def list_workers(self, worker_type: str) -> List[Dict]:
        """List all workers of specified type"""
        if worker_type == "cloud" and self.aws_enabled:
            return self.aws_manager.list_workers()
        elif worker_type == "baremetal" and self.baremetal_enabled:
            return self.baremetal_manager.list_workers()
        else:
            return []
            
    def terminate_worker(self, worker_type: str, worker_id: str) -> str:
        """Terminate a specific worker"""
        if worker_type == "cloud" and self.aws_enabled:
            return self.aws_manager.terminate_worker(worker_id)
        elif worker_type == "baremetal" and self.baremetal_enabled:
            return self.baremetal_manager.terminate_worker(worker_id)
        else:
            return f"Worker type {worker_type} not enabled or invalid"
            
    def list_models(self, provider: Optional[str] = None) -> List[Dict]:
        """List available models"""
        return self.model_provider.list_models(provider)
        
    def get_model_info(self, model_name: str) -> Dict:
        """Get model information"""
        return self.model_provider.get_model_info(model_name)
        
    def load_model(self, model_name: str) -> str:
        """Load a model for inference"""
        return self.inference.load_model(model_name)
        
    def unload_model(self) -> str:
        """Unload the current model"""
        return self.inference.unload_model()
        
    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        num_return_sequences: int = 1,
        stop_sequences: Optional[List[str]] = None
    ) -> List[str]:
        """Generate text from prompt"""
        return self.inference.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences
        )
        
    def batch_generate(
        self,
        prompts: List[str],
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> List[str]:
        """Generate text for multiple prompts"""
        results = []
        for prompt in prompts:
            result = self.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=1
            )
            results.extend(result)
        return results
        
    def start_training(self, config: Dict) -> str:
        """Start a training job"""
        # Create training config
        training_config = TrainingConfig(
            model_name=config['model_name'],
            output_dir=config['output_dir'],
            num_train_epochs=config.get('num_train_epochs', 3),
            per_device_train_batch_size=config.get('batch_size', 8),
            learning_rate=config.get('learning_rate', 5e-5),
            warmup_steps=config.get('warmup_steps', 500),
            gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
            fp16=config.get('fp16', True),
            gradient_checkpointing=config.get('gradient_checkpointing', False)
        )
        
        # Start training
        job_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.active_jobs[job_id] = {
            'status': 'running',
            'config': config,
            'start_time': datetime.now().isoformat(),
            'type': 'training'
        }
        
        try:
            self.trainer.train(training_config)
            self.active_jobs[job_id]['status'] = 'completed'
            return job_id
        except Exception as e:
            self.active_jobs[job_id]['status'] = 'failed'
            self.active_jobs[job_id]['error'] = str(e)
            raise
            
    def get_job_status(self, job_id: str) -> Dict:
        """Get status of a job"""
        return self.active_jobs.get(job_id, {'status': 'not_found'})
        
    def cancel_job(self, job_id: str) -> str:
        """Cancel a job"""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            if job['type'] == 'training':
                self.trainer.cancel_training()
            job['status'] = 'cancelled'
            return f"Job {job_id} cancelled"
        return f"Job {job_id} not found"
        
    def export_job(self, job_id: str, output_path: str) -> str:
        """Export job artifacts"""
        if job_id not in self.active_jobs:
            return f"Job {job_id} not found"
            
        job = self.active_jobs[job_id]
        if job['status'] != 'completed':
            return f"Job {job_id} is not completed"
            
        try:
            # Export model and config
            src_dir = job['config']['output_dir']
            os.makedirs(output_path, exist_ok=True)
            
            # Copy files
            for item in os.listdir(src_dir):
                s = os.path.join(src_dir, item)
                d = os.path.join(output_path, item)
                if os.path.isfile(s):
                    shutil.copy2(s, d)
                else:
                    shutil.copytree(s, d)
                    
            return f"Job {job_id} exported to {output_path}"
        except Exception as e:
            return f"Failed to export job: {str(e)}"

def create_interface():
    """Create Gradio interface"""
    platform = LLMPlatform()
    
    # Model selection dropdown
    model_list = [model.name for model in platform.list_models()]
    
    with gr.Blocks() as interface:
        gr.Markdown("# LLM Training and Inference Platform")
        
        with gr.Tab("Worker Management"):
            with gr.Row():
                worker_type = gr.Dropdown(
                    choices=["cloud", "baremetal"],
                    label="Worker Type"
                )
                worker_id = gr.Textbox(label="Worker ID")
                
            with gr.Row():
                worker_config = gr.JSON(
                    label="Worker Configuration",
                    value={
                        "instance_type": "p3.2xlarge",
                        "region": "us-west-2",
                        "ssh_key": "default"
                    }
                )
                
            with gr.Row():
                deploy_btn = gr.Button("Deploy Worker")
                terminate_btn = gr.Button("Terminate Worker")
                list_btn = gr.Button("List Workers")
                
            worker_status = gr.JSON(label="Worker Status")
        
        with gr.Tab("Inference"):
            with gr.Row():
                with gr.Column():
                    model_dropdown = gr.Dropdown(
                        choices=model_list,
                        label="Select Model"
                    )
                    load_button = gr.Button("Load Model")
                    unload_button = gr.Button("Unload Model")
                    model_info = gr.JSON(label="Model Info")
                    
            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(
                        lines=5,
                        label="Input Text"
                    )
                    max_length = gr.Slider(
                        minimum=1,
                        maximum=4096,
                        value=2048,
                        label="Max Length"
                    )
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.7,
                        label="Temperature"
                    )
                    top_p = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.95,
                        label="Top P"
                    )
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=50,
                        label="Top K"
                    )
                    generate_button = gr.Button("Generate")
                    
                with gr.Column():
                    output_text = gr.Textbox(
                        lines=5,
                        label="Generated Text"
                    )
                    
            with gr.Row():
                gr.Markdown("### Batch Inference")
                
            with gr.Row():
                with gr.Column():
                    batch_input = gr.Textbox(
                        lines=5,
                        label="Batch Input (One prompt per line)"
                    )
                    batch_generate_button = gr.Button("Generate Batch")
                    
                with gr.Column():
                    batch_output = gr.Textbox(
                        lines=5,
                        label="Batch Output"
                    )
                    
        with gr.Tab("Training"):
            with gr.Row():
                with gr.Column():
                    train_model_dropdown = gr.Dropdown(
                        choices=model_list,
                        label="Base Model"
                    )
                    output_dir = gr.Textbox(
                        label="Output Directory",
                        value="./outputs"
                    )
                    training_data = gr.File(
                        label="Training Data",
                        file_types=["csv", "json", "txt"]
                    )
                    
            with gr.Row():
                with gr.Column():
                    num_epochs = gr.Number(
                        value=3,
                        label="Number of Epochs"
                    )
                    batch_size = gr.Number(
                        value=8,
                        label="Batch Size"
                    )
                    learning_rate = gr.Number(
                        value=5e-5,
                        label="Learning Rate"
                    )
                    gradient_checkpointing = gr.Checkbox(
                        label="Use Gradient Checkpointing",
                        value=False
                    )
                    fp16_training = gr.Checkbox(
                        label="Use FP16 Training",
                        value=True
                    )
                    
                with gr.Column():
                    warmup_steps = gr.Number(
                        value=500,
                        label="Warmup Steps"
                    )
                    gradient_accumulation = gr.Number(
                        value=1,
                        label="Gradient Accumulation Steps"
                    )
                    train_button = gr.Button("Start Training")
                    
            with gr.Row():
                with gr.Column():
                    job_id = gr.Textbox(label="Job ID")
                    job_status = gr.JSON(label="Job Status")
                    cancel_button = gr.Button("Cancel Job")
                    
                with gr.Column():
                    export_path = gr.Textbox(
                        label="Export Path",
                        value="./exports"
                    )
                    export_button = gr.Button("Export Model")
                    
        # Event handlers
        def deploy_worker(wtype, config):
            return platform.deploy_worker(wtype, config)
            
        def terminate_worker(wtype, wid):
            return platform.terminate_worker(wtype, wid)
            
        def list_workers(wtype):
            return platform.list_workers(wtype)
            
        def load_model(model_name):
            platform.load_model(model_name)
            return platform.get_model_info(model_name)
            
        def unload_model():
            return platform.unload_model()
            
        def generate(prompt, max_len, temp, p, k):
            return platform.generate(
                prompt=prompt,
                max_length=max_len,
                temperature=temp,
                top_p=p,
                top_k=k
            )[0]  # Return first generation
            
        def batch_generate(text, max_len, temp, p, k):
            prompts = [line.strip() for line in text.split('\n') if line.strip()]
            results = platform.batch_generate(
                prompts=prompts,
                max_length=max_len,
                temperature=temp,
                top_p=p,
                top_k=k
            )
            return '\n\n'.join(results)
            
        def start_training(
            model, outdir, data, epochs, bs, lr, 
            grad_ckpt, fp16, warmup, grad_accum
        ):
            config = {
                'model_name': model,
                'output_dir': outdir,
                'data_path': data.name if data else None,
                'num_train_epochs': epochs,
                'batch_size': bs,
                'learning_rate': lr,
                'gradient_checkpointing': grad_ckpt,
                'fp16': fp16,
                'warmup_steps': warmup,
                'gradient_accumulation_steps': grad_accum
            }
            return platform.start_training(config)
            
        def get_status(job_id):
            return platform.get_job_status(job_id)
            
        def cancel_training(job_id):
            return platform.cancel_job(job_id)
            
        def export_model(job_id, path):
            return platform.export_job(job_id, path)
            
        # Connect event handlers
        deploy_btn.click(
            fn=deploy_worker,
            inputs=[worker_type, worker_config],
            outputs=[worker_status]
        )
        
        terminate_btn.click(
            fn=terminate_worker,
            inputs=[worker_type, worker_id],
            outputs=[worker_status]
        )
        
        list_btn.click(
            fn=list_workers,
            inputs=[worker_type],
            outputs=[worker_status]
        )
        
        load_button.click(
            fn=load_model,
            inputs=[model_dropdown],
            outputs=[model_info]
        )
        
        unload_button.click(
            fn=unload_model,
            inputs=[],
            outputs=[model_info]
        )
        
        generate_button.click(
            fn=generate,
            inputs=[input_text, max_length, temperature, top_p, top_k],
            outputs=[output_text]
        )
        
        batch_generate_button.click(
            fn=batch_generate,
            inputs=[batch_input, max_length, temperature, top_p, top_k],
            outputs=[batch_output]
        )
        
        train_button.click(
            fn=start_training,
            inputs=[
                train_model_dropdown,
                output_dir,
                training_data,
                num_epochs,
                batch_size,
                learning_rate,
                gradient_checkpointing,
                fp16_training,
                warmup_steps,
                gradient_accumulation
            ],
            outputs=[job_id]
        )
        
        cancel_button.click(
            fn=cancel_training,
            inputs=[job_id],
            outputs=[job_status]
        )
        
        export_button.click(
            fn=export_model,
            inputs=[job_id, export_path],
            outputs=[job_status]
        )
        
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="0.0.0.0")
