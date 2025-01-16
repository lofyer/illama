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
import uuid
import time

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from core.model_provider import ModelProvider
from core.trainer import LLMTrainer, TrainingConfig
from core.inference import LLMInference
from core.config_loader import ConfigLoader
from workers.package_agent import create_agent_package

# Load configs
config_path = os.path.join(project_root, "config/config.yaml")
training_config_path = os.path.join(project_root, "config/training_config.yaml")
inference_config_path = os.path.join(project_root, "config/inference_config.yaml")
moe_config_path = os.path.join(project_root, "config/moe_config.yaml")

# Setup logging
def setup_logging():
    """Setup logging configuration"""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'illama_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Create logger
    logger = logging.getLogger('illama')
    logger.setLevel(logging.INFO)
    
    return logger

logger = setup_logging()

class LLMPlatform:
    def __init__(self, config_path: str):
        """Initialize the LLM platform"""
        self.config_loader = ConfigLoader(config_path)
        self.model_provider = ModelProvider(self.config_loader)
        self.inference = LLMInference(self.config_loader)
        self.trainer = LLMTrainer(self.config_loader)
        self.active_jobs = {}
        self.workers = {}
        self.log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
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
        
    def list_workers(self, worker_type: str = "all") -> list:
        """List all workers or workers of specific type"""
        workers = []
        for worker_id, info in self.workers.items():
            if worker_type == "all" or info.get("type") == worker_type:
                workers.append(info)
        return workers
        
    def get_worker_status(self, worker_id: str) -> dict:
        """Get worker status"""
        return self.workers.get(worker_id, {"error": "Worker not found"})
        
    def terminate_worker(self, worker_id: str) -> dict:
        """Terminate a worker"""
        if worker_id in self.workers:
            worker = self.workers.pop(worker_id)
            return {"status": "terminated", "worker": worker}
        return {"error": "Worker not found"}
        
    def start_training(self, config: Dict) -> str:
        """Start a training job"""
        job_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Set training log directory
        training_log_dir = os.path.join(self.log_dir, f"training_{job_id}")
        os.makedirs(training_log_dir, exist_ok=True)
        
        training_config = TrainingConfig(
            model_name=config['model_name'],
            output_dir=config['output_dir'],
            log_dir=training_log_dir,
            num_train_epochs=config.get('num_epochs', 3),
            per_device_train_batch_size=config.get('batch_size', 8),
            learning_rate=config.get('learning_rate', 5e-5),
            warmup_steps=config.get('warmup_steps', 500),
            gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
            fp16=config.get('fp16', True),
            gradient_checkpointing=config.get('gradient_checkpointing', False)
        )
        
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
        
    def save_training_config(
        self,
        model_name: str,
        output_dir: str,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        warmup_steps: int,
        gradient_accumulation: int,
        fp16: bool,
        gradient_checkpointing: bool
    ) -> str:
        """Save training configuration"""
        config = {
            'model': {
                'name': model_name,
                'output_dir': output_dir
            },
            'training': {
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'warmup_steps': warmup_steps,
                'gradient_accumulation_steps': gradient_accumulation,
                'fp16': fp16,
                'gradient_checkpointing': gradient_checkpointing
            }
        }
        with open(training_config_path, 'w') as f:
            yaml.dump(config, f)
        return "Training configuration saved successfully"

    def save_moe_config(
        self,
        num_experts: int,
        expert_capacity: float,
        router_jitter: bool,
        router_z_loss: float,
        load_balancing: bool,
        aux_loss: float
    ) -> str:
        """Save MoE configuration"""
        config = {
            'moe': {
                'num_experts': num_experts,
                'expert_capacity': expert_capacity,
                'router_jitter': router_jitter,
                'router_z_loss': router_z_loss,
                'load_balancing': load_balancing,
                'aux_loss': aux_loss
            }
        }
        with open(moe_config_path, 'w') as f:
            yaml.dump(config, f)
        return "MoE configuration saved successfully"

    def save_inference_config(
        self,
        max_length: int,
        temperature: float,
        top_p: float,
        top_k: int
    ) -> str:
        """Save inference configuration"""
        config = {
            'inference': {
                'max_length': max_length,
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k
            }
        }
        with open(inference_config_path, 'w') as f:
            yaml.dump(config, f)
        return "Inference configuration saved successfully"

def create_training_tab(platform: LLMPlatform) -> gr.Tab:
    """Create the training tab"""
    with gr.Tab("Train") as tab:
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Training Configuration")
                model_name = gr.Dropdown(
                    choices=platform.list_models(),
                    label="Model",
                    interactive=True
                )
                dataset_path = gr.Textbox(
                    label="Dataset Path",
                    placeholder="/path/to/dataset",
                    interactive=True
                )
                output_dir = gr.Textbox(
                    label="Output Directory",
                    placeholder="/path/to/output",
                    interactive=True
                )
                batch_size = gr.Number(
                    label="Batch Size",
                    value=8,
                    interactive=True
                )
                learning_rate = gr.Number(
                    label="Learning Rate",
                    value=2e-5,
                    interactive=True
                )
                num_epochs = gr.Number(
                    label="Number of Epochs",
                    value=3,
                    interactive=True
                )
                start_training_btn = gr.Button("Start Training")
                job_id = gr.Textbox(
                    label="Job ID",
                    interactive=False
                )
                cancel_job_btn = gr.Button("Cancel Training")
                train_status = gr.Textbox(
                    label="Training Status",
                    interactive=False
                )

        def start_training():
            config = {
                "model_name": model_name.value,
                "dataset_path": dataset_path.value,
                "output_dir": output_dir.value,
                "batch_size": batch_size.value,
                "learning_rate": learning_rate.value,
                "num_epochs": num_epochs.value
            }
            try:
                job_id = platform.start_training(config)
                return job_id, f"Training started with job ID: {job_id}"
            except Exception as e:
                return "", f"Error starting training: {str(e)}"
        
        def cancel_training(job_id):
            if not job_id:
                return "No active job to cancel"
            try:
                return platform.cancel_job(job_id)
            except Exception as e:
                return f"Error canceling job: {str(e)}"
        
        start_training_btn.click(
            start_training,
            outputs=[job_id, train_status]
        )
        
        cancel_job_btn.click(
            cancel_training,
            inputs=[job_id],
            outputs=[train_status]
        )
        
        return tab

def create_moe_tab(platform: LLMPlatform) -> gr.Tab:
    """Create the MoE configuration tab"""
    with gr.Tab("MoE Configuration") as tab:
        with gr.Row():
            with gr.Column():
                gr.Markdown("### MoE Configuration")
                moe_enabled = gr.Checkbox(
                    label="Enable MoE",
                    value=True,
                    interactive=True
                )
                batch_wise_balance = gr.Checkbox(
                    label="Batch-wise Load Balancing",
                    value=True,
                    interactive=True
                )
                num_experts = gr.Slider(
                    label="Number of Experts",
                    minimum=1,
                    maximum=128,
                    value=64,
                    step=1
                )
                
                gr.Markdown("### Precision Settings")
                precision_type = gr.Dropdown(
                    label="Training Precision",
                    choices=["fp8", "fp16", "bf16"],
                    value="fp8"
                )
                
                gr.Markdown("### Memory Optimization")
                recompute_activations = gr.Checkbox(
                    label="Recompute Activations",
                    value=True
                )
                activation_dtype = gr.Dropdown(
                    label="Activation Cache Dtype",
                    choices=["fp8", "fp16", "bf16"],
                    value="fp8"
                )
                
            with gr.Column():
                gr.Markdown("### Context Settings")
                context_length = gr.Slider(
                    label="Max Context Length",
                    minimum=2048,
                    maximum=128000,
                    value=128000,
                    step=1024
                )
                
                gr.Markdown("### Multi-Token Prediction")
                enable_mtp = gr.Checkbox(
                    label="Enable Multi-Token Prediction",
                    value=True
                )
                future_tokens = gr.Slider(
                    label="Number of Future Tokens",
                    minimum=1,
                    maximum=8,
                    value=4,
                    step=1
                )
                
                gr.Markdown("### Training Efficiency")
                batch_size = gr.Number(
                    label="Batch Size",
                    value=8
                )
                grad_accum_steps = gr.Number(
                    label="Gradient Accumulation Steps",
                    value=8
                )
                
        save_config_btn = gr.Button("Save Configuration")
        moe_status = gr.Textbox(label="Status", interactive=False)

        def save_moe_config():
            config = {
                'moe': {
                    'enabled': moe_enabled.value,
                    'framework': 'DeepSeekMoE',
                    'load_balancing': {
                        'batch_wise': batch_wise_balance.value,
                        'type': 'auxiliary-loss-free'
                    }
                },
                'distributed_training': {
                    'expert_parallel': {
                        'num_experts': num_experts.value
                    }
                },
                'precision': {
                    'training_dtype': precision_type.value
                },
                'memory_optimization': {
                    'activation_recompute': {
                        'enabled': recompute_activations.value
                    },
                    'activation_cache': {
                        'dtype': activation_dtype.value
                    }
                },
                'context': {
                    'max_length': context_length.value
                },
                'multi_token_prediction': {
                    'enabled': enable_mtp.value,
                    'num_future_tokens': future_tokens.value
                },
                'training_efficiency': {
                    'batch_size': batch_size.value,
                    'gradient_accumulation_steps': grad_accum_steps.value
                }
            }
            return platform.save_moe_config(
                num_experts.value,
                1.0,
                False,
                0.01,
                True,
                0.01
            )

        save_config_btn.click(
            save_moe_config,
            outputs=[moe_status]
        )
        
        return tab

def create_inference_tab(platform: LLMPlatform) -> gr.Tab:
    """Create the inference tab"""
    with gr.Tab("Inference") as tab:
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Model Selection")
                model_dropdown = gr.Dropdown(
                    label="Select Model",
                    choices=platform.list_models(),
                    interactive=True
                )
                load_btn = gr.Button("Load Model")
                unload_btn = gr.Button("Unload Model")
                model_info = gr.JSON(label="Model Information")
                
                gr.Markdown("### Generation Parameters")
                prompt = gr.Textbox(
                    label="Input Prompt",
                    placeholder="Enter your prompt here...",
                    lines=3
                )
                max_length = gr.Slider(
                    label="Max Length",
                    minimum=1,
                    maximum=4096,
                    value=256,
                    step=1
                )
                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1
                )
                top_p = gr.Slider(
                    label="Top P",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.9,
                    step=0.1
                )
                top_k = gr.Slider(
                    label="Top K",
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1
                )
                generate_btn = gr.Button("Generate")
                output_text = gr.Textbox(
                    label="Generated Text",
                    interactive=False,
                    lines=5
                )

        def generate_text():
            try:
                outputs = platform.generate(
                    prompt=prompt.value,
                    max_length=max_length.value,
                    temperature=temperature.value,
                    top_p=top_p.value,
                    top_k=top_k.value
                )
                return outputs[0] if outputs else "No output generated"
            except Exception as e:
                return f"Error generating text: {str(e)}"

        # Connect event handlers
        load_btn.click(
            platform.load_model,
            inputs=[model_dropdown],
            outputs=[model_info]
        )
        
        unload_btn.click(
            platform.unload_model,
            outputs=[model_info]
        )
        
        generate_btn.click(
            generate_text,
            outputs=[output_text]
        )
        
        return tab

def create_worker_tab(platform: LLMPlatform) -> gr.Tab:
    """Create the worker management tab"""
    with gr.Tab("Worker Management") as tab:
        with gr.Row():
            gr.Markdown("""
            ### Worker Setup Instructions
            1. Download the agent package using the link below
            2. Extract and run on your target machine:
               ```bash
               tar xzf illama-agent.tar.gz
               cd illama-agent
               ./setup.sh
               ```
            3. The agent will automatically register with this server
            """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Agent Download")
                agent_file = gr.File(
                    label="Agent Package",
                    visible=True,
                    interactive=False
                )
                download_btn = gr.Button("Download Agent Package")
                download_status = gr.Textbox(label="Download Status", interactive=False)
                
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Connected Workers")
                worker_list = gr.Dataframe(
                    headers=["Worker ID", "Host", "Status", "GPU Count", "Memory", "Last Seen"],
                    interactive=False
                )
                refresh_btn = gr.Button("Refresh Workers")
                
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Worker Actions")
                selected_worker = gr.Dropdown(
                    label="Select Worker",
                    choices=[],
                    interactive=True
                )
                status_btn = gr.Button("Check Status")
                terminate_btn = gr.Button("Terminate Worker")
                
        with gr.Row():
            worker_status = gr.JSON(label="Worker Status")

        def download_agent():
            try:
                package_path = create_agent_package(output_dir=str(platform.config_loader.downloads_dir))
                if os.path.exists(package_path):
                    return [package_path, f"Agent package created successfully at {package_path}"]
                else:
                    return [None, f"Failed to create agent package - file not found"]
            except Exception as e:
                return [None, f"Error creating agent package: {str(e)}"]

        def refresh_workers():
            try:
                workers = platform.list_workers("all")
                worker_data = []
                worker_ids = []
                for worker in workers:
                    worker_data.append([
                        worker.get("id", "N/A"),
                        worker.get("host", "N/A"),
                        worker.get("status", "unknown"),
                        worker.get("gpu_count", 0),
                        f"{worker.get('memory_used', 0)}/{worker.get('memory_total', 0)} GB",
                        worker.get("last_seen", "N/A")
                    ])
                    worker_ids.append(worker["id"])
                return (
                    worker_data, 
                    gr.Dropdown.update(choices=worker_ids), 
                    "Workers refreshed successfully"
                )
            except Exception as e:
                return (
                    [], 
                    gr.Dropdown.update(choices=[]), 
                    f"Error refreshing workers: {str(e)}"
                )

        def check_worker_status(worker_id):
            return platform.get_worker_status(worker_id)

        def terminate_selected_worker(worker_id):
            return platform.terminate_worker(worker_id)

        # Connect event handlers
        download_btn.click(
            download_agent,
            outputs=[agent_file, download_status]
        )
        
        refresh_btn.click(
            refresh_workers,
            outputs=[worker_list, selected_worker, worker_status]
        )
        
        status_btn.click(
            check_worker_status,
            inputs=[selected_worker],
            outputs=[worker_status]
        )
        
        terminate_btn.click(
            terminate_selected_worker,
            inputs=[selected_worker],
            outputs=[worker_status]
        )
        
        return tab

def create_interface():
    """Create the Gradio interface"""
    platform = LLMPlatform(config_path)
    
    with gr.Blocks(title="iLLaMA Platform") as interface:
        gr.Markdown("# iLLaMA Platform")
        
        with gr.Tabs():
            # Training Configuration Tab
            with gr.Tab("Training Config"):
                create_training_config_tab(platform)
            
            # MoE Configuration Tab
            with gr.Tab("MoE Config"):
                create_moe_config_tab(platform)
                
            # Training Tab
            with gr.Tab("Training"):
                create_training_tab(platform)
            
            # Inference Configuration Tab
            with gr.Tab("Inference Config"):
                create_inference_config_tab(platform)
                
            # Inference Tab
            with gr.Tab("Inference"):
                create_inference_tab(platform)
                
            # Worker Management Tab
            with gr.Tab("Workers"):
                create_worker_tab(platform)
    
    return interface

def create_training_config_tab(platform):
    """Create the training configuration tab"""
    with gr.Column():
        gr.Markdown("## Training Configuration")
        
        with gr.Row():
            model_name = gr.Dropdown(
                choices=platform.list_models(),
                label="Base Model",
                info="Select the base model for training"
            )
            
            output_dir = gr.Textbox(
                label="Output Directory",
                placeholder="Path to save the trained model",
                info="Directory where the trained model will be saved"
            )
            
        with gr.Row():
            num_epochs = gr.Number(
                value=3,
                label="Number of Epochs",
                minimum=1,
                maximum=100,
                info="Number of training epochs"
            )
            
            batch_size = gr.Number(
                value=8,
                label="Batch Size",
                minimum=1,
                maximum=128,
                info="Training batch size per device"
            )
            
            learning_rate = gr.Number(
                value=5e-5,
                label="Learning Rate",
                minimum=1e-6,
                maximum=1e-3,
                info="Initial learning rate"
            )
            
        with gr.Row():
            warmup_steps = gr.Number(
                value=500,
                label="Warmup Steps",
                minimum=0,
                maximum=10000,
                info="Number of warmup steps"
            )
            
            gradient_accumulation = gr.Number(
                value=1,
                label="Gradient Accumulation Steps",
                minimum=1,
                maximum=32,
                info="Number of steps to accumulate gradients"
            )
            
        with gr.Row():
            fp16 = gr.Checkbox(
                value=True,
                label="Use FP16",
                info="Enable mixed precision training"
            )
            
            gradient_checkpointing = gr.Checkbox(
                value=False,
                label="Gradient Checkpointing",
                info="Enable gradient checkpointing to save memory"
            )
            
        save_config = gr.Button("Save Configuration")
        
        save_config.click(
            fn=platform.save_training_config,
            inputs=[
                model_name, output_dir, num_epochs, batch_size,
                learning_rate, warmup_steps, gradient_accumulation,
                fp16, gradient_checkpointing
            ],
            outputs=gr.Textbox(label="Status")
        )

def create_moe_config_tab(platform):
    """Create the MoE configuration tab"""
    with gr.Column():
        gr.Markdown("## Mixture of Experts Configuration")
        
        with gr.Row():
            num_experts = gr.Number(
                value=8,
                label="Number of Experts",
                minimum=1,
                maximum=128,
                info="Number of expert models"
            )
            
            expert_capacity = gr.Number(
                value=1.0,
                label="Expert Capacity Factor",
                minimum=0.1,
                maximum=10.0,
                info="Capacity factor for each expert"
            )
            
        with gr.Row():
            router_jitter = gr.Checkbox(
                value=True,
                label="Router Jitter",
                info="Add noise to router decisions during training"
            )
            
            router_z_loss = gr.Number(
                value=0.01,
                label="Router Z-Loss",
                minimum=0.0,
                maximum=1.0,
                info="Additional loss term for router"
            )
            
        with gr.Row():
            load_balancing = gr.Checkbox(
                value=True,
                label="Load Balancing",
                info="Enable expert load balancing"
            )
            
            aux_loss = gr.Number(
                value=0.01,
                label="Auxiliary Loss Weight",
                minimum=0.0,
                maximum=1.0,
                info="Weight for auxiliary load balancing loss"
            )
            
        save_config = gr.Button("Save MoE Configuration")
        
        save_config.click(
            fn=platform.save_moe_config,
            inputs=[
                num_experts, expert_capacity, router_jitter,
                router_z_loss, load_balancing, aux_loss
            ],
            outputs=gr.Textbox(label="Status")
        )

def create_inference_config_tab(platform):
    """Create the inference configuration tab"""
    with gr.Column():
        gr.Markdown("## Inference Configuration")
        
        with gr.Row():
            max_length = gr.Slider(
                minimum=1,
                maximum=4096,
                value=100,
                step=1,
                label="Max Length",
                info="Maximum number of tokens to generate"
            )
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=0.7,
                step=0.1,
                label="Temperature",
                info="Controls randomness in generation"
            )
            
        with gr.Row():
            top_p = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.9,
                step=0.05,
                label="Top P",
                info="Nucleus sampling probability"
            )
            
            top_k = gr.Slider(
                minimum=1,
                maximum=100,
                value=50,
                step=1,
                label="Top K",
                info="Top K tokens to consider"
            )
            
        save_config = gr.Button("Save Inference Configuration")
        
        save_config.click(
            fn=platform.save_inference_config,
            inputs=[max_length, temperature, top_p, top_k],
            outputs=gr.Textbox(label="Status")
        )

if __name__ == "__main__":
    interface = create_interface()
    interface.queue()
    interface.launch(server_name="0.0.0.0", server_port=7860)
