import yaml
import psutil
import json
from typing import List, Dict, Optional
from abc import ABC, abstractmethod

class WorkerManager(ABC):
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.workers = {}

    @staticmethod
    def _load_config(config_path: str) -> Dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    @abstractmethod
    def deploy_worker(self, worker_config: Dict) -> str:
        """Deploy a new worker with given configuration"""
        pass

    @abstractmethod
    def terminate_worker(self, worker_id: str):
        """Terminate a worker by its ID"""
        pass

    @abstractmethod
    def get_worker_status(self, worker_id: str) -> Dict:
        """Get status of a specific worker"""
        pass

    def list_workers(self) -> List[Dict]:
        """List all workers and their status"""
        return [
            {"id": worker_id, "status": self.get_worker_status(worker_id)}
            for worker_id in self.workers
        ]

    def get_worker_metrics(self, worker_id: str) -> Dict:
        """Get resource metrics for a worker"""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent
        }

    def save_state(self, state_file: str):
        """Save worker manager state to file"""
        state = {
            "workers": self.workers,
            "config": self.config
        }
        with open(state_file, 'w') as f:
            json.dump(state, f)

    def load_state(self, state_file: str):
        """Load worker manager state from file"""
        with open(state_file, 'r') as f:
            state = json.load(f)
            self.workers = state["workers"]
            self.config.update(state["config"])
