import yaml
import paramiko
from typing import Dict, List
import os

class BaremetalWorker:
    def __init__(self, config_path: str):
        self.config = yaml.safe_load(open(config_path))
        self.baremetal_config = self.config.get('baremetal', {})
        
        if not self.baremetal_config.get('enabled', False):
            return
            
        self.ssh_key = self.baremetal_config.get('ssh_key', '')
        self.ssh_user = self.baremetal_config.get('ssh_user', 'ubuntu')
        self.hosts = self.baremetal_config.get('hosts', [])
        
        # Initialize SSH client
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    def deploy_worker(self, config: Dict) -> str:
        if not self.baremetal_config.get('enabled', False):
            return "Baremetal deployment is not enabled"
            
        host = config.get('host')
        if not host:
            return "No host specified"
            
        try:
            # Connect to remote host
            self.ssh.connect(
                hostname=host,
                username=self.ssh_user,
                key_filename=self.ssh_key
            )
            
            # Deploy training code and dependencies
            self._setup_environment()
            
            return f"Successfully deployed worker on {host}"
            
        except Exception as e:
            return f"Failed to deploy worker: {str(e)}"
            
    def list_workers(self) -> List[Dict]:
        if not self.baremetal_config.get('enabled', False):
            return []
            
        workers = []
        for host in self.hosts:
            try:
                # Check if host is reachable
                self.ssh.connect(
                    hostname=host['host'],
                    username=self.ssh_user,
                    key_filename=self.ssh_key
                )
                workers.append({
                    'id': host['name'],
                    'host': host['host'],
                    'status': 'running',
                    'gpu': host.get('gpu', 'unknown')
                })
            except:
                workers.append({
                    'id': host['name'],
                    'host': host['host'],
                    'status': 'unreachable',
                    'gpu': host.get('gpu', 'unknown')
                })
        return workers
        
    def terminate_worker(self, worker_id: str):
        if not self.baremetal_config.get('enabled', False):
            return "Baremetal deployment is not enabled"
            
        host = next((h for h in self.hosts if h['name'] == worker_id), None)
        if not host:
            return f"Worker {worker_id} not found"
            
        try:
            # Connect to remote host
            self.ssh.connect(
                hostname=host['host'],
                username=self.ssh_user,
                key_filename=self.ssh_key
            )
            
            # Stop training processes
            self._cleanup_environment()
            
            return f"Successfully terminated worker {worker_id}"
            
        except Exception as e:
            return f"Failed to terminate worker: {str(e)}"
    
    def _setup_environment(self):
        """Set up the training environment on the remote host"""
        commands = [
            "sudo apt-get update",
            "sudo apt-get install -y python3-pip",
            "pip3 install -r requirements.txt"
        ]
        
        for cmd in commands:
            self.ssh.exec_command(cmd)
    
    def _cleanup_environment(self):
        """Clean up the training environment on the remote host"""
        commands = [
            "pkill -f train.py",
            "rm -rf /tmp/training_*"
        ]
        
        for cmd in commands:
            self.ssh.exec_command(cmd)
