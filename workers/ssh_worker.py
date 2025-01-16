import os
import sys
import paramiko
import logging
import json
import time
from typing import Dict, Optional, List
from pathlib import Path
import subprocess
import socket
import threading
from datetime import datetime

logger = logging.getLogger(__name__)

class SSHWorker:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.load_config()
        self.active_connections = {}
        self.health_check_interval = 30  # seconds
        self._start_health_checker()
        
    def load_config(self):
        """Load SSH configuration"""
        with open(self.config_path) as f:
            config = yaml.safe_load(f)
        self.ssh_config = config.get('ssh', {})
        self.agent_path = self.ssh_config.get('agent_path', 'workers/agent')
        
    def _start_health_checker(self):
        """Start background health checker thread"""
        self.health_checker = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self.health_checker.start()
        
    def _health_check_loop(self):
        """Continuously check health of connected workers"""
        while True:
            for host, connection in list(self.active_connections.items()):
                try:
                    if not self.check_host_alive(host):
                        logger.warning(f"Host {host} is down, removing from active connections")
                        self.active_connections.pop(host, None)
                except Exception as e:
                    logger.error(f"Health check failed for {host}: {str(e)}")
            time.sleep(self.health_check_interval)
            
    def check_host_alive(self, host: str, port: int = 22, timeout: int = 5) -> bool:
        """Check if host is reachable"""
        try:
            # Try TCP connection
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except (socket.timeout, ConnectionRefusedError):
            return False
        except Exception as e:
            logger.error(f"Error checking host {host}: {str(e)}")
            return False
            
    def connect(self, host: str, username: str, 
               password: Optional[str] = None, 
               key_path: Optional[str] = None) -> bool:
        """Establish SSH connection to host"""
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            if key_path:
                key = paramiko.RSAKey.from_private_key_file(key_path)
                ssh.connect(host, username=username, pkey=key)
            else:
                ssh.connect(host, username=username, password=password)
                
            self.active_connections[host] = {
                'client': ssh,
                'username': username,
                'connected_at': datetime.now()
            }
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to {host}: {str(e)}")
            return False
            
    def deploy_agent(self, host: str) -> bool:
        """Deploy agent to remote host"""
        if host not in self.active_connections:
            logger.error(f"No active connection to {host}")
            return False
            
        try:
            ssh = self.active_connections[host]['client']
            sftp = ssh.open_sftp()
            
            # Create remote agent directory
            remote_agent_path = '/opt/illama/agent'
            stdin, stdout, stderr = ssh.exec_command(f'mkdir -p {remote_agent_path}')
            if stderr.read():
                raise Exception(f"Failed to create agent directory: {stderr.read()}")
                
            # Copy agent files
            local_agent_path = Path(self.agent_path)
            for file in local_agent_path.rglob('*'):
                if file.is_file():
                    relative_path = file.relative_to(local_agent_path)
                    remote_file = f"{remote_agent_path}/{relative_path}"
                    remote_dir = os.path.dirname(remote_file)
                    ssh.exec_command(f'mkdir -p {remote_dir}')
                    sftp.put(str(file), remote_file)
                    
            # Set permissions and start agent
            ssh.exec_command(f'chmod +x {remote_agent_path}/start.sh')
            stdin, stdout, stderr = ssh.exec_command(
                f'cd {remote_agent_path} && ./start.sh'
            )
            
            error = stderr.read()
            if error:
                raise Exception(f"Failed to start agent: {error}")
                
            logger.info(f"Successfully deployed agent to {host}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy agent to {host}: {str(e)}")
            return False
            
    def execute_command(self, host: str, command: str) -> Dict:
        """Execute command on remote host"""
        if host not in self.active_connections:
            return {'success': False, 'error': 'No active connection'}
            
        try:
            ssh = self.active_connections[host]['client']
            stdin, stdout, stderr = ssh.exec_command(command)
            
            return {
                'success': True,
                'stdout': stdout.read().decode(),
                'stderr': stderr.read().decode(),
                'exit_code': stdout.channel.recv_exit_status()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
            
    def get_worker_status(self, host: str) -> Dict:
        """Get worker status including system metrics"""
        if host not in self.active_connections:
            return {'status': 'disconnected'}
            
        try:
            result = self.execute_command(
                host,
                'cat /proc/cpuinfo | grep processor | wc -l; '
                'free -m | grep Mem; '
                'nvidia-smi --query-gpu=gpu_name,memory.total,memory.used,temperature.gpu '
                '--format=csv,noheader'
            )
            
            if not result['success']:
                raise Exception(result['error'])
                
            lines = result['stdout'].splitlines()
            cpu_count = int(lines[0])
            mem_total = int(lines[1].split()[1])
            mem_used = int(lines[1].split()[2])
            
            gpus = []
            for gpu_line in lines[2:]:
                name, total, used, temp = gpu_line.split(',')
                gpus.append({
                    'name': name.strip(),
                    'memory_total': total.strip(),
                    'memory_used': used.strip(),
                    'temperature': temp.strip()
                })
                
            return {
                'status': 'connected',
                'cpu_count': cpu_count,
                'memory_total_mb': mem_total,
                'memory_used_mb': mem_used,
                'gpus': gpus,
                'connected_since': self.active_connections[host]['connected_at'].isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get status for {host}: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def disconnect(self, host: str):
        """Disconnect from host"""
        if host in self.active_connections:
            try:
                self.active_connections[host]['client'].close()
            except Exception as e:
                logger.error(f"Error disconnecting from {host}: {str(e)}")
            finally:
                self.active_connections.pop(host, None)
