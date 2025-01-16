import os
import sys
import json
import logging
import socket
import threading
import time
import subprocess
import psutil
import GPUtil
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/opt/illama/agent/agent.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class AgentRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            health_data = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'metrics': self.get_system_metrics()
            }
            
            self.wfile.write(json.dumps(health_data).encode())
            
        elif self.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            status_data = {
                'agent_pid': os.getpid(),
                'uptime': time.time() - psutil.Process(os.getpid()).create_time(),
                'system_info': self.get_system_info(),
                'gpu_info': self.get_gpu_info()
            }
            
            self.wfile.write(json.dumps(status_data).encode())
            
        else:
            self.send_response(404)
            self.end_headers()
            
    def do_POST(self):
        """Handle POST requests"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        request = json.loads(post_data.decode())
        
        if self.path == '/execute':
            command = request.get('command')
            if not command:
                self.send_error(400, "Missing command parameter")
                return
                
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True
                )
                
                response = {
                    'success': result.returncode == 0,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'exit_code': result.returncode
                }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                
            except Exception as e:
                self.send_error(500, str(e))
                
        else:
            self.send_response(404)
            self.end_headers()
            
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available': memory.available,
            'disk_percent': disk.percent,
            'disk_free': disk.free
        }
        
        # Add GPU metrics if available
        try:
            gpus = GPUtil.getGPUs()
            metrics['gpus'] = [{
                'id': gpu.id,
                'name': gpu.name,
                'load': gpu.load,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'temperature': gpu.temperature
            } for gpu in gpus]
        except Exception as e:
            logger.warning(f"Failed to get GPU metrics: {str(e)}")
            metrics['gpus'] = []
            
        return metrics
        
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        cpu_info = {
            'physical_cores': psutil.cpu_count(logical=False),
            'total_cores': psutil.cpu_count(logical=True),
            'max_frequency': psutil.cpu_freq().max if psutil.cpu_freq() else None
        }
        
        memory_info = {
            'total': psutil.virtual_memory().total,
            'swap_total': psutil.swap_memory().total
        }
        
        return {
            'hostname': socket.gethostname(),
            'platform': sys.platform,
            'cpu': cpu_info,
            'memory': memory_info,
            'python_version': sys.version
        }
        
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get detailed GPU information"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '-q'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return {'nvidia_smi_output': result.stdout}
            else:
                return {'error': 'nvidia-smi command failed'}
                
        except Exception as e:
            return {'error': str(e)}


def run_agent(port: int = 8080):
    """Run the agent server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, AgentRequestHandler)
    
    logger.info(f"Starting agent on port {port}")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down agent")
        httpd.server_close()


if __name__ == "__main__":
    run_agent()
