import boto3
from typing import Dict
from core.worker_manager import WorkerManager

class CloudWorker(WorkerManager):
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.ec2 = boto3.client('ec2',
                              aws_access_key_id=self.config['aws']['access_key'],
                              aws_secret_access_key=self.config['aws']['secret_key'],
                              region_name=self.config['aws']['region'])

    def deploy_worker(self, worker_config: Dict) -> str:
        """Deploy a new EC2 instance as worker"""
        response = self.ec2.run_instances(
            ImageId=worker_config['ami_id'],
            InstanceType=worker_config['instance_type'],
            MinCount=1,
            MaxCount=1,
            UserData=self._get_init_script(),
            TagSpecifications=[{
                'ResourceType': 'instance',
                'Tags': [
                    {'Key': 'Name', 'Value': f'LLM-Worker-{len(self.workers)}'},
                    {'Key': 'Project', 'Value': 'iLLaMA2'},
                ]
            }]
        )
        instance_id = response['Instances'][0]['InstanceId']
        self.workers[instance_id] = {
            'config': worker_config,
            'status': 'initializing'
        }
        return instance_id

    def terminate_worker(self, worker_id: str):
        """Terminate an EC2 instance"""
        self.ec2.terminate_instances(InstanceIds=[worker_id])
        del self.workers[worker_id]

    def get_worker_status(self, worker_id: str) -> Dict:
        """Get EC2 instance status"""
        response = self.ec2.describe_instances(InstanceIds=[worker_id])
        instance = response['Reservations'][0]['Instances'][0]
        return {
            'state': instance['State']['Name'],
            'public_ip': instance.get('PublicIpAddress'),
            'launch_time': instance['LaunchTime'].isoformat(),
            'config': self.workers[worker_id]['config']
        }

    def _get_init_script(self) -> str:
        """Generate initialization script for worker"""
        return """#!/bin/bash
# Install dependencies
apt-get update
apt-get install -y python3-pip git

# Clone repository and install requirements
git clone https://github.com/your-repo/illama2.git
cd illama2
pip3 install -r requirements.txt

# Start worker service
python3 worker_service.py
"""
