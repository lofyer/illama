import os
import tarfile
import tempfile
import shutil
from pathlib import Path
import time

def create_agent_package(output_dir: str = None) -> str:
    """Create a downloadable agent package"""
    
    try:
        # Create output directory if it doesn't exist
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Define the output package path
        if output_dir:
            package_name = f"illama-agent-{int(time.time())}.tar.gz"
            package_path = os.path.join(output_dir, package_name)
        else:
            package_name = f"illama-agent-{int(time.time())}.tar.gz"
            package_path = os.path.join(os.getcwd(), package_name)
        
        print(f"Creating agent package at {package_path}")
        
        # Create a temporary directory for building
        temp_dir = tempfile.mkdtemp()
        
        agent_dir = Path(temp_dir) / "illama-agent"
        agent_dir.mkdir()
        
        # Copy agent files
        source_dir = Path(__file__).parent / "agent"
        for file in source_dir.glob("**/*"):
            if file.is_file():
                relative_path = file.relative_to(source_dir)
                target_path = agent_dir / relative_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, target_path)
        
        # Create setup script
        setup_script = """#!/bin/bash
# Setup script for Illama Agent

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed. Please install Python 3 first."
    exit 1
fi

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Make agent executable
chmod +x agent.py

# Create systemd service file
sudo tee /etc/systemd/system/illama-agent.service << EOF
[Unit]
Description=Illama Agent Service
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin:$PATH
ExecStart=$(pwd)/venv/bin/python agent.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Start and enable service
sudo systemctl daemon-reload
sudo systemctl enable illama-agent
sudo systemctl start illama-agent

echo "Illama Agent has been installed and started as a system service"
"""
        
        setup_path = agent_dir / "setup.sh"
        setup_path.write_text(setup_script)
        setup_path.chmod(0o755)
        
        # Create tar archive
        with tarfile.open(package_path, "w:gz") as tar:
            tar.add(agent_dir, arcname="illama-agent")
        
        print(f"Agent package created successfully at {package_path}")
        return package_path
        
    except Exception as e:
        print(f"Error creating agent package: {str(e)}")
        raise
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    output = create_agent_package()
    print(f"Agent package created at: {output}")
