#!/bin/bash

# Set environment variables
export ILLAMA_AGENT_HOME="/opt/illama/agent"
export PYTHONPATH="${ILLAMA_AGENT_HOME}:${PYTHONPATH}"

# Check if agent is already running
if [ -f "${ILLAMA_AGENT_HOME}/agent.pid" ]; then
    pid=$(cat "${ILLAMA_AGENT_HOME}/agent.pid")
    if ps -p $pid > /dev/null 2>&1; then
        echo "Agent is already running with PID $pid"
        exit 1
    else
        rm "${ILLAMA_AGENT_HOME}/agent.pid"
    fi
fi

# Start agent in background
nohup python3 "${ILLAMA_AGENT_HOME}/agent.py" > "${ILLAMA_AGENT_HOME}/agent.log" 2>&1 &
echo $! > "${ILLAMA_AGENT_HOME}/agent.pid"

echo "Agent started with PID $(cat ${ILLAMA_AGENT_HOME}/agent.pid)"
