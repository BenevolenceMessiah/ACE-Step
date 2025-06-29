#!/bin/bash

# Change to script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Configure ACE-Step arguments with documentation
ARG_CHECKPOINT_PATH="--checkpoint_path ./checkpoints"
ARG_SERVER_NAME="--server_name 127.0.0.1"
ARG_PORT="--port 7867"
ARG_DEVICE_ID="--device_id 0"
ARG_SHARE="--share false"
ARG_BF16="--bf16 true"
ARG_TORCH_COMPILE="--torch_compile false"
ARG_CPU_OFFLOAD="--cpu_offload true"
ARG_OVERLAPPED_DECODE="--overlapped_decode true"

# Environment setup
VENV_DIR=".venv"

# Create virtual environment if not exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating new .venv environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Check for updates in the repository
echo "Checking the Benevolence Messiah fork for updates via Git..."
git pull

# Install/update dependencies
echo "Installing/updating requirements..."
pip install -e .

# Combine all arguments into a single variable
ACESTEP_ARGS="$ARG_CHECKPOINT_PATH $ARG_SERVER_NAME $ARG_PORT $ARG_DEVICE_ID $ARG_SHARE $ARG_BF16 $ARG_TORCH_COMPILE $ARG_CPU_OFFLOAD $ARG_OVERLAPPED_DECODE"

# Start API service in background
echo "Starting ACE-Step API service..."
python infer-api.py &

# Store API process ID
API_PID=$!

# Launch ACE-Step service
echo "Starting ACE-Step Gradio Web UI with parameters:"
echo "$ACESTEP_ARGS"
acestep $ACESTEP_ARGS

# Cleanup after ACE-Step exits
kill $API_PID
echo "ACE-Step execution completed."