#!/bin/bash
set -euo pipefail

# Ensure API terminal is cleaned up even on Ctrl-C or errors
cleanup() { [[ -n "${API_PID:-}" ]] && kill "$API_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# Change to script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# -----------------------------------------------------------------------------
# Configuration for ACE-Step (Gradio UI)
# -----------------------------------------------------------------------------
ARG_CHECKPOINT_PATH="--checkpoint_path ./checkpoints"
ARG_SERVER_NAME="--server_name 127.0.0.1"
ARG_PORT="--port 7867"
ARG_DEVICE_ID="--device_id 0"
ARG_SHARE="--share false"
ARG_BF16="--bf16 true"
ARG_TORCH_COMPILE="--torch_compile false"
ARG_CPU_OFFLOAD="--cpu_offload false"          # unified: false on both platforms
ARG_OVERLAPPED_DECODE="--overlapped_decode true"

# -----------------------------------------------------------------------------
# API unload toggle (new)
#   Default: 1 (unload after each request)
#   Set API_UNLOAD=0 to keep the model resident and skip unloading.
# -----------------------------------------------------------------------------
API_UNLOAD="${API_UNLOAD:-1}"

# -----------------------------------------------------------------------------
# Conservative CUDA allocator tuning (override via env if you like)
# Docs: https://pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf
# -----------------------------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
VENV_DIR=".venv"

# Create virtual environment if not exists
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating new .venv environment..."
  python3 -m venv "$VENV_DIR"
fi

# Activate the virtual environment
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Upgrade pip to its latest version
python -m pip install --upgrade pip

# Check for updates in the repository
echo "Checking the Benevolence Messiah fork for updates via Git..."
git pull

# -----------------------------------------------------------------------------
# Install/update PyTorch first (GPU if possible), then project deps
# Following PyTorch guidance: choose CUDA build appropriate for your system.
# Will try cu128 -> cu124 -> cu121; fallback to CPU-only if no GPU or all fail.
# -----------------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA GPU detected; installing CUDA-enabled PyTorch (trying cu128 → cu124 → cu121)…"
  set +e
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 || \
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 || \
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  PT_EXIT=$?
  set -e
  if [ $PT_EXIT -ne 0 ]; then
    echo "Falling back to CPU-only PyTorch…"
    pip install torch torchvision torchaudio
  fi
else
  echo "No NVIDIA GPU detected; installing CPU-only PyTorch…"
  pip install torch torchvision torchaudio
fi

# Now install/update ACE-Step (editable)
echo "Installing/updating ACE-Step (editable)…"
pip install -e .

# Combined UI arguments
ACESTEP_ARGS="$ARG_CHECKPOINT_PATH $ARG_SERVER_NAME $ARG_PORT $ARG_DEVICE_ID $ARG_SHARE $ARG_BF16 $ARG_TORCH_COMPILE $ARG_CPU_OFFLOAD $ARG_OVERLAPPED_DECODE"

# --------------------------------------------------------------------
#  Start the API server in its own terminal window
# --------------------------------------------------------------------
echo "Starting ACE-Step API service in a new terminal …"
API_LOG="api_$(date +%Y%m%d_%H%M%S).log"

# Build API extra args for unload behavior (unload by default)
API_EXTRA_ARGS="--unload"
if [ "$API_UNLOAD" = "0" ]; then
  API_EXTRA_ARGS="--no-unload"
fi

# Launch API in separate terminal; ensure env var is exported only for this command
if command -v gnome-terminal >/dev/null 2>&1; then
  gnome-terminal --title="ACEStep-API" -- bash -c "
    source '$VENV_DIR/bin/activate';
    ACESTEP_UNLOAD=$API_UNLOAD PYTORCH_CUDA_ALLOC_CONF='$PYTORCH_CUDA_ALLOC_CONF' \
      python infer-api.py $API_EXTRA_ARGS 2>&1 | tee '$API_LOG';
    read -p 'Press ENTER to close …'
  " &
else
  echo "gnome-terminal not found; starting API in this terminal…"
  ACESTEP_UNLOAD=$API_UNLOAD PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    python infer-api.py $API_EXTRA_ARGS 2>&1 | tee "$API_LOG" &
fi

# Store API process ID (this is the terminal or background python)
API_PID=$!

# Launch ACE-Step service (Gradio UI)
echo "Starting ACE-Step Gradio Web UI with parameters:"
echo "$ACESTEP_ARGS"
acestep $ACESTEP_ARGS

echo "ACE-Step execution completed."
