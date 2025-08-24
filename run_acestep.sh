#!/bin/bash
set -euo pipefail

# Ensure any background API terminal is cleaned up if this script is killed (best-effort)
cleanup() { [[ -n "${API_TERM_PID:-}" ]] && kill "$API_TERM_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

RUN_DIR="${RUN_DIR:-run}"
LOG_DIR="${LOG_DIR:-logs}"        # used by API internals; launcher does not redirect to files
mkdir -p "$RUN_DIR" "$LOG_DIR"

# -----------------------------------------------------------------------------
# Configuration (Gradio UI)
# -----------------------------------------------------------------------------
ARG_CHECKPOINT_PATH="--checkpoint_path ./checkpoints"
ARG_SERVER_NAME="--server_name 127.0.0.1"
ARG_PORT="--port 7867"
ARG_DEVICE_ID="--device_id 0"
ARG_SHARE="--share false"
ARG_BF16="--bf16 true"
ARG_TORCH_COMPILE="--torch_compile false"
ARG_CPU_OFFLOAD="--cpu_offload false"
ARG_OVERLAPPED_DECODE="--overlapped_decode true"

ACESTEP_ARGS="$ARG_CHECKPOINT_PATH $ARG_SERVER_NAME $ARG_PORT $ARG_DEVICE_ID $ARG_SHARE $ARG_BF16 $ARG_TORCH_COMPILE $ARG_CPU_OFFLOAD $ARG_OVERLAPPED_DECODE"

# -----------------------------------------------------------------------------
# Behavior toggles
# -----------------------------------------------------------------------------
API_UNLOAD="${API_UNLOAD:-1}"           # 1 = unload after each request (default), 0 = keep loaded
OPEN_TERMINALS="${OPEN_TERMINALS:-1}"   # 1 = open terminals (default)
WAIT_API_SECS="${WAIT_API_SECS:-90}"
WAIT_UI_SECS="${WAIT_UI_SECS:-90}"
HOST_BIND="${HOST_BIND:-127.0.0.1}"

# CUDA allocator tuning (sane defaults for large models)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

# -----------------------------------------------------------------------------
# Python venv & deps
# -----------------------------------------------------------------------------
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating new .venv environment..."
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

echo "Checking the Benevolence Messiah fork for updates via Git..."
git pull

# Install PyTorch (GPU if possible), then project deps
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

echo "Installing/updating ACE-Step (editable)…"
pip install -e .

# -----------------------------------------------------------------------------
# FFmpeg check & optional install
# -----------------------------------------------------------------------------
ensure_ffmpeg() {
  if command -v ffmpeg >/dev/null 2>&1; then
    echo "FFmpeg found: $(command -v ffmpeg)"
    return 0
  fi
  echo "FFmpeg is not installed or not on PATH."
  if [ "${AUTO_INSTALL_FFMPEG:-0}" = "1" ]; then
    choice="y"
  else
    read -r -p "Install FFmpeg now? [y/N] " choice || choice="n"
  fi
  case "$choice" in
    y|Y)
      if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update && sudo apt-get install -y ffmpeg   # Ubuntu/Debian. 
      elif command -v dnf >/dev/null 2>&1; then
        if ! sudo dnf install -y ffmpeg; then
          echo "Enable RPM Fusion first for full ffmpeg on Fedora: https://rpmfusion.org/Configuration"
          return 1
        fi
      elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -S --noconfirm ffmpeg                        # Arch.
      else
        echo "Unsupported package manager. Please install FFmpeg manually."
        return 1
      fi
      ;;
    *) echo "Skipping FFmpeg install. Discord Opus export will be unavailable."; return 0 ;;
  esac
  command -v ffmpeg >/dev/null 2>&1 && echo "FFmpeg installed: $(command -v ffmpeg)" || echo "FFmpeg still not found on PATH."
}
ensure_ffmpeg

# -----------------------------------------------------------------------------
# Helpers (readiness checks)
# -----------------------------------------------------------------------------
wait_http() { # url timeout_seconds
  local url="$1" timeout="$2" t=0
  while (( t < timeout )); do
    if command -v curl >/dev/null 2>&1; then
      if curl -fsS -m 2 "$url" >/dev/null 2>&1; then return 0; fi
    else
      local host port
      host="$(echo "$url" | sed -E 's#^https?://([^:/]+).*$#\1#')" || true
      port="$(echo "$url" | sed -E 's#^https?://[^:/]+:([0-9]+).*$#\1#' | sed 's/^$/80/')" || true
      (echo >"/dev/tcp/$host/$port") >/dev/null 2>&1 && return 0
    fi
    sleep 1; t=$((t+1))
  done
  return 1
}

wait_port() { # host port timeout_seconds
  local host="$1" port="$2" timeout="$3" t=0
  while (( t < timeout )); do
    (echo >/dev/tcp/"$host"/"$port") >/dev/null 2>&1 && return 0 || true
    sleep 1; t=$((t+1))
  done
  return 1
}

# -----------------------------------------------------------------------------
# Terminal launcher (API & UI) – keeps window open even on crash
# -----------------------------------------------------------------------------
launch_in_terminal() {
  local title="$1" cmd="$2"
  local hold="; code=\$?; echo; echo '---'; echo \"$title exited with code: \$code\"; echo 'Press ENTER to close…'; read"
  if command -v gnome-terminal >/dev/null 2>&1; then
    gnome-terminal --title="$title" -- bash -lc "$cmd$hold; exec bash"
  elif command -v konsole >/dev/null 2>&1; then
    konsole --hold -p tabtitle="$title" -e bash -lc "$cmd$hold"
  elif command -v xfce4-terminal >/dev/null 2>&1; then
    xfce4-terminal --title="$title" --hold -e "bash -lc \"$cmd$hold\""
  elif command -v xterm >/dev/null 2>&1; then
    xterm -T "$title" -hold -e bash -lc "$cmd$hold"
  else
    echo "No graphical terminal emulator found. Running in this shell:"
    echo ">> $cmd"
    bash -lc "$cmd"
  fi
}

API_EXTRA_ARGS="--unload"; [ "$API_UNLOAD" = "0" ] && API_EXTRA_ARGS="--no-unload"

# -----------------------------------------------------------------------------
# Start API (own terminal)
# -----------------------------------------------------------------------------
echo "Starting ACE-Step API service in its own terminal…"
API_CMD="source '$VENV_DIR/bin/activate'; ACESTEP_LOG_DIR='$LOG_DIR' ACESTEP_UNLOAD='$API_UNLOAD' LOG_LEVEL=info PYTORCH_CUDA_ALLOC_CONF='$PYTORCH_CUDA_ALLOC_CONF' python infer-api.py $API_EXTRA_ARGS"
launch_in_terminal 'ACEStep-API' "$API_CMD" || true

# -----------------------------------------------------------------------------
# Start UI (own terminal)
# -----------------------------------------------------------------------------
echo "Starting ACE-Step Gradio UI in its own terminal…"
UI_CMD="source '$VENV_DIR/bin/activate'; acestep $ACESTEP_ARGS"
launch_in_terminal 'ACEStep-UI' "$UI_CMD" || true

# -----------------------------------------------------------------------------
# Readiness checks (from the launcher)
# -----------------------------------------------------------------------------
UI_PORT="$(echo "$ARG_PORT" | awk '{print $2}')"

if wait_http "http://$HOST_BIND:8000/health" "$WAIT_API_SECS"; then
  echo "API is ready → http://$HOST_BIND:8000"
else
  echo "API did not become ready within ${WAIT_API_SECS}s. See the API terminal window."
fi

if wait_port "$HOST_BIND" "$UI_PORT" "$WAIT_UI_SECS"; then
  echo "UI is ready → http://$HOST_BIND:$UI_PORT"
else
  echo "UI did not start listening within ${WAIT_UI_SECS}s. See the UI terminal window."
fi

echo "Done. Two terminals are running (close them to stop the services)."
