@echo off
cd /d "%~dp0"

REM ----------
REM Configure ACE-Step arguments with documentation
REM ----------

REM Model checkpoint path (default: downloads automatically)
set ARG_CHECKPOINT_PATH=--checkpoint_path ./checkpoints

REM Server bind address (default: 127.0.0.1)
set ARG_SERVER_NAME=--server_name 127.0.0.1

REM Server port (default: 7865)
set ARG_PORT=--port 7867

REM GPU device ID (default: 0)
set ARG_DEVICE_ID=--device_id 0

REM Enable Gradio sharing (default: false)
set ARG_SHARE=--share false

REM Use bfloat16 precision (default: true)
set ARG_BF16=--bf16 true

REM Use torch.compile() optimization (default: false)
set ARG_TORCH_COMPILE=--torch_compile false

REM Offload model to CPU (default: false)
set ARG_CPU_OFFLOAD=--cpu_offload true

REM Use overlapped decoding (default: false)
set ARG_OVERLAPPED_DECODE=--overlapped_decode true

REM ----------
REM Environment setup
REM ----------

REM Create virtual environment if not exists
if exist ".venv\Scripts\activate" (
    echo Found .venv environment
) else (
    echo Creating new .venv environment...
    python -m venv .venv
)

REM Activate the virtual environment
call .venv\Scripts\activate

REM Check for updates in the repository
echo Checking the Benevolence Messiah fork for updates via Git...
git pull

REM Install/update Windows/NVIDIA PyTorch packages
:: If you are on Windows and plan to use an NVIDIA GPU, install PyTorch with CUDA support first:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

REM Install/update dependencies
pip install -e .
pip install triton-windows

REM ----------
REM Start API service
REM ----------
echo Starting ACE-Step API service...
start "ACE-Step API" python infer-api.py

REM ----------
REM Build and launch ACE-Step
REM ----------

REM Combine all arguments into a single variable
set ACESTEP_ARGS=%ARG_CHECKPOINT_PATH% %ARG_SERVER_NAME% %ARG_PORT% %ARG_DEVICE_ID% %ARG_SHARE% %ARG_BF16% %ARG_TORCH_COMPILE% %ARG_CPU_OFFLOAD% %ARG_OVERLAPPED_DECODE%

REM Display configuration
echo Starting ACE-Step Gradio Web UI with parameters:
echo %ACESTEP_ARGS%

REM Launch ACE-Step service
call acestep %ACESTEP_ARGS%

REM Keep window open after execution
:end
pause