@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

REM ----------------------------------------------------------------------------
REM Configuration for ACE-Step (Gradio UI)
REM ----------------------------------------------------------------------------
set ARG_CHECKPOINT_PATH=--checkpoint_path ./checkpoints
set ARG_SERVER_NAME=--server_name 127.0.0.1
set ARG_PORT=--port 7867
set ARG_DEVICE_ID=--device_id 0
set ARG_SHARE=--share false
set ARG_BF16=--bf16 true
set ARG_TORCH_COMPILE=--torch_compile false
set ARG_CPU_OFFLOAD=--cpu_offload false    REM unified: false on both platforms
set ARG_OVERLAPPED_DECODE=--overlapped_decode true

REM ----------------------------------------------------------------------------
REM API unload toggle (new)
REM   Default: 1 (unload after each request)
REM   Set API_UNLOAD=0 to keep the model resident and skip unloading.
REM ----------------------------------------------------------------------------
if not defined API_UNLOAD set API_UNLOAD=1

REM ----------------------------------------------------------------------------
REM Conservative CUDA allocator tuning (override if needed)
REM Docs: https://pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf
REM ----------------------------------------------------------------------------
if not defined PYTORCH_CUDA_ALLOC_CONF set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128"

REM ----------
REM Environment setup
REM ----------
if exist ".venv\Scripts\activate.bat" (
  echo Found .venv environment
) else (
  echo Creating new .venv environment...
  python -m venv .venv
)

call ".venv\Scripts\activate.bat"

REM Upgrade pip to its latest version
python -m pip install --upgrade pip

echo Checking the Benevolence Messiah fork for updates via Git...
git pull

REM ----------------------------------------------------------------------------
REM Install/update PyTorch first (GPU if possible), then project deps.
REM Try CUDA 12.8 -> 12.4 -> 12.1 wheels; fall back to CPU-only.
REM ----------------------------------------------------------------------------
where nvidia-smi >nul 2>&1
if %errorlevel%==0 (
  echo NVIDIA GPU detected; installing CUDA-enabled PyTorch (trying cu128 → cu124 → cu121)…
  call :InstallTorchWithIndex https://download.pytorch.org/whl/cu128
  if errorlevel 1 call :InstallTorchWithIndex https://download.pytorch.org/whl/cu124
  if errorlevel 1 call :InstallTorchWithIndex https://download.pytorch.org/whl/cu121
  if errorlevel 1 (
    echo Falling back to CPU-only PyTorch…
    pip install torch torchvision torchaudio
  )
) else (
  echo No NVIDIA GPU detected; installing CPU-only PyTorch…
  pip install torch torchvision torchaudio
)

REM Install/update project dependencies
pip install -e .
pip install triton-windows

REM ----------
REM Start API service
REM ----------
echo Starting ACE-Step API service...

REM Build API args based on unload toggle (unload by default)
set "API_EXTRA_ARGS=--unload"
if "%API_UNLOAD%"=="0" (
  set "API_EXTRA_ARGS=--no-unload"
)

REM Ensure child inherits intended behavior
set "ACESTEP_UNLOAD=%API_UNLOAD%"

REM NOTE: With START, the first quoted token is the window title.
start "ACE-Step API" cmd /c "set PYTORCH_CUDA_ALLOC_CONF=%PYTORCH_CUDA_ALLOC_CONF% && python infer-api.py %API_EXTRA_ARGS%"

REM ----------
REM Build and launch ACE-Step
REM ----------
set ACESTEP_ARGS=%ARG_CHECKPOINT_PATH% %ARG_SERVER_NAME% %ARG_PORT% %ARG_DEVICE_ID% %ARG_SHARE% %ARG_BF16% %ARG_TORCH_COMPILE% %ARG_CPU_OFFLOAD% %ARG_OVERLAPPED_DECODE%

echo Starting ACE-Step Gradio Web UI with parameters:
echo %ACESTEP_ARGS%

call acestep %ACESTEP_ARGS%

:end
pause
goto :eof

:InstallTorchWithIndex
  pip install torch torchvision torchaudio --index-url %1
  exit /b %errorlevel%
