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
set ARG_CPU_OFFLOAD=--cpu_offload false
set ARG_OVERLAPPED_DECODE=--overlapped_decode true

REM Combine into one string (shown in the UI window)
set ACESTEP_ARGS=%ARG_CHECKPOINT_PATH% %ARG_SERVER_NAME% %ARG_PORT% %ARG_DEVICE_ID% %ARG_SHARE% %ARG_BF16% %ARG_TORCH_COMPILE% %ARG_CPU_OFFLOAD% %ARG_OVERLAPPED_DECODE%

REM ----------------------------------------------------------------------------
REM Behavior toggles (mirror Linux)
REM ----------------------------------------------------------------------------
if not defined API_UNLOAD set API_UNLOAD=1
if not defined LOG_DIR set "LOG_DIR=logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Conservative CUDA allocator (good defaults for large models)
if not defined PYTORCH_CUDA_ALLOC_CONF set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128"

REM ----------------------------------------------------------------------------
REM Python environment
REM ----------------------------------------------------------------------------
if exist ".venv\Scripts\activate.bat" (
  echo Found .venv environment
) else (
  echo Creating new .venv environment...
  python -m venv .venv
)

call ".venv\Scripts\activate.bat"
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

REM Install/update the project (editable)
pip install -e .

REM ----------------------------------------------------------------------------
REM Ensure FFmpeg exists; offer to install via WinGet (preferred) or Chocolatey.
REM ----------------------------------------------------------------------------
call :EnsureFFmpeg || (
  echo.
  echo FFmpeg is required for Opus transcoding. Continuing without it (transcoding disabled)…
)

REM ----------------------------------------------------------------------------
REM Build API args based on unload toggle (unload by default)
REM ----------------------------------------------------------------------------
set "API_EXTRA_ARGS=--unload"
if "%API_UNLOAD%"=="0" set "API_EXTRA_ARGS=--no-unload"

REM Ensure API inherits intended behavior + log dir
set "ACESTEP_UNLOAD=%API_UNLOAD%"
set "ACESTEP_LOG_DIR=%LOG_DIR%"

REM ----------------------------------------------------------------------------
REM Launch TWO terminals (API + UI), keep them open with cmd /k
REM ----------------------------------------------------------------------------
echo.
echo Starting ACE-Step API service in its own terminal window…
start "ACE-Step API" cmd /k ^
  "call .venv\Scripts\activate && ^
   set ACESTEP_LOG_DIR=%ACESTEP_LOG_DIR% && ^
   set ACESTEP_UNLOAD=%ACESTEP_UNLOAD% && ^
   set PYTORCH_CUDA_ALLOC_CONF=%PYTORCH_CUDA_ALLOC_CONF% && ^
   set LOG_LEVEL=info && ^
   echo [API] Env: ACESTEP_UNLOAD=%ACESTEP_UNLOAD%  ACESTEP_LOG_DIR=%ACESTEP_LOG_DIR%  PYTORCH_CUDA_ALLOC_CONF=%PYTORCH_CUDA_ALLOC_CONF% && ^
   python infer-api.py %API_EXTRA_ARGS%"

echo.
echo Starting ACE-Step Gradio UI in its own terminal window…
start "ACE-Step UI" cmd /k ^
  "call .venv\Scripts\activate && ^
   echo [UI] acestep %ACESTEP_ARGS% && ^
   acestep %ACESTEP_ARGS%"

echo.
echo Two terminals have been launched:
echo   - ACE-Step API (Uvicorn service on port 8000)
echo   - ACE-Step UI  (Gradio interface on port 7867)
echo Close those windows to stop the services.
echo.
pause
goto :eof


:InstallTorchWithIndex
  pip install torch torchvision torchaudio --index-url %1
  exit /b %errorlevel%


:EnsureFFmpeg
  where ffmpeg >nul 2>&1
  if %errorlevel%==0 (
    for /f "delims=" %%p in ('where ffmpeg') do echo FFmpeg found: %%p
    exit /b 0
  )
  echo FFmpeg is not installed or not on PATH.
  if /I "%AUTO_INSTALL_FFMPEG%"=="1" (
    set "choice=Y"
  ) else (
    set /p choice="Install FFmpeg now via WinGet (preferred) or Chocolatey? [Y/n] "
    if not defined choice set "choice=Y"
  )
  if /I "%choice%"=="Y" (
    where winget >nul 2>&1
    if %errorlevel%==0 (
      echo Installing FFmpeg via WinGet…
      winget install -e --id Gyan.FFmpeg
    ) else (
      where choco >nul 2>&1
      if %errorlevel%==0 (
        echo Installing FFmpeg via Chocolatey…
        choco install ffmpeg -y
      ) else (
        echo Neither WinGet nor Chocolatey found. Please install FFmpeg manually:
        echo   https://ffmpeg.org/download.html
        exit /b 1
      )
    )
    REM Re-check PATH; some systems require a new shell session for PATH to refresh.
    where ffmpeg >nul 2>&1
    if %errorlevel%==0 (
      for /f "delims=" %%p in ('where ffmpeg') do echo FFmpeg found after install: %%p
      exit /b 0
    ) else (
      echo FFmpeg may be installed but PATH not refreshed. Open a NEW terminal before re-running.
      exit /b 1
    )
  ) else (
    echo Skipping FFmpeg install. Opus transcoding will be unavailable.
    exit /b 0
  )
