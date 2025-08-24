# infer-api.py
from __future__ import annotations

import base64
import gc
import logging
import logging.config
import os
import shutil
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple, Callable

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from acestep.pipeline_ace_step import ACEStepPipeline
from acestep.data_sampler import DataSampler  # keep import to avoid breaking external users

# -----------------------------------------------------------------------------
# Log directory
# -----------------------------------------------------------------------------
LOG_DIR = os.getenv("ACESTEP_LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Uvicorn-compatible logging dictConfig (console + rotating files)
# -----------------------------------------------------------------------------
def build_logging_config(log_dir: str) -> Dict[str, Any]:
    uvicorn_log_path = os.path.join(log_dir, "uvicorn.log")
    app_log_path = os.path.join(log_dir, "app.log")

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "uvicorn_default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s [%(asctime)s] %(name)s: %(message)s",
                "use_colors": None,
            },
            "uvicorn_access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s [%(asctime)s] %(client_addr)s - "%(request_line)s" %(status_code)s',
                "use_colors": None,
            },
            "app": {
                "format": "[%(asctime)s] %(levelname)s %(name)s :: %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "uvicorn_default",
                "stream": "ext://sys.stdout",
            },
            "console_access": {
                "class": "logging.StreamHandler",
                "formatter": "uvicorn_access",
                "stream": "ext://sys.stdout",
            },
            "uvicorn_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": uvicorn_log_path,
                "maxBytes": 10 * 1024 * 1024,
                "backupCount": 5,
                "encoding": "utf-8",
                "formatter": "uvicorn_default",
            },
            "uvicorn_access_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": uvicorn_log_path,
                "maxBytes": 10 * 1024 * 1024,
                "backupCount": 5,
                "encoding": "utf-8",
                "formatter": "uvicorn_access",
            },
            "app_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": app_log_path,
                "maxBytes": 10 * 1024 * 1024,
                "backupCount": 5,
                "encoding": "utf-8",
                "formatter": "app",
            },
        },
        "loggers": {
            "uvicorn.error": {
                "handlers": ["console", "uvicorn_file"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["console_access", "uvicorn_access_file"],
                "level": "INFO",
                "propagate": False,
            },
            "acestep.api": {
                "handlers": ["console", "app_file"],
                "level": "INFO",
                "propagate": False,
            },
        },
        "root": {"handlers": ["console", "app_file"], "level": "INFO"},
    }

uvicorn_logger = logging.getLogger("uvicorn.error")
api_logger = logging.getLogger("acestep.api")
for lg in (uvicorn_logger, api_logger):
    lg.setLevel(logging.INFO)

def _console(msg: str) -> None:
    print(msg, flush=True)

def _log_both(msg: str) -> None:
    uvicorn_logger.info(msg)
    api_logger.info(msg)
    _console(msg)

# -----------------------------------------------------------------------------
# Optional Prometheus metrics (v7.1.0 API)
# -----------------------------------------------------------------------------
METRICS_AVAILABLE = True
try:
    from prometheus_fastapi_instrumentator import Instrumentator, metrics as p_metrics
    from prometheus_fastapi_instrumentator.metrics import Info
    from prometheus_client import Gauge
except Exception:
    METRICS_AVAILABLE = False

# -----------------------------------------------------------------------------
# Runtime config
# -----------------------------------------------------------------------------
UNLOAD_AFTER_RUN: bool = os.getenv("ACESTEP_UNLOAD", "1") != "0"
SERVER_BOOT: Dict[str, Any] = {
    "host": os.getenv("HOST", "0.0.0.0"),
    "port": int(os.getenv("PORT", "8000")),
    "cli_unload": None,  # set in __main__
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _fmt_bytes(n: int) -> str:
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(x) < 1024.0 or unit == "TiB":
            return f"{int(x)} {unit}" if unit == "B" else f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{x:.2f} TiB"

def _cuda_mem() -> Dict[str, Any]:
    if torch.cuda.is_available():
        allocated = int(torch.cuda.memory_allocated())
        reserved = int(torch.cuda.memory_reserved())
        device = torch.cuda.current_device()
        name = torch.cuda.get_device_name(device)
        return {
            "cuda": True,
            "device_index": device,
            "device_name": name,
            "allocated_bytes": allocated,
            "reserved_bytes": reserved,
            "allocated_hr": _fmt_bytes(allocated),
            "reserved_hr": _fmt_bytes(reserved),
        }
    return {"cuda": False}

def _cuda_device_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"cuda_available": torch.cuda.is_available()}
    if not info["cuda_available"]:
        return info
    devs = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        devs.append({
            "index": i,
            "name": props.name,
            "total_memory_bytes": int(props.total_memory),
            "total_memory_hr": _fmt_bytes(int(props.total_memory)),
            "major": props.major,
            "minor": props.minor,
        })
    info["devices"] = devs
    return info

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class ACEStepInput(BaseModel):
    """
    Input schema for the ACEStep inference API.

    Key additions for Discord/file delivery:
      * discord_voice: when true, transcode to Ogg/Opus 48 kHz stereo (libopus).
      * opus_bitrate_kbps: desired Opus bitrate (we may downshift to fit a target size).
      * target_max_bytes: size cap for attachments (e.g., 10 MiB for non-Nitro).
    """
    format: str = "wav"

    checkpoint_path: str
    bf16: bool = True
    torch_compile: bool = False
    device_id: int = 0
    output_path: Optional[str] = None
    audio_duration: float
    prompt: str
    lyrics: str
    infer_step: int
    guidance_scale: float
    scheduler_type: str
    cfg_type: str
    omega_scale: float
    manual_seeds: List[int]
    guidance_interval: float
    guidance_interval_decay: float
    min_guidance_scale: float
    use_erg_tag: bool
    use_erg_lyric: bool
    use_erg_diffusion: bool
    oss_steps: List[int]
    guidance_scale_text: float = 0.0
    guidance_scale_lyric: float = 0.0

    # Discord/file delivery helpers (optional)
    discord_voice: bool = False
    opus_bitrate_kbps: int = 96
    target_max_bytes: Optional[int] = None  # e.g., 10 * 1024 * 1024

class ACEStepOutput(BaseModel):
    status: str
    output_path: Optional[str]
    message: str
    audio_data: Optional[str] = None
    deliverable_path: Optional[str] = None
    deliverable_format: Optional[str] = None
    deliverable_codec: Optional[str] = None
    deliverable_bitrate_kbps: Optional[int] = None
    duration_s: Optional[float] = None
    file_size_bytes: Optional[int] = None

# -----------------------------------------------------------------------------
# Pipeline init / teardown
# -----------------------------------------------------------------------------
def initialize_pipeline(
    checkpoint_path: str, bf16: bool, torch_compile: bool, device_id: int
) -> ACEStepPipeline:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    dtype = "bfloat16" if bf16 else "float32"
    _log_both(
        f"[init] Initializing ACEStepPipeline on device_id={device_id} dtype={dtype} "
        f"(torch_compile={torch_compile})"
    )
    return ACEStepPipeline(
        checkpoint_dir=checkpoint_path,
        dtype=dtype,
        torch_compile=torch_compile,
    )

def _unload_pipeline(model: Optional[ACEStepPipeline]) -> None:
    before = _cuda_mem()
    devmap: Dict[str, Any] = {}
    try:
        if model is not None:
            for name in ("ace_step_transformer", "text_encoder_model", "music_dcae"):
                mod = getattr(model, name, None)
                if mod is not None and hasattr(mod, "device"):
                    try:
                        devmap[name] = str(mod.device)
                    except Exception:
                        devmap[name] = "unknown"
    except Exception:
        pass

    _log_both(
        "[unload] BEFORE  "
        f"devices={devmap}  allocated={before.get('allocated_hr','?')} "
        f"reserved={before.get('reserved_hr','?')}"
    )

    try:
        if model is not None:
            try:
                if getattr(model, "ace_step_transformer", None) is not None:
                    model.ace_step_transformer.to("cpu")
            except Exception:
                pass
            try:
                if getattr(model, "text_encoder_model", None) is not None:
                    model.text_encoder_model.to("cpu")
            except Exception:
                pass
            try:
                if getattr(model, "music_dcae", None) is not None:
                    model.music_dcae.to("cpu")
            except Exception:
                pass
    finally:
        try:
            del model
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    after = _cuda_mem()
    _log_both(
        "[unload] AFTER   "
        f"allocated={after.get('allocated_hr','?')}  reserved={after.get('reserved_hr','?')}"
    )

# -----------------------------------------------------------------------------
# Discord Opus transcode helpers
# -----------------------------------------------------------------------------
def _guess_bitrate_to_fit(duration_s: float, target_max_bytes: int, requested_kbps: int) -> int:
    if duration_s <= 0:
        return requested_kbps
    max_kbps = int((target_max_bytes * 8) / duration_s / 1000) - 8  # header overhead
    max_kbps = max(16, min(max_kbps, 256))
    return min(requested_kbps, max_kbps)

def _transcode_to_opus(
    src_path: str,
    bitrate_kbps: int,
    out_path: Optional[str] = None,
) -> Tuple[str, int]:
    if shutil.which("ffmpeg") is None:
        raise HTTPException(status_code=500, detail="FFmpeg not found in PATH, cannot transcode to Opus.")

    if out_path is None:
        base, _ = os.path.splitext(src_path)
        out_path = f"{base}.ogg"

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", src_path, "-vn", "-ac", "2", "-ar", "48000",
        "-c:a", "libopus", "-b:a", f"{bitrate_kbps}k",
        "-vbr", "on", "-compression_level", "10", "-application", "audio",
        out_path,
    ]
    _log_both(f"[opus] FFmpeg cmd: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"FFmpeg failed to encode Opus: {e}")

    return out_path, bitrate_kbps

# -----------------------------------------------------------------------------
# FastAPI app (lifespan)
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    conf = {
        "effective_unload": UNLOAD_AFTER_RUN,
        "env.ACESTEP_UNLOAD": os.getenv("ACESTEP_UNLOAD", None),
        "env.PYTORCH_CUDA_ALLOC_CONF": os.getenv("PYTORCH_CUDA_ALLOC_CONF", None),
        "env.ACESTEP_LOG_DIR": os.getenv("ACESTEP_LOG_DIR", None),
        "server": {"host": SERVER_BOOT.get("host"), "port": SERVER_BOOT.get("port")},
        "cli_unload": SERVER_BOOT.get("cli_unload"),
    }
    cuda_info = _cuda_device_info()
    mem = _cuda_mem()
    _log_both("======== ACE-Step API Startup ========")
    _log_both(f"Config: {conf}")
    _log_both(f"CUDA:   {cuda_info}")
    _log_both(f"Memory: allocated={mem.get('allocated_hr','?')} reserved={mem.get('reserved_hr','?')}")
    _log_both("======================================")
    yield

# Create the FastAPI app ONCE with lifespan
app = FastAPI(title="ACEStep Pipeline API", lifespan=lifespan)

# ---- Prometheus metrics: instrument BEFORE startup (v7.1.0 API) ----
if METRICS_AVAILABLE:
    try:
        instrumentator = Instrumentator(
            should_group_status_codes=True,
            should_ignore_untemplated=True,
            should_respect_env_var=False,                 # toggle via env if you prefer
            should_instrument_requests_inprogress=True,   # <- correct flag name
            excluded_handlers=["/metrics"],
            inprogress_name="inprogress",
            inprogress_labels=True,                       # <- correct kwarg (no extra underscore)
        )

        # Default helpful metrics
        instrumentator.add(p_metrics.request_size())
        instrumentator.add(p_metrics.response_size())
        instrumentator.add(p_metrics.latency(buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5)))

        # GPU memory gauges (current device)
        GA = Gauge("acestep_cuda_allocated_bytes", "Current CUDA allocated bytes", ["device_index", "device_name"])
        GR = Gauge("acestep_cuda_reserved_bytes", "Current CUDA reserved bytes", ["device_index", "device_name"])

        def cuda_memory_metrics() -> Callable[[Info], None]:
            def instrumentation(_: Info) -> None:
                if torch.cuda.is_available():
                    idx = torch.cuda.current_device()
                    name = torch.cuda.get_device_name(idx)
                    GA.labels(str(idx), name).set(float(torch.cuda.memory_allocated()))
                    GR.labels(str(idx), name).set(float(torch.cuda.memory_reserved()))
            return instrumentation

        instrumentator.add(cuda_memory_metrics())

        # Register middleware & expose endpoint now (pre-startup)
        instrumentator.instrument(app)
        instrumentator.expose(app, include_in_schema=False, endpoint="/metrics", should_gzip=True)
        _console("[metrics] Prometheus instrumented at /metrics")
    except Exception as e:
        _console(f"[metrics] Prometheus not available or failed to initialize: {e}")
else:
    _console("[metrics] Prometheus not available (install prometheus-fastapi-instrumentator + prometheus-client).")

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
class GenerateRequest(ACEStepInput):
    pass

@app.post("/generate", response_model=ACEStepOutput)
async def generate_audio(input_data: GenerateRequest):
    model_demo: Optional[ACEStepPipeline] = None
    try:
        model_demo = initialize_pipeline(
            input_data.checkpoint_path,
            input_data.bf16,
            input_data.torch_compile,
            input_data.device_id,
        )
        params = (
            input_data.format,
            input_data.audio_duration,
            input_data.prompt,
            input_data.lyrics,
            input_data.infer_step,
            input_data.guidance_scale,
            input_data.scheduler_type,
            input_data.cfg_type,
            input_data.omega_scale,
            input_data.manual_seeds,
            input_data.guidance_interval,
            input_data.guidance_interval_decay,
            input_data.min_guidance_scale,
            input_data.use_erg_tag,
            input_data.use_erg_lyric,
            input_data.use_erg_diffusion,
            input_data.oss_steps,
            input_data.guidance_scale_text,
            input_data.guidance_scale_lyric,
        )
        output_path = input_data.output_path or f"output_{uuid.uuid4().hex}.{input_data.format}"

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        before = _cuda_mem()
        _log_both(
            "[generate] START  "
            f"allocated={before.get('allocated_hr','?')}  reserved={before.get('reserved_hr','?')}  "
            f"output={output_path}  discord_voice={input_data.discord_voice}"
        )

        with torch.inference_mode():
            model_demo(*params, save_path=output_path)

        elapsed = time.perf_counter() - t0
        peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        base_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0

        after = _cuda_mem()
        _log_both(
            "[generate] END    "
            f"elapsed={elapsed:.2f}s  peak_allocated={(_fmt_bytes(peak) if peak is not None else 'n/a')}  "
            f"allocated={after.get('allocated_hr','?')}  reserved={after.get('reserved_hr','?')}  "
            f"file_size={_fmt_bytes(base_size)}  path={output_path}"
        )

        deliverable_path = output_path
        deliverable_format = os.path.splitext(output_path)[1].lstrip(".").lower()
        deliverable_codec = None
        deliverable_bitrate = None

        if input_data.discord_voice:
            bitrate_kbps = input_data.opus_bitrate_kbps
            if input_data.target_max_bytes:
                bitrate_kbps = _guess_bitrate_to_fit(
                    duration_s=float(input_data.audio_duration),
                    target_max_bytes=int(input_data.target_max_bytes),
                    requested_kbps=int(input_data.opus_bitrate_kbps),
                )
                _log_both(
                    f"[opus] targeting size ≤{_fmt_bytes(input_data.target_max_bytes)}; "
                    f"selected bitrate ≈ {bitrate_kbps} kbps for duration {input_data.audio_duration:.2f}s"
                )
            deliverable_path, deliverable_bitrate = _transcode_to_opus(
                src_path=output_path,
                bitrate_kbps=bitrate_kbps,
            )
            deliverable_format = "ogg"
            deliverable_codec = "opus"
            out_size = os.path.getsize(deliverable_path) if os.path.exists(deliverable_path) else 0
            _log_both(
                f"[opus] wrote {deliverable_path}  size={_fmt_bytes(out_size)}  "
                f"bitrate={deliverable_bitrate} kbps  sr=48000  ch=2"
            )

        audio_data: Optional[str] = None
        try:
            with open(deliverable_path, "rb") as f:
                audio_bytes = f.read()
                audio_data = base64.b64encode(audio_bytes).decode("ascii")
        except Exception:
            audio_data = None

        final_size = os.path.getsize(deliverable_path) if os.path.exists(deliverable_path) else None

        return ACEStepOutput(
            status="success",
            output_path=output_path,
            message="Audio generated successfully",
            audio_data=audio_data,
            deliverable_path=deliverable_path,
            deliverable_format=deliverable_format,
            deliverable_codec=deliverable_codec,
            deliverable_bitrate_kbps=deliverable_bitrate,
            duration_s=float(input_data.audio_duration),
            file_size_bytes=final_size,
        )

    except HTTPException:
        raise
    except Exception as e:
        uvicorn_logger.exception("Error during /generate")
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")
    finally:
        if UNLOAD_AFTER_RUN:
            _unload_pipeline(model_demo)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/debug/memory")
def debug_memory():
    if not torch.cuda.is_available():
        return {"cuda": False}
    try:
        summary = torch.cuda.memory_summary(abbreviated=True)
    except Exception:
        summary = "<unavailable>"
    snap = _cuda_mem()
    snap["summary"] = summary
    return snap

@app.get("/debug/config")
def debug_config():
    return {
        "unload_after_run": UNLOAD_AFTER_RUN,
        "env": {
            "ACESTEP_UNLOAD": os.getenv("ACESTEP_UNLOAD", None),
            "PYTORCH_CUDA_ALLOC_CONF": os.getenv("PYTORCH_CUDA_ALLOC_CONF", None),
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES", None),
            "ACESTEP_LOG_DIR": os.getenv("ACESTEP_LOG_DIR", None),
        },
        "server_boot": SERVER_BOOT,
        "cuda": _cuda_device_info(),
    }

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="ACEStep Pipeline API server")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--unload", dest="unload", action="store_true",
                       help="Unload model after each /generate call (default).")
    group.add_argument("--no-unload", dest="unload", action="store_false",
                       help="Keep model loaded between calls (override default).")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"),
                        help="Bind host (default: 0.0.0.0 or $HOST).")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")),
                        help="Bind port (default: 8000 or $PORT).")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "info"),
                        help="Uvicorn log level (default: info).")

    parser.set_defaults(unload=(os.getenv("ACESTEP_UNLOAD", "1") != "0"))
    args = parser.parse_args()

    UNLOAD_AFTER_RUN = bool(args.unload)
    SERVER_BOOT.update({"host": args.host, "port": args.port, "cli_unload": bool(args.unload)})

    _log_both(
        f"[boot] Effective unload_after_run={UNLOAD_AFTER_RUN}  "
        f"(cli_unload={args.unload}  env.ACESTEP_UNLOAD={os.getenv('ACESTEP_UNLOAD')})  "
        f"PYTORCH_CUDA_ALLOC_CONF={os.getenv('PYTORCH_CUDA_ALLOC_CONF')}  "
        f"ACESTEP_LOG_DIR={LOG_DIR}"
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        log_config=build_logging_config(LOG_DIR),
    )
