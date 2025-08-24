# infer-api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import gc
import base64
import uuid
import time
import logging

import torch  # required for explicit CUDA cache management

from acestep.pipeline_ace_step import ACEStepPipeline
from acestep.data_sampler import DataSampler  # kept to avoid breaking external imports

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
# Use Uvicorn's error logger for server-side logs so they appear with the server.
uvicorn_logger = logging.getLogger("uvicorn.error")
api_logger = logging.getLogger("acestep.api")  # extra channel if you want to filter later
for lg in (uvicorn_logger, api_logger):
    lg.setLevel(logging.INFO)

# Also mirror to console explicitly if the environment strips handlers.
def _console(msg: str) -> None:
    print(msg, flush=True)

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(title="ACEStep Pipeline API")

# Default to unload after each request. Can be overridden by CLI or env.
# ACESTEP_UNLOAD=1 (default) -> unload; ACESTEP_UNLOAD=0 -> keep loaded
UNLOAD_AFTER_RUN: bool = os.getenv("ACESTEP_UNLOAD", "1") != "0"

# Persist CLI/bootstrap data so startup event can print them
SERVER_BOOT: Dict[str, Any] = {
    "host": os.getenv("HOST", "0.0.0.0"),
    "port": int(os.getenv("PORT", "8000")),
    "cli_unload": None,  # will be set in __main__
}

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _fmt_bytes(n: int) -> str:
    # human readable bytes
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TiB"

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

def _log_both(msg: str) -> None:
    uvicorn_logger.info(msg)
    api_logger.info(msg)
    _console(msg)

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class ACEStepInput(BaseModel):
    """
    Input schema for the ACEStep inference API.

    The pipeline expects arguments in a strict positional order.  To make
    HTTP clients intuitive and prevent accidental misalignment, this
    dataclass mirrors the signature of :meth:`ACEStepPipeline.__call__`.

    Key points:

    * ``format`` comes first and defaults to ``"wav"``.  This determines
      the container for the generated audio (e.g. ``mp3``, ``wav``, ``flac``).
    * ``manual_seeds`` replaces the previous ``actual_seeds`` field.
      It should be a list of integers and will be forwarded directly to the
      underlying pipeline's ``manual_seeds`` parameter.
    * ``oss_steps`` accepts a list of integers for one-shot sample steps.
    """

    # Audio file format (wav, mp3, flac, ogg).  Defaults to "wav".
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


class ACEStepOutput(BaseModel):
    status: str
    output_path: Optional[str]
    message: str
    # Base64-encoded audio data (no data: URI prefix).
    audio_data: Optional[str] = None

# -----------------------------------------------------------------------------
# Pipeline init / teardown
# -----------------------------------------------------------------------------
def initialize_pipeline(
    checkpoint_path: str, bf16: bool, torch_compile: bool, device_id: int
) -> ACEStepPipeline:
    """
    Initialise the ACEStep pipeline with the given checkpoint and device.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    # Log device + dtype intent
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
    """
    Forcefully free GPU memory used by the pipeline and log before/after stats.

    Correct order to reclaim VRAM:
      1) Move large modules to CPU (drop device allocations)
      2) Delete Python references
      3) gc.collect()
      4) torch.cuda.empty_cache()  (releases unoccupied cached blocks)

    Reference:
      - empty_cache() and allocator notes (reserved vs allocated). See PyTorch docs.
    """
    import inspect
    # Snapshot BEFORE
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

    # Move to CPU then drop references
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

    # Snapshot AFTER
    after = _cuda_mem()
    _log_both(
        "[unload] AFTER   "
        f"allocated={after.get('allocated_hr','?')}  reserved={after.get('reserved_hr','?')}"
    )

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.post("/generate", response_model=ACEStepOutput)
async def generate_audio(input_data: ACEStepInput):
    """
    Generate an audio file from the supplied parameters.

    This function constructs a tuple of positional arguments in the order
    expected by `ACEStepPipeline.__call__` and forwards them to the pipeline.
    """
    model_demo: Optional[ACEStepPipeline] = None

    try:
        # Initialise the underlying diffusion pipeline
        model_demo = initialize_pipeline(
            input_data.checkpoint_path,
            input_data.bf16,
            input_data.torch_compile,
            input_data.device_id,
        )

        # Construct the argument tuple for the call.  See
        # ``acestep/pipeline_ace_step.py:ACEStepPipeline.__call__`` for parameter ordering.
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

        # Generate output_path if not provided; default to input format
        output_path = input_data.output_path or f"output_{uuid.uuid4().hex}.{input_data.format}"

        # Per-request memory/latency instrumentation
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        before = _cuda_mem()
        _log_both(
            "[generate] START  "
            f"allocated={before.get('allocated_hr','?')}  reserved={before.get('reserved_hr','?')}  "
            f"output={output_path}"
        )

        # Execute the pipeline under inference_mode for extra perf/memory benefits
        with torch.inference_mode():
            model_demo(*params, save_path=output_path)

        dur = time.perf_counter() - start
        peak = None
        if torch.cuda.is_available():
            peak = int(torch.cuda.max_memory_allocated())
        size_bytes = 0
        try:
            size_bytes = os.path.getsize(output_path)
        except Exception:
            pass

        after = _cuda_mem()
        _log_both(
            "[generate] END    "
            f"elapsed={dur:.2f}s  peak_allocated={( _fmt_bytes(peak) if peak is not None else 'n/a' )}  "
            f"allocated={after.get('allocated_hr','?')}  reserved={after.get('reserved_hr','?')}  "
            f"file_size={_fmt_bytes(size_bytes)}  path={output_path}"
        )

        # After generation, attempt to read the produced file and encode Base64 (optional heavy)
        audio_data: Optional[str] = None
        try:
            with open(output_path, "rb") as f:
                audio_bytes = f.read()
                audio_data = base64.b64encode(audio_bytes).decode("ascii")
        except Exception:
            audio_data = None

        return ACEStepOutput(
            status="success",
            output_path=output_path,
            message="Audio generated successfully",
            audio_data=audio_data,
        )

    except Exception as e:
        # Wrap any exception in an HTTPException for FastAPI
        uvicorn_logger.exception("Error during /generate")
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")

    finally:
        # Ensure memory is reclaimed if configured.
        if UNLOAD_AFTER_RUN:
            _unload_pipeline(model_demo)

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "healthy"}

@app.get("/debug/memory")
def debug_memory():
    """
    Report CUDA memory (allocated/reserved) and an abbreviated memory summary.
    """
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
    """
    Report effective server/runtime config including unload policy & allocator knobs.
    """
    return {
        "unload_after_run": UNLOAD_AFTER_RUN,
        "env": {
            "ACESTEP_UNLOAD": os.getenv("ACESTEP_UNLOAD", None),
            "PYTORCH_CUDA_ALLOC_CONF": os.getenv("PYTORCH_CUDA_ALLOC_CONF", None),
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES", None),
        },
        "server_boot": SERVER_BOOT,
        "cuda": _cuda_device_info(),
    }

# -----------------------------------------------------------------------------
# App lifecycle (startup banner)
# -----------------------------------------------------------------------------
@app.on_event("startup")
async def _startup_banner():
    conf = {
        "effective_unload": UNLOAD_AFTER_RUN,
        "env.ACESTEP_UNLOAD": os.getenv("ACESTEP_UNLOAD", None),
        "env.PYTORCH_CUDA_ALLOC_CONF": os.getenv("PYTORCH_CUDA_ALLOC_CONF", None),
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

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="ACEStep Pipeline API server")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--unload",
        dest="unload",
        action="store_true",
        help="Unload model after each /generate call (default).",
    )
    group.add_argument(
        "--no-unload",
        dest="unload",
        action="store_false",
        help="Keep model loaded between calls (override default).",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("HOST", "0.0.0.0"),
        help="Bind host (default: 0.0.0.0 or $HOST).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8000")),
        help="Bind port (default: 8000 or $PORT).",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "info"),
        help="Uvicorn log level (default: info).",
    )

    # Default comes from env (ACESTEP_UNLOAD), but CLI can override.
    parser.set_defaults(unload=(os.getenv("ACESTEP_UNLOAD", "1") != "0"))
    args = parser.parse_args()

    # Sync CLI to module-level toggle and persist for startup banner
    UNLOAD_AFTER_RUN = bool(args.unload)
    SERVER_BOOT.update({"host": args.host, "port": args.port, "cli_unload": bool(args.unload)})

    # Also print an immediate pre-boot verification line to stdout & Uvicorn logger
    _log_both(
        f"[boot] Effective unload_after_run={UNLOAD_AFTER_RUN}  "
        f"(cli_unload={args.unload}  env.ACESTEP_UNLOAD={os.getenv('ACESTEP_UNLOAD')})  "
        f"PYTORCH_CUDA_ALLOC_CONF={os.getenv('PYTORCH_CUDA_ALLOC_CONF')}"
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
