"""KittenTTS 2 local Web UI - FastAPI backend.

Startup never blocks serving: the lifespan handler starts a background thread
that loads the model. Endpoints that need the model return 503
{"model_state": ...} until it is ready.

The model weights are not part of this repository. They are downloaded from
Hugging Face (KittenML/kitten-tts-2) on first run and cached under HF_HOME.
They are covered by their own license; see README.md.

Security: the server has no authentication. It binds to 127.0.0.1 by default.
Do not expose it to a network you do not trust.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------- configuration
# Repo root = the folder that contains webui/. Every default is relative to it.
ROOT = Path(__file__).resolve().parent.parent

try:  # optional: load ROOT/.env if python-dotenv is installed
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def _path_env(name: str, default: Path) -> Path:
    """Env var as a path; relative values are resolved against the repo root."""
    v = (os.environ.get(name) or "").strip()
    p = Path(v).expanduser() if v else default
    return p if p.is_absolute() else (ROOT / p).resolve()


HOST = os.environ.get("KITTEN_WEBUI_HOST", "127.0.0.1").strip() or "127.0.0.1"
PORT = int(os.environ.get("KITTEN_WEBUI_PORT", "7860"))
OUTPUT = _path_env("KITTEN_OUTPUT_DIR", ROOT / "output")
CACHE = _path_env("KITTEN_CACHE_DIR", ROOT / "cache")
TMP = _path_env("KITTEN_TMP_DIR", ROOT / "tmp")
os.environ["HF_HOME"] = str(_path_env("HF_HOME", CACHE / "hf"))
os.environ.setdefault("TORCH_HOME", str(CACHE / "torch"))
os.environ.setdefault("PYTHONUTF8", "1")  # only affects child processes; see _utf8_text_open
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TEMP", str(TMP))
os.environ.setdefault("TMP", str(TMP))

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

# torch / kittenml are imported inside the loader thread so the server can bind
# and answer immediately (importing torch alone takes several seconds).
torch = None

WEBUI = Path(__file__).resolve().parent
STATIC = WEBUI / "static"
MODEL_ID = "KittenML/kitten-tts-2"
DEFAULT_WEIGHTS, DEFAULT_DECODER, DEFAULT_PRESET = "packed", "default", "auto"
SIDECAR_FIELDS = ("text", "voice", "preset", "generation_seconds", "audio_duration", "created",
                  "weights", "decoder", "seed")

for d in (OUTPUT, TMP, Path(os.environ["HF_HOME"])):
    d.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- model state
_state_lock = threading.Lock()   # guards STATE
_load_lock = threading.Lock()    # one load/reload at a time
_gen_lock = threading.Lock()     # GPU work is serialized
_model = None
STATE = {
    "phase": "starting",          # starting | loading | ready | error
    "message": "server starting",
    "error": None,
    "weights": None,
    "decoder": None,
    "target_weights": DEFAULT_WEIGHTS,
    "target_decoder": DEFAULT_DECODER,
    "device": None,
    "load_started": None,
    "load_seconds": None,
    "ready_at": None,
    "steps": [],                  # [{"step", "seconds"}] timing breakdown of the last load
    "job": None,
    "gpu_name": None,
    "device_warning": None,
    "download": None,             # {"downloaded_bytes", "total_bytes", "percent", "estimated"}
}

def _set(**kw):
    with _state_lock:
        STATE.update(kw)


def _step(name: str, t0: float):
    dt = round(time.perf_counter() - t0, 2)
    with _state_lock:
        STATE["steps"].append({"step": name, "seconds": dt})
        STATE["message"] = f"{name} done ({dt}s)"
    log(f"load step: {name} {dt}s")


def snapshot_state() -> dict:
    with _state_lock:
        s = json.loads(json.dumps(STATE))
    if s["phase"] == "loading" and s["load_started"]:
        s["loading_for"] = round(time.time() - s["load_started"], 1)
    s["ready"] = s["phase"] == "ready"
    return s


class ModelNotReady(Exception):
    pass


LOADING_MSG = "Model is still loading, try again in a moment"


def not_ready_response() -> JSONResponse:
    s = snapshot_state()
    if s["phase"] == "error":
        err = s.get("error") or "The model failed to load. See the server window for details."
    else:
        err = LOADING_MSG
    return JSONResponse(
        status_code=503,
        content={"error": err, "detail": err, "state": s["phase"], "model_state": s},
        headers={"Retry-After": "5"},
    )


# ---------------------------------------------------------------- friendly errors
_NETWORK_HINTS = ("connection", "timed out", "timeout", "max retries", "getaddrinfo", "name resolution",
                  "temporary failure", "offline", "localentrynotfound", "ssl", "proxy", "network",
                  "huggingface.co", "hfhub", "repositorynotfound", "gated", "401", "403", "429")


def friendly_load_error(e: BaseException) -> str:
    """Turn a model-load exception into a message a non-developer can act on."""
    raw = f"{type(e).__name__}: {e}"
    low = raw.lower()
    if _is_oom(e):
        return ("The GPU ran out of memory while loading the model. It needs about 4.7 GB of free VRAM. "
                "Close other GPU apps (for example unload the model in LM Studio, or close games and "
                "video editors), then restart start.bat.")
    if isinstance(e, ModuleNotFoundError) or "no module named" in low:
        return (f"A Python package is missing ({e}). Run setup.bat again to reinstall the requirements.")
    if any(h in low for h in _NETWORK_HINTS):
        return ("Could not download the model from Hugging Face. Check that this PC is online, that "
                "https://huggingface.co opens in your browser, and that a firewall, VPN or proxy is not "
                "blocking Python. Also check there is enough free disk space for the model cache "
                f"(HF_HOME). Then restart start.bat. Details: {raw[:300]}")
    if "no space left" in low or "errno 28" in low:
        return ("The disk is full. Free some space on the drive that holds the cache folder "
                "(HF_HOME), then restart start.bat.")
    return f"The model failed to load. Details: {raw[:400]}"


def _utf8_text_open(file, mode="r", *args, **kwargs):
    """open() that defaults to UTF-8 for text mode. kittenml reads voices.json and
    config.json with a bare open(), which uses the Windows ANSI code page (cp1252)
    unless Python runs in UTF-8 mode, and crashes on the non-ASCII voice data."""
    if "b" not in mode and len(args) < 2 and "encoding" not in kwargs:
        kwargs["encoding"] = "utf-8"
    return open(file, mode, *args, **kwargs)


def _patch_kittenml_open():
    """Make kittenml's own file reads UTF-8 without relying on PYTHONUTF8."""
    import importlib
    for name in ("kittenml.repo", "kittenml.kittentts2.model"):
        try:
            importlib.import_module(name).open = _utf8_text_open
        except Exception as e:  # module layout changed: leave it, PYTHONUTF8 still helps
            log(f"could not patch {name} file encoding: {e}")


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _repo_cache_dir(repo_id: str) -> Path:
    return Path(os.environ["HF_HOME"]) / "hub" / ("models--" + repo_id.replace("/", "--"))


def _model_cache_dir() -> Path:
    return _repo_cache_dir(MODEL_ID)


def _decoder_repo() -> tuple:
    """(repo_id, filename) of the vocoder weights kittenml fetches on first use."""
    try:
        from kittenml.kittentts2 import vocoder
        return vocoder.CHATTERBOX_REPO, vocoder.S3GEN_WEIGHTS
    except Exception:
        return "ResembleAI/chatterbox-turbo", "s3gen_meanflow.safetensors"


def _sum_matching(sizes: dict, patterns) -> int:
    import fnmatch
    return sum(sz for path, sz in sizes.items()
               if any(fnmatch.fnmatch(path, pat) for pat in patterns))


def _repo_file_sizes(repo_id: str) -> dict:
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo_id, files_metadata=True, timeout=10)
    return {s.rfilename: s.size for s in (info.siblings or []) if getattr(s, "size", None)}


def _remote_total_bytes(weights: Optional[str] = None, decoder: Optional[str] = None) -> Optional[int]:
    """Bytes the first run will actually download, or None when it cannot be worked out.

    The model repo also holds weight variants and runtimes this app never fetches (about
    7 GB in total), so only the files kittenml asks for are counted: the ones matching its
    own allow-list for the chosen weights/decoder, plus the ResembleAI vocoder file."""
    try:
        from huggingface_hub import hf_hub_download
        from kittenml.get_model import _kitten2_files
        cfg_path = hf_hub_download(MODEL_ID, "config.json")
        with open(cfg_path, encoding="utf-8") as f:
            config = json.load(f)
        total = _sum_matching(_repo_file_sizes(MODEL_ID), _kitten2_files(config, decoder, weights))
        dec_repo, dec_file = _decoder_repo()
        total += _repo_file_sizes(dec_repo).get(dec_file, 0)
        return total or None
    except Exception as e:
        log(f"could not work out the download size ({type(e).__name__}: {e}); showing bytes only")
        return None


class _DownloadWatcher:
    """Polls the size of the cache folders (model repo + vocoder repo) while a download runs.
    Simple and robust: works whatever library does the downloading. The total counts only
    the files that will be fetched; the percentage is still capped at 99 until loading finishes."""

    def __init__(self, weights: Optional[str] = None, decoder: Optional[str] = None):
        self._stop = threading.Event()
        self._start_bytes = 0
        self._total = None
        self._weights, self._decoder = weights, decoder
        self._dirs = [_model_cache_dir(), _repo_cache_dir(_decoder_repo()[0])]

    def _size(self) -> int:
        return sum(_dir_size(d) for d in self._dirs)

    def __enter__(self):
        self._start_bytes = self._size()
        self._total = _remote_total_bytes(self._weights, self._decoder)
        _set(download={"downloaded_bytes": self._start_bytes, "total_bytes": self._total,
                       "percent": self._percent(self._start_bytes), "estimated": True})
        threading.Thread(target=self._run, daemon=True, name="download-watch").start()
        return self

    def _percent(self, n: int) -> Optional[float]:
        if not self._total:
            return None
        return round(min(99.0, 100.0 * n / self._total), 1)

    def _run(self):
        while not self._stop.wait(1.0):
            n = self._size()
            _set(download={"downloaded_bytes": n, "total_bytes": self._total,
                           "percent": self._percent(n), "estimated": True})

    def __exit__(self, *exc):
        self._stop.set()
        return False


@contextlib.contextmanager
def _hf_offline():
    """Prefer the local HF cache (no network round-trips); caller falls back online."""
    import huggingface_hub.constants as hc
    prev_c = hc.HF_HUB_OFFLINE
    prev_e = os.environ.get("HF_HUB_OFFLINE")
    hc.HF_HUB_OFFLINE = True
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        yield
    finally:
        hc.HF_HUB_OFFLINE = prev_c
        if prev_e is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev_e


_queue_lock = threading.Lock()                 # guards _loader
_loader = {"running": False, "pending": None}  # v5: one loader thread; reloads queue (latest wins)
_loaded_key = None                             # (weights, decoder) of the model in _model


def _load_model_blocking(weights: str, decoder: str):
    """Load (or reload) the model. Runs in the loader thread (_loader_loop)."""
    global _model, torch, _loaded_key
    with _load_lock:
        if _model is not None and _loaded_key == (weights, decoder):
            _set(phase="ready", weights=weights, decoder=decoder, target_weights=weights,
                 target_decoder=decoder, target={"weights": weights, "decoder": decoder},
                 loaded={"weights": weights, "decoder": decoder}, error=None,
                 message=f"ready (weights={weights}, decoder={decoder} already loaded)")
            return
        prev = {"weights": _loaded_key[0], "decoder": _loaded_key[1]} if _loaded_key else None
        _set(phase="loading", error=None, steps=[], load_started=time.time(),
             target_weights=weights, target_decoder=decoder,
             target={"weights": weights, "decoder": decoder}, loaded=None, previous=prev,
             weights=weights, decoder=decoder,
             message=f"loading model (weights={weights}, decoder={decoder})")
        log(f"model load start weights={weights} decoder={decoder}")
        t_all = time.perf_counter()
        try:
            _set(message="importing torch")
            t0 = time.perf_counter()
            import torch as _torch
            torch = _torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            gpu_name = None
            if device == "cuda":
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                except Exception:
                    gpu_name = None
                device_warning = None
            else:
                device_warning = ("No NVIDIA GPU is available to PyTorch, so the model runs on the CPU. "
                                  "It works, but generation is much slower. Update the NVIDIA driver "
                                  "(570 or newer, check with nvidia-smi) and run setup.bat again.")
                log("WARNING: " + device_warning)
            _set(device=device, gpu_name=gpu_name, device_warning=device_warning)
            _step("import torch", t0)

            _set(message="importing kittenml")
            t0 = time.perf_counter()
            from kittenml import KittenTTS
            import kittenml.kittentts2  # noqa: F401  (pulls transformers etc.)
            _patch_kittenml_open()
            _step("import kittenml", t0)

            # Free the previous model before allocating a new one (reload case).
            if _model is not None:
                _set(message="freeing the previous model (waits for a running generation)")
                with _gen_lock:
                    _model = None
                    _loaded_key = None
                    torch.cuda.empty_cache() if device == "cuda" else None

            _set(message=f"building model (weights={weights}, decoder={decoder}) from the local cache")
            t0 = time.perf_counter()
            try:
                with _hf_offline():
                    m = KittenTTS(MODEL_ID, device=device, weights=weights, decoder=decoder)
                _step("build model (offline cache)", t0)
            except Exception as e:
                if _is_oom(e):
                    raise
                log(f"not in the local cache yet ({type(e).__name__}); downloading from Hugging Face")
                _set(phase="downloading",
                     message=f"downloading the model from Hugging Face ({MODEL_ID})")
                t0 = time.perf_counter()
                with _DownloadWatcher(weights, decoder):
                    m = KittenTTS(MODEL_ID, device=device, weights=weights, decoder=decoder)
                _step("download + build model", t0)

            _model = m
            _loaded_key = (weights, decoder)
            try:
                _remember_model_choices(m)
            except Exception:
                pass
            total = round(time.perf_counter() - t_all, 2)
            dl = STATE.get("download")
            if dl:
                dl = dict(dl, percent=100.0 if dl.get("percent") is not None else None)
            _set(phase="ready", weights=weights, decoder=decoder, load_seconds=total, download=dl,
                 ready_at=time.time(), loaded={"weights": weights, "decoder": decoder},
                 message=f"ready ({total}s load on {device})")
            log(f"model ready in {total}s")
        except Exception as e:
            traceback.print_exc()
            sys.stderr.flush()
            _model = None
            _loaded_key = None
            if _is_oom(e):
                _empty_cuda_cache()
            _set(phase="error", error=friendly_load_error(e), error_raw=f"{type(e).__name__}: {e}"[:1000],
                 loaded=None, message="model load failed")
            log(f"model load FAILED: {type(e).__name__}: {e}")


def _loader_loop(weights: str, decoder: str):
    """Load, then keep loading whatever reload was queued meanwhile (latest wins)."""
    while True:
        try:
            _load_model_blocking(weights, decoder)
        except Exception:
            traceback.print_exc()
        with _queue_lock:
            nxt = _loader["pending"]
            _loader["pending"] = None
            if nxt is None:
                _loader["running"] = False
                _set(queued=None)
                return
        weights, decoder = nxt
        _set(queued=None)


def start_model_load(weights: str, decoder: str) -> bool:
    """Start a background (re)load. Returns True if it starts now; False if a load is
    already running, in which case this one is queued (latest wins). /api/status flips
    to phase=loading (target = the requested model, loaded = null) immediately."""
    with _queue_lock:
        _set(phase="loading", target_weights=weights, target_decoder=decoder,
             target={"weights": weights, "decoder": decoder}, loaded=None,
             weights=weights, decoder=decoder, error=None, load_started=time.time(),
             message=f"reload accepted (weights={weights}, decoder={decoder})")
        if _loader["running"]:
            _loader["pending"] = (weights, decoder)
            _set(queued={"weights": weights, "decoder": decoder},
                 message=f"reload queued (weights={weights}, decoder={decoder}); "
                         f"starts when the current load finishes")
            return False
        _loader["running"] = True
    threading.Thread(target=_loader_loop, args=(weights, decoder),
                     daemon=True, name="model-loader").start()
    return True


def get_ready_model():
    """Return the loaded model or raise ModelNotReady (-> 503)."""
    if _model is None or STATE["phase"] != "ready":
        raise ModelNotReady()
    return _model



def _startup_worker():
    with _queue_lock:
        _loader["running"] = True
    _loader_loop(DEFAULT_WEIGHTS, DEFAULT_DECODER)   # packed / default (+ reloads queued meanwhile)


@contextlib.asynccontextmanager
async def lifespan(app_: FastAPI):
    if os.environ.get("KITTEN_SKIP_MODEL_LOAD", "").strip() == "1":
        log("lifespan: KITTEN_SKIP_MODEL_LOAD=1, not loading the model (test mode)")
        _set(phase="loading", message="model loading disabled (KITTEN_SKIP_MODEL_LOAD=1)")
    else:
        log("lifespan: starting background model loader; server is accepting requests")
        _set(phase="loading", message="starting loader", load_started=time.time())
        threading.Thread(target=_startup_worker, daemon=True, name="startup-worker").start()
    yield
    log("lifespan: shutdown")


app = FastAPI(title="KittenTTS 2 Web UI", lifespan=lifespan)


@app.exception_handler(ModelNotReady)
async def _model_not_ready_handler(request, exc):
    return not_ready_response()


@app.exception_handler(HTTPException)
async def _http_error(request: Request, exc: HTTPException):
    """API errors are JSON {"error": "...", "detail": "..."} with a plain-text message."""
    msg = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"error": msg, "detail": msg},
                        headers=getattr(exc, "headers", None))


@app.exception_handler(Exception)
async def _unhandled_error(request: Request, exc: Exception):
    """Last resort: never send a stack trace to the page."""
    traceback.print_exc()
    if _is_oom(exc):
        _empty_cuda_cache()
        return oom_response("this request")
    msg = f"Something went wrong ({type(exc).__name__}). See the server window for details."
    return JSONResponse(status_code=500, content={"error": msg, "detail": msg})


@app.exception_handler(RequestValidationError)
async def _api_validation_error(request: Request, exc: RequestValidationError):
    """Request validation errors under /api/ -> 400 {"detail": "<plain text>"} (not 422)."""
    if not request.url.path.startswith("/api/"):
        return await request_validation_exception_handler(request, exc)
    msgs = []
    for e in exc.errors() or []:
        loc = [str(x) for x in (e.get("loc") or []) if x not in ("body", "query", "path", "header", "form")]
        field = ".".join(loc) or "request"
        if e.get("type") == "missing":
            if field == "text" and request.url.path.rstrip("/") == "/api/generate":
                msgs.append("Text is required")
            else:
                msgs.append(f"Missing required field: {field}")
        else:
            msgs.append(f"Invalid value for {field}: {e.get('msg', 'invalid value')}")
    detail = "; ".join(dict.fromkeys(msgs)) or "Invalid request"
    return JSONResponse(status_code=400, content={"error": detail, "detail": detail})



SAMPLES = [
    {
        "id": "narration",
        "label": "Plain narration",
        "group": "English",
        "text": (
            "In a quiet town by the river, morning light spilled across the cobblestones. "
            "Shopkeepers opened their doors, and the smell of fresh bread drifted down the street."
        ),
        "voice": "Bruno",
        "normalize": True,
        "preset": "stable",
    },
    {
        "id": "conversation",
        "label": "Conversational",
        "group": "English",
        "text": (
            "Hey, did you get my message earlier? I was thinking we could grab coffee "
            "around three, unless that does not work for you."
        ),
        "voice": "Bella",
        "normalize": True,
        "preset": "stable",
    },
    {
        "id": "emotions",
        "label": "Emotion tags",
        "group": "Expressive",
        "text": (
            "[excited] You will not believe what just happened! "
            "[stern] Please listen carefully. "
            "[tender] I am really glad you are here."
        ),
        "voice": "Kiki",
        "normalize": True,
        "preset": "expressive",
    },
    {
        "id": "events",
        "label": "Vocal events",
        "group": "Expressive",
        "text": (
            "Well <um> I was going to say something clever, but then <sigh> "
            "I forgot. <laugh> Anyway, here we are."
        ),
        "voice": "Jasper",
        "normalize": True,
        "preset": "expressive",
    },
    {
        "id": "emphasis",
        "label": "Emphasis",
        "group": "Expressive",
        "text": (
            "I said (((please))) do not touch that. It is (((extremely))) fragile, "
            "and we only have (((one))) left."
        ),
        "voice": "Claire",
        "normalize": True,
        "preset": "expressive",
    },
    {
        "id": "normalize",
        "label": "Numbers & dates (normalize)",
        "group": "English",
        "text": (
            "Meet me on March 15th 2026 at 3:30 PM. Bring $12.50 and two tickets "
            "for the 7:00 show on 5th Avenue."
        ),
        "voice": "Matthew",
        "normalize": True,
        "preset": "stable",
    },
    {
        "id": "lang_ar",
        "label": "Arabic",
        "group": "Languages",
        "text": "مرحبا، كيف حالك اليوم؟ أتمنى أن تستمتع بهذا الصوت الجميل.",
        "voice": "Arabic",
        "normalize": False,
        "preset": "stable",
    },
    {
        "id": "lang_zh",
        "label": "Chinese",
        "group": "Languages",
        "text": "你好，今天天气真不错。希望你喜欢这个声音。",
        "voice": "Chinese",
        "normalize": False,
        "preset": "stable",
    },
    {
        "id": "lang_fr",
        "label": "French",
        "group": "Languages",
        "text": "Bonjour, comment allez-vous aujourd'hui ? J'espère que vous appréciez cette voix.",
        "voice": "French",
        "normalize": False,
        "preset": "stable",
    },
    {
        "id": "lang_de",
        "label": "German",
        "group": "Languages",
        "text": "Guten Tag, wie geht es Ihnen heute? Ich hoffe, Ihnen gefällt diese Stimme.",
        "voice": "German",
        "normalize": False,
        "preset": "stable",
    },
    {
        "id": "lang_hi",
        "label": "Hindi",
        "group": "Languages",
        "text": "नमस्ते, आप आज कैसे हैं? मुझे आशा है कि आपको यह आवाज़ पसंद आएगी।",
        "voice": "Hindi",
        "normalize": False,
        "preset": "stable",
    },
    {
        "id": "lang_it",
        "label": "Italian",
        "group": "Languages",
        "text": "Ciao, come stai oggi? Spero che ti piaccia questa voce.",
        "voice": "Italian",
        "normalize": False,
        "preset": "stable",
    },
    {
        "id": "lang_pt",
        "label": "Portuguese",
        "group": "Languages",
        "text": "Olá, como você está hoje? Espero que goste desta voz.",
        "voice": "Portuguese",
        "normalize": False,
        "preset": "stable",
    },
    {
        "id": "lang_ru",
        "label": "Russian",
        "group": "Languages",
        "text": "Здравствуйте, как ваши дела сегодня? Надеюсь, вам нравится этот голос.",
        "voice": "Russian",
        "normalize": False,
        "preset": "stable",
    },
    {
        "id": "lang_es",
        "label": "Spanish",
        "group": "Languages",
        "text": "Hola, ¿cómo estás hoy? Espero que te guste esta voz.",
        "voice": "Spanish",
        "normalize": False,
        "preset": "stable",
    },
]




# ---------------------------------------------------------------- helpers
_GENERIC_TAG_RE = re.compile(r"\[[A-Za-z][A-Za-z _-]{1,30}\]|<[A-Za-z][A-Za-z _-]{1,30}>|\(\(\(.+?\)\)\)")


class GpuOOM(RuntimeError):
    """CUDA ran out of memory; surfaced as HTTP 503 with a retry hint."""


def _is_oom(e: BaseException) -> bool:
    oom_cls = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None) if torch else None
    if oom_cls is not None and isinstance(e, oom_cls):
        return True
    s = str(e).lower()
    return "out of memory" in s or "cublas_status_alloc_failed" in s


def _empty_cuda_cache():
    try:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def oom_response(where: str) -> JSONResponse:
    return JSONResponse(status_code=503, headers={"Retry-After": "10"}, content={
        "error": f"The GPU ran out of memory during {where}. Close other GPU apps (for example unload "
                 f"the model in LM Studio) or try shorter text, then try again.",
        "detail": f"The GPU ran out of memory during {where}. Close other GPU apps (for example unload "
                  f"the model in LM Studio) or try shorter text, then try again.",
        "oom": True, "model_state": snapshot_state()})


def _safe_name(text: str, voice: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", (voice or "voice")[:24]).strip("_") or "clip"
    short = re.sub(r"[^a-zA-Z0-9_-]+", "_", text[:40]).strip("_") or "speech"
    name, n = f"{stamp}_{slug}_{short}.wav", 2
    while (OUTPUT / name).exists():
        name = f"{stamp}_{slug}_{short}_{n}.wav"
        n += 1
    return name


def _safe_child(base: Path, name: str) -> Path:
    """Return base/name only if it is a plain file name that resolves inside base."""
    if (not name or name in (".", "..") or ".." in name or "/" in name or "\\" in name
            or ":" in name or "\x00" in name or Path(name).is_absolute()):
        raise HTTPException(400, "Invalid name")
    base_r = base.resolve()
    p = (base / name).resolve()
    if p.parent != base_r:
        raise HTTPException(400, "Invalid name")
    return p


def _wav_meta(path: Path) -> dict:
    info = sf.info(str(path))
    return {
        "name": path.name,
        "url": f"/output/{path.name}",
        "size": path.stat().st_size,
        "duration": round(info.frames / info.samplerate, 2),
        "sample_rate": info.samplerate,
        "mtime": path.stat().st_mtime,
    }


def _strip_markup(s: str) -> str:
    return _GENERIC_TAG_RE.sub(" ", s or "")


def has_markup(text: str) -> bool:
    try:
        from kittenml.kittentts2.text import has_expression_tag
        if has_expression_tag(text):
            return True
    except Exception:
        pass
    return bool(_GENERIC_TAG_RE.search(text or ""))


def resolve_preset(preset: Optional[str], text: str, m) -> str:
    """'auto' (default): 'expressive' when the text has [tags], <events> or (((emphasis))),
    else 'stable'. Other values must be a model preset (400 otherwise)."""
    p = (preset or DEFAULT_PRESET).strip().lower()
    if p == "auto":
        return "expressive" if has_markup(text) else "stable"
    presets = list(getattr(m, "decode_presets", {}) or {"stable": {}, "expressive": {}})
    if p not in presets:
        raise HTTPException(400, f"Unknown preset {preset!r}. Choose from: {['auto', *presets]}")
    return p


def preset_reason(preset: Optional[str], text: str) -> str:
    """'tags' / 'no_tags' for auto, 'manual' when a preset was chosen explicitly."""
    if (preset or DEFAULT_PRESET).strip().lower() != "auto":
        return "manual"
    return "tags" if has_markup(text) else "no_tags"


_KNOWN_CHOICES = {"weights": None, "decoders": None}


def _remember_model_choices(m):
    for key, attr in (("weights", "available_weights"), ("decoders", "available_decoders")):
        try:
            v = getattr(m, attr, None)
            if v is not None:
                _KNOWN_CHOICES[key] = list(v)
        except Exception:
            pass


def validate_model_choice(weights: str, decoder: str, m=None):
    """400 for unknown weights/decoder (instead of starting a reload that fails and leaves
    the server without a model). Uses the lists remembered from the last loaded model
    while a reload is running."""
    if m is None:
        try:
            m = get_ready_model()
        except Exception:
            m = None
    if m is not None:
        _remember_model_choices(m)
    for kind, val, avail in (("weights", weights, _KNOWN_CHOICES["weights"]),
                             ("decoder", decoder, _KNOWN_CHOICES["decoders"])):
        if avail and val not in avail:
            raise HTTPException(400, f"Unknown {kind} {val!r}. Choose from: {avail}")




# ---- request validation
MAX_TEXT_CHARS = 2000
MAX_NEW_TOKENS_CAP = 4000
# temperature must be > 0, top_k 0 = off, top_p 1.0 = off, min_p 0 = off;
# model.generate default max_new_tokens=1000 (about 25 tokens per second of audio).
PARAM_RANGES = {
    "temperature": {"type": "float", "min": 0.0, "min_exclusive": True, "max": 2.0, "default": None,
                    "text": "a number greater than 0 and at most 2.0 (leave empty for the preset value)"},
    "top_k": {"type": "int", "min": 0, "max": 1000, "default": None, "off_value": 0,
              "text": "a whole number from 0 to 1000 (0 turns top-k off)"},
    "top_p": {"type": "float", "min": 0.0, "min_exclusive": True, "max": 1.0, "default": None,
              "off_value": 1.0, "text": "a number greater than 0 and at most 1 (1 turns top-p off)"},
    "min_p": {"type": "float", "min": 0.0, "max": 1.0, "default": None, "off_value": 0.0,
              "text": "a number from 0 to 1 (0 turns min-p off)"},
    "max_new_tokens": {"type": "int", "min": 1, "max": MAX_NEW_TOKENS_CAP, "default": 1000,
                       "text": f"a whole number from 1 to {MAX_NEW_TOKENS_CAP} (25 tokens is about 1 s of audio)"},
    "seed": {"type": "int", "min": 0, "max": 2 ** 32 - 1, "default": None,
             "text": f"a whole number from 0 to {2 ** 32 - 1}"},
    "normalize": {"type": "bool", "default": True},
    "text": {"type": "str", "max_chars": MAX_TEXT_CHARS},
}
_BOOL_TRUE, _BOOL_FALSE = {"true", "1", "yes", "on"}, {"false", "0", "no", "off"}


def parse_bool(v, field: str, default=None):
    """Strict boolean form field: true/false, 1/0, yes/no, on/off (any case); empty = default."""
    if v is None or isinstance(v, bool):
        return default if v is None else v
    s = str(v).strip().lower()
    if s == "":
        return default
    if s in _BOOL_TRUE:
        return True
    if s in _BOOL_FALSE:
        return False
    raise HTTPException(400, f"{field} must be a boolean (true or false); got {str(v)[:40]!r}.")


def parse_number(v, field: str):
    """Parse + range-check a sampling parameter per PARAM_RANGES; empty = None (default)."""
    r = PARAM_RANGES[field]
    if v is None or str(v).strip() == "":
        return None
    s = str(v).strip()
    bad = HTTPException(400, f"{field} must be {r['text']}; got {s[:40]!r}.")
    try:
        x = int(s) if r["type"] == "int" else float(s)
    except ValueError:
        raise bad
    if r["type"] == "float" and not np.isfinite(x):
        raise bad
    lo, hi = r["min"], r["max"]
    if (x <= lo if r.get("min_exclusive") else x < lo) or x > hi:
        raise bad
    return x


def check_text_length(text: str, field: str = "text"):
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(400, f"{field} is too long ({len(text)} characters); the maximum is "
                                 f"{MAX_TEXT_CHARS}. Split it into shorter parts.")



def _spoken_len(m, text: str) -> int:
    try:
        s = m.normalize_text(text)
    except Exception:
        s = text
    return len(_strip_markup(s).strip())


def loop_suspected(m, text: str, duration: float) -> bool:
    """Heuristic: audio far longer than the text needs."""
    expected = _spoken_len(m, text) / 14.0
    return duration > max(4.0, expected * 2.2 + 1.5)


def _run_generate(m, text, kw, out_path):
    with _gen_lock:
        try:
            audio = m.generate(text, **kw)
        except Exception as e:
            if _is_oom(e):
                _empty_cuda_cache()
                raise GpuOOM(str(e))
            raise
        sf.write(str(out_path), audio, m.sample_rate)
        return float(len(audio)) / float(m.sample_rate)


# ---------------------------------------------------------------- routes
@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(STATIC / "index.html")


STATIC_EXTS = {".html", ".htm", ".js", ".mjs", ".css", ".svg", ".png", ".jpg", ".jpeg", ".gif",
               ".ico", ".webp", ".woff", ".woff2", ".ttf", ".otf", ".map", ".webmanifest"}


@app.api_route("/static/{path:path}", methods=["GET", "HEAD"])
def static_asset(path: str):
    """Only the app's own assets (allow-listed types); never *.bak*, hidden files or paths
    outside webui/static."""
    parts = path.replace("\\", "/").split("/")
    if any(p in ("", ".", "..") or p.startswith(".") or ":" in p or "\x00" in p
           or ".bak" in p.lower() for p in parts):
        raise HTTPException(404, "Not found")
    base = STATIC.resolve()
    f = base.joinpath(*parts).resolve()
    if base not in f.parents or not f.is_file() or f.suffix.lower() not in STATIC_EXTS:
        raise HTTPException(404, "Not found")
    return FileResponse(f)


@app.api_route("/output/{path:path}", methods=["GET", "HEAD"])
def output_audio(path: str):
    """Generated clips: output/<name>.wav only (no .json sidecars)."""
    try:
        f = _safe_child(OUTPUT, path)
    except HTTPException:
        raise HTTPException(404, "Not found")
    if f.suffix.lower() != ".wav" or not f.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(f, media_type="audio/wav")


@app.get("/api/status")
def status():
    s = snapshot_state()
    s["model_loaded"] = s["ready"]
    s["status"] = s["message"]
    s["cuda"] = s.get("device") == "cuda"
    s["port"] = PORT
    s.setdefault("loaded", {"weights": s.get("weights"), "decoder": s.get("decoder")} if s["ready"] else None)
    s.setdefault("target", {"weights": s.get("target_weights"), "decoder": s.get("target_decoder")})
    s.setdefault("queued", None)
    s["defaults"] = {"weights": DEFAULT_WEIGHTS, "decoder": DEFAULT_DECODER, "preset": DEFAULT_PRESET}
    # Simple fields for the first-run banner and the header badge.
    s["state"] = {"starting": "loading"}.get(s["phase"], s["phase"])   # downloading|loading|ready|error
    dev = s.get("device")
    s["device_label"] = (f"GPU: {s['gpu_name']}" if dev == "cuda" and s.get("gpu_name")
                         else "GPU" if dev == "cuda" else "CPU" if dev == "cpu" else None)
    s["warnings"] = [s["device_warning"]] if s.get("device_warning") else []
    s.pop("error_raw", None)
    return s


@app.get("/api/samples")
def samples():
    return {"samples": SAMPLES, "max_text_chars": MAX_TEXT_CHARS}


LANG_VOICES = {"Arabic", "Chinese", "French", "German", "Hindi",
               "Italian", "Portuguese", "Russian", "Spanish"}


@app.get("/api/meta")
def meta():
    m = get_ready_model()
    voices = [v for v in m.available_voices if not str(v).startswith("hybrid:")]
    model_presets = list(getattr(m, "decode_presets", {}) or {"stable": {}, "expressive": {}})
    return {
        "voices": voices,
        "voices_english": [v for v in voices if v not in LANG_VOICES],
        "voices_language": [v for v in voices if v in LANG_VOICES],
        "presets": ["auto", *[p for p in model_presets if p != "auto"]],
        "default_preset": DEFAULT_PRESET,
        "default_voice": "Bruno",
        "weights": list(m.available_weights),
        "decoders": list(m.available_decoders),
        "sample_rate": m.sample_rate,
        "current_weights": STATE["weights"],
        "current_decoder": STATE["decoder"],
        "emotions": ["angry", "contemplative", "excited", "joyful", "mundane",
                     "nervous", "sad", "stern", "surprised", "tender"],
        "events": ["pause", "sigh", "gasp", "laugh", "giggle",
                   "sob", "scoff", "growl", "um", "gulp"],
        "advanced_defaults": m.advanced_options,
        "defaults": {"weights": DEFAULT_WEIGHTS, "decoder": DEFAULT_DECODER, "preset": DEFAULT_PRESET},
        "max_text_chars": MAX_TEXT_CHARS,
        "param_ranges": PARAM_RANGES,
        "loaded": {"weights": STATE["weights"], "decoder": STATE["decoder"]},
    }


def _sidecar_path(wav: Path) -> Path:
    return wav.with_suffix(".json")


def _write_sidecar(wav: Path, data: dict):
    """Write {name}.json next to the wav (atomic replace). Never fails the request."""
    try:
        side = _sidecar_path(wav)
        tmp = side.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, side)
    except Exception as e:
        log(f"sidecar write failed for {wav.name}: {type(e).__name__}: {e}")


def _read_sidecar(wav: Path) -> Optional[dict]:
    side = _sidecar_path(wav)
    if not side.is_file():
        return None
    try:
        data = json.loads(side.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None  # corrupt sidecar: treat as missing


def _history_item(p: Path) -> dict:
    item = _wav_meta(p)
    side = _read_sidecar(p)
    item["has_sidecar"] = side is not None
    for k in SIDECAR_FIELDS:
        item[k] = (side or {}).get(k)
    return item


@app.get("/api/history")
def history(limit: Optional[str] = None):
    """Generated clips, newest first. ?limit=N returns at most N items."""
    n = None
    if limit not in (None, ""):
        try:
            n = int(limit)
        except ValueError:
            raise HTTPException(400, "limit must be a positive integer")
        if n < 1:
            raise HTTPException(400, "limit must be a positive integer")
    files = []
    for p in OUTPUT.glob("*.wav"):
        try:
            files.append((p.stat().st_mtime, p))
        except OSError:
            pass
    files.sort(key=lambda t: t[0], reverse=True)
    out = []
    for _, p in files:
        if n is not None and len(out) >= n:
            break
        try:
            out.append(_history_item(p))
        except Exception:
            pass
    return out


@app.delete("/api/history/{name}")
def history_delete(name: str):
    """Delete one generated clip (and its sidecar) from the output folder."""
    path = _safe_child(OUTPUT, name)
    if path.suffix.lower() != ".wav" or not path.is_file():
        raise HTTPException(404, "Not found")
    path.unlink()
    _sidecar_path(path).unlink(missing_ok=True)
    return {"deleted": name}


@app.get("/api/download/{name}")
def download(name: str):
    path = _safe_child(OUTPUT, name)
    if path.suffix.lower() != ".wav" or not path.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(path, media_type="audio/wav", filename=name)


@app.post("/api/reload")
def reload_model(weights: str = Form(DEFAULT_WEIGHTS), decoder: str = Form(DEFAULT_DECODER)):
    """The only way to change the model. 202 {started, queued, target, model_state}: a reload
    sent while another is running is queued (latest wins)."""
    weights, decoder = (weights or DEFAULT_WEIGHTS).strip(), (decoder or DEFAULT_DECODER).strip()
    validate_model_choice(weights, decoder)
    started = start_model_load(weights, decoder)
    return JSONResponse(status_code=202, content={
        "started": started, "queued": not started,
        "target": {"weights": weights, "decoder": decoder}, "model_state": snapshot_state()})


@app.post("/api/generate")
async def generate(
    text: str = Form(...),
    voice: Optional[str] = Form(None),
    preset: str = Form(DEFAULT_PRESET),
    normalize: Optional[str] = Form(None),
    weights: Optional[str] = Form(None),
    decoder: Optional[str] = Form(None),
    temperature: Optional[str] = Form(None),
    top_k: Optional[str] = Form(None),
    top_p: Optional[str] = Form(None),
    min_p: Optional[str] = Form(None),
    max_new_tokens: Optional[str] = Form(None),
    seed: Optional[str] = Form(None),
):
    """Generate speech with a built-in voice and save it to output/ (plus a {name}.json sidecar).

    preset 'auto' (default) -> expressive when the text has expression tags, else stable.
    If the output looks like a loop, it is retried once without the reference prompt."""
    text = (text or "").strip()
    if not text:
        raise HTTPException(400, "Text is required")
    check_text_length(text, "Text")
    normalize = parse_bool(normalize, "normalize", True)
    temperature = parse_number(temperature, "temperature")
    top_k = parse_number(top_k, "top_k")
    top_p = parse_number(top_p, "top_p")
    min_p = parse_number(min_p, "min_p")
    max_new_tokens = parse_number(max_new_tokens, "max_new_tokens") or PARAM_RANGES["max_new_tokens"]["default"]
    seed = parse_number(seed, "seed")
    warnings: list = []

    m = get_ready_model()
    # Generate never switches models; only POST /api/reload does.
    loaded = {"weights": STATE["weights"], "decoder": STATE["decoder"]}
    req_w, req_d = (weights or "").strip() or None, (decoder or "").strip() or None
    if (req_w and req_w != loaded["weights"]) or (req_d and req_d != loaded["decoder"]):
        requested = {"weights": req_w or loaded["weights"], "decoder": req_d or loaded["decoder"]}
        validate_model_choice(requested["weights"], requested["decoder"], m)
        return JSONResponse(status_code=409, content={
            "detail": (f"The loaded model is weights={loaded['weights']}, decoder={loaded['decoder']}, "
                       f"but this request asked for weights={requested['weights']}, "
                       f"decoder={requested['decoder']}. Switch with POST /api/reload first, "
                       f"or leave weights/decoder out."),
            "loaded": loaded, "requested": requested})
    voice = (voice or "Bruno").strip()
    if voice not in set(m.available_voices) or voice.startswith("hybrid:"):
        raise HTTPException(400, f"Unknown voice {voice!r}.")
    preset_used = resolve_preset(preset, text, m)
    p_reason = preset_reason(preset, text)

    kw = {"voice": voice, "preset": preset_used, "normalize": normalize, "max_new_tokens": max_new_tokens}
    for k, v in (("temperature", temperature), ("top_k", top_k), ("top_p", top_p), ("min_p", min_p)):
        if v is not None:
            kw[k] = v
    if seed is not None:
        kw["advanced"] = {"seed": seed}

    out_path = OUTPUT / _safe_name(text, voice)
    retried, loop_detected = False, False
    t0 = time.perf_counter()
    _set(job={"kind": "generate", "started": datetime.now().isoformat(timespec="seconds")})
    try:
        duration = await run_in_threadpool(_run_generate, m, text, kw, out_path)
        if loop_suspected(m, text, duration):
            loop_detected = True
            first = round(duration, 2)
            kw["use_reference_prompt"] = False
            duration = await run_in_threadpool(_run_generate, m, text, kw, out_path)
            retried = True
            warnings.append(f"The first take looked like a loop ({first} s); "
                            f"retried without the reference prompt.")
    except GpuOOM:
        return oom_response("generation")
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Generation failed ({type(e).__name__}: {str(e)[:300]}). "
                                 f"See the server window for details.")
    finally:
        _set(job=None)
    gen_s = round(time.perf_counter() - t0, 2)
    side = {
        "text": text, "voice": voice, "preset": preset_used, "preset_requested": preset,
        "preset_reason": p_reason, "generation_seconds": gen_s, "audio_duration": round(duration, 2),
        "created": datetime.now().isoformat(timespec="seconds"),
        "weights": loaded["weights"], "decoder": loaded["decoder"], "seed": seed, "retried": retried,
    }
    _write_sidecar(out_path, side)
    out = _wav_meta(out_path)
    out.update(side)
    out.update({"preset_used": preset_used, "loop_detected": loop_detected, "warnings": warnings,
                "sidecar": _sidecar_path(out_path).name})
    return JSONResponse(out)


if __name__ == "__main__":
    import uvicorn
    if HOST not in ("127.0.0.1", "localhost", "::1"):
        log(f"WARNING: binding {HOST}. This server has no authentication; anyone who can reach "
            f"this address can use your GPU and read/delete generated clips.")
    log(f"binding http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
