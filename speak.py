"""KittenTTS 2 command-line example.

    python speak.py "Hello there." --voice Bruno --out output/hello.wav

The model (KittenML/kitten-tts-2) is downloaded from Hugging Face on first run
and cached under HF_HOME (default: ./cache/hf next to this script).
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
_hf = Path(os.environ.get("HF_HOME") or "cache/hf")
os.environ["HF_HOME"] = str(_hf if _hf.is_absolute() else ROOT / _hf)
_out_dir = Path(os.environ.get("KITTEN_OUTPUT_DIR") or "output")
_out_dir = _out_dir if _out_dir.is_absolute() else ROOT / _out_dir

p = argparse.ArgumentParser(description="KittenTTS 2 command-line speaker")
p.add_argument("text", nargs="?", help="text to speak (wrap in quotes)")
p.add_argument("--voice", default="Bruno", help="built-in voice name (default Bruno)")
p.add_argument("--out", default=str(_out_dir / "speech.wav"), help="output wav path")
p.add_argument("--preset", help="decode preset, e.g. stable or expressive")
p.add_argument("--no-normalize", action="store_true", help="use for non-English text")
p.add_argument("--list-voices", action="store_true", help="print voices and exit")
a = p.parse_args()
if not a.text and not a.list_voices:
    p.error("text is required (or use --list-voices)")

import soundfile as sf
import torch
from kittenml import KittenTTS

dev = "cuda" if torch.cuda.is_available() else "cpu"
try:
    m = KittenTTS("KittenML/kitten-tts-2", device=dev)
except TypeError:
    m = KittenTTS("KittenML/kitten-tts-2")
if a.list_voices:
    print("\n".join(m.available_voices))
    sys.exit(0)

kw = {"voice": a.voice}
if a.preset:
    kw["preset"] = a.preset
if a.no_normalize:
    kw["normalize"] = False

t = time.perf_counter()
audio = m.generate(a.text, **kw)
out = os.path.abspath(a.out)
os.makedirs(os.path.dirname(out), exist_ok=True)
sf.write(out, audio, m.sample_rate)
print(f"Saved {out} ({len(audio)/m.sample_rate:.1f}s audio, {time.perf_counter()-t:.1f}s on {dev})")
