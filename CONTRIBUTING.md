# Contributing

Thanks for helping. Bug reports, fixes and small improvements are all welcome.

## Workflow

1. Fork the repository and create a branch for your change, for example `fix/history-delete`.
2. Make your change. Keep it focused; one fix or feature per pull request.
3. Run the checks (see below).
4. Open a pull request describing what you changed and how you tested it. The CI check must pass before it can be merged.

## Running the checks

The tests do not need a GPU, PyTorch or the model; they use a stand-in model.

```
python -m venv .venv
.venv\Scripts\activate          (on Linux/macOS: source .venv/bin/activate)
pip install -r requirements-dev.txt
python tools/guard.py
python -m pytest -q tests
```

If you changed `webui/static/index.html`, also check the page in a browser with the real model if you can.

## Rules

- Never commit model weights (`*.onnx`, `*.safetensors`, `*.bin`, `*.pt`), audio files (`*.wav`, `*.mp3`), `.env` files, tokens or passwords.
- No personal or absolute paths such as a drive letter and your user folder. Use paths relative to the repository or settings in `.env`.
- Keep the server bound to `127.0.0.1` by default.
- `tools/guard.py` enforces most of this and runs in CI.

## Reporting bugs

Use the bug report template. Please include your GPU, NVIDIA driver version, the output of `setup.bat` and the console log from `start.bat`.
