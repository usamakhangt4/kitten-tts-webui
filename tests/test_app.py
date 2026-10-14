def test_index_returns_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "KittenTTS" in r.text


def test_status_while_loading(client):
    s = client.get("/api/status").json()
    assert s["state"] == "loading"
    assert s["ready"] is False
    for key in ("device", "download", "error", "warnings"):
        assert key in s


def test_status_downloading_progress(client, app_module):
    app_module._set(phase="downloading",
                    download={"downloaded_bytes": 500, "total_bytes": 1000, "percent": 50.0, "estimated": True})
    s = client.get("/api/status").json()
    assert s["state"] == "downloading"
    assert s["download"]["percent"] == 50.0


def test_status_ready_reports_gpu(client, ready):
    s = client.get("/api/status").json()
    assert s["state"] == "ready"
    assert s["device"] == "cuda"
    assert s["gpu_name"] == "Test GPU"


def test_status_error_is_friendly(client, app_module):
    app_module._set(phase="error", error=app_module.friendly_load_error(
        ConnectionError("Max retries exceeded with url: huggingface.co")))
    s = client.get("/api/status").json()
    assert s["state"] == "error"
    assert "Hugging Face" in s["error"]
    assert "Traceback" not in s["error"]


def test_friendly_errors(app_module):
    oom = app_module.friendly_load_error(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    assert "LM Studio" in oom
    missing = app_module.friendly_load_error(ModuleNotFoundError("No module named 'kittenml'"))
    assert "setup.bat" in missing


def test_generate_503_while_loading(client):
    r = client.post("/api/generate", data={"text": "Hello", "voice": "Bruno"})
    assert r.status_code == 503
    assert r.json()["error"] == "Model is still loading, try again in a moment"


def test_generate_when_ready(client, ready):
    r = client.post("/api/generate", data={"text": "Hello [joyful] there", "voice": "Bruno"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["preset_used"] == "expressive"
    assert client.get(body["url"]).status_code == 200
    assert client.get("/api/download/" + body["name"]).status_code == 200


def test_generate_unknown_voice_is_400_with_error(client, ready):
    r = client.post("/api/generate", data={"text": "Hello", "voice": "Nobody"})
    assert r.status_code == 400
    assert "Unknown voice" in r.json()["error"]


def test_generate_oom_is_friendly(client, ready):
    def boom(text, **kw):
        raise RuntimeError("CUDA out of memory")
    ready.generate = boom
    r = client.post("/api/generate", data={"text": "Hello", "voice": "Bruno"})
    assert r.status_code == 503
    assert "LM Studio" in r.json()["error"]


def test_path_traversal_blocked(client):
    for url in ("/api/download/..%2Fapp.py", "/api/download/..%5Capp.py", "/output/..%2F..%2FREADME.md",
                "/static/..%2Fapp.py", "/static/../app.py"):
        r = client.get(url)
        assert r.status_code in (400, 404), url
        assert "fastapi" not in r.text.lower()


def test_private_features_are_gone(client):
    assert client.get("/api/custom-voices").status_code == 404
