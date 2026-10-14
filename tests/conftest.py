import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def app_module(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("kitten")
    os.environ["KITTEN_SKIP_MODEL_LOAD"] = "1"
    os.environ["KITTEN_OUTPUT_DIR"] = str(tmp / "output")
    os.environ["KITTEN_TMP_DIR"] = str(tmp / "tmp")
    os.environ["KITTEN_CACHE_DIR"] = str(tmp / "cache")
    os.environ["HF_HOME"] = str(tmp / "cache" / "hf")
    sys.path.insert(0, str(ROOT / "webui"))
    import app  # noqa: E402
    return app


class FakeModel:
    """Stands in for KittenTTS so tests run without torch, kittenml or a GPU."""
    available_voices = ["Bruno", "Bella", "Spanish"]
    available_weights = ["packed", "full"]
    available_decoders = ["default"]
    decode_presets = {"stable": {}, "expressive": {}}
    sample_rate = 24000
    advanced_options = {}

    def normalize_text(self, text):
        return text

    def generate(self, text, **kw):
        self.last_kwargs = kw
        return np.zeros(self.sample_rate // 2, dtype=np.float32)


@pytest.fixture()
def client(app_module):
    from fastapi.testclient import TestClient
    with TestClient(app_module.app) as c:
        app_module._model = None
        app_module._set(phase="loading", message="test", error=None, device=None, gpu_name=None,
                        device_warning=None, download=None)
        yield c


@pytest.fixture()
def ready(app_module):
    app_module._model = FakeModel()
    app_module._set(phase="ready", weights="packed", decoder="default", device="cuda",
                    gpu_name="Test GPU", error=None)
    return app_module._model
