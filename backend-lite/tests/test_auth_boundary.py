import importlib.util
from pathlib import Path

from fastapi.testclient import TestClient


def load_lite(monkeypatch):
    monkeypatch.delenv("FIREBASE_ADMIN_JSON_BASE64", raising=False)
    monkeypatch.delenv("FIREBASE_ADMIN_JSON", raising=False)
    monkeypatch.delenv("FIREBASE_ADMIN_JSON_PATH", raising=False)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/nonexistent/firebase-admin.json")
    path = Path(__file__).resolve().parents[1] / "app" / "main.py"
    spec = importlib.util.spec_from_file_location("fusion_lite_main", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_unconfigured_firebase_never_accepts_mock_or_arbitrary_bearer_tokens(monkeypatch):
    lite = load_lite(monkeypatch)
    client = TestClient(lite.app)

    response = client.get("/api/me", headers={"Authorization": "Bearer mock"})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "IDENTITY_VERIFICATION_UNAVAILABLE"
