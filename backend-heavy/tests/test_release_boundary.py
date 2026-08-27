import importlib

from fastapi.testclient import TestClient


def load_heavy(monkeypatch):
    monkeypatch.delenv("MODEL_RELEASE_APPROVED", raising=False)
    monkeypatch.delenv("MODEL_ARTIFACT_REVISION", raising=False)
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.delenv("HEAVY_SHARED_SECRET", raising=False)
    import app.main as heavy

    return importlib.reload(heavy)


def test_unapproved_or_unloaded_model_is_never_ready(monkeypatch):
    heavy = load_heavy(monkeypatch)
    client = TestClient(heavy.app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is False
    assert health.json()["release_configured"] is False

    readiness = client.get("/ready")
    assert readiness.status_code == 503
    assert readiness.json()["detail"]["code"] == "MODEL_NOT_READY"


def test_prediction_requires_internal_credential_and_approved_loaded_model(monkeypatch):
    heavy = load_heavy(monkeypatch)
    client = TestClient(heavy.app)
    payload = {"sequence": "ACDEFG"}

    no_credential = client.post("/predict", json=payload)
    assert no_credential.status_code == 503
    assert no_credential.json()["detail"]["code"] == "INTERNAL_AUTH_NOT_CONFIGURED"

    monkeypatch.setenv("HEAVY_SHARED_SECRET", "test-internal-credential")
    no_release = client.post("/predict", json=payload, headers={"X-Render-Secret": "test-internal-credential"})
    assert no_release.status_code == 503
    assert no_release.json()["detail"]["code"] == "MODEL_NOT_READY"
    assert heavy._model is None
