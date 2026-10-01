"""Phase 0 smoke tests: prove the environment and the serving image are sane."""

import importlib

import pytest
from fastapi.testclient import TestClient

from api.main import app

RUNTIME_PACKAGES = ["polars", "pyarrow", "lightgbm", "holidays", "fastapi", "pydantic", "uvicorn"]
TRAINING_PACKAGES = ["pandas", "sklearn", "shap", "mlflow", "geopandas", "shapely", "matplotlib"]


@pytest.mark.parametrize("name", RUNTIME_PACKAGES + TRAINING_PACKAGES)
def test_dependency_importable(name):
    assert importlib.import_module(name) is not None


def test_tripeta_package_importable():
    import tripeta

    assert tripeta.__version__ == "0.1.0"


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": None}
