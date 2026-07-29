"""Smoke tests for the application factory and health check route."""

import pytest

from app import create_app


@pytest.fixture
def client():
    app = create_app("testing")
    with app.test_client() as test_client:
        yield test_client


def test_health_check_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_app_factory_uses_testing_config():
    app = create_app("testing")
    assert app.config["TESTING"] is True
    assert app.config["DEBUG"] is True