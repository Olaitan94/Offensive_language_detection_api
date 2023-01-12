from typing import Generator

import pytest
from fastapi.testclient import TestClient

from app.main import app


#This fixture will be used to provide argument value to the prediction test
@pytest.fixture()
def input_text():
    return "Yo! This is crazy."


@pytest.fixture()
def client() -> Generator:
    with TestClient(app) as _client:
        yield _client
        app.dependency_overrides = {}
