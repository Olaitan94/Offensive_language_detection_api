import math

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient


def test_make_prediction(client: TestClient, input_text: str) -> None:
    # Given
    payload = {
        "inputs": input_text
    }

    # When
    response = client.post(
        "http://localhost:8001/api/v1/predict",
        json=payload,
    )

    # Then
    assert response.status_code == 200
    prediction_data = response.json()
    assert prediction_data["ACTUAL_TEXT"]
    assert prediction_data["PREDICTED_CLASS"]
    assert prediction_data["Probability"]
    assert prediction_data["PREDICTED_CLASS"] == 'OFFENSIVE'
