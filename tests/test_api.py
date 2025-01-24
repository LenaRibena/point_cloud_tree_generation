import numpy as np
import pytest
from fastapi.testclient import TestClient

from tree.app import app

client = TestClient(app)


@pytest.mark.skip(reason="This would require mounting the models.")  # type: ignore
def test_read_generate() -> None:
    with TestClient(app) as client:
        response = client.get("/generate/flow")
        tree = np.array(response.json()["tree"])
        assert response.status_code == 200
        assert tree.shape == (4096, 3), "Flow tree shape is not correct"

        response = client.get("/generate/gauss")
        tree = np.array(response.json()["tree"])
        assert response.status_code == 200
        assert tree.shape == (4096, 3), "Gaussian tree shape is not correct"
