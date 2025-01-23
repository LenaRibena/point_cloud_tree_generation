from collections.abc import Generator
from enum import Enum

from fastapi import FastAPI
from pydantic import BaseModel

from tree.models.vae_flow import FlowVAE
from tree.models.vae_gaussian import GaussianVAE

# See s7_deployment\exercise_files\fastapi_solution.py
# and s8_monitoring\exercise_files\iris_fastapi_solution.py
# and s8_monitoring\exercise_files\sentiment_api.py
# and s8_monitoring\exercise_files\sentiment_client.py

models: dict[str, FlowVAE | GaussianVAE] = {}


class ModelEnum(Enum):
    flow = "flow"
    gauss = "gauss"


class GenerationOutput(BaseModel):
    tree: list[list[float]]


def lifespan(app) -> Generator[None]:
    """Load generator."""
    models[ModelEnum.flow] = FlowVAE.load("models/flow_model.pth")
    models[ModelEnum.gauss] = GaussianVAE.load("models/gaussian_model.pth")

    yield

    models.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/generate/{item_id}")
def generate(item_id: ModelEnum = ModelEnum.flow):
    tree = models[item_id].generate()
    return GenerationOutput(tree=tree.tolist())
