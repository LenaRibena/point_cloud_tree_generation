from contextlib import asynccontextmanager
from enum import Enum
from typing import AsyncGenerator, List

from fastapi import FastAPI
from pydantic import BaseModel

from tree.modules.vae_flow import FlowVAE
from tree.modules.vae_gaussian import GaussianVAE

# See s7_deployment\exercise_files\fastapi_solution.py
# and s8_monitoring\exercise_files\iris_fastapi_solution.py
# and s8_monitoring\exercise_files\sentiment_api.py
# and s8_monitoring\exercise_files\sentiment_client.py


class ModelEnum(Enum):
    flow = "flow"
    gauss = "gauss"


models: dict[ModelEnum, FlowVAE | GaussianVAE] = {}


class GenerationOutput(BaseModel):  # type: ignore
    tree: List[List[float]]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load generator."""
    models[ModelEnum.flow] = FlowVAE.load("models/flow_model.pth")
    models[ModelEnum.gauss] = GaussianVAE.load("models/gaussian_model.pth")

    yield

    models.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/generate/{item_id}")  # type: ignore
def generate(item_id: ModelEnum = ModelEnum.flow) -> GenerationOutput:
    tree = models[item_id].generate()
    return GenerationOutput(tree=tree.tolist())
