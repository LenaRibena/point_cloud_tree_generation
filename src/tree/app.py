import pickle
from collections.abc import Generator

from fastapi import FastAPI

from tree.models.vae_flow import FlowVAE
from tree.models.vae_gaussian import GaussianVAE

generator: FlowVAE | GaussianVAE = None


def lifespan(app: FastAPI) -> Generator[None]:
    """Load generator."""
    global generator
    with open("generator.pkl", "rb") as file:
        generator = pickle.load(file)

    yield

    del generator


app = FastAPI(lifespan=lifespan)
