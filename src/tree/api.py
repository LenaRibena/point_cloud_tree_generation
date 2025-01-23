from __future__ import annotations

from collections.abc import Generator
from http import HTTPStatus

import anyio
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

from tree.generate_tree import TreeGenerator

generator = None


def lifespan(app: FastAPI) -> Generator[None]:
    """Load generator"""
    global generator
    generator = TreeGenerator()

    yield

    del model


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    """Simple root endpoint."""
    return {"Hello": "World"}


@app.get("/generate_tree/")
def generate_tree():
    """Generate a tree."""
    tree = generator()
    np.save("tree.npy", tree)
    return FileResponse("tree.npy", media_type="application/x-npy", filename="tree.npy")


@app.post("/cv_model/")
async def cv_model(data: UploadFile = File(...), h: None | int = 28, w: None | int = 28):
    """Simple function using open-cv to resize an image."""
    async with await anyio.open_file("image.jpg", "wb") as image:
        content = await data.read()
        image.write(content)
        image.close()

    img = cv2.imread("image.jpg")
    res = cv2.resize(img, (h, w))

    cv2.imwrite("image_resize.jpg", res)

    return {
        "input": data,
        "output": FileResponse("image_resize.jpg"),
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }


import os
import signal

import uvicorn
from fastapi import Response


@app.post("/shutdown")
def shutdown():
    print("Server shutting down...")
    os.kill(os.getpid(), signal.SIGTERM)
    return Response(status_code=200, content="Server shutting down...")


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
