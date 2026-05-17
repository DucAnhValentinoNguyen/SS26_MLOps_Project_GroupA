"""still need to modify."""

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def root():
    """A simple root endpoint."""
    return {"message": "ok"}
