from fastapi import FastAPI
from contextlib import asynccontextmanager

from vLLM_server import init_vLLM, stop_vLLM
from geo_recog import GeoRecog

geo_recog = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print('Starting vLLM server...')
    api_pool = init_vLLM()
    print('All vLLM server started.')
    global geo_recog
    geo_recog = GeoRecog(api_pool)
    yield
    print('Stopping vLLM server...')
    stop_vLLM()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/query")
def query(content: str):
    return geo_recog.query(content)
