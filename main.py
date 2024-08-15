import threading
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

from utils.vLLM_server import init_vLLM, stop_vLLM
from utils.geo_recog import GeoRecog

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

class TaskItem(BaseModel):
    id: str
    content: str

@app.post("/geo/extract/text/batch")
def extract_text_batch(tasks: List[TaskItem]):
    try:
        res = {
            'code': 200,
            'message': 'Success',
            'data': []
        }

        threads = []
        ans_list = [None] * len(tasks)
        def worker(task, idx):
            ans_list[idx] = geo_recog.query(task.content)
        for i, task in enumerate(tasks):
            t = threading.Thread(target=worker, args=(task, i))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        for i in range(len(tasks)):
            ans = ans_list[i]
            if type(ans['province']) is not str:
                ans['province'] = ''
            if type(ans['city']) is not str:
                ans['city'] = ''
            res['data'].append({
                'id': tasks[i].id,
                'geo_code': ans['code'],
                'geo_text': ans['province'] + ans['city']
            })
    except:
        res = {
            'code': 500,
            'message': 'Internal Server Error',
            'data': []
        }
    return res
