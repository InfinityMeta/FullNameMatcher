import logging

from fastapi import FastAPI

from src.const import LOGS_FILE_PATH
from src.endpoints.base_endpoints import router as base_router
from src.endpoints.train_endpoints import router as train_router
from src.endpoints.eval_endpoints import router as eval_router


logging.basicConfig(
    filename=LOGS_FILE_PATH,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

app = FastAPI()
app.include_router(base_router)
app.include_router(train_router)
app.include_router(eval_router)
