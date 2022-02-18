import git
import os
import logging
import time
from pathlib import Path
from time import strftime

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.logger import logger as flogger
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from labml_app import handlers
from labml_app.logger import logger
from labml_app import db


def get_static_path():
    package_path = Path(os.path.dirname(os.path.abspath(__file__)))
    app_path = package_path.parent.parent

    static_path = app_path / 'static'
    if static_path.exists():
        return static_path

    static_path = package_path.parent / 'static'
    if not static_path.exists():
        static_path = package_path / 'static'
    if not static_path.exists():
        raise RuntimeError(f'Static folder not found. Package path: {str(package_path)}')

    return static_path


STATIC_PATH = get_static_path()


def create_app():
    # disable logger
    flogger.setLevel(logging.ERROR)

    _app = FastAPI()

    db.init_db()

    def run_on_start():
        logger.info('initializing labml_app')

        try:
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha
            logger.error(f'THIS IS NOT AN ERROR: Server Deployed SHA : {sha}')
        except git.InvalidGitRepositoryError:
            pass

    run_on_start()

    return _app


app = create_app()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['https://app.labml.ai'],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

handlers.add_handlers(app)


@app.get('/')
def index():
    file_path = str(STATIC_PATH) + '/' + 'index.html'

    return FileResponse(file_path)


@app.get('/{file_path:path}')
def send_js(file_path: str):
    file_path = str(STATIC_PATH) + '/' + file_path
    if not Path(file_path).exists():
        file_path = str(STATIC_PATH) + '/' + 'index.html'

    return FileResponse(file_path)


@app.middleware('http')
async def log_process_time(request: Request, call_next):
    """
    Save time before each request
    TODO: Track time and content size in tracker. No need of logs"""
    timestamp = strftime('[%Y-%b-%d %H:%M]')
    request_start_time = time.time()
    logger.debug(f'time: {timestamp} uri: {request.url}')

    response: Response = await call_next(request)

    """Calculate and log execution time"""
    request_time = time.time() - request_start_time

    logger.info(f'PERF time: {request_time * 1000:.2f}ms uri: {request.url} method:{request.method}')

    # TODO check this; otherwise network.ts:43 Refused to get unsafe header "Authorization"
    response.headers['Access-Control-Expose-Headers'] = 'Authorization'
    # response.headers['Access-Control-Max-Age'] = str(60 * 24)

    return response


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5000)
