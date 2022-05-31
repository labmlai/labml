import subprocess

from settings import PORT


def start_server():
    try:
        subprocess.run(
            [f"gunicorn --bind 0.0.0.0:{PORT} -w 1 -k uvicorn.workers.UvicornWorker labml_app.flask_app:app"],
            shell=True,
        )
    except KeyboardInterrupt:
        pass
