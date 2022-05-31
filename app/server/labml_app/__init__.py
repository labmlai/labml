import subprocess


def start_server():
    try:
        subprocess.run(
            ["gunicorn --bind 0.0.0.0:5005 -w 1 -k uvicorn.workers.UvicornWorker labml_app.flask_app:app"],
            shell=True,
        )
    except KeyboardInterrupt:
        pass
