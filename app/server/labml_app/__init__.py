import os
import subprocess


def start_server(ip: str, port: int):
    try:
        environment = os.environ.copy()
        environment['LABML_APP_SERVER_IP'] = ip
        environment['LABML_APP_SERVER_PORT'] = str(port)
        subprocess.run(
            [f"gunicorn --bind {ip}:{port} -w 1 -k uvicorn.workers.UvicornWorker labml_app.flask_app:app"],
            env=environment,
            shell=True,
        )
    except KeyboardInterrupt:
        pass
