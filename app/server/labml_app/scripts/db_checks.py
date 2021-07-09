from labml_app.logger import logger
from labml_app.db import Models

for s, m in Models:
    logger.info('checking: ' + str(m))
    model_keys = m.get_all()

    for model_key in model_keys:
        try:
            m = model_key.load()
        except Exception as e:
            logger.error(e)

# TODO add more checks
