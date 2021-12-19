from labml import monit
from labml_app.db import init_db, app_token

init_db()

count = 0

token_keys = app_token.AppToken.get_all()
for token_key in monit.iterate('Tokens', token_keys):
    if not token_key:
        continue
    at: app_token.AppToken = token_key.load()
    if not at:
        continue
    if at.user:
        continue

    app_token.delete(at)
