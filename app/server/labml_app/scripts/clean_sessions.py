from labml import monit
from labml_app.db import session, init_db

init_db()

session_keys = session.Session.get_all()
for session_key in monit.iterate('Sessions', session_keys):
    s = session_key.load()
    session.delete(s.session_uuid)
