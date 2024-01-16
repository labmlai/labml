from labml_app.db import run, init_mongo_db

init_mongo_db()

r = run.get('5372ad8ab45311eeb035946dae1e1468')

print(r)
