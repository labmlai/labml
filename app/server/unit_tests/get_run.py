from labml_app.db import run, init_mongo_db

init_mongo_db()

r = run.get('f8850c6c99a211eebae53275fba9aeb4')

print(r.stdout)
print(r.stdout_unmerged)
