from labml_app.db import run, init_mongo_db

init_mongo_db()

r = run.get('f8850c6c99a211eebae53275fba9aeb4')

print(repr(r.stdout))
print(repr(r.stdout_unmerged))
