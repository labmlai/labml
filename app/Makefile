.PHONY: help
.DEFAULT_GOAL := help

setup: ## install server and ui dependencies
	pip install pipenv
	sudo apt-get update
	sudo apt-get install npm
	cd ui && npm install
	cd server && pipenv install --ignore-pipfile

server-dev: ## start and watch server
	cd server &&  pipenv run uvicorn labml_app.flask_app:app --reload --host 0.0.0.0 --port 5005

server-prod: ## compile and start server in prod
	# pkill gunicorn
    # export PATH=~/miniconda/bin:$PATH
	cd server && pipenv install --ignore-pipfile && pipenv run gunicorn --bind 0.0.0.0:5000 -w 2 -k uvicorn.workers.UvicornWorker labml_app.flask_app:app --daemon

clean-db: ## clean float-project and move samples to float-project
	cd server &&  pipenv run python -m labml_app.scripts.clean_ups

compile: ## Compile JS
	rm -rf static
	mkdir -p static/js
	cp ui/src/index.html static/index.html
	cp ui/src/site.webmanifest static/site.webmanifest
	cp ui/images/favicon.ico static/favicon.ico
	cp -r ui/images static/
	cd ui && npm run build

clean-pycache:
	find . -depth -name "__pycache__" -type d -exec rm -rf {} +
	find . -depth -name ".DS_Store" -type f -exec rm -rf {} +
	find . -depth -name ".ipynb_checkpoints" -type d -exec rm -rf {} +

compile-prod: compile
	$(eval JS_CHECKSUM := $(shell md5sum static/js/bundle.min.js | cut -f 1 -d " "))
	$(eval CSS_CHECKSUM := $(shell md5sum static/css/style.css | cut -f 1 -d " "))
	sed -i 's/bundle.min.js/$(JS_CHECKSUM).min.js/g' static/index.html
	sed -i 's/bundle.min.js.map/$(JS_CHECKSUM).min.js.map/g' static/js/bundle.min.js
	sed -i 's/style.css/$(CSS_CHECKSUM).css/g' static/index.html
	sed -i 's/style.css.map/$(CSS_CHECKSUM).css.map/g' static/css/style.css
	mv static/js/bundle.min.js static/js/$(JS_CHECKSUM).min.js
	mv static/js/bundle.min.js.map static/js/$(JS_CHECKSUM).min.js.map
	mv static/css/style.css static/css/$(CSS_CHECKSUM).css
	mv static/css/style.css.map static/css/$(CSS_CHECKSUM).css.map


watch-ui: compile ## Compile and Watch JS & CSS
	cd ui && npm run watch

build-ui: compile ## build production ui

build-ui-package: ## build production ui for the package
	rm -rf static
	rm -rf ui/src/env.ts
	cp local_server_settings/ui/env.ts ui/src/
	mkdir -p static/js
	cp ui/src/index.html static/index.html
	cp ui/src/site.webmanifest static/site.webmanifest
	cp ui/images/favicon.ico static/favicon.ico
	cp -r ui/images static/
	cd ui && npm run build

clean: clean-pycache ## Cleanup builds
	rm -rf server/labml_app/static
	rm -rf server/build
	rm -rf server/dist
	rm -rf static

package: build-ui-package clean-pycache ## Build PIPy Package
	rm -rf server/labml_app/static
	cp -r static/ server/labml_app/static
	rm -f server/labml_app/analyses_settings.py
	rm -f server/labml_app/block_uuids.py
	rm -f server/labml_app/settings.py
	cp local_server_settings/server/*.py server/labml_app/
	cd server && python setup.py sdist bdist_wheel

check-package:  ## List contents of PIPy Package
	cd server && tar -tvf dist/*.tar.gz

upload-package: package  ## Upload PIPy Package
	twine upload server/dist/*

install: compile  ## Install from repo
	cd server && pip install -e .

uninstall: ## Uninstall
	pip uninstall labml_app labml


help: ## Show this help.
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

