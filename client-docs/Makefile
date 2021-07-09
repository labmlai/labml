sphinx-help: ## Sphinx help
	@sphinx-build -M help source build

prepare: ## Prepare environment for build
	@pip3 install -r requirements.txt

build: ## Build docs
	@sphinx-build -M html source build

open: ## Open docs
	@open build/html/index.html

clean: ## Clean
	@echo "Removing everything under 'build'.."
	@rm -rf build/html/ build/doctrees
	@rm -rf ../docs

rebuild: clean build ## Rebuild docs

pages: rebuild ## Create pages
	@mkdir -p ../docs
	@cp -r build/html/* ../docs

help: ## Show this help.
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

.PHONY: build clean help sphinx-help pages rebuild prepare
.DEFAULT_GOAL := help
